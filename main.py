import os
import random
import pickle
import logging
import pandas as pd
import argparse
from utils.utils import *
from time import time
from data.complex_dataset import MDDTADataset
from data.complex_dataloader import MDDTADataLoader
from model.predictor import MDDTAPredictor
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
import os
import pandas as pd
import torch
from time import time
import logging
from sklearn.model_selection import train_test_split

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

def save_metrics_to_csv(file_path, metrics):
    df = pd.DataFrame([metrics])
    if not os.path.exists(file_path):
        df.to_csv(file_path, mode='w', index=False, header=True)
    else:
        df.to_csv(file_path, mode='a', index=False, header=False)


def affinity_criterion_pos(pred_affinity, y_affinity, native_mask, decoy_weight, decoy_gap=1.0):
    decoy_weight_1 = 1.0 - decoy_weight

    pos_loss = (pred_affinity[native_mask] - y_affinity[native_mask]) ** 2

    adjusted_target = y_affinity[~native_mask] - decoy_gap * decoy_weight_1[~native_mask]
    adjusted_target = adjusted_target.relu()
    
    neg_loss = (pred_affinity[~native_mask] - adjusted_target) ** 2

    pos_loss_mean = pos_loss.sum()  / (pos_loss.size(0) + 1e-8)
    neg_loss_mean = neg_loss.sum() / (neg_loss.size(0) + 1e-8)

    return pos_loss_mean,neg_loss_mean


def train(epoch, model, device, loader, optimizer, scheduler, clip_norm, beta1=1.0, beta2=1.0, beta3=1.0, threshold=10.0, accum_steps=10):
    model.train()
    affinity_batch_loss = 0.0

    optimizer.zero_grad()
    for step, batch in tqdm(enumerate(loader),total=len(loader)):

        prot_batch, lig_batch, native_batch, affinity_batch, weight_batch = batch

        native_batch = native_batch.to(device)
        affinity_batch = affinity_batch.to(device)
        weight_batch = weight_batch.to(device)

        prot_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in prot_batch]
        lig_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in lig_batch]

        affinity_pre = model(
            prot_batch, lig_batch
        )

        affinity_pos_loss,neg_loss = affinity_criterion_pos(affinity_pre, affinity_batch, native_batch, weight_batch) 

        affinity_loss = affinity_pos_loss +  0.9*neg_loss

        affinity_loss.backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        affinity_batch_loss += affinity_loss.item()

    scheduler.step()

    metrics = {
        "epoch": epoch,

        "affinity_loss": round(affinity_batch_loss/(step+1), 5),
    }

    return metrics

@torch.no_grad()
def test(dataset, epoch, model, device, loader, beta1=1.0, beta2=1.0, beta3=2.0, threshold=10.0):
    model.eval()

    affinity_list, affinity_pred_list = [], []

    affinity_batch_loss = 0.0

    for step, batch in tqdm(enumerate(loader),total=len(loader)):

        prot_batch, lig_batch,  native_batch, affinity_batch, _ = batch
        #contact_map = contact_map.to(device)
        native_batch = native_batch.to(device)
        affinity_batch = affinity_batch.to(device)
        #weight_batch = weight_batch.to(device)
        prot_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in prot_batch]
        lig_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in lig_batch]

        affinity_pre = model(
            prot_batch, lig_batch
        )

        affinity_loss = torch.nn.MSELoss()(affinity_pre, affinity_batch)

        affinity_batch_loss += affinity_loss.item()

        affinity_list.append(affinity_batch[native_batch])
        affinity_pred_list.append(affinity_pre[native_batch].detach())


    affinity = torch.cat(affinity_list)
    affinity_pred = torch.cat(affinity_pred_list)

    metrics = {
        "epoch": epoch,

        "affinity_loss": round(affinity_batch_loss/(step+1), 5),
    }


    metrics.update(affinity_metrics(affinity_pred.detach().cpu(), affinity.detach().cpu()))

    return metrics

def read_data(data, decoy_weight, batch_size, protein_emb_path, protein_coords_path, 
              ligand_emb_path, cache_path, device=0, shuffle=True):
    dataset = MDDTADataset(
        data, decoy_weight, cache_path, protein_emb_path,
        protein_coords_path, ligand_emb_path, save_hetero=True, device=device
    )
    
    return MDDTADataLoader(
        dataset, protein_emb_path, protein_coords_path, ligand_emb_path,
        batch_size=batch_size, shuffle=shuffle
    )


def run(seed, args):
    save_path = os.path.join(args.save_path, f'seed_{seed}')
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    data_train_valid = pd.read_csv(args.data_path)
    test_data = pd.read_csv(args.data_test_path)

    process_data_train_valid = data_train_valid[~(((data_train_valid['pocket_len'] < 10) | (data_train_valid['pocket_len'] > 300) | (data_train_valid['ligand_len'] > 100)))]

    print('process_data_train_valid:',len(process_data_train_valid))
    train_data, val_data = train_test_split(process_data_train_valid[process_data_train_valid['native'] == True], test_size=0.1, random_state=42)

    train_data = pd.concat([train_data, process_data_train_valid[process_data_train_valid['native'] == False]])
    print(f"Train data: {len(train_data)}, Val data: {len(val_data)}, Test data: {len(test_data)}")
    s = time()
    #train_val_loder = read_data(process_data_train_valid, args.decoy_weight, args.batch_size, args.protein_emb_path, args.protein_coords_path, args.ligand_emb_path, args.cache_path["train_val"], device=device, shuffle=True)
    train_loader = read_data(train_data, args.decoy_weight, args.batch_size, args.protein_emb_path, args.protein_coords_path, args.ligand_emb_path, args.cache_path["train"], device=device, shuffle=True)
    val_loader = read_data(val_data, args.decoy_weight, args.eval_batch_size, args.protein_emb_path, args.protein_coords_path, args.ligand_emb_path, args.cache_path["val"], device=device, shuffle=False)
    test_loader = read_data(test_data, args.decoy_weight, args.eval_batch_size, args.protein_emb_path, args.protein_coords_path, args.ligand_emb_path, args.cache_path["test"], device=device, shuffle=False)    

    logging.info(f"Data loading time: {time() - s:.4f}s")

    model = MDDTAPredictor(args).to(device)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_curve, val_curve, test_curve = [], [], []
    best_val_epoch, best_val_metric = 0, float('inf')
    epoch_not_improved = 0

    for epoch in range(1, args.epochs + 1):
        train_perf = train(epoch, model, device, train_loader, optimizer, scheduler, args.clip_norm, args.contact_loss_weight, args.ligpos_loss_weight, args.affinity_loss_weight, args.threshold)
        train_curve.append(train_perf)

        if epoch % args.eval_step == 0:
            val_perf = test('val', epoch, model, device, val_loader, args.contact_loss_weight, args.ligpos_loss_weight, args.affinity_loss_weight, args.threshold)
            test_perf = test('test', epoch, model, device, test_loader, args.contact_loss_weight, args.ligpos_loss_weight, args.affinity_loss_weight, args.threshold)

            print_metrics("Train", train_perf)
            print_metrics("Val", val_perf)
            print_metrics("Test", test_perf)

            save_metrics_to_csv(f"{save_path}/train_metric.csv", train_perf)
            save_metrics_to_csv(f"{save_path}/val_metric.csv", val_perf)
            save_metrics_to_csv(f"{save_path}/test_metric.csv", test_perf)
            val_curve.append(val_perf)
            test_curve.append(test_perf)
            if val_perf["rmse"] < best_val_metric:
                epoch_not_improved = 0
                best_val_epoch = epoch
                best_val_metric = val_perf["rmse"]
                if args.save_checkpoint:
                    best_val_model_path = os.path.join(save_path, f"{epoch}.pt")
                    torch.save({"model": model.state_dict()}, best_val_model_path)
            else:
                epoch_not_improved += 1
                print(f"Epoch not improved: {epoch}")
            if epoch_not_improved >=10:
                break
            
            print(f"Best val rmse score: {best_val_metric}, best epoch: {best_val_epoch}")
            
        else:
            print(f"Train epoch {epoch}: {train_perf}")
        print('-------------------------------------------------')

    plot_loss_curves(train_curve, val_curve, save_path)
    print('-------------------------------------------------')

    print_metrics('Best val', val_curve[best_val_epoch - 1])

    print_metrics('Best test', test_curve[best_val_epoch - 1])

    logging.info(f"Overall time: {time() - s:.4f}s")
    return test_curve[best_val_epoch - 1]


def main(args):

    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    root = os.path.dirname(os.path.abspath(__file__))
    print(root)

    checkpoint_folder = "MDDTA"+'-'+ str(args.batch_size) + '-' + str(args.lr) + '-'+ timestamp
    args.save_path = os.path.join(root, 'checkpoints', checkpoint_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    args.dataset_root = os.path.join(root, 'dataset')
    if not os.path.exists(os.path.join(args.dataset_root, 'cache')):
        os.makedirs(os.path.join(args.dataset_root, 'cache'), exist_ok=True)

    args.protein_emb_path = os.path.join(args.dataset_root, f"protein_emb_data.h5")
    args.protein_coords_path = os.path.join(args.dataset_root, f"protein_coords_data.h5")
    args.ligand_emb_path = os.path.join(args.dataset_root, f"ligand_data.h5")

    args.data_path = os.path.join(args.dataset_root, f"PDBbind_2020_MD.csv")
    args.data_test_path = os.path.join(args.dataset_root, f"CASF2016.csv")
    args.cache_path = {"train": os.path.join(args.dataset_root, f"cache/train.pkl"),
                        "val": os.path.join(args.dataset_root, f"cache/val.pkl"),
                        "test": os.path.join(args.dataset_root, f"cache/test.pkl")}

    logging.info(f"Save path: {args.save_path}")
    print(args)

    test_metric = run(args.seed, args)
    print(f"Test metric: {test_metric}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, nargs="+", default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--save_checkpoint", action="store_true", default=True)

    """training parameter"""
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128) #128
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=256) #256
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--clip_norm', type=float, default=1.5)
    parser.add_argument('--pos_weight', type=float, default=4.)
    parser.add_argument('--threshold', type=float, default=10.0)
    parser.add_argument('--contact_loss_weight', type=float, default=0.1)
    parser.add_argument('--ligpos_loss_weight', type=float, default=0.5)
    parser.add_argument('--affinity_loss_weight', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    """model parameter"""
    parser.add_argument('--model', type=str, default='faformer')
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--protein_n_layers', type=int, default=2)
    parser.add_argument('--ligand_n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--contact_hidden_dim', type=int, default=128)
    parser.add_argument('--contact_edge_hdim', type=int, default=128)
    parser.add_argument('--affinity_hidden_dim', type=int, default=128)
    parser.add_argument('--affinity_edge_hdim', type=int, default=128)
    parser.add_argument('--drop_ratio', type=float, default=0.2)
    parser.add_argument('--attn_drop_ratio', type=float, default=0.2)
    parser.add_argument('--top_k_neighbors', type=int, default=30)
    parser.add_argument('--embedding_grad_frac', type=float, default=1)
    parser.add_argument('--max_dist', type=float, default=1e5)
    parser.add_argument('--act', type=str, default='swiglu')
    parser.add_argument("--edge_residue", default=True)

    parser.add_argument('--decoy_weight', type=float, default=1)
    parser.add_argument('--prot_emb_dim', type=int, default=1280)
    parser.add_argument('--lig_emb_dim', type=int, default=300)

    args = parser.parse_args()
    main(args)

