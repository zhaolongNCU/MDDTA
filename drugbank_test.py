import os
import random
import pickle
import logging
import pandas as pd
import argparse
from utils.utils import *
from time import time
from data.complex_dataset import MDDTADataset_FDA
from data.complex_dataloader import MDDTADataLoader_FDA
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


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 


def affinity_criterion_pos(pred_affinity, y_affinity, native_mask, decoy_weight, decoy_gap=1.0):
    decoy_weight_1 = 1.0 - decoy_weight

    pos_loss = (pred_affinity[native_mask] - y_affinity[native_mask]) ** 2

    adjusted_target = y_affinity[~native_mask] - decoy_gap * decoy_weight_1[~native_mask]
    adjusted_target = adjusted_target.relu()
    
    neg_loss = (pred_affinity[~native_mask] - adjusted_target) ** 2

    pos_loss_mean = pos_loss.sum()  / (pos_loss.size(0) + 1e-8)
    neg_loss_mean = neg_loss.sum() / (neg_loss.size(0) + 1e-8)

    return pos_loss_mean,neg_loss_mean


@torch.no_grad()
def test(dataset, model, device, loader, beta1=1.0, beta2=1.0, beta3=2.0, threshold=10.0):
    model.eval()
    affinity_list, affinity_pred_list = [], []

    affinity_batch_loss = 0.0
    
    for step, batch in tqdm(enumerate(loader),total=len(loader)):
        prot_batch, lig_batch,affinity_batch = batch

        affinity_batch = affinity_batch.to(device)
        prot_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in prot_batch]
        lig_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in lig_batch]

        affinity_pre = model(
            prot_batch, lig_batch
        )

        affinity_loss = torch.nn.MSELoss()(affinity_pre, affinity_batch)

        affinity_batch_loss += affinity_loss.item()

        affinity_list.append(affinity_batch)
        affinity_pred_list.append(affinity_pre.detach())

    affinity = torch.cat(affinity_list)
    affinity_pred = torch.cat(affinity_pred_list)

    return affinity_pred.squeeze(-1).detach().cpu(), affinity.squeeze(-1).detach().cpu()

def read_data(data, decoy_weight, batch_size, protein_emb_path, protein_coords_path, 
              ligand_emb_path, cache_path, device=0, shuffle=True):
    dataset = MDDTADataset_FDA(
        data, decoy_weight, cache_path, protein_emb_path,
        protein_coords_path, ligand_emb_path, save_hetero=True, device=device
    )
    
    return MDDTADataLoader_FDA(
        dataset, protein_emb_path, protein_coords_path, ligand_emb_path,
        batch_size=batch_size, shuffle=shuffle
    )

def run(seed, args):
    save_path = os.path.join(args.save_path, f'seed_{seed}')
    os.makedirs(save_path, exist_ok=True)
    device = args.device
    test_data = pd.read_csv(args.data_test_path)

    print(f"Test data: {len(test_data)}")

    s = time()
    test_loader = read_data(test_data, args.decoy_weight, args.eval_batch_size, args.protein_emb_path, args.protein_coords_path, args.ligand_emb_path, args.cache_path["test"], device=device, shuffle=False)    

    logging.info(f"Data loading time: {time() - s:.4f}s")

    # 初始化模型
    model = MDDTAPredictor(args).to(device)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    checkpoint = torch.load(args.pt)
    model.load_state_dict(checkpoint["model"])

    affinity_pre,affinity = test('test', model, device, test_loader, args.contact_loss_weight, args.ligpos_loss_weight, args.affinity_loss_weight, args.threshold)
    print(affinity_pre.shape,affinity.shape)

    test_data['affinity_pre'] = affinity_pre.numpy()
    test_data.to_csv(os.path.join(save_path, 'test_result.csv'), index=False)
    return affinity_pre,affinity


def save_metrics_to_csv(file_path, metrics):
    df = pd.DataFrame([metrics])
    if not os.path.exists(file_path):
        df.to_csv(file_path, mode='w', index=False, header=True)
    else:
        df.to_csv(file_path, mode='a', index=False, header=False)


def main(args):
    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    root = os.path.dirname(os.path.abspath(__file__))
    print(root)
    checkpoint_folder = "drugbank_test"+'-'+ str(args.batch_size) + '-' + str(args.lr) + '-'+ timestamp
    args.save_path = os.path.join(root, 'checkpoints', checkpoint_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    args.dataset_root = os.path.join(root, 'drugbank_dataset')
    if not os.path.exists(os.path.join(args.dataset_root, 'cache')):
        os.makedirs(os.path.join(args.dataset_root, 'cache'), exist_ok=True)

    args.save_path = os.path.join(args.save_path, checkpoint_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    if not os.path.exists(os.path.join(args.dataset_root, 'cache')):
        os.makedirs(os.path.join(args.dataset_root, 'cache'), exist_ok=True)

    args.protein_emb_path = os.path.join(args.dataset_root, f"protein_emb_data.h5")
    args.protein_coords_path = os.path.join(args.dataset_root, f"protein_coords_data.h5")
    args.ligand_emb_path = os.path.join(args.dataset_root, f"ligand_data.h5")
    args.data_test_path = os.path.join(args.dataset_root, f"drugbank_test.csv")
    args.pt = os.path.join(root, f"MDDTA.pt")

    args.cache_path = {"test": os.path.join(args.dataset_root, f"cache/drugbank_test.pkl")}

    logging.info(f"Save path: {args.save_path}")
    print(args)
    affinity_pre,affinity = run(args.seeds, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs="+", default=[0])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--save_checkpoint", action="store_true", default=True)

    """training parameter"""
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=256)
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
    #parser.add_argument('--edge_hidden_dim', type=int, default=64)
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

