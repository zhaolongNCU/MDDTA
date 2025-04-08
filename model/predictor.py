import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder.transformer import FAFormer
from model.encoder.config import FAFormerConfig


def get_geo_encoder(args,emb_dim,hidden_dim,edge_hidden_dim):
    if args.model == "faformer":
        return FAFormer(
            FAFormerConfig(
                d_input=emb_dim,
                d_model=hidden_dim,
                d_edge_model=edge_hidden_dim,
                activation=args.act,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                proj_drop=args.drop_ratio,
                attn_drop=args.attn_drop_ratio,
                n_neighbors=args.top_k_neighbors,
                valid_radius=args.max_dist,
                embedding_grad_frac=args.embedding_grad_frac,
                n_pos=300,
            )
        )
    else:
        raise ValueError("Invalid geo framework: {}".format(args.model))

class AffinityDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(AffinityDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)
        self.dropout = nn.Dropout(0.4)
        self.leakrelu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))
        return x

class MDDTAPredictor(nn.Module):
    def __init__(self, args):
        super(MDDTAPredictor, self).__init__()

        self.prot_emb_mlp = nn.Sequential(
            nn.Linear(args.prot_emb_dim, 1024),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 256),
        )

        self.lig_emb_mlp = nn.Sequential(
            nn.Linear(args.lig_emb_dim, 256),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
        )
        self.protein_FAformer = FAFormer(
            FAFormerConfig(
                d_input=256,
                d_model=128,
                d_edge_model=128,
                n_layers=2,
                n_neighbors=30,
                n_heads=args.n_heads,
                proj_drop=0.3,
                attn_drop=0.3,
                valid_radius=args.max_dist,
                activation=args.act,
                embedding_grad_frac=args.embedding_grad_frac,
                n_pos=300,
            ))
        self.ligand_FAformer = FAFormer(
            FAFormerConfig(
                d_input=256,
                d_model=128,
                d_edge_model=128,
                n_layers=2,
                n_neighbors=30,
                n_heads=args.n_heads,
                proj_drop=0.3,
                attn_drop=0.3,
                valid_radius=args.max_dist,
                activation=args.act,
                embedding_grad_frac=args.embedding_grad_frac,
                n_pos=100,
            ))

        self.threshold = args.threshold   #6.0

        self.complex_FAformer = FAFormer(
            FAFormerConfig(
                d_input=128,
                d_model=128,
                d_edge_model=128,
                n_layers=2,
                n_neighbors=30,
                n_heads=args.n_heads,
                proj_drop=0.3,
                attn_drop=0.3,
                valid_radius=args.max_dist,
                activation=args.act,
                embedding_grad_frac=args.embedding_grad_frac,
                n_pos=400,
            ))

        #self.bias = torch.nn.Parameter(torch.ones(1))
        self.leaky = torch.nn.LeakyReLU()
        self.aff1_linear = nn.Linear(1, 1)
        self.linear_energy = nn.Linear(128, 1)
        self.gate_linear = nn.Linear(128, 1)

        self.affinity_predictor = AffinityDecoder(128,1280,512,1)
        self.layernorm = torch.nn.LayerNorm(128)

    def get_distance_map(self, h_mol1, h_mol2): 
        B, N_mol1, N_mol2 = h_mol1.shape[0], h_mol1.shape[1], h_mol2.shape[1]
        h_mol1 = h_mol1.unsqueeze(2).expand(-1, -1, N_mol2, -1)
        h_mol2 = h_mol2.unsqueeze(1).expand(-1, N_mol1, -1, -1)
        return torch.cat([h_mol1, h_mol2], dim=-1)

    def _decenter(self, X, mask):
        # X: [B, N, 3]
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X_decenter = X - center.unsqueeze(1) * mask
        return X_decenter, center

    def forward(self, mol1, mol2, ):
        prot_embs, prot_coords = mol1

        lig_embs, lig_coords = mol2

        B, N_prot, N_lig = prot_embs.shape[0], prot_embs.shape[1], lig_embs.shape[1]
        prot_mask,lig_mask = (prot_embs.sum(dim=-1) != 0).float(), (lig_embs.sum(dim=-1) != 0).float()

        decenter_coords_prot, center_coords_prot = self._decenter(prot_coords, prot_mask)
        decenter_coords_lig, center_coords_lig = self._decenter(lig_coords, lig_mask)
        
        prot_embs = self.prot_emb_mlp(prot_embs)
        lig_embs = self.lig_emb_mlp(lig_embs)

        FA_prot_embs,FA_prot_coords = self.protein_FAformer(prot_embs, decenter_coords_prot, ~prot_mask.bool())
        FA_lig_embs,FA_lig_coords = self.ligand_FAformer(lig_embs, decenter_coords_lig, ~lig_mask.bool())
        FA_prot_embs = self.layernorm(FA_prot_embs)
        FA_lig_embs = self.layernorm(FA_lig_embs)

        complex_embs = torch.cat([FA_prot_embs, FA_lig_embs], dim=1)
        #complex_embs = self.layernorm_com(self.complex_nn(complex_embs))
        complex_coords = torch.cat([prot_coords, lig_coords], dim=1)
        #complex_coords = torch.cat([decenter_coords_prot, decenter_coords_lig], dim=1)
        #complex_coords = torch.cat([prot_coords_output, lig_coords_output], dim=1)  #torch.Size([8, 84, 3])
        complex_mask = torch.cat([prot_mask, lig_mask], dim=1)
        complex_coords_decenter, _ = self._decenter(complex_coords, complex_mask)
        complex_embs1, _ = self.complex_FAformer(complex_embs, complex_coords_decenter, ~complex_mask.bool())
        #complex_embs2, _ = self.complex_FAformer2(complex_embs1+complex_embs, complex_coords_decenter, ~complex_mask.bool())
        #complex_embs3, _ = self.complex_FAformer3(complex_embs2, complex_coords_decenter, ~complex_mask.bool())
        #complex_embs = complex_embs[complex_mask.unsqueeze(-1).expand(-1, -1, complex_embs.size(-1)).bool()].view(B,-1,complex_embs.size(-1))  #torch.Size([8, 84, 128])
        complex_embs = (complex_embs1*complex_mask.unsqueeze(-1)).sum(dim=1)/complex_mask.unsqueeze(-1).sum(dim=1)
        affinity = self.affinity_predictor(complex_embs)

        return affinity

