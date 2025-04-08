import h5py
import torch
from torch.utils.data import Dataset
import os
import pickle
import os
import logging
import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from tqdm.auto import tqdm
import torch.multiprocessing as tmp
import multiprocessing as mp
from contextlib import contextmanager
from itertools import starmap


class MDDTADataset(Dataset):
    def __init__(self, data, decoy_weight, hetero_cache_path, protein_emb_path=None, protein_coords_path=None, ligand_emb_path=None, save_hetero=True, device=0):
        self.data = data
        self.device = device
        
        if hetero_cache_path is None or not os.path.exists(hetero_cache_path):
            logging.info("Preprocessing data...")
            self.complex_graphs = self._preprocess(self.data, hetero_cache_path, decoy_weight, protein_emb_path, protein_coords_path, ligand_emb_path, save_hetero=save_hetero)
        else:
            with open(hetero_cache_path, 'rb') as f:
                self.complex_graphs = pickle.load(f)

        print(f"Total {len(self.complex_graphs)} complexes")  

    def _preprocess(self, data, save_path, decoy_weight, protein_embeddings_path=None, protein_coords_path=None, ligand_embeddings_path=None, save_hetero=True):
        protein_coords = h5py.File(protein_coords_path, 'r')
        protein_emb = h5py.File(protein_embeddings_path, 'r')
        ligand_coords = h5py.File(ligand_embeddings_path, 'r')

        complex_graphs_all = []
        for pdb in tqdm(data['pdb']):  
            data_pdb = data[data['pdb'] == pdb]

            complex_graph = HeteroData()
            complex_graph.protein_coord = protein_coords['coords'][pdb][:]
            complex_graph.ligand_coord =ligand_coords['coords'][pdb][:]
            complex_graph.protein_emb = protein_emb['embedding'][pdb][:]
            complex_graph.ligand_emb = ligand_coords['embeddings'][pdb][:]
            
            complex_graph.protein_len = complex_graph.protein_coord.shape[0]
            complex_graph.ligand_len = complex_graph.ligand_coord.shape[0]
            complex_graph.pdb = pdb
            complex_graph.affinity = torch.tensor(data_pdb['affinity'].values, dtype=torch.float)
            complex_graph.native = torch.tensor(data_pdb['native'].values, dtype=torch.bool)
            complex_graph.weight = torch.tensor(data_pdb['decoy_weight'].values * decoy_weight, dtype=torch.float) if complex_graph.native != 1 else torch.tensor(data_pdb['decoy_weight'].values, dtype=torch.float)
            # 保存到缓存
            complex_graphs_all.append(complex_graph)

        if save_hetero:
            with open(save_path, 'wb') as f:
                pickle.dump(complex_graphs_all, f)
        return complex_graphs_all

    def __len__(self):
        return len(self.complex_graphs)

    def __getitem__(self, idx):
        return self.complex_graphs[idx]



class MDDTADataset_FDA(Dataset):
    def __init__(self, data, decoy_weight, hetero_cache_path, protein_emb_path=None, protein_coords_path=None, ligand_emb_path=None, save_hetero=True, device=0):
        self.data = data
        self.device = device

        if hetero_cache_path is None or not os.path.exists(hetero_cache_path):
            logging.info("Preprocessing data...")
            self.complex_graphs = self._preprocess(self.data, hetero_cache_path, decoy_weight, protein_emb_path, protein_coords_path, ligand_emb_path, save_hetero=save_hetero)
        else:
            with open(hetero_cache_path, 'rb') as f:
                self.complex_graphs = pickle.load(f)

        print(f"Total {len(self.complex_graphs)} complexes")

    def _preprocess(self, data, save_path, decoy_weight, protein_embeddings_path=None, protein_coords_path=None, ligand_embeddings_path=None, save_hetero=True):
        protein_coords = h5py.File(protein_coords_path, 'r')
        protein_emb = h5py.File(protein_embeddings_path, 'r')
        ligand_coords = h5py.File(ligand_embeddings_path, 'r')

        complex_graphs_all = []
        for pdb in tqdm(data['Drugbank ID']):
            data_pdb = data[data['Drugbank ID'] == pdb]

            complex_graph = HeteroData()
            complex_graph.protein_coord = protein_coords['coords'][pdb][:]
            complex_graph.ligand_coord =ligand_coords['coords'][pdb][:]
            complex_graph.protein_emb = protein_emb['embedding'][pdb][:]
            complex_graph.ligand_emb = ligand_coords['embeddings'][pdb][:]
            
            complex_graph.protein_len = complex_graph.protein_coord.shape[0]
            complex_graph.ligand_len = complex_graph.ligand_coord.shape[0]
            complex_graph.pdb = pdb
            complex_graph.affinity = torch.tensor(data_pdb['Y'].values, dtype=torch.float)
            #complex_graph.native = torch.tensor(data_pdb['native'].values, dtype=torch.bool)
            #complex_graph.weight = torch.tensor(data_pdb['decoy_weight'].values * decoy_weight, dtype=torch.float) if complex_graph.native != 1 else torch.tensor(data_pdb['decoy_weight'].values, dtype=torch.float)
            #complex_graph.entropy_weight = torch.tensor(data_pdb['entropy_weight'].values, dtype=torch.float) if complex_graph.native != 1 else torch.tensor(data_pdb['entropy_weight'].values, dtype=torch.float)
            complex_graphs_all.append(complex_graph)

        if save_hetero:
            with open(save_path, 'wb') as f:
                pickle.dump(complex_graphs_all, f)
        return complex_graphs_all

    def __len__(self):
        return len(self.complex_graphs)

    def __getitem__(self, idx):
        return self.complex_graphs[idx]
