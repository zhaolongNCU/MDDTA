from torch.utils.data import DataLoader
import h5py
import torch
import torch.nn.functional as F

class MDDTADataLoader(DataLoader):
    def __init__(self, complex_dataset, protein_emb_path=None, protein_coords_path=None, ligand_emb_path=None, **kwargs):
        self.complex_dataset = complex_dataset
        self.protein_emb_path = protein_emb_path
        self.protein_coords_path = protein_coords_path
        self.ligand_coords_path = ligand_emb_path

        array = torch.arange(len(complex_dataset)).long()

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=array, **kwargs)

    def __collate_fn__(self, batch_idx):

        B = len(batch_idx)
        sampled_batch = [self.complex_dataset[i] for i in batch_idx]

        protein_max1 = max([b.protein_len for b in sampled_batch])
        ligand_max2 = max([b.ligand_len for b in sampled_batch])

        prot_embs, prot_coords, lig_embs, lig_coords = [], [], [], []
        natives, affinitys, weights = [], [], []

        for i, complex in enumerate(sampled_batch):
            pdb_id = complex.pdb
            n_prot, n_lig = complex.protein_len, complex.ligand_len

            prot_embs.append(F.pad(torch.tensor(complex.protein_emb, dtype=torch.float32), (0, 0, 0, protein_max1 - n_prot)))
            lig_embs.append(F.pad(torch.tensor(complex.ligand_emb, dtype=torch.float32), (0, 0, 0, ligand_max2 - n_lig)))
            prot_coords.append(F.pad(torch.from_numpy(complex.protein_coord).float(), (0, 0, 0, protein_max1 - n_prot)))
            lig_coords.append(F.pad(torch.from_numpy(complex.ligand_coord).float(), (0, 0, 0, ligand_max2 - n_lig)))
            #contact_map.append(F.pad(torch.cdist(prot_coord, lig_coord).float().t(), (0, protein_max1 - n_prot, 0, ligand_max2 - n_lig)))  # lig, pro
            natives.append(complex.native)
            affinitys.append(complex.affinity)
            weights.append(complex.weight)

        prot_embs = torch.stack(prot_embs)
        prot_coords = torch.stack(prot_coords)
        lig_embs = torch.stack(lig_embs)
        lig_coords = torch.stack(lig_coords)
        #contact_map = torch.stack(contact_map)
        #contact_map[contact_map>10] = 10
        natives = torch.stack(natives)
        affinitys = torch.stack(affinitys)
        weights = torch.stack(weights)

        return (prot_embs, prot_coords), (lig_embs, lig_coords),  natives, affinitys, weights

class MDDTADataLoader_FDA(DataLoader):
    def __init__(self, complex_dataset, protein_emb_path=None, protein_coords_path=None, ligand_emb_path=None, **kwargs):
        self.complex_dataset = complex_dataset
        self.protein_emb_path = protein_emb_path
        self.protein_coords_path = protein_coords_path
        self.ligand_coords_path = ligand_emb_path

        array = torch.arange(len(complex_dataset)).long()

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=array, **kwargs)

    def __collate_fn__(self, batch_idx):
        B = len(batch_idx)
        sampled_batch = [self.complex_dataset[i] for i in batch_idx]

        protein_max1 = max([b.protein_len for b in sampled_batch])
        ligand_max2 = max([b.ligand_len for b in sampled_batch])

        prot_embs, prot_coords, lig_embs, lig_coords, contact_map = [], [], [], [], []
        natives, affinitys, weights,entropy_weights = [], [], [],[]
        

        for i, complex in enumerate(sampled_batch):
            pdb_id = complex.pdb
            n_prot, n_lig = complex.protein_len, complex.ligand_len

            prot_embs.append(F.pad(torch.tensor(complex.protein_emb, dtype=torch.float32), (0, 0, 0, protein_max1 - n_prot)))
            lig_embs.append(F.pad(torch.tensor(complex.ligand_emb, dtype=torch.float32), (0, 0, 0, ligand_max2 - n_lig)))
            prot_coords.append(F.pad(torch.from_numpy(complex.protein_coord).float(), (0, 0, 0, protein_max1 - n_prot)))
            lig_coords.append(F.pad(torch.from_numpy(complex.ligand_coord).float(), (0, 0, 0, ligand_max2 - n_lig)))
            #contact_map.append(F.pad(torch.cdist(prot_coord, lig_coord).float().t(), (0, protein_max1 - n_prot, 0, ligand_max2 - n_lig)))  # lig, pro
            #natives.append(complex.native)
            affinitys.append(complex.affinity)
            #weights.append(complex.weight)
            #entropy_weights.append(complex.entropy_weight)

        prot_embs = torch.stack(prot_embs)
        prot_coords = torch.stack(prot_coords)
        lig_embs = torch.stack(lig_embs)
        lig_coords = torch.stack(lig_coords)
        #contact_map = torch.stack(contact_map)
        #contact_map[contact_map>10] = 10
        #natives = torch.stack(natives)
        affinitys = torch.stack(affinitys)
        #weights = torch.stack(weights)
        #entropy_weights = torch.stack(entropy_weights)

        return (prot_embs, prot_coords), (lig_embs, lig_coords),affinitys
