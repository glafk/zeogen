# DEPRECATED
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm

import torch
import torch_geometric as tg
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

from data_utils.utils import get_atoms, distance


class EdgeData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if 'edge_index' in key:
            return len(self.x)
        return super().__inc__(key, value, *args, **kwargs)


class ZeoData(Dataset):

    def __init__(self, zeo_idxes, l_uc, cutoff=6.5, n_samples=1500):
        self.n_samples = n_samples
        self.l_uc = l_uc
        self.cutoff = cutoff
        self.n_samples = n_samples        

        self.ats_zeo = []
        self.pos_zeo = []
        
        for idx in tqdm.tqdm(zeo_idxes):
            ats_z, pos_z = get_atoms(f'../data/MOR_dataloader_test_100/RHO_clusters_{idx}.cif')

            self.ats_zeo.append(ats_z)
            self.pos_zeo.append(pos_z)
        
    def __len__(self):

        return len(self.ats_zeo)*self.n_samples
    
    def __getitem__(self, idx):
        zeo_idx = idx // self.n_samples

        atoms = self.ats_zeo[zeo_idx]
        position = self.pos_zeo[zeo_idx]

        pos = position

        d = distance(pos, pos, self.l_uc).fill_diagonal_(np.inf)

        idx1, idx2 = torch.where(d<self.cutoff)
        edge_index = torch.vstack([idx1, idx2])

        graph = EdgeData(x=atoms,
                         edge_index=edge_index,
                         pos=pos*self.l_uc) # transform to cartesian coordinates

        return graph


def create_zeo(atoms, pos, l_uc, cutoff=6.5):

    n_al = sum(atoms)

    ats = F.one_hot(torch.cat([torch.tensor(atoms), torch.tensor(sum(atoms)*[2] + [3])]))

    zero = torch.tensor([0.,0.,0.])
    pos_nac = torch.vstack([zero for _ in range(n_al+1)])

    pos = torch.cat([pos,pos_nac],0)

    dist = distance(pos, pos, l_uc.cpu())
    dist = dist.fill_diagonal_(np.inf)
    dist[n_al+1:] = np.inf
    dist[:,n_al+1:] = np.inf

    idx1, idx2 = torch.where(dist<cutoff)
    edge_index = torch.vstack([idx1, idx2])

    graph = EdgeData(x=ats, pos=pos*l_uc.cpu(), edge_index=edge_index)

    return graph