import pickle
import os

import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data

from data_utils.crystal_utils import (
    preprocess, preprocess_tensors, add_scaled_lengths_prop)

ZEOLITE_CODES_MAPPING = {'LTL': 0, 'TON3': 1, 'CHA': 2, 'MER': 3, 'TONch': 4, 'BEC': 5, 'RHO': 6, 'MOR': 7, 'ERI': 8, 'MEL': 9, 'FAUch': 10, 'MFI': 11, 'TON4': 12, 'HEU': 13, 'MTW': 14, 'FAU': 15, 'ITW': 16, 'TON2': 17, 'TON': 18, 'YFI': 19, 'DDRch1': 20, 'FER': 21, 'DDRch2': 22, 'NAT': 23, 'MELch': 24, 'LTA': 25}

class TensorCrystDataset(Dataset):
    def __init__(self, path, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, prop, num_records=None, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.path = path
        self.prop = prop
        self.num_records = num_records

        # Read the tensors from path to crystal_array_list
        crystal_array_list = pickle.load(open(path, 'rb'))
        print(f"Pickle file {path} was loaded successfully.")
        print(f"Number of crystals: {len(crystal_array_list)}")
        self.cached_data = preprocess_tensors(
            crystal_array_list,
            graph_method=self.graph_method,
            num_records=self.num_records,
            prop_name=self.prop)

        add_scaled_lengths_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        prop = self.scaler.transform(data_dict[self.prop])
        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            zeolite_code=data_dict["zeolite_code"],
            zeolite_code_enc=torch.Tensor(ZEOLITE_CODES_MAPPING[data_dict["zeolite_code"]]).float(),
            hoa=torch.Tensor(prop).view(1, -1),
            hoa_mu=torch.Tensor([data_dict['hoa_mu']]).view(1, -1),
            hoa_sigma=torch.Tensor([data_dict['hoa_sigma']]).view(1, -1),
            norm_hoa=torch.Tensor([data_dict['norm_hoa']]).view(1, -1),
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"