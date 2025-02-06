import pickle
import os

import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data

from data_utils.crystal_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)


ZEOLITE_CODES_MAPPING = {'DDRch1': 0, 'DDRch2': 1, 'FAU': 2, 
                         'FAUch': 3, 'ITW': 4, 'MEL': 5, 
                         'MELch': 6, 'MFI': 7, 'MOR': 8, 
                         'RHO': 9, 'TON': 10, 'TON2': 11, 
                         'TON3': 12, 'TON4': 13, 'TONch': 14, 
                         'BEC': 15, 'CHA': 16, 'ERI': 17, 
                         'FER': 18, 'HEU': 19, 'LTA': 20, 
                         'LTL': 21, 'MER': 22, 'MTW': 23, 
                         'NAT': 24, 'YFI': 25, "DDR": 26}

ZEOLITE_CODES_MAPPING_SMALL = {'DDR': 1, 'FAU': 2, 'ITW': 3, 'MEL': 4, 'MFI': 5, 'MOR': 6, 
                               'RHO': 7, 'TON': 8, 'BEC': 9, 'CHA': 10, 'ERI': 11, 'FER': 12, 
                               'HEU': 13, 'LTA': 14, 'LTL': 15, 'MER': 16, 'MTW': 17, 'NAT': 18, 'YFI': 19}
ZEOLITE_CODES_MAPPING_ALL_CODES = {'ACO': 1,
 'AEI': 2,
 'AEL': 3,
 'AEN': 4,
 'AET': 5,
 'AFG': 6,
 'AFI': 7,
 'AFN': 8,
 'AFO': 9,
 'AFR': 10,
 'AFS': 11,
 'AFT': 12,
 'AFV': 13,
 'AFX': 14,
 'AFY': 15,
 'AHT': 16,
 'ANO': 17,
 'APC': 18,
 'APD': 19,
 'AST': 20,
 'ASV': 21,
 'ATN': 22,
 'ATO': 23,
 'ATS': 24,
 'ATT': 25,
 'ATV': 26,
 'AVE': 27,
 'AVL': 28,
 'AWO': 29,
 'AWW': 30,
 'BCT': 31,
 'BIK': 32,
 'BOF': 33,
 'BOG': 34,
 'BOZ': 35,
 'BPH': 36,
 'BRE': 37,
 'BSV': 38,
 'CAN': 39,
 'CAS': 40,
 'CDO': 41,
 'CFI': 42,
 'CGF': 43,
 'CGS': 44,
 'CON': 45,
 'CSV': 46,
 'CZP': 47,
 'DAC': 48,
 'DFO': 49,
 'DFT': 50,
 'DOH': 51,
 'DON': 52,
 'EAB': 53,
 'EEI': 54,
 'EMT': 55,
 'EON': 56,
 'EOS': 57,
 'EPI': 58,
 'ESV': 59,
 'ETL': 60,
 'ETR': 61,
 'ETV': 62,
 'EUO': 63,
 'EWF': 64,
 'EWO': 65,
 'EWS': 66,
 'EZT': 67,
 'FAR': 68,
 'FRA': 69,
 'GIS': 70,
 'GIU': 71,
 'GME': 72,
 'GON': 73,
 'GOO': 74,
 'IFO': 75,
 'IFR': 76,
 'IFW': 77,
 'IFY': 78,
 'IHW': 79,
 'IRN': 80,
 'IRR': 81,
 'ISV': 82,
 'ITE': 83,
 'ITG': 84,
 'ITH': 85,
 'ITR': 86,
 'ITT': 87,
 'ITW': 88,
 'IWR': 89,
 'IWS': 90,
 'IWV': 91,
 'IWW': 92,
 'JNT': 93,
 'JOZ': 94,
 'JRY': 95,
 'JSN': 96,
 'JSR': 97,
 'JST': 98,
 'JSW': 99,
 'JSY': 100,
 'JZT': 101,
 'KFI': 102,
 'LAU': 103,
 'LEV': 104,
 'LIO': 105,
 'LOS': 106,
 'LOV': 107,
 'LTF': 108,
 'LTJ': 109,
 'MAR': 110,
 'MAZ': 111,
 'MEI': 112,
 'MEP': 113,
 'MFS': 114,
 'MON': 115,
 'MOZ': 116,
 'MRT': 117,
 'MSE': 118,
 'MSO': 119,
 'MTF': 120,
 'MTN': 121,
 'MTT': 122,
 'MVY': 123,
 'MWW': 124,
 'NAB': 125,
 'NES': 126,
 'NON': 127,
 'NPT': 128,
 'NSI': 129,
 'OBW': 130,
 'OFF': 131,
 'OKO': 132,
 'OSI': 133,
 'OSO': 134,
 'OWE': 135,
 'PCR': 136,
 'PHI': 137,
 'PON': 138,
 'POR': 139,
 'POS': 140,
 'PSI': 141,
 'PTF': 142,
 'PTO': 143,
 'PTT': 144,
 'PTY': 145,
 'PWW': 146,
 'RFE': 147,
 'RRO': 148,
 'RTE': 149,
 'RTH': 150,
 'RUT': 151,
 'RWR': 152,
 'RWY': 153,
 'SAF': 154,
 'SAO': 155,
 'SAS': 156,
 'SAT': 157,
 'SAV': 158,
 'SBE': 159,
 'SBN': 160,
 'SBS': 161,
 'SBT': 162,
 'SEW': 163,
 'SFE': 164,
 'SFF': 165,
 'SFG': 166,
 'SFH': 167,
 'SFN': 168,
 'SFO': 169,
 'SFS': 170,
 'SFW': 171,
 'SGT': 172,
 'SIV': 173,
 'SOD': 174,
 'SOF': 175,
 'SOR': 176,
 'SOS': 177,
 'SOV': 178,
 'SSF': 179,
 'SSY': 180,
 'STF': 181,
 'STI': 182,
 'STT': 183,
 'STW': 184,
 'SVV': 185,
 'SWY': 186,
 'SZR': 187,
 'TER': 188,
 'THO': 189,
 'TOL': 190,
 'TUN': 191,
 'UEI': 192,
 'UFI': 193,
 'UOS': 194,
 'UOV': 195,
 'UOZ': 196,
 'USI': 197,
 'UTL': 198,
 'UWY': 199,
 'VET': 200,
 'VFI': 201,
 'VNI': 202,
 'VSV': 203,
 'WEI': 204,
 'YUG': 205,
 'ZON': 206,
 'FER': 207,
 'BEC': 208,
 'FAU': 209,
 'DDR': 210,
 'MTW': 211,
 'MFI': 212,
 'TON': 213,
 'RHO': 214,
 'MEL': 215,
 'MOR': 216,
 'HEU': 217,
 'ERI': 218,
 'LTA': 219,
 'YFI': 220,
 'CHA': 221,
 'LTL': 222,
 'NAT': 223,
 'MER': 224}


class CrystDataset(Dataset):
    def __init__(self, name: str, path: str,
                 prop: str, niggli: bool, primitive: bool,
                 graph_method: str, preprocess_workers: int,
                 lattice_scale_method: str,
                 num_records: int = None,
                 **kwargs):
        super().__init__()
        # The path should point to a pickle file with an index of the CIF files that are to be used
        # Could also change in the future
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.num_records = num_records

        cwd = os.getcwd()

        input_files = [f"{os.path.join(cwd, '../../data/MOR_dataloader_test_100/')}{file}" for file in pickle.load(self.path)]
        self.cached_data = preprocess(
            input_files,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop])

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

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
            y=prop.view(1, -1),
            zeolite_code=data_dict["zeolite_code"],
            zeolite_code_enc=ZEOLITE_CODES_MAPPING_ALL_CODES[data_dict["zeolite_code"]]
        )
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, path, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, prop, num_records=None,
                 top_k=None, max_zeolite_size=None, sort="smallest", **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.path = path
        self.prop = prop
        self.num_records = num_records
        self.top_k = top_k
        self.sort = sort
        self.max_zeolite_size = max_zeolite_size


        # Read the tensors from path to crystal_array_list
        crystal_array_list = pickle.load(open(path, 'rb'))
        self.cached_data = preprocess_tensors(
            crystal_array_list,
            graph_method=self.graph_method,
            num_records=self.num_records,
            top_k=self.top_k,
            sort=self.sort,
            max_zeolite_size=self.max_zeolite_size)

        self.zeolite_codes = [data['zeolite_code'] for data in self.cached_data]
        # print(f"Set of zeolite codes in dataset {set(self.zeolite_codes)}")
        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
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
            y=prop.view(1, -1),
            zeolite_code=data_dict["zeolite_code"],
            zeolite_code_enc=ZEOLITE_CODES_MAPPING_ALL_CODES[data_dict["zeolite_code"]],
            norm_hoa=torch.Tensor([data_dict['norm_hoa']]).view(1, -1)
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"