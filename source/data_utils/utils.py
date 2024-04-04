import numpy as np
import torch
from torch_geometric.data import Batch

def difference(X1, X2):
    diff = (X2[:,None] - X1)
    
    # Since we are dealing with partial coordinates,
    # here if we have a po
    diff = torch.where(diff > .5, diff-1, diff)
    diff = torch.where(diff < -.5, diff+1, diff)

    return diff


def distance(X1, X2, l):
    # Uncomment for distance in partial coordinates
    # X1 = X1/l
    # X2 = X2/l

    diff = difference(X1, X2)
    diff *= l
    dist = diff.pow(2).sum(-1).pow(0.5)
    return dist


def get_atoms(file, get_o=False):
    with open(file) as f:
        lines = f.readlines()
    lines = [i.strip().split() for i in lines]
    lines = [i for i in lines if len(i)>1]


    at_pos = [i[1:5] for i in lines if i[1] in ['Si', 'Al']]
    atom = np.array([1 if i[0]=='Al' else 0 for i in at_pos])
    X = np.array([list(map(float, i[1:])) for i in at_pos])

    if not get_o:

        
        return torch.tensor(atom), torch.tensor(X)    
    else:
        o_pos = [i[1:5] for i in lines if i[1] == 'O']
        X_o = np.array([list(map(float, i[1:])) for i in o_pos])

        return torch.tensor(atom), torch.tensor(X), torch.tensor(X_o)


def get_cif_str(file):
    with open(file) as f:
        cif_str = f.read()
    
    return cif_str


def collate_fn(batch):

    return Batch.from_data_list(batch)

def get_transform_matrix(a, b, c, alpha, beta, gamma):

    alpha = alpha*np.pi/180
    beta = beta*np.pi/180
    gamma = gamma*np.pi/180
    zeta = (np.cos(alpha) - np.cos(gamma) * np.cos(beta))/np.sin(gamma)
    
    h = np.zeros((3,3))
    
    h[0,0] = a
    h[0,1] = b * np.cos(gamma)
    h[0,2] = c * np.cos(beta)

    h[1,1] = b * np.sin(gamma)
    h[1,2] = c * zeta

    h[2,2] = c * np.sqrt(1 - np.cos(beta)**2 - zeta**2)

    return h    


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)