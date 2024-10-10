import copy
import os
from itertools import islice

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import torch
from p_tqdm import p_umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from data_utils.utils import abs_cap

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=True)


# Tensor of unit cells. Assumes 27 cells in -1, 0, 1 offsets in the x and y dimensions
# Note that differing from OCP, we have 27 offsets here because we are in 3D
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]
EPSILON = 1e-5

def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def lattice_matrix_to_params(matrix):
    lengths = np.sqrt(np.sum(matrix ** 2, axis=1)).tolist()

    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) /
                            (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def build_crystal(cif_str, primitive=False):
    """Build crystal from cif string.
    
    Params:
    crystal_str: str
    String containing the CIF

    primitive: bool
    Whether to consider the primitive structure of the crystal
    """
    crystal = Structure.from_str(cif_str, fmt='cif')

    if primitive:
        crystal = crystal.get_primitive_structure()
    else:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False
    )

    return canonical_crystal


def build_crystal_graph(crystal, graph_method='crystalnn'):
    """
    """

    # crystalnn stands for crystal nearest neighbor. A strategy for generating the neighbors from positions
    if graph_method == 'crystalnn':
        crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, CrystalNN)
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms


def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    pos = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)  # cart coords

    return pos


def cart_to_frac_coords(
    cart_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
    return (frac_coords % 1.)


def get_pbc_distances(
    coords,
    edge_index,
    lengths,
    angles,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
    return_offsets=False,
    return_distance_vec=False,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
        pos = torch.einsum('bi,bij->bj', coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattice, num_bonds, dim=0)
    offsets = torch.einsum('bi,bij->bj', to_jimages.float(), lattice_edges)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out


def radius_graph_pbc_wrapper(data, radius, max_num_neighbors_threshold, device):
    cart_coords = frac_to_cart_coords(
        data.frac_coords, data.lengths, data.angles, data.num_atoms)
    return radius_graph_pbc(
        cart_coords, data.lengths, data.angles, data.num_atoms, radius,
        max_num_neighbors_threshold, device)


def radius_graph_pbc(cart_coords, lengths, angles, num_atoms,
                     radius, max_num_neighbors_threshold, device,
                     topk_per_pair=None):
    """Computes pbc graph edges under pbc.

    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair

    Note: topk should take into account self-self edge for (i, i)
    """
    batch_size = len(num_atoms)

    # position of the atoms
    atom_pos = cart_coords

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        (atom_count_sqr // num_atoms_per_image_expand)
    ).long() + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ).long() + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)

    if topk_per_pair is not None:
        assert topk_per_pair.size(0) == num_atom_pairs
        atom_distance_sqr_sort_index = torch.argsort(atom_distance_sqr, dim=1)
        assert atom_distance_sqr_sort_index.size() == (num_atom_pairs, num_cells)
        atom_distance_sqr_sort_index = (
            atom_distance_sqr_sort_index +
            torch.arange(num_atom_pairs, device=device)[:, None] * num_cells).view(-1)
        topk_mask = (torch.arange(num_cells, device=device)[None, :] <
                     topk_per_pair[:, None])
        topk_mask = topk_mask.view(-1)
        topk_indices = atom_distance_sqr_sort_index.masked_select(topk_mask)

        topk_mask = torch.zeros(num_atom_pairs * num_cells, device=device)
        topk_mask.scatter_(0, topk_indices, 1.)
        topk_mask = topk_mask.bool()

    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask)

    num_neighbors = torch.zeros(len(cart_coords), device=device)
    num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device))
    num_neighbors = num_neighbors.long()
    max_num_neighbors = torch.max(num_neighbors).long()

    # Compute neighbors per image
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[
        _max_neighbors > max_num_neighbors_threshold
    ] = max_num_neighbors_threshold
    _num_neighbors = torch.zeros(len(cart_coords) + 1, device=device).long()
    _natoms = torch.zeros(num_atoms.shape[0] + 1, device=device).long()
    _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0)
    _natoms[1:] = torch.cumsum(num_atoms, dim=0)
    num_neighbors_image = (
        _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
    )

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        if topk_per_pair is None:
            return torch.stack((index2, index1)), unit_cell, num_neighbors_image
        else:
            return torch.stack((index2, index1)), unit_cell, num_neighbors_image, topk_mask

    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
    distance_sort = torch.zeros(
        len(cart_coords) * max_num_neighbors, device=device
    ).fill_(radius * radius + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index1 * max_num_neighbors
        + torch.arange(len(index1), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(len(cart_coords), max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with distances greater than the radius
    mask_within_radius = torch.le(distance_sort, radius * radius)
    index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index1), device=device).bool()
    mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
    index1 = torch.masked_select(index1, mask_num_neighbors)
    index2 = torch.masked_select(index2, mask_num_neighbors)
    unit_cell = torch.masked_select(
        unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)

    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask_num_neighbors)

    edge_index = torch.stack((index2, index1))

    if topk_per_pair is None:
        return edge_index, unit_cell, num_neighbors_image
    else:
        return edge_index, unit_cell, num_neighbors_image, topk_mask


def min_distance_sqr_pbc(cart_coords1, cart_coords2, lengths, angles,
                         num_atoms, device, return_vector=False,
                         return_to_jimages=False):
    """Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each data point.
    returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from cart_coords1 to cart_coords2, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of cart_coord2 relative to cart_coord1 in pbc
    """
    batch_size = len(num_atoms)

    # Get the positions for each atom
    pos1 = cart_coords1
    pos2 = cart_coords2

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(cart_coords2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the vector between atoms
    # shape (num_atom_squared_sum, 3, 27)
    atom_distance_vector = pos1 - pos2
    atom_distance_sqr = torch.sum(atom_distance_vector ** 2, dim=1)

    min_atom_distance_sqr, min_indices = atom_distance_sqr.min(dim=-1)

    return_list = [min_atom_distance_sqr]

    if return_vector:
        min_indices = min_indices[:, None, None].repeat([1, 3, 1])

        min_atom_distance_vector = torch.gather(
            atom_distance_vector, 2, min_indices).squeeze(-1)

        return_list.append(min_atom_distance_vector)

    if return_to_jimages:
        to_jimages = unit_cell.T[min_indices].long()
        return_list.append(to_jimages)

    return return_list[0] if len(return_list) == 1 else return_list


class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def inverse_transform_backprob_compat(self, X):
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )


def get_scaler_from_data_list(data_list, key):
    targets = torch.tensor([d[key] for d in data_list])
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler


# This is for loading data straigth from CIF files, so I'll ignore it for now
def preprocess(input_files, num_workers, niggli, primitive, graph_method,
               prop_list):
    # Currently this method would not work
    def process_one(file, niggli, primitive, graph_method, prop_list):
        with open(file, "r") as cif_file:
            crystal_str = cif_file.read()

        crystal = build_crystal(
            crystal_str, niggli=niggli, primitive=primitive)
        graph_arrays = build_crystal_graph(crystal, graph_method)
        # TODO: Leaving out the properties for now. To be reused later.
        # properties = {k: row[k] for k in prop_list if k in row.keys()}
        result_dict = {
            # This method of identifying the files should also be updated in the future
            'mp_id': file.split("_")[2],
            'cif': crystal_str,
            'graph_arrays': graph_arrays,
        }
        # result_dict.update(properties)
        return result_dict

    unordered_results = p_umap(
        process_one,
        [file for file in input_files],
        [niggli] * len(input_files),
        [primitive] * len(input_files),
        [graph_method] * len(input_files),
        [prop_list] * len(input_files),
        num_cpus=num_workers)

    mpid_to_results = {result['mp_id']: result for result in unordered_results}
    #ordered_results = [mpid_to_results[df.iloc[idx]['material_id']]
    #                   for idx in range(len(df))]

    #return ordered_results


# Array shape
# arr = [{frac_coords: [list], atom_types: [list], lengths: [list], angles: [list], adsorption_cap: float]}]
# lengths = [a,b,c]; angles = [alpha, beta, gamma)
def preprocess_tensors(crystal_dict_list, graph_method, num_records=None, prop_name='hoa'):
    def process_one(batch_idx, crystal_dict, graph_method, prop_name):
        frac_coords = crystal_dict['frac_coords']
        atom_types = crystal_dict['atom_types']
        lengths = crystal_dict['lengths']
        angles = crystal_dict['angles']
        hoa = crystal_dict[prop_name]
        crystal = Structure(
            lattice=Lattice.from_parameters(
                *(lengths + angles)),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False)
        graph_arrays = build_crystal_graph(crystal, graph_method)
        result_dict = {
            'batch_idx': batch_idx,
            'graph_arrays': graph_arrays,
            'hoa': hoa,
            'hoa_mu': crystal_dict['hoa_mu'],
            'hoa_std': crystal_dict['hoa_std'],
            'norm_hoa': crystal_dict['norm_hoa'],
            'zeolite_code': crystal_dict['zeolite_code'],
            # 'zeolite_code_enc': crystal_dict['zeolite_code_enc'],
        }
        return result_dict

    # Extract HOA and zeolite codes
    hoa = np.array([entry['hoa'] for entry in crystal_dict_list])
    zeo_code = np.array([entry['zeolite_code'] for entry in crystal_dict_list])

    # Find unique zeolite codes
    unique_zeo_codes = np.unique(zeo_code)

    # Find the maximum HOA for each zeolite type
    mean_hoa_per_zeo_code = {}
    std_hoa_per_zeo_code = {}
    for code in unique_zeo_codes:
        # Get the HOA values corresponding to the current zeolite code
        hoa_values_for_code = np.array([entry['hoa'] for entry in crystal_dict_list if entry['zeolite_code'] == code])
        mean_hoa_per_zeo_code[code] = np.mean(hoa_values_for_code)
        std_hoa_per_zeo_code[code] = np.std(hoa_values_for_code)

    # Add normalized HOA
    for entry in crystal_dict_list:
        mean_hoa = mean_hoa_per_zeo_code[entry['zeolite_code']]
        std_hoa = std_hoa_per_zeo_code[entry['zeolite_code']]
        entry['hoa_mu'] = mean_hoa
        entry['hoa_std'] = std_hoa
        entry['norm_hoa'] = (entry['hoa'] - mean_hoa) / std_hoa

    # Optionally clean up if needed
    del hoa
    del zeo_code

    # Limit number of items temporarily for testing purporses
    if num_records is not None:
        crystal_dict_list = crystal_dict_list[:num_records]

    unordered_results = p_umap(
        process_one,
        list(range(len(crystal_dict_list))),
        crystal_dict_list,
        [graph_method] * len(crystal_dict_list),
        [prop_name] * len(crystal_dict_list),
        num_cpus=4
    )
    ordered_results = list(
        sorted(unordered_results, key=lambda x: x['batch_idx']))
    return ordered_results


def add_scaled_lengths_prop(data_list, lattice_scale_method):
    for dict in data_list:
        graph_arrays = dict['graph_arrays']
        # the indexes are brittle if more objects are returned
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert lengths.shape[0] == angles.shape[0] == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == 'scale_length':
            lengths = lengths / float(num_atoms)**(1/3)

        dict['scaled_lengths'] = lengths


def mard(targets, preds):
    """Mean absolute relative difference."""
    assert torch.all(targets > 0.)
    return torch.mean(torch.abs(targets - preds) / targets)

# For now I'll comment this out as I'm not sure I'll measure theese targets
"""
def batch_accuracy_precision_recall(
    pred_edge_probs,
    edge_overlap_mask,
    num_bonds
):
    if (pred_edge_probs is None and edge_overlap_mask is None and
            num_bonds is None):
        return 0., 0., 0.
    pred_edges = pred_edge_probs.max(dim=1)[1].float()
    target_edges = edge_overlap_mask.float()

    start_idx = 0
    accuracies, precisions, recalls = [], [], []
    for num_bond in num_bonds.tolist():
        pred_edge = pred_edges.narrow(
            0, start_idx, num_bond).detach().cpu().numpy()
        target_edge = target_edges.narrow(
            0, start_idx, num_bond).detach().cpu().numpy()

        accuracies.append(accuracy_score(target_edge, pred_edge))
        precisions.append(precision_score(
            target_edge, pred_edge, average='binary'))
        recalls.append(recall_score(target_edge, pred_edge, average='binary'))

        start_idx = start_idx + num_bond

    return np.mean(accuracies), np.mean(precisions), np.mean(recalls)
"""
    

def compute_volume(batch_lattice):
    """Compute volume from batched lattice matrix

    batch_lattice: (N, 3, 3)
    """
    vector_a, vector_b, vector_c = torch.unbind(batch_lattice, dim=1)
    return torch.abs(torch.einsum('bi,bi->b', vector_a,
                                  torch.cross(vector_b, vector_c, dim=1)))


def lengths_angles_to_volume(lengths, angles):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return compute_volume(lattice)

class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none


def write_cif(structure: Structure, path: str, decimals: int = 3, *args, **kwargs):
    '''
    Write a structure to a CIF file

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Structure to write
    filename : str
        Name of the CIF file
    '''
    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c

    alpha = structure.lattice.alpha
    beta = structure.lattice.beta
    gamma = structure.lattice.gamma

    vol = structure.volume

    sga = SpacegroupAnalyzer(structure, symprec=0.001)
    try:
        crystal_system = sga.get_crystal_system()
    except:
        crystal_system = 'triclinic'


    print(f"Saving to {path}.")
    with open(path, 'w') as f:
        
        f.write(f"_cell_length_a {a:.{decimals}f}\n")
        f.write(f"_cell_length_b {b:.{decimals}f}\n")
        f.write(f"_cell_length_c {c:.{decimals}f}\n")
        f.write(f"_cell_angle_alpha {alpha:.{decimals}f}\n")
        f.write(f"_cell_angle_beta {beta:.{decimals}f}\n")
        f.write(f"_cell_angle_gamma {gamma:.{decimals}f}\n")
        f.write(f"_cell_volume {vol:.{decimals}f}\n")
        f.write("\n")
        f.write(f"_symmetry_cell_setting {crystal_system}\n")
        f.write(f"_symmetry_space_group_name_Hall 'P 1'\n")
        f.write(f"_symmetry_space_group_name_H-M 'P 1'\n")
        f.write("_symmetry_Int_Tables_number 1\n")
        f.write("_symmetry_equiv_pos_as_xyz 'x,y,z'\n")
        f.write("\n")
        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        f.write("_atom_site_charge\n")

        for site in structure:
            # for zeolites:
            if site.species_string == 'Si':
                f.write(f"{site.species_string} {site.species_string} {site.frac_coords[0]:.{decimals}f} {site.frac_coords[1]:.{decimals}f} {site.frac_coords[2]:.{decimals}f} -0.393\n")

            else:
                f.write(f"{site.species_string} {site.species_string} {site.frac_coords[0]:.{decimals}f} {site.frac_coords[1]:.{decimals}f} {site.frac_coords[2]:.{decimals}f} 0.000\n")


def sample2cif(sample: dict, path: str, trajectory_path: str=None, save_trajectory=False, downsample_trajectory=False, downsample_frame_rate=10):
    a,b,c = sample["lengths"]
    alpha, beta, gamma = sample["angles"]
    lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    species = sample["atom_types"]
    coords = sample["frac_coords"]
    structure = Structure(lattice, species, coords)

    write_cif(structure, path)
    if save_trajectory:
        if not downsample_trajectory:
            counter = 1
            for step_coords, step_atoms in zip(sample["all_frac_coords"], ["all_atom_types"]):
                step_structure = Structure(lattice, step_atoms, step_coords)
                step_path = os.path.join(trajectory_path, f"step_{counter}.cif")
                write_cif(step_structure, step_path)
                counter+=1
        else:
            counter=1
            for step_coords, step_atoms in islice(zip(sample["all_frac_coords"], sample["all_atom_types"]), 0, None, downsample_frame_rate):
                step_structure = Structure(lattice, step_atoms, step_coords)
                step_path = os.path.join(trajectory_path, f"step_{counter}.cif")
                write_cif(step_structure, step_path)
                counter+=downsample_frame_rate


def reconstruction2cif(reconstruction: dict, path: str, trajectory_path: str=None, save_trajectory=False, downsample_trajectory=False, downsample_frame_rate=5):
    reconstruction["atom_types"] = reconstruction["atom_types"].cpu()
    reconstruction["angles"] = reconstruction["angles"].cpu()[0]
    reconstruction["lengths"] = reconstruction["lengths"].cpu()[0]
    reconstruction["num_atoms"] = reconstruction["num_atoms"].cpu()
    reconstruction["frac_coords"] = reconstruction["frac_coords"].cpu()
    if save_trajectory:
        reconstruction["all_frac_coords"] = reconstruction["all_frac_coords"].cpu()
        reconstruction["all_atom_types"] = reconstruction["all_atom_types"].cpu()

    a,b,c = reconstruction["lengths"]
    alpha, beta, gamma = reconstruction["angles"]
    lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    final_species = reconstruction["atom_types"]
    final_coords = reconstruction["frac_coords"]
    final_structure = Structure(lattice, final_species, final_coords)

    write_cif(final_structure, path)
    if save_trajectory:
        if not downsample_trajectory:
            counter = 1
            for step_coords, step_atoms in zip(reconstruction["all_frac_coords"], reconstruction["all_atom_types"]):
                step_structure = Structure(lattice, step_atoms, step_coords)
                step_path = os.path.join(trajectory_path, f"step_{counter}.cif")
                write_cif(step_structure, step_path)
                counter+=1
        else:
            counter=1
            for step_coords, step_atoms in islice(zip(reconstruction["all_frac_coords"], reconstruction["all_atom_types"]), 0, None, downsample_frame_rate):
                step_structure = Structure(lattice, step_atoms, step_coords)
                step_path = os.path.join(trajectory_path, f"step_{counter}.cif")
                write_cif(step_structure, step_path)
                counter+=downsample_frame_rate



def save_samples_as_cifs(samples: dict, directory: str, 
                         save_trajectory=False, 
                         downsample_trajectory=True,
                         downsample_frame_rate=10):
    if not os.path.exists(directory):
        os.makedirs(directory)

    # samples["atom_types"] = samples["atom_types"]
    # samples["angles"] = samples["angles"]
    # samples["lengths"] = samples["lengths"]
    # samples["num_atoms"] = samples["num_atoms"]
    # samples["frac_coords"] = samples["frac_coords"]

    individual_samples = []

    # With the current setup, we have one item per domain in the list
    for item in samples:
        # Split atom types
        split_atom_types = np.split(item["atom_types"], np.cumsum(item["num_atoms"])[:-1])
    # Split fractional coordinates
        split_frac_coords = np.split(item["frac_coords"], np.cumsum(item["num_atoms"])[:-1])

        split_all_atoms = np.split(item["all_atom_types"], np.cumsum(item["num_atoms"])[:-1])
        split_all_coords = np.split(item["all_frac_coords"], np.cumsum(item["num_atoms"])[:-1])

        for i in range(len(item["num_atoms"])):
            individual_samples.append({"atom_types": split_atom_types[i], "frac_coords": split_frac_coords[i], "lengths": item["lengths"][i], "angles": item["angles"][i], "domain": item["domains"][i], "norm_hoa": item["norm_hoas"][i], "pred_hoa": item["pred_hoas"][i], "all_atom_types": split_all_atoms[i], "all_frac_coords": split_all_coords[i]})

    for sample in individual_samples: # individual_samples:
        filename = os.path.join(directory, f"sample_{sample["domain"]}_{str(sample["norm_hoa"]).replace('.', '_')}.cif")

        if sample.get("is_traj", True) and save_trajectory:
            traj_directory = os.path.join(directory, f"reconstruction_{sample["domain"]}_{sample['norm_hoa']}_traj")
            if not os.path.exists(traj_directory):
                os.makedirs(traj_directory)
            sample2cif(sample, filename, traj_directory, save_trajectory=True, downsample_trajectory=downsample_trajectory, downsample_frame_rate=downsample_frame_rate)
        else:
            sample2cif(sample, filename)


def save_reconstructions_as_cifs(reconstructions: list, directory: str, ground_truth: bool = False, save_trajectory=False, downsample_trajectory=False, downsample_frame_rate=5):
    if not os.path.exists(directory):
        os.makedirs(directory)

    counter = 1
    for reconstruction in reconstructions:
        filename = os.path.join(directory, f"reconstruction_{counter}.cif")
        if ground_truth:
            filename = os.path.join(directory, f"reconstruction_{counter}_gt.cif")
        
        if reconstruction.get("is_traj", False) and save_trajectory:
            traj_directory = os.path.join(directory, f"reconstruction_{counter}_traj")
            if not os.path.exists(traj_directory):
                os.makedirs(traj_directory)
            reconstruction2cif(reconstruction, filename, traj_directory, save_trajectory=True, downsample_trajectory=downsample_trajectory, downsample_frame_rate=downsample_frame_rate)
        else:
            reconstruction2cif(reconstruction, filename)

        counter += 1  


def visualize_trajectory(fractional_coords, lattice_vectors):
    """
    Visualize the trajectory of an atom given its fractional coordinates and lattice vectors using Plotly.

    Parameters:
    - fractional_coords: List of fractional coordinates (each element is [x, y, z]).
    - lattice_vectors: 3x3 array of lattice vectors defining the unit cell.
    """

    # Convert fractional coordinates to Cartesian coordinates
    cartesian_coords = np.array(fractional_coords)
    # lattice_vectors = np.array(lattice_vectors)
    # cartesian_coords = np.dot(fractional_coords, lattice_vectors)

    # Create 3D scatter plot with optimized performance
    fig = go.Figure(data=[go.Scatter3d(
        x=cartesian_coords[:, 0],
        y=cartesian_coords[:, 1],
        z=cartesian_coords[:, 2],
        mode='markers+lines',
        marker=dict(size=5),
        line=dict(width=2)
    )])

    # Set plot title and axis labels
    fig.update_layout(
        title='Atom Trajectory in 3D Space',
        scene=dict(
            xaxis=dict(title='X (Å)', range=[0, 1]),
            yaxis=dict(title='Y (Å)', range=[0, 1]),
            zaxis=dict(title='Z (Å)', range=[0, 1]),
            aspectmode='cube'
        ),
        scene_camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    )

    # Optimize rendering mode
    fig.update_traces(marker=dict(size=5, line=dict(width=2)),
                      selector=dict(mode='markers+lines'))

    # Show the plot
    fig.show()
