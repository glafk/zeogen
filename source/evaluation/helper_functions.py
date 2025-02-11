import os
import pickle
from pathlib import Path
import numpy as np

# import required module

from pymatgen.core.structure import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.local_env import MinimumDistanceNN


def split_samples_to_individual_zdivae(batched_samples):
    """
        Parameters:
            batched_samples: list of dicts, where each dict is a batch of samples
        Returns:
            list of dicts, with each sample individually
    """

    individual_sample_dicts = []

    for batch in batched_samples:
        batch.pop("is_traj", None)
        batch["norm_hoas"] = [norm_hoa for norm_hoa in batch["norm_hoas"] for _ in range(len(batch["domains"])//len(batch["norm_hoas"]))]
    
        coordinate_splits = np.split(batch["frac_coords"], batch["num_atoms"].cumsum())
        atom_type_splits = np.split(batch["atom_types"], batch["num_atoms"].cumsum())

        batch["frac_coords"] = coordinate_splits[:-1]
        batch["atom_types"] = atom_type_splits[:-1]

        for i in range(len(batch["domains"])):
            sample = {k:batch[k][i] for k in batch.keys()}
            individual_sample_dicts.extend([sample])

    return individual_sample_dicts

def split_samples_to_individual_cdvae(batched_samples):
    """
        Parameters:
            batched_samples: list of dicts, where each dict is a batch of samples
        Returns:
            list of dicts, with each sample individually
    """

    individual_sample_dicts = []

    for batch in batched_samples:
        batch.pop("is_traj", None)
        keys_to_include = list(batch.keys())
        if "hoas" in batch.keys() and (batch["hoas"] is None or len(batch["hoas"]) == 0):
            keys_to_include.remove("hoas")

        if "domains" in batch.keys() and (batch["domains"] is None or len(batch["domains"]) == 0):
            keys_to_include.remove("domains")

        # batch["norm_hoas"] = [norm_hoa for norm_hoa in batch["norm_hoas"] for _ in range(len(batch["domains"])//len(batch["norm_hoas"]))]
    
        coordinate_splits = np.split(batch["frac_coords"], batch["num_atoms"].cumsum())
        atom_type_splits = np.split(batch["atom_types"], batch["num_atoms"].cumsum())

        batch["frac_coords"] = coordinate_splits[:-1]
        batch["atom_types"] = atom_type_splits[:-1]

        for i in range(len(batch["z"])):
            sample = {k:batch[k][i] for k in keys_to_include}
            individual_sample_dicts.extend([sample])

    return individual_sample_dicts

def split_recons_cdvae(batched_recons):
    """
        Parameters:
            batched_recons: list of dicts, where each dict is a batch of reconstructions
        Returns:
            list of dicts, with each reconstruction individually
    """ 
    individual_recons_dicts = []

    for batch in batched_recons:
        batch.pop("is_traj")
        keys_to_include = list(batch.keys())

        coordinate_splits = np.split(batch["frac_coords"], batch["num_atoms"].cumsum())
        atom_type_splits = np.split(batch["atom_types"], batch["num_atoms"].cumsum())

        batch["frac_coords"] = coordinate_splits[:-1]
        batch["atom_types"] = atom_type_splits[:-1]    

        for i in range(len(batch["z"])):
            recon = {k:batch[k][i] for k in keys_to_include}
            individual_recons_dicts.extend([recon])

    return individual_recons_dicts

def split_recons_gt_cdvae(batched_recons):
    """
        Parameters:
            batched_recons: list of dicts, where each dict is a batch of reconstructions
        Returns:
            list of dicts, with each reconstruction individually
    """ 
    individual_recons_dicts = []

    for batch in batched_recons:
        keys_to_include = ['frac_coords','atom_types', 'lengths', 'angles', 'num_atoms', 'zeolite_code', 'norm_hoa', 'y']

        coordinate_splits = np.split(batch["frac_coords"], batch["num_atoms"].cpu().numpy().cumsum())
        atom_type_splits = np.split(batch["atom_types"], batch["num_atoms"].cpu().numpy().cumsum())

        batch["frac_coords"] = coordinate_splits[:-1]
        batch["atom_types"] = atom_type_splits[:-1]    

        for i in range(len(batch["num_atoms"])):
            recon = {k:batch[k][i] for k in keys_to_include}
            individual_recons_dicts.extend([recon])

    return individual_recons_dicts


def split_recons_zdivae(batched_recons):
    """
        Parameters:
            batched_recons: list of dicts, where each dict is a batch of reconstructions
        Returns:
            list of dicts, with each reconstruction individually
    """ 
    individual_recons_dicts = []

    for batch in batched_recons:
        if "is_traj" in batch.keys():
            batch.pop("is_traj")
        keys_to_include = list(batch.keys())

        coordinate_splits = np.split(batch["frac_coords"], batch["num_atoms"].cumsum())
        atom_type_splits = np.split(batch["atom_types"], batch["num_atoms"].cumsum())

        batch["frac_coords"] = coordinate_splits[:-1]
        batch["atom_types"] = atom_type_splits[:-1]    

        for i in range(len(batch["zd"])):
            recon = {k:batch[k][i] for k in keys_to_include}
            individual_recons_dicts.extend([recon])

    return individual_recons_dicts


def split_recons_gt_zdivae(batched_recons):
    """
        Parameters:
            batched_recons: list of dicts, where each dict is a batch of reconstructions
        Returns:
            list of dicts, with each reconstruction individually
    """ 
    individual_recons_dicts = []

    for batch in batched_recons:
        keys_to_include = ['frac_coords','atom_types', 'lengths', 'angles', 'num_atoms', 'zeolite_code', 'norm_hoa', 'hoa', 'hoa_mu', 'hoa_std']

        coordinate_splits = np.split(batch["frac_coords"], batch["num_atoms"].cpu().numpy().cumsum())
        atom_type_splits = np.split(batch["atom_types"], batch["num_atoms"].cpu().numpy().cumsum())

        batch["frac_coords"] = coordinate_splits[:-1]
        batch["atom_types"] = atom_type_splits[:-1]    

        for i in range(len(batch["num_atoms"])):
            recon = {k:batch[k][i] for k in keys_to_include}
            individual_recons_dicts.extend([recon])

    return individual_recons_dicts


# Function to create a structure from unit cell parameters and coordinates
def create_structure(coords, lengths, angles):
    """
    Create a pymatgen Structure object.
    
    Parameters:
        coords (list of list): Fractional coordinates of atoms (e.g., [[0, 0, 0], [0.5, 0.5, 0.5]]).
        lengths (list): Unit cell lengths [a, b, c].
        angles (list): Unit cell angles [alpha, beta, gamma] in degrees.
    
    Returns:
        pymatgen.core.structure.Structure: Generated structure.
    """
    # Create lattice
    lattice = Lattice.from_parameters(*lengths, *angles)
    
    # Create structure (all atoms are Si)
    structure = Structure(lattice, ["Si"] * len(coords), coords)
    return structure


def parse_cif(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        lattice = []
        for i in range(6):
            lattice.append(float(lines[i].strip().split(" ")[-1]))

        coords = []
        for line in lines:
            if line.startswith("Si") or line.startswith("Al"):
                coords.append(list(map(float, line.strip().split(" ")[2:5])))

    return lattice, coords

# def check_zeolite_validity(structure, distance_tol=3.5):
#     """Check that the first 4 atoms closest to a given one are at a distance around 3 
#     because that suggests they from the tetrahedra that zeolites usually have in their structure"""
#     sorted_distance_matrix = np.sort(structure.distance_matrix, axis=1)

#     return np.all([sorted_distance_matrix[i][1:5] < distance_tol for i in range(0, len(sorted_distance_matrix))])

def check_zeolite_validity(structure, tolerance=3.5):
    """
    Check that each atom in the structure has exactly 4 nearest neighbors
    within a reasonable tetrahedral bond distance (around 3 Å, with tol).
    
    Parameters:
        structure (pymatgen.core.structure.Structure): Pymatgen structure object.
        tol (float): Maximum allowed T-T distance (default: 3.5 Å).
    
    Returns:
        bool: True if all atoms have 4 closest neighbors within tol, else False.
    """
    # Compute distance matrix and sort each row (self-distance is always 0 at index 0)
    sorted_distances = np.sort(structure.distance_matrix, axis=1)

    # Extract the four closest neighbors (ignoring the self-distance at index 0)
    nearest_distances = sorted_distances[:, 1:5]  # Shape (N_atoms, 4)

    # Check if all nearest distances are within the given tolerance
    return np.all(nearest_distances < tolerance)

# def compute_bond_angle(v1, v2):
#     """Compute bond angle between two vectors using the dot product formula."""
#     cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#     return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Clip to avoid numerical issues

# def check_zeolite_validity(structure, distance_tol=3.5, angle_tol=(104, 115)):
#     """
#     Check if a zeolite structure satisfies tetrahedral coordination:
#       1. Each Si (or Al) has exactly 4 nearest neighbors within `distance_tol`.
#       2. All O-Si-O angles fall within `angle_tol` (default: 104°–115°).
    
#     Parameters:
#         structure (pymatgen.core.structure.Structure): The structure to validate.
#         distance_tol (float): Maximum allowed bond length for tetrahedral coordination.
#         angle_tol (tuple): Acceptable range for bond angles.
    
#     Returns:
#         bool: True if structure is valid, else False.
#     """
#     nn_finder = MinimumDistanceNN()  # Nearest neighbor finder
    
#     for site in structure:
#         # Find the 4 closest atoms
#         neighbors = nn_finder.get_nn_info(structure, structure.index(site))
        
#         # Distance check: Must be exactly 4 neighbors within tolerance
#         if len(neighbors) != 4:
#             return False  # Incorrect tetrahedral coordination
        
#         # Extract positions and compute distances
#         center = site.coords
#         positions = [n["site"].coords for n in neighbors]
#         distances = [np.linalg.norm(pos - center) for pos in positions]
        
#         if not all(d < distance_tol for d in distances):
#             return False  # One or more bonds exceed tolerance
        
#         # Angle check: Compute all O-Si-O angles
#         vectors = [pos - center for pos in positions]
#         for i in range(4):
#             for j in range(i + 1, 4):
#                 angle = compute_bond_angle(vectors[i], vectors[j])
#                 if not (angle_tol[0] <= angle <= angle_tol[1]):
#                     return False  # Invalid bond angle found
    
#     return True  # Passed both distance and angle checks