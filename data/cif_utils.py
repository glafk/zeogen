import os

import pickle
import random
from pymatgen.io.cif import CifParser
import pandas
import numpy as np


ATOM_TYPES = {
    'Si': 13,
    'Al': 14
}

def parse_cif_manually(file_path, hoa=0.0):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lengths = []
    angles = []
    frac_coords = []
    atom_types = []
    current_loop = None

    for line in lines:
        if line.startswith('_cell_length_a'):
            lengths.append(float(line.split()[1]))
        elif line.startswith('_cell_length_b'):
            lengths.append(float(line.split()[1]))
        elif line.startswith('_cell_length_c'):
            lengths.append(float(line.split()[1]))
        elif line.startswith('_cell_angle_alpha'):
            angles.append(float(line.split()[1]))
        elif line.startswith('_cell_angle_beta'):
            angles.append(float(line.split()[1]))
        elif line.startswith('_cell_angle_gamma'):
            angles.append(float(line.split()[1]))
        elif line.startswith('loop_'):
            current_loop = 'atom_sites'
        elif current_loop == 'atom_sites' and line.startswith('_atom_site_label'):
            continue
        elif current_loop == 'atom_sites' and line.startswith('_atom_site_type_symbol'):
            continue
        elif current_loop == 'atom_sites' and line.startswith('_atom_site_fract_x'):
            continue
        elif current_loop == 'atom_sites' and line.startswith('_atom_site_fract_y'):
            continue
        elif current_loop == 'atom_sites' and line.startswith('_atom_site_fract_z'):
            continue
        elif current_loop == 'atom_sites' and len(line.split()) >= 5:
            tokens = line.split()
            atom_type = tokens[1]
            if atom_type != 'O':
                atom_types.append(ATOM_TYPES[atom_type])
                frac_coords.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])

    return {
        "frac_coords": frac_coords,
        "atom_types": atom_types,
        "lengths": lengths,
        "angles": angles,
        "hoa": hoa
    }


def parse_cif(file_path, hoa=0.0):
    print(f"Parsing {file_path}.")
    parser = CifParser(file_path)
    structure = parser.parse_structures()[0]
    
    frac_coords = []
    atom_types = []
    lengths = [structure.lattice.a, structure.lattice.b, structure.lattice.c]
    angles = [structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma]
    
    for site in structure.sites:
        if site.species_string != 'O':
            frac_coords.append(site.frac_coords.tolist())
            atom_types.append(site.species_string)
    
    return {
        "frac_coords": frac_coords,
        "atom_types": atom_types,
        "lengths": lengths,
        "angles": angles,
        "hoa": hoa
    }

def read_cif_files(root_dir):
    zeolite_data = {}

    for root, dirs, files in os.walk(root_dir):
        basename = os.path.basename(root)
        if 'data' not in basename:
            zeolite_code = os.path.basename(root)
            dirname = os.path.dirname(root)
            with open(f"{dirname}/hoa_{zeolite_code}.dat", 'rb') as f:
                lines = f.readlines()
                # Ignore first line (Header)
                lines = lines[1:]

            hoa = np.array([float(line.split()[1]) for line in lines])
            for file in files:
                if file.endswith('.cif'):
                    dirname = os.path.dirname(root)
                    zeolite_index = int(file.split('_')[1])

                    if zeolite_code not in zeolite_data:
                        zeolite_data[zeolite_code] = []

                    file_path = os.path.join(root, file)
                    cif_data = parse_cif_manually(file_path, hoa[zeolite_index])
                    zeolite_data[zeolite_code].append(cif_data)
    
    print(zeolite_data.keys()) 
    return zeolite_data

def split_data(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

def save_data_splits(zeolite_data, output_dir):
    for zeolite_code, data in zeolite_data.items():
        train_data, val_data, test_data = split_data(data)
        
        zeolite_folder = os.path.join(output_dir, zeolite_code)
        os.makedirs(zeolite_folder, exist_ok=True)
        
        with open(os.path.join(zeolite_folder, 'train.pkl'), 'wb') as f:
            pickle.dump(train_data, f)
        with open(os.path.join(zeolite_folder, 'val.pkl'), 'wb') as f:
            pickle.dump(val_data, f)
        with open(os.path.join(zeolite_folder, 'test.pkl'), 'wb') as f:
            pickle.dump(test_data, f)

def clean_cif(input_file, output_file):
    # Open the input CIF file
    with open(input_file, 'r') as f:
        cif_data = f.readlines()

    # Filter out rows containing oxygen atoms (O)
    cleaned_cif_data = [line for line in cif_data if not line.startswith('O')]

    # Save the cleaned CIF data to the output file
    with open(output_file, 'w') as f:
        f.writelines(cleaned_cif_data)

def clean_all_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".cif"):
                input_cif_file = os.path.join(directory, root, filename)
                output_cif_file = input_cif_file.split('.')[0] + '_clean.' + input_cif_file.split('.')[1]
                clean_cif(input_cif_file, output_cif_file)
                print(f"File '{input_cif_file}' has been cleaned and saved as '{output_cif_file}'.")

def convert_cif_to_txt(input_file, output_file):
    # Open the input CIF file
    with open(input_file, 'r') as f:
        cif_data = f.read()

    # Append ".txt" to the output file name
    output_file_name = output_file.split('.')[0] + '.txt'

    # Save the CIF data as plain text to the output file
    with open(output_file_name, 'w') as f:
        f.write(cif_data)