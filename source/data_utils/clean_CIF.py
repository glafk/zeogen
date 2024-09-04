import argparse
import os

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
    for filename in os.listdir(directory):
        if filename.endswith(".cif"):
            input_cif_file = os.path.join(directory, filename)
            output_cif_file = input_cif_file.split('.')[0] + '_clean.' + input_cif_file.split('.')[1]
            clean_cif(input_cif_file, output_cif_file)
            print(f"File '{input_cif_file}' has been cleaned and saved as '{output_cif_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean CIF file by removing rows with oxygen atoms.")
    parser.add_argument("--input_file", help="Path to the input CIF file or directory")
    parser.add_argument("--all_in_directory", help="Clean all CIF files in the specified directory")

    args = parser.parse_args()

    if args.all_in_directory:
        clean_all_in_directory(args.all_in_directory)
    else:
        input_cif_file = args.input_file
        output_cif_file = input_cif_file.split('.')[0] + '_clean.' + input_cif_file.split('.')[1]
        clean_cif(input_cif_file, output_cif_file)
        print(f"File '{input_cif_file}' has been cleaned and saved as '{output_cif_file}'.")
