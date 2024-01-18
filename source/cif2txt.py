import argparse

def convert_cif_to_txt(input_file, output_file):
    # Open the input CIF file
    with open(input_file, 'r') as f:
        cif_data = f.read()

    # Append ".txt" to the output file name
    output_file_name = output_file.split('.')[0] + '.txt'

    # Save the CIF data as plain text to the output file
    with open(output_file_name, 'w') as f:
        f.write(cif_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CIF file to plain text.")
    parser.add_argument("input_file", help="Path to the input CIF file")
    parser.add_argument("output_file", help="Path to the output text file")

    args = parser.parse_args()

    if not args.input_file.endswith(".cif"):
        print("Error: The input file must have a '.cif' extension.")
        exit(1)

    convert_cif_to_txt(args.input_file, args.output_file)
    print(f"File '{args.input_file}' has been converted to '{args.output_file}'.")
