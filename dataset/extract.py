# Define the input and output file names
from pathlib import Path


input_file = "example.molden"
output_file = "first_molecule.xyz"

def extract(input_file):
    # Initialize variables to store atomic coordinates and number of atoms
    atomic_coordinates = []
    num_atoms = None

    # Open the input file and read its contents
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Loop through the lines to extract atomic coordinates and number of atoms
    for i, line in enumerate(lines):
        if "[GEOMETRIES] (XYZ)" in line:
            # Find the line number where the atomic coordinates start
            start_line = i + 3
            # Extract the number of atoms from the line above the atomic coordinates
            num_atoms = int(lines[i + 1].strip())
            energy = float(lines[i + 2].strip())

    # Check if the number of atoms was found
    if num_atoms is None:
        print("Number of atoms not found in the input file.")
    else:
        # Extract the atomic coordinates for the first molecule
        for i in range(start_line, start_line + num_atoms):
            atom_symbol, x, y, z = lines[i].split()
            x, y, z = map(float, [x, y, z])
            atomic_coordinates.append((atom_symbol, x, y, z))

        # Write the first molecule's atomic coordinates to the output XYZ file
        output_file = input_file.stem + ".xyz"
        with open(output_file, 'w') as outfile:
            outfile.write(f"{num_atoms}\n")
            outfile.write(f"{energy:.12f}\n")
            for atom, x, y, z in atomic_coordinates:
                outfile.write(f"{atom:2s} {x:12.7f} {y:12.7f} {z:12.7f}\n")

        # print(f"The first molecule's geometry has been saved as '{output_file}'.")

if __name__ == "__main__":
    # Define the input and output file names
    # folder = Path("S22")
    folder = Path("e-Baker")
    for file in folder.glob("*.molden"):
        extract(file)