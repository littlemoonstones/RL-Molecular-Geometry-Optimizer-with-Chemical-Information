from pathlib import Path
def save(atoms=None, coords=None, energy=None, fileName=None):
    if coords is None:
        # coords = self.cart_coords.copy()
        raise Exception("no coords")
    
    # coords *= BOHR2ANG
    if fileName is None:
        # fileName = self.file_name
        pass
    else:
        file_name = fileName

    # energy = self.calculator.get_energy(atoms=self.atoms, coords=coords)
    # print(self.atoms)
    with open(f"{Path(file_name)}.xyz", "w") as output:
        output.write(f"{len(atoms)}\n")
        output.write(f"{energy}\n")
        for index, coord in enumerate(coords.reshape(-1, 3)):
            output.write(f"{atoms[index]}\t{coord[0]:18.15f}\t{coord[1]:18.15f}\t{coord[2]:18.15f}\n")