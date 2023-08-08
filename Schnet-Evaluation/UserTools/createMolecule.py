import rdkit.Chem
import rdkit.Chem.AllChem
import numpy as np
import random
import math

def createMolecule(smile, seed=1):
    if seed <= 0:
        raise Exception("seed must larger than 0")
    if type(seed) != int:
        raise Exception("the type of seed must be integer")
    if smile is None or smile == "":
        raise Exception("smile cannot be empty")
    if '[Fe+3]' in smile:
        print("Unrecognized atom type: Fe3 ")
        smile = "CCC"
    
    m = rdkit.Chem.rdmolfiles.MolFromSmiles(smile)
    

    mol = rdkit.Chem.AddHs(m)
    atoms = []
    
    for i in rdkit.Chem.rdchem.Mol.GetAtoms(mol):
        atoms.append(i.GetSymbol())
    
    # set random seed: https://github.com/rdkit/rdkit/issues/2575
    rdkit.Chem.AllChem.EmbedMolecule(mol, randomSeed=seed)
    pyMP = rdkit.Chem.AllChem.MMFFGetMoleculeProperties(mol)
    try:
        pyFF = rdkit.Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, pyMP)
    except:
        raise ValueError
    return smile, atoms, np.array(pyFF.Positions())

def getMolecule(smile: str, seed: int=1):
    i = 0
    while True:
        if i > 100:
            smile = 'CCC'
            i = 0
            print('faild build:', smile)
        try:
            smile, atoms, coords = createMolecule(smile, seed)
            break
        except ValueError:
            seed = random.randint(2, math.pow(2, 30))
        i += 1

    return smile, atoms, coords, seed