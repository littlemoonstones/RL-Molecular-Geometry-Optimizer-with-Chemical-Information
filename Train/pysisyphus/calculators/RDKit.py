from typing import List
import rdkit.Chem
import rdkit.Chem.AllChem
from pysisyphus.constants import BOHR2ANG, ANG2BOHR, AU2KCALPERMOL
from pysisyphus.calculators.Calculator import Calculator
# from nptyping import NDArray

import numpy as np

class Molecule:
    def __init__(self, smi, seed=1):
        m = rdkit.Chem.rdmolfiles.MolFromSmiles(smi)
        self.mol = rdkit.Chem.AddHs(m)
        self.atoms: List[str] = []
        
        for i in rdkit.Chem.rdchem.Mol.GetAtoms(self.mol):
            self.atoms.append(i.GetSymbol())
        # set random seed: https://github.com/rdkit/rdkit/issues/2575
        rdkit.Chem.AllChem.EmbedMolecule(self.mol, randomSeed=seed)
        self.pyMP = rdkit.Chem.AllChem.MMFFGetMoleculeProperties(self.mol)
        self.pyFF = rdkit.Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, self.pyMP)
        # rdkit.Chem.rdmolfiles.MolToXYZFile(self.mol, filename="default.xyz")

    def getEnergy(self, x=None) -> float:                 # The unit of CalcEnergy is "kcal/mol" from https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html
        # x: angstrom
        _coords = x * BOHR2ANG
        return self.pyFF.CalcEnergy(list(_coords)) / AU2KCALPERMOL      # Energy unit is hartree


    def getForces(self, x=None):        # The unit of getForces is "hartree/bohr"
        self.pyFF.Initialize()
        _coords = x * BOHR2ANG
        return -np.array(self.pyFF.CalcGrad(list(_coords))) / AU2KCALPERMOL / ANG2BOHR # 1 hartree = 627.5 kcal/mol

    def getCoordinates(self):
        return np.array(self.pyFF.Positions()) * ANG2BOHR
    

class MMFF(Calculator):

    def __init__(self, smile, seed=1, **kwargs):
        super().__init__(**kwargs)
        self.molecule = Molecule(smile, seed)
        self.results = dict()

    def get_forces(self, atoms: list, coords) -> dict:
        if self.molecule is None:
            raise("Not initialized yet")        
        
        calc_type = "grad"
        return {
            "forces": self.molecule.getForces(coords),
            "energy": self.molecule.getEnergy(coords),         
        }

    def get_energy(self, atoms: list, coords) -> dict:
        if self.molecule is None:
            raise("Not initialized yet")
        
        calc_type = "energy"
       
        return {
            "energy": self.molecule.getEnergy(coords)
        }

    def get_atoms(self)->str:
        return self.molecule.atoms

    def get_atoms_coords(self):
        return self.molecule.getCoordinates()
