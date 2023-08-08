from typing import List, Optional, Tuple, Union, Dict
from pysisyphus.Geometry import Geometry
from pysisyphus.intcoords.Primitive import Primitive
from pysisyphus.intcoords.Bend import Bend
from pysisyphus.intcoords.Stretch import Stretch
from pysisyphus.intcoords.Torsion import Torsion
from pysisyphus.intcoords.LinearBend import LinearBend
from rdkit import Chem
from collections import Counter
import abc
import numpy as np

ATOMS_TYPE: List[Optional[str]] = [
    "H",
    "C",
    "O",
    "N",
]

ATOMS_ONEHOT = {
    "H": np.array([1, 0, 0, 0]),
    "C": np.array([0, 1, 0, 0]),
    "O": np.array([0, 0, 1, 0]),
    "N": np.array([0, 0, 0, 1]),
}

def getNeighborAtomsEncoded(smile: str, geometry: Geometry) ->  Dict[int, np.ndarray]:
    data: Dict[int, np.ndarray] = {}
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    for i, atom in enumerate(geometry.atoms):
        rdkit_atom = mol.GetAtomWithIdx(i)
        
        # Calculate neighbor atoms
        counter = Counter([x.GetSymbol() for x in rdkit_atom.GetNeighbors()])
        atom_num_encode = np.zeros(len(ATOMS_TYPE))
        for j, atom_type in enumerate(ATOMS_TYPE):
            atom_num_encode[j] = counter[atom_type]
        
        data[i] = np.hstack([
            ATOMS_ONEHOT[atom], # atom name
            atom_num_encode     # atom type
        ])

    return data

def Encoded(geometry: Geometry):
    array = []
    primitive: Primitive
    for primitive in geometry.internal.primitives:
        encoded = np.zeros(3)
        if isinstance(primitive, Stretch):
            encoded[0] = 1
        elif isinstance(primitive, (Bend, LinearBend)):
            encoded[1] = 1
        elif isinstance(primitive, Torsion):
            encoded[2] = 1
        else:
            raise Exception()
        array.append(encoded)
    return np.array(array)

def AtomEncoded(geometry: Geometry):
        array = []
        primitive: Primitive
        atoms: List[str] = geometry.atoms

        for primitive in geometry.internal.primitives:
            encoded = np.zeros(len(ATOMS_ONEHOT["H"]))
            for index in primitive.indices:
                atom = atoms[index]
                encoded += ATOMS_ONEHOT[atom]
            array.append(encoded)
        return np.array(array)

class EncodeClass(abc.ABC):
    def __init__(self, geometry: Geometry, smile: Optional[str]=None) -> None:
        self.geometry = geometry
        self.smile = smile
        super().__init__()
    
    @abc.abstractmethod
    def getEncode(self):
        raise NotImplemented

class BasicEncode(EncodeClass):
    def __init__(self, geometry: Geometry, smile=None) -> None:
        super().__init__(geometry)
        self.basic_encode = Encoded(self.geometry)
    
    def getEncode(self):
        return self.basic_encode

class AtomEncode(BasicEncode):
    def __init__(self, geometry: Geometry, smile=None) -> None:
        super().__init__(geometry)
        self.encode = AtomEncoded(geometry)

    def getEncode(self):
        return np.hstack([
            self.basic_encode,
            self.encode
        ])
    

class NeighborEncode(BasicEncode):
    def __init__(self, geometry: Geometry, smile: str) -> None:
        super().__init__(geometry, smile)
        self.neighbor_encode = getNeighborAtomsEncoded(smile, geometry)
        self.encode = self.transform()
    
    def getEncode(self):
        return np.hstack([
            self.basic_encode,
            self.encode
        ])
    
    def transform(self):
        array = []
        primitive: Primitive
        for primitive in self.geometry.internal.primitives:
            encoded = np.zeros(len(self.neighbor_encode[0]))
            for index in primitive.indices:
                encoded += self.neighbor_encode[index]
            array.append(encoded)
        return np.array(array)

    
