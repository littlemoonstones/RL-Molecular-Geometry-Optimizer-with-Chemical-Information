from dataclasses import dataclass
from typing import List
import numpy as np
import torch

@dataclass
class SchNetMetaData:
    n_atom_basis: int
    n_filters: int
    n_interactions: int

@dataclass
class ModelMetaData:
    input_size: int
    act_size: int
    crt_size: int
    hidden_size: int
    n_head: int
    schnet_metadata: SchNetMetaData
    device: torch.device


@dataclass
class TrainingData:
    atoms: List[str]
    coords: np.ndarray
    pre_features: np.ndarray