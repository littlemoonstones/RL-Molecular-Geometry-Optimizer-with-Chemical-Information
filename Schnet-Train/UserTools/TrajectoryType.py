from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class TrajectoryType:
    smile: str
    coord_type: str
    gradients: List[np.ndarray]
    coords: List[np.ndarray]
    actions: List[np.ndarray]