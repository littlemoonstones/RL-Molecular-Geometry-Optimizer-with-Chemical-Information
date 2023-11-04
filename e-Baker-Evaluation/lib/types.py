from enum import Enum
import json
import numpy as np
from pathlib import Path
import pickle
from typing import List, Optional, NamedTuple
from dataclasses import dataclass
import prettytable as pt

class PurtabationType(Enum):
    PERTURBED = 'per'
    UNPERTURBED = 'unper'


@dataclass
class OptimizerResult:
    name: str
    steps: float
    time: float  # unit (s)


@dataclass
class Results:
    time: float
    steps: int
    isConverged: bool
    error: bool

Opt_Result = NamedTuple("Opt_Result", [
    ("steps", int), 
    ("error", bool), 
    ("rebuild", int),
    ("time_avg", float),
    ("time_std", float),
    ("isConverged", bool),
])