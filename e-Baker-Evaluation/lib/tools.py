from typing import Callable, NamedTuple, List
from lib.types import Opt_Result, Results
import time
from dataclasses import dataclass

class OptResult:
    def __init__(self, smile: str, index: int, N_atoms: int, coord_type: str, results: List[Opt_Result]) -> None:
        self.smile = smile
        self.index = index
        self.N_atoms = N_atoms
        self.coord_type = coord_type
        self.results: List[Opt_Result] = results

    def toJson(self):
            results = []
            for result in self.results:
                results.append(result._asdict())
            return {
                "smile": self.smile,
                "index": self.index,
                "N_atoms": self.N_atoms,
                "coord_type": self.coord_type,
                "results": results
            }

    @staticmethod
    def toObject(data):
        smile = data["smile"]
        index = data["index"]
        N_atoms = data["N_atoms"]
        coord_type = data["coord_type"]
        results: List[Opt_Result] = []
        for result in data["results"]:
            results.append(Opt_Result(**result))
        return OptResult(smile, index, N_atoms, coord_type, results)

def measureTime(func: Callable):
    def wrap(*args, **kw):
        start_time = time.time()
        results = func(*args, **kw)
        end_time = time.time()
        return Results(
            time=(end_time - start_time),
            **results
        )
    return wrap