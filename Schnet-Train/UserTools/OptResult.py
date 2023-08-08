from typing import NamedTuple, List

Opt_Result = NamedTuple("Opt_Result", [("steps", int), ("error", bool), ("rebuild", int)])
class OptResult:
    def __init__(self, smile: str, index: int, N_atoms: int, coord_type: str, results: List[Opt_Result]) -> None:
        self.smile = smile
        self.index = index
        self.N_atoms = N_atoms
        self.coord_type = coord_type
        self.results: List[Opt_Result] = results
        pass
    pass

    def toJson(self):
            results = []
            for result in self.results:
                results.append({
                    "steps": result.steps,
                    "error": result.error,
                    "rebuild": result.rebuild
                })
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
            steps = result["steps"]
            error = result["error"]
            rebuild = result["rebuild"]
            results.append(Opt_Result(steps, error, rebuild))
        return OptResult(smile, index, N_atoms, coord_type, results)