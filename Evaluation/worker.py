import os
THREADS = 1
os.environ["MKL_NUM_THREADS"] = str(THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)
os.environ["OMP_NUM_THREADS"] = str(THREADS)

from typing import List, Optional
import yaml
from lib.tools import (
    Results,
    OptResult,
    Opt_Result,
    measureTime,
)
import numpy as np
from lib.args_builder import WorkerArgsBuilder
from pysisyphus.drivers.opt import get_opt_cls
from pysisyphus.run import get_calc_closure
from UserTools.createMolecule import getMolecule
from pysisyphus.optimizers import Optimizer
from pysisyphus.Geometry import Geometry
from pysisyphus.constants import ANG2BOHR

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import tempfile


@dataclass
class ModelMeta:
    name: str
    version: str
    seed: str


@measureTime
def optimize(opt: Optimizer):
    error = False
    try:
        opt.run()
    except:
        error = True
    return dict(
        steps=opt.cur_cycle,
        isConverged=opt.is_converged,
        error=error,
    )


def run(
    smile: str,
    index: int,
    coords_group: List[np.ndarray],
    opt_key: str,
    coord_type="redund",
    calc_key: str = 'mmff',
    calc_kwargs: dict = {},
    modelMeta: Optional[ModelMeta] = None,
) -> OptResult:

    with open("perfect_primitives.pk", "rb")as fs:
        default_primitives: dict = pickle.load(fs)
    _, atoms, _, _seed = getMolecule(smile)

    results: OptResult = OptResult(
        smile=smile,
        index=index,
        N_atoms=len(atoms),
        coord_type=coord_type,
        results=[]
    )
    # create a temporary directory because some calculators will generate temporary files
    tmp_dir_root = Path("qm_calcs")
    tmp_dir_root.mkdir(exist_ok=True)
    for coords in coords_group:
        time_list = []
        for i in range(args.repeat_num):
            tmp_dir = tempfile.TemporaryDirectory(
                prefix=f"{opt_key}-",
                dir=tmp_dir_root
            )

            if coord_type == "redund":
                coord_kwargs = {
                    "typed_prims": default_primitives.get(smile),
                }
            elif coord_type == "cart":
                coord_kwargs = None
            else:
                raise Exception("no coordinate system")

            geometry = Geometry(
                atoms,
                coords*ANG2BOHR,
                coord_type=coord_type,
                coord_kwargs=coord_kwargs
            )

            if calc_key == 'mmff':
                calc_kwargs['smile'] = smile
                calc_kwargs['seed'] = _seed

            calc_kwargs['out_dir'] = tmp_dir.name
            calculator = get_calc_closure(
                'calculator', calc_key, calc_kwargs)()

            geometry.set_calculator(calculator)

            if opt_key == 'rl':
                kwargs = vars(modelMeta)
            else:
                kwargs = {}

            optimizer = get_opt_cls(opt_key)(
                geometry,
                max_force_only=True,
                max_cycles=300,
                **kwargs
            )

            result: Results = optimize(optimizer)
            
            
            time_list.append(result.time)

            # remember to clean up the temporary directory
            tmp_dir.cleanup()
        tmp = vars(result)
        tmp.pop("time")
        results.results.append(
            Opt_Result(
                rebuild=0,
                time_avg=np.array(time_list).mean(),
                time_std=np.array(time_list).std(),
                **tmp,
            )
        )

    return results


def GetData(file_name: str, prefix: str):
    with open(os.path.join('data', 'test', prefix + '-' + file_name), 'rb') as pk:
        data_dict = pickle.load(pk)
    return data_dict


def ExportResult(folder_path: Path, index: int, result: OptResult):
    with open(Path(folder_path, f"data-{index}.json"), "w")as fs:
        json.dump(result.toJson(), fs, indent=4)


args = WorkerArgsBuilder()

with open(f'test-configs/{args.name}/{args.opt_key}.yml', 'r') as stream:
    configs: dict = yaml.load(stream, Loader=yaml.CLoader)

file_name = configs.get("file_name")
prefix = "per" if args.perturbation else "unper"
opt_key = configs.get("optimizer").pop("key")
calc_key = configs.get("calculator").get("key")
calc_kwargs = {}

coord_type = configs.get("coord_type")

if opt_key == 'rl':
    modelMeta = ModelMeta(
        **configs.get("optimizer")
    )
else:
    modelMeta = None

print("select test file:", file_name)
print("select test file:", modelMeta)

folder = Path(f"Results-{args.name}", opt_key, prefix)
folder.mkdir(parents=True, exist_ok=True)

data = GetData(
    file_name,
    prefix,
)
for index, (smile, coords_group) in enumerate(data):
    results: OptResult = run(
        smile,
        index,
        coords_group,
        opt_key,
        coord_type=coord_type,
        calc_key=calc_key,
        modelMeta=modelMeta,
    )
    ExportResult(folder, index, results)
