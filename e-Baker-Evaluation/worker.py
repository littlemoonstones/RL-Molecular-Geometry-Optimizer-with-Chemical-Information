import os
THREADS = 1
os.environ["MKL_NUM_THREADS"] = str(THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)
os.environ["OMP_NUM_THREADS"] = str(THREADS)

from dataclasses import dataclass
import yaml

from lib.tools import (
    OptResult,
)
from pysisyphus.drivers.opt import get_opt_cls
from pysisyphus.run import get_calc_closure
from pysisyphus.optimizers import Optimizer
from pysisyphus.helpers import geom_from_xyz_file

from pathlib import Path
import tempfile
import argparse

@dataclass
class ModelMeta:
    name: str
    version: str
    seed: str

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--file_name", help="source name",
                    type=str)
parser.add_argument("--config_name", help="source name",
                    type=str)
parser.add_argument("--group_name", help="source name",
                    type=str)
parser.add_argument("--method", help="source name",
                    type=str)
parser.add_argument("--basis", help="source name",
                    type=str)

args = parser.parse_args()

def run(
    file_name: str,
    group_name: str,
    config_name: str,
    coord_type="redund",
    calc_key: str = 'psi4',
    calc_kwargs: dict = {},
) -> OptResult:
    with open(f'test-configs/{group_name}/{config_name}.yml', 'r') as stream:
        configs: dict = yaml.load(stream, Loader=yaml.CLoader)
    opt_key = configs.get("optimizer").pop("key")

    # create a temporary directory because some calculators will generate temporary files
    tmp_dir_root = Path("qm_calcs")
    tmp_dir_root.mkdir(exist_ok=True)
    tmp_dir = tempfile.TemporaryDirectory(
        prefix=f"{group_name}-{opt_key}-",
        dir=tmp_dir_root
    )


    geometry = geom_from_xyz_file(file_name, coord_type=coord_type)
    print(geometry.coord_type)

    calc_kwargs['out_dir'] = tmp_dir.name

    if  Path(file_name).stem.startswith("32_histamine_H+"):
        calc_kwargs['charge'] = 1
        
    calculator = get_calc_closure(
        'calculator', calc_key, calc_kwargs)()

    geometry.set_calculator(calculator)
    print('calculator', geometry.calculator.__class__.__name__)
    print('calculator', calc_kwargs)

    if opt_key == 'rl' or opt_key == 'rls':
        kwargs = configs.get("optimizer")
    else:
        kwargs = {}

    optimizer: Optimizer = get_opt_cls(opt_key)(
        geometry,
        thresh='gau',
        max_cycles=300,
        **kwargs
    )

    optimizer.run()

    # remember to clean up the temporary directory
    tmp_dir.cleanup()

# print('running...',)
# print(args.file_name)
# energy, gradients = run(
#     args.file_name, 
#     args.group_name,
#     args.config_name,
#     calc_key="psi4",
#     calc_kwargs=dict(
#         method=args.method,
#         basis=args.basis
#     )
# )
