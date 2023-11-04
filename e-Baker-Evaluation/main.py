import os
THREADS = 1
os.environ["MKL_NUM_THREADS"] = str(THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)
os.environ["OMP_NUM_THREADS"] = str(THREADS)

from pathlib import Path
from natsort import natsorted
import worker

ROOT_FOLDER = Path(".").resolve()
CONFIG_ROOT_DIR = Path(ROOT_FOLDER, "test-configs")

group_name = "group-paperFinetune01"
folder_path = "dataset/e-Baker-XYZ"
config_name = "v1_708700354"

file_list = [(file_path.stem, file_path.resolve()) for file_path in Path('..', folder_path).glob("*.xyz")]
# sort by file name by natural sort
natsorted(file_list, key=lambda x: x[0])
for file_name, file_path in file_list:
    print("="*10 + file_name + "="*10)
    worker.run(
        file_name=file_path,
        group_name=group_name,
        config_name=config_name,
        calc_key="psi4",
        calc_kwargs=dict(
            method="b3lyp",
            basis="def2-svp",
        )
    )
