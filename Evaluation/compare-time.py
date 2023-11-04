import copy
from pathlib import Path
from typing import List
from lib.check import Check, CheckDiverge, CheckError
from lib.optimization_result_handling import \
    PurtabationType, OptimizationResultLoader,\
    AvgStepDataProcessor, TotalStepDataProcessor, display_result_by_table

from lib.types import PurtabationType

ROOT_FOLDER = Path(".").resolve()


_check_list: List[Check] = [
    CheckError(),
    CheckDiverge(max_iter=299),
]

optimizer_names: List[str] = [
    "rl",
    "rl-schnet",
    "rfo",
    "cnewton",
    "lbfgs",
    "bfgs",
]

result_names = [
    "Results-group-C",
]

perturbation_types = [
    PurtabationType.PERTURBED,
    PurtabationType.UNPERTURBED
]

for result_name in result_names:
    for perturbation_type in perturbation_types:
        print(f"# {perturbation_type.value}turbation {result_name}")
        current_check_list = copy.deepcopy(_check_list)
        test_file_name = "test_dict.pk"

        result_loader = OptimizationResultLoader(
            Path(".").resolve(),
            test_file_name,
            result_name,
            perturbation_type,
            optimizer_names
        )
        datas = result_loader.load_all_result_sets()
        try:
            processor = TotalStepDataProcessor(optimizer_names, current_check_list)
            # processor = AvgStepDataProcessor(optimizer_names, current_check_list)
            results = processor.process_data(datas)
        except Exception as e:
            print("all failed", e)

        display_result_by_table(results, ["name", "Total Step", "Total Time"])

        for check in current_check_list:
            print(check)
        print("="*10)
