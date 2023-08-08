import abc
import pickle
import numpy as np
from typing import List
from lib.check import Check, CheckDiverge, CheckError
from Evaluation.lib.optimization_result_handling_original import \
    PurtabationType, OptResult, OptimizationResultLoader
from dataclasses import dataclass

@dataclass
class OptimizerResult:
    name: str
    average_steps: float
    average_time: float  # unit (s)

class DataProcessor(abc.ABC):

    def __init__(
            self, 
            all_files: List[str], 
            check_list: List[Check]
        ) -> None:
        """
        all_files: list of file names or optimizer names
        check_list: list of Check objects
        """

        self.all_files = all_files
        self.check_list = check_list
        self.total = 0 # total number of structures
        self.results: List[List[int]] = [[] for _ in range(len(all_files))]
        """
        results: list of list of steps for each optimizer
        for example: results[0] is a list of steps for optimizer 0
        """
        
        self.time_results: List[List[int]] = [[] for _ in range(len(all_files))]

        self.records = [[[]] for _ in range(len(self.all_files))]

    def process_data(self, data: List[List[OptResult]]) -> List[OptimizerResult]:
        """
        Process data from different optimizers
        """

        # process each smile
        for smile_index, datas in enumerate(data):
            try:
                if datas is None:
                    continue  # if one of results from different optimizers fails, skip this molecule
                
                # each smile has 10 different structures
                for j in range(max([len(data.results) for data in datas])):
                    
                    # check a structure for each optimizer if the structure is diverged or error
                    for check in self.check_list:
                        check.setData(datas, j)
                    checks: List[bool] = [check.do() for check in self.check_list]

                    for optimizer_index, _ in enumerate(datas):
                        # if new smile appears, create a new list to store the steps of the structure
                        if(len(self.records[optimizer_index]) <= smile_index):
                            self.records[optimizer_index].append([])
                    
                    self.total += 1

                    # Skip this structure if one of the structure in optimizers is diverged or error.
                    if any(checks): continue

                    self.collect_results(smile_index, datas, j)
            except AttributeError:
                print("error")
                continue

        return self.prepare_output()

    @abc.abstractmethod
    def collect_results(self, smile_index: int, datas: List[OptResult], j: int):
        '''
        Collect results from each optimizer for a specific structure
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_output(self):
        raise NotImplementedError


class AvgStepDataProcessor(DataProcessor):
    def collect_results(self, smile_index: int, datas: List[OptResult], j: int):
        for index, tmp_data in enumerate(datas):
            try:
                self.results[index].append(tmp_data.results[j].steps)
                self.time_results[index].append(
                    tmp_data.results[j].time_avg
                )
            except:
                print(index)
                raise Exception

    def prepare_output(self):
        result: List[int]
        avg_list: List[OptimizerResult] = []
        for index, result in enumerate(self.results):
            clip_result = np.clip(result, 0, 300)
            name = self.all_files[index]
            avg_list.append(
                OptimizerResult(
                    name=name,
                    average_steps=np.average(clip_result),
                    average_time=np.average(self.time_results[index]),
                )
            )
        print(
            f"Completion: {len(result)}/{self.total} -> {round(len(result)/self.total*100, 2)}%"
        )

        return avg_list


class StepPerSmileDataProcessor(DataProcessor):
    def __init__(
            self, 
            all_files: List[str], 
            check_list: List[Check]
        ) -> None:
        super().__init__(all_files, check_list)
        with open("perfect_primitives.pk", "rb")as fs:
            self.default_primitives: dict = pickle.load(fs)
        
        self.ans = []
        # self.records = [[[]] for _ in range(len(self.all_files))]
        '''
        [a][b][c]
        a: optimizer_index
        b: smile_index
        c: steps_index
        '''
        

    def collect_results(self, smile_index: int, datas: List[OptResult], j: int):
        tmp_results: List[List[int]] = [[] for _ in range(len(self.all_files))]
        # print(f" {datas[0].smile}, ", datas[0].results)
        for index, tmp_data in enumerate(datas):
            tmp_results[index].append(tmp_data.results[j].steps)
            try:
                self.records[index][smile_index].append(tmp_data.results[j].steps)
            except:
                print(index, smile_index, len(self.records[index]))
                print(j, len(tmp_data.results))
                raise Exception
        for _i in range(len(self.all_files)):
            if len(tmp_results[_i]) == 0:
                break
            average = np.average(tmp_results[_i])
            self.results[_i].append(average)

    def prepare_output(self):
        result: List[int]
        avg_list: List[float] = []
        for index, result in enumerate(self.results):
            clip_result = np.clip(result, 0, 300)
            metaFile = self.all_files[index]
            name = f"{metaFile}"
            print(f"{index:02} {name:<20} avg:{round(np.average(clip_result), 2):>7}, max: {max(result)}")
            avg_list.append(np.average(clip_result))

        print(f"Completion: {len(result)}/{self.total} -> {round(len(result)/self.total*100, 2)}%")
        return self.records