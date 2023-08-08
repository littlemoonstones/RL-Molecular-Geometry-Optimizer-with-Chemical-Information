from lib.tools import Results
from typing import List, Optional
from lib.tools import OptResult
import abc
import numpy as np


class Check(abc.ABC):
    def __init__(self, name) -> None:
        '''
        Parameters:
        name (str): The name of the check, used for identification and reporting.

        Attributes:
        index (int): Index of data to check in the provided data list
        datas (List[OptResult]): List of data on which checks will be performed
        count (int): Counter for the number of times a check returns True (indicates a problem)
        name (str): The name of the check, used for identification and reporting.
        """
        '''
        self.index: int = None
        self.datas: List[OptResult] = None
        self.count = 0
        self.name = name

    def setData(self, datas: List[OptResult], index: int):
        self.index = index
        self.datas = datas

    def do(self):
        '''
        Perform the check. If there is a problem, return True.
        Raise an exception if the check is attempted without setting the data or index.
        This method should be overridden in subclasses to implement specific checks.
        '''
        if self.datas is None or self.index is None:
            raise Exception("Data or index not set")

    def check(self, condition: bool):
        if condition:
            self.record()

    def record(self):
        self.count += 1

    def __str__(self) -> str:
        return f"{self.name}: {self.count: 4}"


class CheckError(Check):
    def __init__(self) -> None:
        super().__init__("Error")

    def do(self):
        '''
        Perform the "CheckError" operation.

        This method checks for errors in the optimization results of the specified index
        within the provided list of datasets (OptResult objects).

        Returns:
        bool: True if any of the datasets have an error at the specified index, False otherwise.
        '''
        super().do()
        check_errors: List[bool] = [
            data.results[self.index].error for data in self.datas]
        '''
        any(f, f, f) -> f
        any(f, t, f) -> t
        '''
        result = any(check_errors)
        # print(check_errors, result)
        self.check(result)
        return result


class CheckRebuild(Check):
    def __init__(self) -> None:
        super().__init__("Rebuild")

    def do(self):
        super().do()
        check_rebuilds = np.array(
            [data.results[self.index].rebuild for data in self.datas])
        '''
        all(t, t, t) -> t
        all(f, t, t) -> f
        '''
        result = np.any(check_rebuilds > 0)
        self.check(result)
        return result


class CheckDiverge(Check):
    def __init__(self, max_iter=999) -> None:
        self.max_iter = max_iter
        super().__init__("Diverge")

    def do(self):
        """
        Perform the "CheckDiverge" operation.

        This method checks if any of the optimization results of the specified index
        within the provided list of datasets (OptResult objects) have reached the maximum
        allowed number of iterations for convergence.

        Returns:
        bool: True if any of the datasets have exceeded the maximum allowed iterations, False otherwise.
        """
        super().do()
        check_maxs = np.array(
            [data.results[self.index].steps for data in self.datas])
        '''
        all(t, t, t) -> t
        all(f, t, t) -> f
        '''
        result = np.any(check_maxs >= self.max_iter)
        self.check(result)
        return result
