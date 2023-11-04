from lib.tools import Results
from typing import List, Optional
from lib.tools import OptResult
import abc
import numpy as np

class Check(abc.ABC):
    def __init__(self, name) -> None:
        self.index: int = None
        self.datas: List[OptResult] = None
        self.count = 0
        self.name = name

    def setData(self, datas: List[OptResult], index: int) -> bool:
        self.index = index
        self.datas = datas

    def do(self):
        '''
        如果有問題，回傳True
        '''
        if self.datas is None or self.index is None:
            raise Exception("0")

    def check(self, a: bool):
        if a == True:
            self.record()

    def record(self):
        self.count += 1

    def __str__(self) -> str:
        return f"{self.name}: {self.count: 4}"

class CheckError(Check):
    def __init__(self) -> None:
        super().__init__("Error")

    def do(self):
        super().do()
        check_errors: List[bool] = [data.results[self.index].error for data in self.datas]
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
        check_rebuilds = np.array([data.results[self.index].rebuild for data in self.datas])
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
        super().do()
        check_maxs = np.array([data.results[self.index].steps for data in self.datas])
        '''
        all(t, t, t) -> t
        all(f, t, t) -> f
        '''
        result = np.any(check_maxs >= self.max_iter)
        self.check(result)
        return result