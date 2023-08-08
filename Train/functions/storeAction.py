import abc
from pysisyphus.Geometry import Geometry
from typing import Optional
import numpy as np

class StoreActionMethod(abc.ABC):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__()
        self._n = 0
        self.geo = geometry

    def getSize(self):
        return self._n
    
    def setSize(self, new_size: int):
        self._n = new_size

    def getInitAction(self):
        return np.zeros(shape=(len(self.geo.coords), self.getSize()))
    
    @abc.abstractmethod
    def transform(self, array: np.ndarray) -> Optional[np.ndarray]:
        return NotImplementedError
    
    @staticmethod
    def splitToLog(array: np.ndarray):
        sign = np.sign(array)
        array = np.clip(np.abs(array), 1e-20, 20)
        array = np.log(np.abs(array))
        return sign, array

class StoreNoneAction(StoreActionMethod):
    def __init__(self, geometry) -> None:
        super().__init__(geometry)
        self.setSize(0)
    def transform(self, array: np.ndarray):
        return None

class StoreNaiveAction(StoreActionMethod):
    def __init__(self, geometry) -> None:
        super().__init__(geometry)
        self.setSize(1)
    
    def transform(self, array: np.ndarray):
        return array[:, None]

class StoreLogAction(StoreActionMethod):
    def __init__(self, geometry) -> None:
        super().__init__(geometry)
        self.setSize(2)
    
    def transform(self, array: np.ndarray):
        sign, array = StoreActionMethod.splitToLog(array)
        return np.hstack([
            sign[:, None],
            array[:, None],
        ]).astype(np.float32)

class StoreLogActionNormalized(StoreActionMethod):
    def __init__(self, geometry) -> None:
        super().__init__(geometry)
        self.setSize(2)
    
    def transform(self, array: np.ndarray):
        sign, array = StoreActionMethod.splitToLog(array)
        return np.hstack([
            sign[:, None],
            array[:, None]/10,
        ]).astype(np.float32)
    