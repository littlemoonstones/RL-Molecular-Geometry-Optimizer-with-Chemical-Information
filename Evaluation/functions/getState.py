from pysisyphus.Geometry import Geometry
import numpy as np
from functions.lib import splitLog

import abc

class State(abc.ABC):
    def __init__(self, geometry: Geometry) -> None:
        self.geo = geometry

    @abc.abstractmethod
    def getState(self) -> np.ndarray:
        return NotImplemented
    
    def getSize(self):
        return self.getState().shape[-1]

class LogGradientAndLogCoordinateState(State):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)
    
    def getState(self):
        g_sign, g_array = splitLog(self.geo.gradient)
        c_sign, c_array = splitLog(self.geo.coords)
        return np.hstack([
            g_sign[:, None],
            g_array[:, None],
            c_sign[:, None],
            c_array[:, None],
        ]) 
    

class NaiveGradientState(State):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)
    
    def getState(self):
        return self.geo.gradient[:, None]


class NormalizedGradientState(State):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)
    
    def getState(self):
        n_g = np.linalg.norm(self.geo.gradient)
        return (self.geo.gradient.copy()/n_g)[:, None]

class NaiveGradientAndCoordateState(State):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)
    
    def getState(self):
        return np.hstack([
            self.geo.gradient[:, None], 
            self.geo.coords[:, None],
        ])

class LogGradienttState(State):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)
    
    def getState(self):
        g_sign, g_array = splitLog(self.geo.gradient)
        return np.hstack([
            g_sign[:, None],
            g_array[:, None],
        ]) 

class LogGradientAndNaiveCoordinateState(State):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)
    
    def getState(self):
        g_sign, g_array = splitLog(self.geo.gradient)
        return np.hstack([
            g_sign[:, None],
            g_array[:, None],
            self.geo.coords[:, None],
        ]) 

class StandardizedLogGradientAndNaiveCoordinateState(State):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)
    
    def getState(self):
        array: np.ndarray = self.geo.gradient
        sign = np.sign(array)
        array = np.clip(np.abs(array), 1e-8, 20)
        array = np.log(np.abs(array))
        std = array.std()
        mean = array.mean()
        standardizedGradient =  np.hstack([
            sign[:, None],
            ((array - mean) / std)[:, None],
        ]).astype(np.float32)

        return np.hstack([
            standardizedGradient, 
            self.geo.coords[:, None],
        ])

class StandardizedLogGradient(State):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)
    
    def getState(self):
        array: np.ndarray = self.geo.gradient
        sign = np.sign(array)
        array = np.clip(np.abs(array), 1e-8, 20)
        array = np.log(np.abs(array))
        std = array.std()
        mean = array.mean()

        return np.hstack([
            sign[:, None],
            ((array - mean) / std)[:, None],
        ]).astype(np.float32)



class LogGradienttStateNormalized(State):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)
    
    def getState(self):
        g_sign, g_array = splitLog(self.geo.gradient)
        return np.hstack([
            g_sign[:, None],
            g_array[:, None]/10,
        ]) 