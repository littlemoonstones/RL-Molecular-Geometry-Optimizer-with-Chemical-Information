import abc
from typing import List, Tuple, Union
from pysisyphus.Geometry import Geometry
import numpy as np


class Reward(abc.ABC):
    def __init__(self) -> None:
        pass
    @abc.abstractmethod
    def getReward(self, pre_gradients: np.ndarray, cur_gradients: np.ndarray, steps: np.ndarray):
        return NotImplemented

class Only1(Reward):
    def __init__(self) -> None:
        super().__init__()
    
    def getReward(self, pre_gradients: np.ndarray, cur_gradients: np.ndarray, steps: np.ndarray):
        return -1

class Reward1(Reward):
    def __init__(self) -> None:
        super().__init__()
    
    def getReward(self, pre_gradients: np.ndarray, cur_gradients: np.ndarray, steps: np.ndarray):
        force = np.abs(cur_gradients).max()
        reward = (np.log(force/2.5e-3))/np.log(2.5e-3) - 1
        return reward

class Reward2(Reward):
    def __init__(self) -> None:
        super().__init__()
    
    def getReward(self, pre_gradients: np.ndarray, cur_gradients: np.ndarray, steps: np.ndarray):
        force = np.abs(cur_gradients).max()
        reward = (np.log(force//4.5e-4))/np.log(4.5e-4) - 1
        return reward

class Reward3(Reward):
    def __init__(self) -> None:
        super().__init__()
    
    def getReward(self, pre_gradients: np.ndarray, cur_gradients: np.ndarray, steps: np.ndarray):
        pre_grads = self.getAbsMax(pre_gradients)
        cur_grads = self.getAbsMax(cur_gradients)
        max_steps = self.getAbsMax(steps)

        log_steps = np.log(max_steps)

        eta = cur_grads/pre_grads
        log_eta = np.log10(eta)

        sign = -np.sign(log_eta)
        sign = (sign == -1).astype(np.float64)
        reward: np.ndarray = 0.6*(0.2*np.log(cur_grads/4.5*10**-4)/np.log(4.5*10**-4)-1) + \
            0.4*((-log_steps)/10-log_eta*((1-sign) * 0.1 + sign*3))

        return reward

    def getAbsMax(self, array: np.ndarray) -> np.ndarray:
        return np.max(np.abs(array))

class Reward3Avg(Reward):
    def __init__(self) -> None:
        super().__init__()
    
    def getReward(self, pre_gradients: np.ndarray, cur_gradients: np.ndarray, steps: np.ndarray):
        pre_grads = abs(pre_gradients)
        cur_grads = abs(cur_gradients)
        max_steps = abs(steps)

        log_steps = np.log(max_steps)

        eta = cur_grads/pre_grads
        log_eta = np.log10(eta)

        sign = -np.sign(log_eta)
        sign = (sign == -1).astype(np.float64)
        reward: np.ndarray = 0.6*(0.2*np.log(cur_grads/4.5*10**-4)/np.log(4.5*10**-4)-1) + \
            0.4*((-log_steps)/10-log_eta*((1-sign) * 0.1 + sign*3))

        return reward.mean()
