import abc
from typing import List, Tuple, Union
from pysisyphus.Geometry import Geometry
import numpy as np
from functions.lib import splitLog


def transformLog(array: np.ndarray):
    array = np.clip(array, 1e-8, 20)
    sign = np.sign(array)
    array = np.log(np.abs(array))
    return np.concatenate([
        sign,
        array,
    ]).astype(np.float32)


# low_val = -8
# high_val = -2

low_val = -21
high_val = -3
def transform(x): return (high_val-low_val)/2 * (x+1)+low_val


def getActionFromNaive(geometry: Geometry, action: np.ndarray):
    return action


def getActionFromLog(geometry: Geometry, action: np.ndarray):
    exp, sign = action[:len(geometry.coords)], action[len(geometry.coords):]
    exp = transform(exp)
    sign[sign >= 0.5] = 1
    sign[sign < 0.5] = -1
    return np.exp(exp) * sign


class Action(abc.ABC):
    def __init__(self, geometry: Geometry) -> None:
        self.geo = geometry
        self.coord_length = len(geometry.coords)

    @abc.abstractmethod
    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        return NotImplemented

    # def getInputSize(self):
    #     low_space, high_space = self.getActionSpace()
    #     return len(low_space)

    @abc.abstractmethod
    def getActionSpace(self) -> Tuple[List[int], List[int]]:
        return NotImplemented

    # @abc.abstractmethod
    # def getTrainingAction(self, action: np.ndarray, gradients: np.ndarray):
    #     return NotImplemented

    def transform(self, x: Union[float, np.ndarray], high_val: float, low_val: float):
        return (high_val-low_val)/2 * (x+1)+low_val


class NaiveAction(Action):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        return action[:self.coord_length].squeeze(-1) # 2D([N_atoms, action_space]) to 1D([N_atoms])

    def getTrainingAction(self, action: np.ndarray, gradients: np.ndarray):
        # action is 1D
        return action[:, None] # 2D

    # def getInputSize(self):
    #     return len(self.geo.coords)

    def getActionSpace(self):
        low_space = [-1]
        high_space = [1]
        return low_space, high_space


class LogAction(Action):
    def __init__(self, geometry: Geometry, low_val: float = -8, high_val: float = -1) -> None:
    # def __init__(self, geometry: Geometry, low_val: float = -21, high_val: float = -3) -> None:
    # def __init__(self, geometry: Geometry, low_val: float = -21, high_val: float = -3) -> None:
    # def __init__(self, geometry: Geometry, low_val: float = -8, high_val: float = -2) -> None:
        super().__init__(geometry)
        self.low_val = low_val
        self.high_val = high_val

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        # print(action.shape)
        # raise Exception
        action = action[:self.coord_length]
        exp = action[:, 0]
        # -1 ~ 1
        sign = np.sign(action[:, 1])
                           
        exp = self.transform(exp, self.high_val, self.low_val)
        # sign[sign >= 0.5] = 1
        # sign[sign < 0.5] = -1
        return sign * np.exp(exp)
    
    def getTrainingAction(self, action: np.ndarray, gradients: np.ndarray):
        # action is 1D
        sign, array = splitLog(action)

        return np.hstack([
           sign[:, None], 
           array[:, None], 
        ]) # 2D

    # def getInputSize(self):
    #     return len(self.geo.coords) * 2

    # def transform(self, x: float):
    #     return (self.high_val-self.low_val)/2 * (x+1)+self.low_val

    def getActionSpace(self):
        low_space: List[int] = [-1]
        high_space: List[int] = [1]
        low_space.extend([-1])
        high_space.extend([1])
        return low_space, high_space


class NaiveActionFactor(NaiveAction):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)
        self.high_val = 0
        self.low_val = -8

    def getActionSpace(self):
        low_space, high_space = super().getActionSpace()
        low_space.append(-1)
        high_space.append(1)
        return low_space, high_space

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        action = action[:self.coord_length]
        action, factor = action[:, 0], action[:, 1]
        print("factor", np.exp(self.transform(factor, self.high_val, self.low_val)))
        return action * np.exp(self.transform(factor, self.high_val, self.low_val))


class LogActionFactor(LogAction):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        action = action[:self.coord_length]
        exp = action[:, 0]
        sign = action[:, 1]
        factor = action[:, 2]

        exp = self.transform(exp, self.high_val, self.low_val)
        sign[sign >= 0.5] = 1
        sign[sign < 0.5] = -1
        return sign * np.exp(exp) * np.exp(self.transform(factor, 0, -8))

    def getActionSpace(self):
        low_space, high_space = super().getActionSpace()
        low_space.append(-1)
        high_space.append(1)
        return low_space, high_space


def standarized(array: np.ndarray):
    error = 1e-40
    std = array.std()
    mean = array.mean()
    # try:
        # t = (array - mean) / std
    # except:
    # print("error")
    # print("array", array)
    # print("mean", mean)
    # print("std", std)
    return (array - mean) / (std+error), mean, std

class StandardizedActionFactor(Action):
    def __init__(self, geometry: Geometry, mean_factor: float = 1, std_factor: float = 1) -> None:
        super().__init__(geometry)
        self.mean_factor = mean_factor
        self.std_factor = std_factor
    
    def destandardize(self, array: np.ndarray, mean: float, std: float, mean_factor: float, std_factor: float):
        return array * std * std_factor + mean * mean_factor

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        # print("np.isnan", np.isnan(action.sum()))
        if np.isnan(action.sum()) == True:
            raise Exception("there is nana in action")
        sign = np.sign(gradients)
        exp_raw = action[:self.coord_length].squeeze()
        exp_standardized, _, _ = standarized(exp_raw)
        log_grad = np.log(np.abs(gradients))
        _, grad_mean, grad_std = standarized(log_grad)

        # print("raw", exp_raw)
        # print("stand raw", exp_standardized)

        mean_factor, std_factor = self.mean_factor, self.std_factor
        # mean_factor, std_factor = action[-2], action[-1]
        # exp = exp_standardized * grad_std * std_factor + grad_mean * mean_factor
        exp = self.destandardize(exp_standardized, grad_mean, grad_std, mean_factor, std_factor)
        if np.isnan(np.exp(exp).sum()):
            print("exp_raw", exp_raw)
            print("exp_standardized", exp_standardized)
            print("log_grad", log_grad)
            print("grad_mean", grad_mean)
            print("grad_std", grad_std)
            print("exp", exp)
            raise Exception("transform Error")
        return -sign * np.exp(exp)

    def getActionSpace(self):
        low_space = [-1]
        high_space = [1]
        return low_space, high_space

    def getTrainingAction(self, action: np.ndarray, gradients: np.ndarray):
        # transform optimizer's action into output. 1D
        sign = np.sign(gradients)
        exp_raw = np.log(np.abs(action))
        exp_standardized, _, _ = standarized(exp_raw)
        log_grad = np.log(np.abs(gradients))
        _, grad_mean, grad_std = standarized(log_grad)


        mean_factor, std_factor = self.mean_factor, self.std_factor
        
        return np.hstack([
           sign[:, None], 
        #    array[:, None], 
        ]) # 2D

class StandardizedAction08Factor(StandardizedActionFactor):
    def __init__(self, geometry: Geometry, mean_factor: float = 0.8, std_factor: float = 0.8) -> None:
        super().__init__(geometry, mean_factor, std_factor)
  
    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        return super().getAction(action, gradients)

    def getActionSpace(self):
        return super().getActionSpace()


class StandardizedActionAutoFactor(StandardizedActionFactor):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        sign = np.sign(gradients)
        action = action[:self.coord_length]
        exp_raw = action[:, 0]
        exp_standardized, _, _ = standarized(exp_raw)
        log_grad = np.log(np.abs(gradients))
        _, grad_mean, grad_std = standarized(log_grad)

        raw_mean_factor, raw_std_factor = action[:, 1], action[:, 2]
        mean_factor = self.transform(raw_mean_factor, 1, 0.7)
        std_factor = self.transform(raw_std_factor, 1, 0.7)
        exp = self.destandardize(exp_standardized, grad_mean, grad_std, mean_factor, std_factor)
        return -sign * np.exp(exp)

    def getActionSpace(self):
        low_space, high_space = super().getActionSpace()
        low_space.extend([-1, -1])
        high_space.extend([1, 1])
        return low_space, high_space

class StandardizedActionAutoLinearFactor(StandardizedActionFactor):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        '''

        '''
        sign = np.sign(gradients)
        action = action[:self.coord_length]
        exp_raw = action[:, 0]
        exp_standardized, _, _ = standarized(exp_raw)
        log_grad = np.log(np.abs(gradients))
        # grad_array, grad_mean, grad_std = standarized(log_grad)

        raw_a, raw_b = action[:, 1], action[:, 2]
        a = self.transform(raw_a, 0.9, 0.3)
        # b = self.transform(raw_b, 1, 0.7)
        exp = log_grad * a + raw_b

        return -sign * np.exp(exp)

    def getActionSpace(self):
        low_space, high_space = super().getActionSpace()
        low_space.extend([-1, -1])
        high_space.extend([1, 1])
        return low_space, high_space


class ActionFactor(Action):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)
    
    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        action = action[:self.coord_length].squeeze()
        print(self.transform(action, 0.5, 0))
        return -gradients*self.transform(action, 0.5, 0)

    def getActionSpace(self):
        low_space = [-1]
        high_space = [1]
        return low_space, high_space

class LogActionNoSign(Action):
    def __init__(self, geometry: Geometry, low_val: float = -12, high_val: float = 0) -> None:
        # def __init__(self, geometry: Geometry, low_val: float = -8, high_val: float = -2) -> None:
        super().__init__(geometry)
        self.low_val = low_val
        self.high_val = high_val

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        # print(action.shape)
        # raise Exception
        action = action[:self.coord_length].squeeze(-1)
        print("raw", action)
        exp = action
        # -1 ~ 1
        sign = np.sign(gradients)
                           
        exp = self.transform(exp, self.high_val, self.low_val)
        return -sign * np.exp(exp)
    
    def getActionSpace(self):
        low_space: List[int] = [-1]
        high_space: List[int] = [1]
        return low_space, high_space

class StandardizedActionAutoLinearFactorFixed(StandardizedActionFactor):
    def __init__(self, geometry: Geometry, a_low=0.3, a_high=0.9, b_low=-1, b_high=1) -> None:
        self.a_low = a_low
        self.a_high = a_high
        self.b_low = b_low
        self.b_high = b_high
        super().__init__(geometry)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        '''

        '''
        sign = np.sign(gradients)
        action = action[:self.coord_length]
        log_grad = np.log(np.abs(gradients))
        # grad_array, grad_mean, grad_std = standarized(log_grad)

        raw_a, raw_b = action[:, 0], action[:, 1]
        # a = self.transform(raw_a, 0.9, 0.3)
        a = self.transform(raw_a, self.a_high, self.a_low)
        b = self.transform(raw_b, self.b_high, self.b_low)
        exp = log_grad * a + b

        return -sign * np.exp(exp)

    def getActionSpace(self):
        low_space, high_space = super().getActionSpace()
        low_space.extend([-1])
        high_space.extend([1])
        return low_space, high_space

class StandardizedActionAutoLinearFactorFixedSign(StandardizedActionFactor):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        '''

        '''
        sign = np.sign(gradients)
        action = action[:self.coord_length]
        log_grad = np.log(np.abs(gradients))
        # grad_array, grad_mean, grad_std = standarized(log_grad)

        sign_probability = (action[:, 0]+1)/2
        sign[sign_probability < 0.2] *= -1
        raw_a, raw_b = action[:, 1], action[:, 2]
        a = self.transform(raw_a, 0.9, 0.3)
        # b = self.transform(raw_b, 1, 0.7)
        exp = log_grad * a + raw_b

        return -sign * np.exp(exp)

    def getActionSpace(self):
        low_space, high_space = super().getActionSpace()
        low_space.extend([-1, -1])
        high_space.extend([1, 1])
        return low_space, high_space

class StandardizedActionAutoLinearFactorOpt(StandardizedActionFactor):
    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        '''

        '''
        sign = np.sign(gradients)
        action = action[:self.coord_length]
        log_grad = np.log(np.abs(gradients))
        # grad_array, grad_mean, grad_std = standarized(log_grad)

        raw_a, raw_b = action[:, 0], action[:, 1]
        a = self.transform(raw_a, 0.63297486, -0.63297486)
        b = self.transform(raw_b, 2.39260476, -2.39260476)
        C = 1.0371923
        D = 0.98302403
        exp = log_grad * a *np.power(D, log_grad) + b*np.power(C, log_grad)

        return -sign * np.exp(exp)

    def getActionSpace(self):
        low_space, high_space = super().getActionSpace()
        low_space.extend([-1])
        high_space.extend([1])
        return low_space, high_space


class StandardizedActionAutoLinearFactorFixed2(StandardizedActionAutoLinearFactorFixed):
    def __init__(self, geometry: Geometry) -> None:
        parameters = dict(
            a_low = 0.8,
            a_high = 1.1,
            b_low = 2,
            b_high = -2,
        )
        super().__init__(geometry, **parameters)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        return super().getAction(action, gradients)

    def getActionSpace(self):
        return super().getActionSpace()
        
class StandardizedActionAutoLinearFactorFixed3(StandardizedActionAutoLinearFactorFixed):
    def __init__(self, geometry: Geometry) -> None:
        parameters = dict(
            a_low = 0.8,
            a_high = 1.1,
            b_low = 2,
            b_high = -2,
        )
        super().__init__(geometry, **parameters)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        return super().getAction(action, gradients)

    def getActionSpace(self):
        return super().getActionSpace()

class StandardizedActionAutoLinearFactorFixed4(StandardizedActionAutoLinearFactorFixed):
    def __init__(self, geometry: Geometry) -> None:
        parameters = dict(
            a_low = 0.3,
            a_high = 1.1,
            b_low = 1,
            b_high = -1,
        )
        super().__init__(geometry, **parameters)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        return super().getAction(action, gradients)

    def getActionSpace(self):
        return super().getActionSpace()

class StandardizedActionAutoLinearFactorFixed5(StandardizedActionAutoLinearFactorFixed):
    def __init__(self, geometry: Geometry) -> None:
        parameters = dict(
            a_low = 0.4,
            a_high = 1.1,
            b_low = 2,
            b_high = -2,
        )
        super().__init__(geometry, **parameters)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        return super().getAction(action, gradients)

    def getActionSpace(self):
        return super().getActionSpace()

class StandardizedActionAutoLinearFactorFixed6(StandardizedActionAutoLinearFactorFixed):
    def __init__(self, geometry: Geometry) -> None:
        parameters = dict(
            a_low = 0.4,
            a_high = 1.0,
            b_low = 3,
            b_high = -2,
        )
        super().__init__(geometry, **parameters)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        return super().getAction(action, gradients)

    def getActionSpace(self):
        return super().getActionSpace()

class StandardizedActionAutoLinearFactorFixed7(StandardizedActionFactor):
    def __init__(self, geometry: Geometry, a_low=0.3, a_high=1.2) -> None:
        self.a_low = a_low
        self.a_high = a_high
        super().__init__(geometry)

    def getAction(self, action: np.ndarray, gradients: np.ndarray):
        '''

        '''
        sign = np.sign(gradients)
        action = action[:self.coord_length]
        log_grad = np.log(np.abs(gradients))
        # grad_array, grad_mean, grad_std = standarized(log_grad)

        raw_a = action[:, 0]
        a = self.transform(raw_a, self.a_high, self.a_low)
        exp = log_grad * a

        return -sign * np.exp(exp)

    def getActionSpace(self):
        low_space, high_space = super().getActionSpace()
        return low_space, high_space