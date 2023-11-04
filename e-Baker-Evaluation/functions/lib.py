from typing import Tuple
import numpy as np

def splitLog(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    return sign(array), ln(|array|)
    '''
    sign = np.sign(array)
    array = np.clip(np.abs(array), 1e-8, 20)
    array = np.log(np.abs(array))
    return sign, array