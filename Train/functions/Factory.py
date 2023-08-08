from functions.getState import State,\
    NaiveGradientState,\
    NormalizedGradientState,\
    NaiveGradientAndCoordateState,\
    LogGradienttState,\
    LogGradientAndLogCoordinateState,\
    StandardizedLogGradientAndNaiveCoordinateState,\
    LogGradienttStateNormalized,\
    StandardizedLogGradient

from functions.getAction import Action, ActionFactor, \
    LogAction,\
    NaiveAction,\
    NaiveActionFactor,\
    LogActionFactor,\
    StandardizedActionFactor, StandardizedAction08Factor, StandardizedActionAutoFactor,\
    ActionFactor,\
    StandardizedActionAutoLinearFactor,\
    LogActionNoSign,\
    StandardizedActionAutoLinearFactorFixed,\
    StandardizedActionAutoLinearFactorFixedSign,\
    StandardizedActionAutoLinearFactorOpt,\
    StandardizedActionAutoLinearFactorFixed2,\
    StandardizedActionAutoLinearFactorFixed3,\
    StandardizedActionAutoLinearFactorFixed4,\
    StandardizedActionAutoLinearFactorFixed5,\
    StandardizedActionAutoLinearFactorFixed6,\
    StandardizedActionAutoLinearFactorFixed7

from functions.storeAction import StoreActionMethod,\
    StoreNoneAction,\
    StoreNaiveAction,\
    StoreLogAction,\
    StoreLogActionNormalized

from functions.getReward import Reward,\
    Only1,\
    Reward1,\
    Reward2,\
    Reward3,\
    Reward3Avg

from functions.getEncode import EncodeClass,\
    BasicEncode,\
    AtomEncode,\
    NeighborEncode

from typing import Dict, List, Optional, Union
import numpy as np


reward_function_dict = {
    "0": lambda _: -1,
    "1": lambda force: (np.log(force/2.5e-3))/np.log(2.5e-3) - 1,
    "2": lambda force: (np.log(force/4.5e-4))/np.log(4.5e-4) - 1
}

REWARD_CLASS_DICT: Dict[str, Reward] = {
    "0": Only1,
    "1": Reward1,
    "2": Reward2,
    "3": Reward3,
    "4": Reward3Avg,
}

STATE_CLASS_DICT: Dict[str, State] = {
    "0": NaiveGradientState,
    "1": NormalizedGradientState,
    "2": NaiveGradientAndCoordateState,
    "3": LogGradienttState,
    "4": LogGradientAndLogCoordinateState,
    "5": StandardizedLogGradientAndNaiveCoordinateState,
    "6": LogGradienttStateNormalized,
    "7": StandardizedLogGradient,
}

ACTION_CLASS_DICT: Dict[str, Action] = {
    "0": NaiveAction,
    "1": LogAction,
    "2": NaiveActionFactor,
    "3": LogActionFactor,
    "4": StandardizedActionFactor,
    "5": StandardizedAction08Factor,
    "6": StandardizedActionAutoFactor,
    "7": ActionFactor,
    "8": StandardizedActionAutoLinearFactor,
    "9": LogActionNoSign,
    "10": StandardizedActionAutoLinearFactorFixed,
    # "11": StandardizedActionAutoLinearFactorFixedSign,
    # "12": StandardizedActionAutoLinearFactorOpt,
    # "13": StandardizedActionAutoLinearFactorFixed2,
    # "14": StandardizedActionAutoLinearFactorFixed3,
    # "15": StandardizedActionAutoLinearFactorFixed4,
    # "16": StandardizedActionAutoLinearFactorFixed5,
    # "17": StandardizedActionAutoLinearFactorFixed6,
    # "18": StandardizedActionAutoLinearFactorFixed7,
}

STORE_ACTION_CLASS_DICT: Dict[str, StoreActionMethod] = {
    "0": StoreNoneAction,
    "1": StoreNaiveAction,
    "2": StoreLogAction,
    "3": StoreLogActionNormalized,
}

ENCODE_CLASS_DICT: Dict[str, EncodeClass] = {
    "0": BasicEncode,
    "1": AtomEncode,
    "2": NeighborEncode,
}

