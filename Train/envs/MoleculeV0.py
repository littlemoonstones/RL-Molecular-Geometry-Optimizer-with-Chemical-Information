import copy
import json
import random
from typing import Callable, List, Optional, Tuple
import gym
from gym import spaces
import pickle
import numpy as np
import configparser
from pathlib import Path
from UserTools.createMolecule import getMolecule

from pysisyphus.Geometry import Geometry
from pysisyphus.optimizers.MyOptimizer import MyOptimizer
from pysisyphus.optimizers.BFGS import BFGS
from pysisyphus.calculators.RDKit import MMFF
from pysisyphus.intcoords.PrimTypes import PrimTypes
from pysisyphus.constants import ANG2BOHR
from pysisyphus.drivers.opt import get_opt_cls

from functions.storeAction import StoreActionMethod
from functions.getState import State
from functions.getAction import Action
from functions.HistoryState import HistoryState
from functions.getEncode import EncodeClass
from functions.Factory import reward_function_dict, \
    STATE_CLASS_DICT,\
    ACTION_CLASS_DICT,\
    STORE_ACTION_CLASS_DICT,\
    ENCODE_CLASS_DICT

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)


class MoleculeV0(gym.Env):
    '''
    many kinds of molecules
    '''

    def __init__(self,
                 configs: configparser.ConfigParser
                 ) -> None:
        super().__init__()
        print("Using env", self.__class__.__name__)
        with open("envs/perfect_primitives.pk", "rb")as fs:
            self.default_primitives: dict = pickle.load(fs)

        with open("envs/smiles_258.json", "r")as fs:
            self.smiles: List[str] = json.load(fs)

        self.env_iteration: int = 0

        smile = "C"
        # _smile, atoms, coords, _seed = getMolecule(smile, seed=1)

        # geometry = Geometry(atoms=atoms, coords=coords*ANG2BOHR, coord_type="redund", coord_kwargs={
        #     "typed_prims": self.default_primitives.get(smile),
        # })
        self.set_params(configs)
        atoms, geometry = self.createGeometry(
            smile,
            self.coord_type,
            None,
            0
        )

        # geometry.set_calculator(MMFF(smile=smile, seed=0))

        self.Action: Action = self.action_class(geometry)
        self.State: State = self.state_class(geometry)
        self.StoreActionMethod: StoreActionMethod = self.store_action_class(
            geometry)
        
        
        self.Encode: Optional[EncodeClass] = None
        if self.coord_type == 'redund':
            self.Encode = self.encode_class(geometry, smile)
        elif self.coord_type == 'cart':
            self.Encode = None

        print("using Encode class:", self.Encode.__class__.__name__)
        print("using State class:", self.State.__class__.__name__)
        print("using Action class:", self.Action.__class__.__name__)
        print("using StoreActionMethod class:",
              self.StoreActionMethod.__class__.__name__)

        self.history_state = HistoryState(
            n_size=self.n_history,
            state=self.State,
            action=self.Action,
            store_action=self.StoreActionMethod,
            encode=self.Encode,
            max_coords_length=self.max_coord_length,
            coord_type=self.coord_type,
            need_encode=self.need_encode,
        )

        low, high = self.Action.getActionSpace()

        self.action_space = spaces.Box(
            low=np.array(low),
            high=np.array(high),
            shape=np.array(low).shape,
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.history_state.state_size, ),
            dtype=np.float32
        )

        print("action_space:", self.action_space.shape[-1])
        print("observation_space:", self.observation_space.shape[-1])

    def set_params(
        self,
        configs: configparser.ConfigParser
    ):
        # coord_type
        self.coord_type: str = configs.get("coord_type")

        self.max_force: float = configs.getfloat("max_force")
        self.max_step: float = configs.getfloat("max_step")
        self.reward_function: Callable = reward_function_dict[configs.get(
            "reward_index")]
        self.state_class = STATE_CLASS_DICT[configs.get("state_index")]
        self.action_class = ACTION_CLASS_DICT[configs.get("action_index")]
        self.store_action_class = STORE_ACTION_CLASS_DICT[configs.get(
            "store_action_index")]
        self.encode_class = ENCODE_CLASS_DICT[configs.get("encode_index")]
        self.n_history: int = configs.getint("n_history")
        self.max_iter: int = configs.getint("max_iter")
        self.max_coord_length: int = configs.getint("max_coord_length")
        # self.opt_key: str = configs.get("opt_key")
        self.need_encode: bool = configs.getboolean("need_encode")

        self.isLimitMaxStep: bool = configs.getboolean("isLimitMaxStep")
        print("Limit Max Step", self.isLimitMaxStep)

    def createGeometry(
        self,
        smile: str,
        coord_type: str,
        coord: Optional[np.ndarray],
        noise_probability: float = 0.2
    ) -> Tuple[List[str], Geometry]:

        _smile, atoms, coords, _seed = getMolecule(
            smile, random.randint(1, 2**30))
        if type(coord) == np.ndarray:
            coords = coord

        # if there is coord(for validation), don't add noise
        if coord is None and random.random() < noise_probability:
            coords += np.random.normal(0, 0.1, len(coords))

        if coord_type == "redund":
            coord_kwargs = {
                "typed_prims": self.default_primitives.get(smile),
            }
        elif coord_type == "cart":
            coord_kwargs = None

        geometry = Geometry(
            atoms=atoms,
            coords=coords * ANG2BOHR,
            coord_type=coord_type,
            coord_kwargs=coord_kwargs
        )

        geometry.set_calculator(MMFF(smile=smile, seed=_seed))
        return atoms, geometry

    def step(self, action: np.ndarray):
        self.env_iteration += 1
        error = False

        if np.sum(np.isnan(action)) != 0:
            print("prvious state", self.history_state.getState())
            raise Exception("action nan")

        try:
            rl_action = self.Action.getAction(action, self.geometry.gradient)

            if np.sum(np.isnan(rl_action)) != 0:
                raise Exception("rl_action nan")

            # take a step
            (cur_cycle, optimizer_action) = next(
                self.iterator
            )

            if self.isLimitMaxStep is True:
                rl_action = self.optimizer.scale_by_max_step(
                    copy.deepcopy(rl_action)
                ) - optimizer_action

            # after the step
            is_converged = self.iterator.send(
                rl_action
            )

            reward = self.reward_function(self.optimizer.max_forces[-1])

            self.iter += 1
            is_converged = next(self.iterator)
            self.history_state.append(self.State.getState(), rl_action)
            self.state = self.history_state.getState()
            self.max_forces.append(self.geometry.forces.max())

            if self.optimizer.max_forces[-1] < self.max_force:
                print("converged")
                raise StopIteration

            if is_converged or self.iter >= self.max_iter:
                # next(self.iterator)
                raise StopIteration

        except StopIteration as result:
            print("End Smile:", self.smile, ", result:", result)
            print("stop StopIteration", self.iter)
            reward = +5 if self.iter < self.max_iter else 0
            is_converged = True
            error = False
        except (ValueError, RuntimeWarning) as e:
            print("haha", e)
            reward = -self.max_iter+1
            is_converged = True
            error = True

        except Exception as e:
            # bfgs reset() is not defined
            # print(f"action {action.shape}:", action)
            print(f"rl_action {rl_action.shape}:", rl_action)
            print(f"geometry {len(self.geometry.coords)}:",
                  self.geometry.coords)
            print("Exception type:", e.__class__.__name__)
            print(e.__traceback__)
            print('-------------Attr:', e)
            trace = []
            tb = e.__traceback__
            while tb is not None:
                trace.append({
                    "filename": tb.tb_frame.f_code.co_filename,
                    "name": tb.tb_frame.f_code.co_name,
                    "lineno": tb.tb_lineno
                })
                tb = tb.tb_next
            print(str({
                'type': type(e).__name__,
                'message': str(e),
                'trace': trace
            }))
            # raise Exception()
            reward = -self.max_iter+1
            is_converged = True
            error = True

        finally:
            state = self.state

        return state, reward, bool(is_converged), {
            "error": error,
            "rebuild": 0,
        }

    def reset(self, smile: str = None, coord=None):
        self.iter = 0

        noise_probability = 0.5
        self.max_forces: List[float] = []

        if smile != None:
            self.smile = smile
        else:
            self.smile = random.choice(self.smiles)
        print("using smile:", self.smile)
        print("using coord_type:", self.coord_type)

       # initial geoemtry force needs larger then threshold
        atoms, tmp_geometry = self.createGeometry(
            self.smile,
            self.coord_type,
            coord,
            noise_probability
        )
        
        while type(coord) != np.ndarray and np.abs(tmp_geometry.gradient).max() <= self.max_force or \
              ( False if self.coord_type == 'cart' else len(tmp_geometry.coords) != len(self.default_primitives.get(self.smile))):
            atoms, tmp_geometry = self.createGeometry(
                self.smile,
                self.coord_type,
                coord,
                noise_probability
            )

        self.geometry = tmp_geometry
        self.Action: Action = self.action_class(self.geometry)
        self.State: State = self.state_class(self.geometry)
        self.StoreActionMethod: StoreActionMethod = self.store_action_class(
            self.geometry)
        if self.coord_type == 'redund':
            self.Encode: EncodeClass = self.encode_class(self.geometry, self.smile)
        else:
            self.Encode = None

        # self.history_state.reset()
        self.history_state = HistoryState(
            n_size=self.n_history,
            state=self.State,
            action=self.Action,
            store_action=self.StoreActionMethod,
            encode=self.Encode,
            max_coords_length=self.max_coord_length,
            need_encode=self.need_encode,
        )

        self.optimizer = MyOptimizer(self.geometry,
                                     max_cycles=self.max_iter-1,
                                     thresh="gau",
                                     max_force_only=True,
                                     max_step=self.max_step,
                                     )
        print("using Optimizer:", self.optimizer.__class__.__name__)
        self.iterator = self.optimizer.run()
        self.history_state.first_append(self.State.getState())
        self.env_iteration += 1
        return self.history_state.getState()
