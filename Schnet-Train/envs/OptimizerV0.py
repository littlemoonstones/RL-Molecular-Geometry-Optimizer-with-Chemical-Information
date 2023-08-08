import copy
import json
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
import gym
from gym import spaces
import pickle
import numpy as np
import configparser
from pathlib import Path
from UserTools.createMolecule import getMolecule

from pysisyphus.Geometry import Geometry
from pysisyphus.calculators.RDKit import MMFF
from pysisyphus.intcoords.PrimTypes import PrimTypes
from pysisyphus.constants import ANG2BOHR

from pysisyphus.drivers.opt import OPT_DICT

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)
# warnings.filterwarnings("error::RuntimeWarning")

class OptimizerV0(gym.Env):
    '''
    many kinds of molecules
    '''
    def __init__(self,
                configs: configparser.ConfigParser
                 ) -> None:
        super().__init__()
        with open("envs/perfect_primitives.pk", "rb")as fs:
            self.default_primitives: dict = pickle.load(fs)
        
        self.env_iteration: int = 0

        self.set_params(configs)
       

    def set_params(self,
                configs: configparser.ConfigParser
                   ):
        self.max_force: float = configs.getfloat("max_force")
        self.max_step: float = configs.getfloat("max_step")
        self.max_iter: int = configs.getint("max_iter")
        self.max_coord_length: int = configs.getint("max_coord_length")
        self.key: str = configs.get("key")

        self.expert_participation: float = configs.getfloat("expert_participation")
        assert 1 >= self.expert_participation >= 0, "expert_participation is not in range 0~1"

        self.isLimitMaxStep: bool = configs.getboolean("isLimitMaxStep")
        print("Limit Max Step", self.isLimitMaxStep)
    
    def createGeometry(self, smile: str, coord: Optional[np.ndarray], noise_probability: float=0.2) -> Tuple[List[str], Geometry]:
        _smile, atoms, coords, _seed = getMolecule(smile, random.randint(1, 2**30))
        if type(coord) == np.ndarray:
            coords = coord
        # if there is coord, don't add noise
        if coord is None and random.random() < noise_probability:
            coords += np.random.normal(0, 0.1, len(coords))
     
        geometry = Geometry(atoms=atoms, coords=coords*ANG2BOHR, coord_type="redund", coord_kwargs={
            "typed_prims": self.default_primitives.get(smile),
        })
        geometry.set_calculator(MMFF(smile=smile, seed=_seed))
        return atoms, geometry

    def step(self, action: np.ndarray):
        self.env_iteration += 1
        error = False
        # print("raw_action:", action)
        try:
            (cur_cycle, optimizer_action) = next(
                self.iterator)         # take a step
            if self.isLimitMaxStep is True:
                rl_action = self.optimizer.scale_by_max_step(copy.deepcopy(optimizer_action)) - optimizer_action
            is_converged = self.iterator.send(rl_action)                # after the step

            self.max_forces.append(self.geometry.forces.max())

            self.iter += 1
            if self.optimizer.max_forces[-1] < self.max_force:
                print("converged")
                raise StopIteration
            if is_converged or self.iter >= self.max_iter:
                next(self.iterator)
                raise StopIteration()
            is_converged = next(self.iterator)

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
            try:
                reward = -self.max_iter+1
                is_converged = True
                error = True

            except TimeoutError as e:
                print("TimeoutError")
                reward = -self.max_iter+1
                is_converged = True
                error = True
            except RuntimeError as e:
                print(e)
                reward = -self.max_iter+1
                is_converged = True
                error = True

        finally:
            state = self.state
        return state, reward, bool(is_converged), {
            "error": error,
            "rebuild": 0,
        }

    def reset(self, smile: str=None, coord=None):
        self.iter = 0
        
        self.max_forces: List[float] = []

        if smile != None:
            self.smile = smile
        else:
            self.smile = random.choice(self.smiles)
        print("using smile:", self.smile)
        
       # initial geoemtry force needs larger then threshold
        atoms, tmp_geometry = self.createGeometry(self.smile, coord, 0)
        self.geometry = tmp_geometry
   
        self.optimizer = OPT_DICT[self.key](self.geometry,
                                    max_cycles=self.max_iter-1,
                                    thresh="gau_loose",
                                    max_force_only=True,
                                    max_step=self.max_step,
                                )
        print("using Optimizer:", self.optimizer.__class__.__name__)
        self.iterator = self.optimizer.run()
        self.env_iteration += 1
        return self.geometry.gradient
