from functions.getAction import Action
from functions.storeAction import StoreActionMethod
from functions.getState import State
from pysisyphus.Geometry import Geometry
from pysisyphus.intcoords.Primitive import Primitive
from pysisyphus.intcoords.Bend import Bend
from pysisyphus.intcoords.Stretch import Stretch
from pysisyphus.intcoords.Torsion import Torsion
from pysisyphus.intcoords.LinearBend import LinearBend
import numpy as np

def Encoded(geometry: Geometry):
    array = []
    # atoms = geometry.atoms
    primitive: Primitive
    for primitive in geometry.internal.primitives:
        # encoded = np.zeros(8)
        encoded = np.zeros(3)
        # print(primitive.indices)
        if isinstance(primitive, Stretch):
            encoded[0] = 1
        elif isinstance(primitive, (Bend, LinearBend)):
            encoded[1] = 1
        elif isinstance(primitive, Torsion):
            encoded[2] = 1
        else:
            raise Exception()
        array.append(encoded)
    return np.array(array)

def getPrimitiveIndices(geometry: Geometry):
    array = np.zeros((len(geometry.coords), 4))-1
    p: Primitive
    for i, p in enumerate(geometry.internal.primitives):
        indices = np.array(p.indices)
        array[i, :len(indices)] = indices
    return array

class HistoryState:
    def __init__(self,
                 n_size: int,
                 state: State,
                 action: Action,
                 store_action: StoreActionMethod,
                 max_coords_length: int,
                 coord_type: str = "redund",
        ) -> None:
        self.n_size = n_size
        self.State = state
        self.Action = action
        self.StoreAction = store_action

        self._state_size = self.State.getSize()
        self.State.getSize()
        self._state_size += self.StoreAction.getSize()
        
        self.max_coords_length: int = max_coords_length
        self.coords_length: int = len(self.State.geo.coords)
        if coord_type != "redund":
            self.encode: np.ndarray = np.array([], dtype=np.float64).reshape(self.coords_length, 0)
        else:
            self.encode: np.ndarray = np.hstack([
                Encoded(self.State.geo),
                getPrimitiveIndices(self.State.geo),
            ])
        self.input_size = self.encode.shape[-1] + self._state_size * self.n_size

        self.padding = np.zeros([self.max_coords_length - self.coords_length, self.input_size])
        
        self.mask = np.vstack([
            np.ones((self.coords_length, 1)),
            np.zeros((self.max_coords_length - self.coords_length, 1)),
        ])

        self.reset()

    def reset(self):
        self.hisotry_state = np.zeros((self.n_size, self.coords_length, self._state_size))
    
    @property
    def state_size(self):
        return self.input_size
    
    def first_append(self, state: np.ndarray):
        psudo_action = self.StoreAction.getInitAction() # (a)
        if self.StoreAction.getSize() != 0:
            state_and_actioin = np.hstack([
                state,
                psudo_action
            ])
            self.hisotry_state = np.vstack([
                state_and_actioin[None, :],
                self.hisotry_state[:-1],
            ])
        else:
            self.hisotry_state = np.vstack([
                state[None, :],
                self.hisotry_state[:-1],
            ])
    
    def append(self, state: np.ndarray, action: np.ndarray):

        if self.StoreAction.getSize() != 0:
            state_and_actioin = np.hstack([
                state,
                self.StoreAction.transform(action)
            ])

            self.hisotry_state = np.vstack([
                state_and_actioin[None, :],
                self.hisotry_state[:-1],
            ])
        # Only store state
        else:
            # print("state", state[None, :].shape)
            self.hisotry_state = np.vstack([
                state[None, :],
                self.hisotry_state[:-1],
            ])
            # print("self.hisotry_state", self.hisotry_state.shape)
            # print(self.hisotry_state[:-1].shape)
    
    def getState(self):
        state = self.hisotry_state.transpose(1, 0, 2).astype(np.float32)
        state = state.reshape(self.coords_length, -1)
        state = np.hstack([
            self.encode,
            state,
        ])
        state = np.vstack([
            state,
            self.padding,
        ])
        state = np.hstack([
            state,
            self.mask
        ])

        if np.isnan(state.sum()):
            print(f"state: {state}")
            # raise ValueError("@@@@@@")
        return state
    
    
