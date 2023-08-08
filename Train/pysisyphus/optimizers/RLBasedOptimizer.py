import copy
import numpy as np
from pysisyphus.optimizers.restrict_step import scale_by_max_step
from pysisyphus.optimizers.Optimizer import Optimizer
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Literal
from pysisyphus.Geometry import Geometry
import torch
import configparser
import ptan
import models.model as model
from functions.storeAction import StoreActionMethod
from functions.getState import State
from functions.getAction import Action
from functions.HistoryState import HistoryState

from gym import spaces
from functions.Factory import\
    STATE_CLASS_DICT,\
    ACTION_CLASS_DICT,\
    STORE_ACTION_CLASS_DICT

class RLBasedOptimizer(Optimizer):

    def __init__(self, 
        geometry: Geometry, 
        name,
        seed,
        standard_lesson,
        version,
        external_optimzer: Optimizer,
        max_step: float = 0.5, 
        **kwargs
        ):
        super().__init__(geometry, max_step=max_step, **kwargs)
        assert self.geometry.coord_type == "redund", "RL Optimizer is only suitable for redund."
        assert self.thresh == "gau_loose"
        assert external_optimzer.__name__ in ["BFGS", "LBFGS", "CubicNewton", "RFOptimizer"]
        
        self.device = torch.device("cpu")
        self.optimizer: Optimizer = external_optimzer(geometry, max_step=max_step, **kwargs)
        self.optimizer.prepare_opt()

        config_path = Path("configs", name, f"lesson{standard_lesson}", f'v{version}.ini')
        if config_path.exists() is False: raise Exception("config file does not exists")

        config = configparser.ConfigParser()
        config.read(config_path)
        configs = config["ENVIRONMENT"]

        self.max_coord_length = len(self.geometry.coords)
        self.state_class = STATE_CLASS_DICT[configs.get("state_index")]
        self.action_class = ACTION_CLASS_DICT[configs.get("action_index")]
        self.store_action_class = STORE_ACTION_CLASS_DICT[configs.get("store_action_index")]
        self.n_history: int = configs.getint("n_history")
        self.expert_participation: float = configs.getfloat("expert_participation")
        assert 1 >= self.expert_participation >= 0, "expert_participation is not in range 0~1"

        self.isLimitMaxStep: bool = configs.getboolean("isLimitMaxStep")

        self.Action: Action = self.action_class(geometry)
        self.State: State = self.state_class(geometry)
        self.StoreActionMethod: StoreActionMethod = self.store_action_class(
            geometry)

        self.history_state = HistoryState(
            n_size=self.n_history,
            state=self.State,
            action=self.Action,
            store_action=self.StoreActionMethod,
            max_coords_length=self.max_coord_length,
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


        # Model hyperparameters
        n_head = config["MODEL"].getint("n_head")
        hidden_size = config["MODEL"].getint("hidden_size")

        self.act_net = model.ModelActor(
            self.observation_space.shape[-1],
            self.action_space.shape[-1],
            hidden_size = hidden_size,
            n_head = n_head,
        ).to(self.device)

        model_path = Path("saves", f"RL-{name}", f"lesson{standard_lesson}", f'v{version}', f'seed-{seed}', 'best_model.pt')
        checkpoint = torch.load(model_path, map_location = self.device)
        self.act_net.load_state_dict(checkpoint["act_net"])



    def optimize(self):
        forces = self.geometry.forces
        energy = self.geometry.energy
        if self.cur_cycle == 0:
            self.history_state.first_append(self.State.getState())
        else:
            self.history_state.append(self.State.getState(), self.rl_action)

        state = self.history_state.getState()
        obs_v = ptan.agent.float32_preprocessor([state]).to(self.device)
        mu_v = self.act_net(obs_v)[0]
        action = mu_v.squeeze(dim=0).data.cpu().numpy()
        action = np.clip(action, -1, 1)
        
        # Step restriction
        optimizer_action = self.optimizer.optimize()
        rl_action = self.Action.getAction(action, self.geometry.gradient)
        self.rl_action = (1-self.expert_participation)*rl_action + self.expert_participation * optimizer_action
        if self.isLimitMaxStep is True:
            self.rl_action = scale_by_max_step(copy.deepcopy(self.rl_action), self.max_step) - optimizer_action
       
        self.forces.append(forces)
        self.energies.append(energy)

        return self.rl_action + optimizer_action
