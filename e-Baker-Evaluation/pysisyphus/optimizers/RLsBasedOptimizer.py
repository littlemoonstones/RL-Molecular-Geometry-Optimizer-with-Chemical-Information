import copy
from typing import Union, List
import numpy as np
from pysisyphus.optimizers.restrict_step import scale_by_max_step
from pysisyphus.optimizers.Optimizer import Optimizer
import numpy as np
from pysisyphus.Geometry import Geometry
from lib.Agent import Agent


class RLsBasedOptimizer(Optimizer):

    def __init__(self,
                 geometry: Geometry,
                 name,
                 seeds: Union[int, List[int]],
                 version,
                 max_step: float = 0.5,
                 **kwargs
                 ):
        super().__init__(geometry, max_step=max_step, **kwargs)

        if self.thresh != "gau_loose":
            print("Warning: RL Optimizer is only suitable for gau_loose.")

        if type(seeds) == int:
            seeds = [seeds]

        assert type(seeds) == list, \
            f"seeds must be list[int], but got {type(seeds)}"
        self.agents = []
        for seed in seeds:
            agent = Agent(name, seed, version, geometry)
            self.agents.append(agent)
        
        print("total agents:", len(self.agents))

    def optimize(self):
        forces = self.geometry.forces
        energy = self.geometry.energy

        actions = []
        agent: Agent
        for agent in self.agents:
            current_state = agent.get_current_state()
            if self.cur_cycle == 0:
                agent.init_state(current_state)
            else:
                agent.append_state(current_state, self.rl_action)

            state = agent.get_state()
            action = agent.get_action(state, self.geometry)

            if agent.isLimitMaxStep is True:
                action = scale_by_max_step(
                    copy.deepcopy(action),
                    self.max_step
                )
            actions.append(action)
        rl_actions = np.array(actions)

        self.rl_action = rl_actions.mean(0)

        self.forces.append(forces)
        self.energies.append(energy)
        # debug
        self.cart_forces.append(self.geometry.cart_forces)

        return self.rl_action
