from pysisyphus.optimizers.restrict_step import scale_by_max_step
from pysisyphus.optimizers.Optimizer import Optimizer
import numpy as np
from typing import Dict, List, Tuple
from pysisyphus.Geometry import Geometry

class RandomOptimizer(Optimizer):

    def __init__(self, geometry: Geometry, max_step: float = 0.5, **kwargs):
        super().__init__(geometry, max_step=max_step, **kwargs)    
        self.max_coord_length = len(self.geometry.coords)
        self.act_net_a = lambda : np.random.random(self.max_coord_length)
        self.act_net_b = lambda : np.random.random(self.max_coord_length)

    def optimize(self):
        forces = self.geometry.forces
        energy = self.geometry.energy
        a = self.act_net_a()*0.6+0.3
        b = self.act_net_b()*2-1
        log_grad = np.log(np.abs(self.geometry.gradient))
        action = -np.sign(self.geometry.gradient)*np.exp(log_grad*a + b)
        step = scale_by_max_step(action, self.max_step)
        self.forces.append(forces)
        self.energies.append(energy)

        return step
