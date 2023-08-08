import numpy as np

from pysisyphus.optimizers.Optimizer import Optimizer

class MyOptimizer(Optimizer):

    def __init__(self, geometry, *args, **kwargs):
        super().__init__(geometry, *args, **kwargs)

        assert self.align == False, \
            "align=True does not work with this optimizer! Consider using LBFGS."

    def optimize(self):
        forces = self.geometry.forces
        energy = self.geometry.energy
        self.forces.append(forces)
        self.energies.append(energy)
        return np.zeros_like(self.geometry.coords) 
    
    # def reset(self):
        # return NotImplementedError()
        # return Exception()
