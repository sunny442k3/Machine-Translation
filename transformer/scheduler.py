import numpy as np
from . import constant as cf

class SchedulerAdam():

    def __init__(self, optimizer):
        self.optimizer = optimizer 
        self.lr = cf.d_model ** (-0.5)
        self.current_steps = 0
        self.warm_steps = cf.warm_steps


    def _get_scale(self):
        return np.min([
            self.current_steps ** (-0.5),
            self.current_steps * (self.warm_steps ** (-0.5))
        ])


    def step(self):
        self.current_steps += 1
        lr = self.lr * self._get_scale()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.optimizer.step()
    

    def zero_grad(self):
        self.optimizer().zero_grad()