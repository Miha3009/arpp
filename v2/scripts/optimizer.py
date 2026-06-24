import torch
from saver import saveable

class Optimizer:
    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

@saveable
class AdamOptimizer(Optimizer):
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.args = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }

    def setup(self, params):
        self.optimizer = torch.optim.Adam(params, **self.args)
