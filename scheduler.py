import numpy as np
from matplotlib import pyplot as plt
import torch


class Scheduler():
    def __init__(self, optimizer, num_steps, max_lr=1e-3, min_lr=1e-5,
                 pct_start=0.2, pct_max=0.1, annealing='cosine', conditional_wd=False) -> None:
        self.__optimizer: torch.optim.Optimizer = optimizer
        self.__num_steps = num_steps
        self.__max_lr = max_lr
        self.__min_lr = min_lr
        self.__pct_start = pct_start
        self.__pct_max = pct_max
        self.__annealing = annealing
        self.__conditional_wd = conditional_wd
        self.__wd = self.__optimizer.param_groups[0]['weight_decay']
        self.__step_num = 0
        self.restart()

    def lr_function(self, x):
        if x <= self.__num_steps * self.__pct_start and self.__pct_start > 0:
            # increase the learning rate up to max_lr starting at min_lr
            lr_range = self.__max_lr - self.__min_lr
            lr_func = (
                1 + np.cos(np.pi * x / (self.__num_steps * self.__pct_start) + np.pi)) / 2
            return self.__min_lr + lr_range * lr_func
        elif x <= self.__num_steps * (self.__pct_start + self.__pct_max) and self.__pct_max > 0:
            # keep the learning rate at max_lr
            return self.__max_lr
        else:
            # decrease the learning rate down to 0
            if self.__annealing == 'cosine':
                # cosine annealing
                top = (x - (self.__pct_start + self.__pct_max)
                       * self.__num_steps) * np.pi
                bottom = (1 - (self.__pct_start + self.__pct_max)) * \
                    self.__num_steps
                return self.__max_lr * (1 + np.cos(top / bottom)) / 2
            elif self.__annealing == 'linear':
                # linear annealing
                top = (x - (self.__pct_start + self.__pct_max) * self.__num_steps)
                bottom = ((1 - (self.__pct_start + self.__pct_max))
                          * self.__num_steps)
                return self.__max_lr * (1 - top / bottom)
            else:
                raise ValueError(
                    f"annealing must be either 'cosine' or 'linear', not {self.__annealing}")

    def step(self):
        self.__step_num += 1
        self.__optimizer.param_groups[0]['lr'] = self.lr_function(
            self.__step_num)
        if self.__conditional_wd:
            if self.__optimizer.param_groups[0]['lr'] < self.min_lr:
                self.__optimizer.param_groups[0]['weight_decay'] = 0
            else:
                self.__optimizer.param_groups[0]['weight_decay'] = self.__wd

    def restart(self):
        self.__step_num = 0
        self.__optimizer.param_groups[0]['lr'] = self.lr_function(
            self.__step_num)

    def state_dict(self):
        return {
            'step_num': self.__step_num,
            'num_steps': self.__num_steps,
            'optimizer': self.__optimizer.state_dict(),
            'max_lr': self.__max_lr,
            'min_lr': self.__min_lr,
            'pct_start': self.__pct_start,
            'pct_max': self.__pct_max,
            'annealing': self.__annealing,
        }

    def load_state_dict(self, state_dict):
        self.__step_num = state_dict['step_num']
        self.__num_steps = state_dict['num_steps']
        self.__optimizer.load_state_dict(state_dict['optimizer'])
        self.__max_lr = state_dict['max_lr']
        self.__min_lr = state_dict['min_lr']
        self.__pct_start = state_dict['pct_start']
        self.__pct_max = state_dict['pct_max']
        self.__annealing = state_dict['annealing']

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def max_lr(self):
        return self.__max_lr

    @max_lr.setter
    def max_lr(self, max_lr):
        self.__max_lr = max_lr

    @property
    def min_lr(self):
        return self.__min_lr

    @min_lr.setter
    def min_lr(self, min_lr):
        self.__min_lr = min_lr


if __name__ == "__main__":
    opt = torch.optim.ASGD([torch.zeros(1)], lr=1e-3)
    params = opt.param_groups[0]
    sch = Scheduler(opt, 4000,
                    max_lr=1e-1,
                    min_lr=5e-2,
                    pct_start=0.1,
                    pct_max=0.1,
                    annealing='cosine',)
    lrs = []
    for i in range(4000):
        sch.step()
        lrs.append(params['lr'])
    plt.plot(lrs)
    plt.show(block=True)
