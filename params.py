
import torch
import torch.nn as nn
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import numpy as np
import argparse


class EpochBaseLR(_LRScheduler):
    def __init__(self, optimizer, milestones, lrs, last_epoch=-1, ):
        if len(milestones)+1 != len(lrs):
            raise ValueError('The length of milestones must equal to the '
                             ' length of lr + 1. Got {} and {} separately', len(milestones)+1, len(lrs))
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)

        self.milestones = milestones
        self.lrs = lrs
        super(EpochBaseLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.lrs[bisect_right(self.milestones, self.last_epoch)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.get_lr()

        for g in self.optimizer.param_groups:
            g['lr'] = lr


class ParamsControl(object):
    def __init__(self, opts, model):

        model_params = list(model.parameters())

        lrs = [float(value) for value in opts.lr.split(',')]
        milestones = [float(value) for value in opts.milestone.split(',')]

        self.optimizers = [
            torch.optim.SGD(model_params, lr=0.0, momentum=0.9, weight_decay=5e-4)
        ]

        self.schedulers = [
            EpochBaseLR(self.optimizers[0], milestones=milestones, lrs=lrs)
        ]

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def back_grad(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def update(self, epoch):
        for scheduler in self.schedulers:
            scheduler.step(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test params')
    parser.add_argument('--lr', type=str, default='0.1,0.01,0.001,1e-4')
    parser.add_argument('--milestone', type=str, default='20,40,60')
    opts = parser.parse_args()

    net = nn.Linear(12,4)
    params = ParamsControl(opts, net)

    lr = []

    for epoch in range(0, 60):
        params.update(epoch)
        lr.append(params.schedulers[0].get_lr())
    if 1:
        plt.plot(np.arange(0, 60), lr)
        plt.show()
