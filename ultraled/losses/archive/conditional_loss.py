import torch
from torch import nn as nn
from torch.nn import functional as F

from ultraled.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class EMAConditionalLoss(nn.Module):
    def __init__(self, branch_num, ema_decay=0.999, ema_time=1, core='l1', *, conv_count=1, reduce='mean', loss_weight=1.0, init='eye') -> None:
        super().__init__()
        if init == 'eye':
            gt = torch.eye(branch_num)
        elif init == 'rand':
            gt = torch.rand((branch_num, branch_num))
            gt = torch.softmax(gt, 1)
        elif init == 'randn':
            gt = torch.randn((branch_num, branch_num))
            gt = torch.softmax(gt, 1)
        # gt = self.label_smoothing(torch.eye(branch_num))
        gt = gt.repeat(1, conv_count)
        self.gts = nn.parameter.Parameter(gt, requires_grad=False)
        self.ema_decay = ema_decay
        self.ema_time = ema_time
        self.counter = nn.parameter.Parameter(torch.zeros(branch_num, dtype=torch.int), requires_grad=False)

        assert reduce in ['mean', 'sum']
        reduce_fn = eval(f'torch.{reduce}')
        if core == 'l1':
            self.loss = lambda x, y: reduce_fn(torch.abs(x - y))
        elif core == 'l2' or core == 'mse':
            self.loss = lambda x, y: reduce_fn((x - y) * (x - y))
        elif core == 'cross_entropy':
            self.loss = lambda x, y: F.cross_entropy(x, y, reduce=reduce)
        else:
            raise NotImplementedError()

        self.loss_weight = loss_weight

    def label_smoothing(self, x):
        x[x == 1] = 0.8
        x[x == 0] = 0.05
        return x

    def ema(self, x, index):
        self.counter[index] += 1
        if self.counter[index] % self.ema_time == 0:
            self.counter[index] = 0
            self.gts[index].data.mul_(self.ema_decay).add_(x, alpha=1-self.ema_decay)
            print(self.gts)

    def forward(self, x, index):
        loss = self.loss(x, self.gts[index])
        if self.ema_decay > 0:
            self.ema(x, index)
        return loss * self.loss_weight
