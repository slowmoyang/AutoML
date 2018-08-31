from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F

# FIXME in the binary classification case, we can expect initial accuracy of # of
class ExpoMovingAverage(object):
    def __init__(self, decay_factor=0.9999):
        if 0 >= decay_factor or decay_factor >= 1:
            raise ValueError
        # the degree of weighting decrease, a constant smoothing factor between
        # 0 and 1. A higher \alpha discounts older observations faster
        self._a = decay_factor
        # time eriod
        self._time_period = 0
        # the vale of the EMA at any time period t
        self._s = 0

    def __call__(self, y):
        self._time_period += 1

        if self._time_period == 1:
            self._s = y
        else:
            self._s = self._a * y + (1 - self._a) * self._s

        return self._s

    def reset(self):
        self._time_period = 0
        self._s = 0


class REINFORCELossWithEMA(nn.Module):
    def __init__(self, decay_factor=0.9999):
        super(REINFORCELossWithEMA, self).__init__()
        self._ema = ExpoMovingAverage(
            decay_factor=decay_factor
        )
        self._baseline = 0

    def forward(self, reward, log_prob):
        advantage = reward - self._baseline
        loss = -log_prob * advantage
        loss = loss.mean()

        # update baseline
        self._baseline = self._ema(reward)
        return loss


def _test():
    controller_loss = REINFORCELossWithEMA()

    for _ in range(10): 
        reward = torch.empty(100).uniform_(0, 1)
        log_prob = torch.log(torch.empty(100).uniform_(0, 1))
        loss = controller_loss(reward, log_prob)
        print(loss)
    

if __name__ == "__main__":
    _test()
