from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F

# FIXME in the binary classification case, we expect 50 % of initial accuracy
class ExpoMovingAverage(object):
    def __init__(self, decay_factor=0.9999):
        if 0 >= decay_factor or decay_factor >= 1:
            raise ValueError
        self._a = decay_factor
        self._t = 0
        self._s = 0

    def __call__(self, y):
        self._t += 1

        if self._t == 1:
            self._s = y
        else:
            self._s = self._a * y + (1 - self._a) * self._s

        return self._s

    def reset(self):
        self._t = 0
        self._s = 0


class REINFORCELossWithEMA(nn.Module):
    def __init__(self, decay_factor=0.9999):
        super(REINFORCELossWithEMA, self).__init__()
        self._ema = ExpoMovingAverage(
            decay_factor=decay_factor
        )
        self._baseline = 0

    def forward(self, reward, log_probs):
        adv = reward - self._baseline
        loss = -log_probs.sum() * adv
        loss = loss.mean()

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
