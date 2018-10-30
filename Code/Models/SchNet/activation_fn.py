import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSP(nn.Module):

    def __init__(self, shift=0.5):
        super(SSP, self).__init__()
        self.shift = shift

    def forward(self, x):
        return F.softplus(x) - np.log(2)
