import torch
import torch.nn as nn


class SSP(nn.Module):

    def __init__(self, shift=0.5):
        super(SSP, self).__init__()
        self.shift = shift

    def forward(self, x):
        return torch.log(self.shift*torch.exp(x) + self.shift)