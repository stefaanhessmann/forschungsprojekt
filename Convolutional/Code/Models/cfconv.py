import torch
import torch.nn as nn
import numpy as np
from Convolutional.Code.Models.activation_functions import SSP

def rbf(d_ij, gamma=10, n_rbf=300):
    return torch.stack([torch.exp(-gamma*(d_ij-0.1*mu_k)**2) for mu_k in range(n_rbf)])


class Cfconv(nn.Module):

    def __init__(self, n_rbf=300):
        super(Cfconv, self).__init__()
        self.n_rbf = n_rbf
        self.convnet = nn.Sequential(nn.Linear(self.n_rbf, 64), SSP(), nn.Linear(64, 64), SSP()).double()

    def forward(self, x, distance):
        gamma = 10
        e_k = torch.stack([rbf(d_ij, gamma=gamma) for d_ij in distance])
        w_l = self.convnet(e_k)
        return x * w_l.float()





