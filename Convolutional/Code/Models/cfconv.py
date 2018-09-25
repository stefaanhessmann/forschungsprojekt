import torch
import torch.nn as nn
from Code.Models.activation_functions import SSP


def rbf(d_ij, gamma=10, n_rbf=300, use_cuda=False):
    mu_k = 0.1 * torch.DoubleTensor(range(n_rbf))
    if use_cuda:
        mu_k.cuda()
    result = torch.exp(-gamma*(d_ij-mu_k)**2)
    return result


class Cfconv(nn.Module):

    def __init__(self, n_rbf=300, use_cuda=False):
        super(Cfconv, self).__init__()
        self.use_cuda = use_cuda
        self.n_rbf = n_rbf
        self.convnet = nn.Sequential(nn.Linear(self.n_rbf, 64), SSP(), nn.Linear(64, 64), SSP())

    def forward(self, x, distances):
        gamma = 10
        e_k = rbf(distances.unsqueeze(-1), gamma, self.n_rbf, self.use_cuda)
        w_l = self.convnet(e_k)
        return torch.sum(x.unsqueeze(2) * w_l, 2)





