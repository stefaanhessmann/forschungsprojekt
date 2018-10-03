import torch.nn as nn
import torch.nn.functional as F
from DeepPotential.Code.Models.base_model import BaseNet

class SubNetwork(nn.Module):
    def __init__(self, input_dim):
        # 600-400-200-100-80-40-20
        super(SubNetwork, self).__init__()
        self.h1 = nn.Linear(input_dim, 160)
        self.h1_bn = nn.BatchNorm1d(160)
        self.h2 = nn.Linear(160, 40)
        self.h2_bn = nn.BatchNorm1d(40)
        self.h3 = nn.Linear(40, 10)
        self.h3_bn = nn.BatchNorm1d(10)
        self.h4 = nn.Linear(10, 10)
        self.h4_bn = nn.BatchNorm1d(10)
        self.h5 = nn.Linear(10, 1)

    def forward(self, X):
        a1 = F.relu(self.h1_bn(self.h1(X)))
        a2 = F.relu(self.h2_bn(self.h2(a1)))
        a3 = F.relu(self.h3_bn(self.h3(a2)))
        a4 = F.relu(self.h4_bn(self.h4(a3)))
        out = F.relu(self.h5(a4))

        return out


class DeepPotential(BaseNet):

    def __init__(self, optim, loss, cuda=False, lr_scheduler=None):
        super().__init__(optim, loss, cuda, lr_scheduler)

    def _setup(self):
        # one subnetwork for every layer:
        sub_dim = 2 * 4
        self.h_net = SubNetwork(sub_dim)
        self.o_net = SubNetwork(sub_dim)
        self.c_net = SubNetwork(sub_dim)

    def forward(self, X):
        a1 = F.relu(self.h_net.forward(X[:, 0]))
        a2 = F.relu(self.c_net.forward(X[:, 1]))
        a3 = F.relu(self.o_net.forward(X[:, 2]))

        out = a1 + a2 + a3
        return out


def update_lr(abc, x):
    a, b, c = abc
    return a * b ** (x / c)
