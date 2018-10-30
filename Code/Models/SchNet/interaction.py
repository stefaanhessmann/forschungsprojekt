import torch
from torch import nn
from Code.Models.SchNet.cfconv import Cfconv
from Code.Models.SchNet.activation_fn import SSP

class Interaction(nn.Module):

    def __init__(self):
        super(Interaction, self).__init__()
        self.atomwise_1 = nn.Linear(64, 64)
        self.cfconv = Cfconv()
        self.atomwise_2 = nn.Linear(64, 64)
        self.ssp = SSP()
        self.atomwise_3 = nn.Linear(64, 64)

    def forward(self, x_l, distance):
        x_interaction = self.atomwise_1(x_l)
        x_interaction = self.cfconv(x_interaction, distance)
        x_interaction = self.atomwise_2(x_interaction)
        x_interaction = self.ssp(x_interaction)
        x_interaction = self.atomwise_3(x_interaction)
        return torch.add(x_l, x_interaction)
