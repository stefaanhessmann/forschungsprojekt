import torch
from torch import nn
from Code.Models.SchNet.interaction import Interaction
from Code.Models.SchNet.activation_fn import SSP


class SchNet(nn.Module):

    def __init__(self, n_atoms=19, n_features=64):
        super(SchNet, self).__init__()
        self.embedding = nn.Embedding(n_atoms, n_features)
        self.interaction_1 = Interaction()
        self.interaction_2 = Interaction()
        self.interaction_3 = Interaction()
        self.atomwise_1 = nn.Linear(64, 32)
        self.ssp = SSP()
        self.atomwise_2 = nn.Linear(32, 1)

    def forward(self, x_data):
        atom_ids = x_data[:, :, -1].long()
        distances = x_data[:, :, :-1]
        embedded_atom_ids = self.embedding(atom_ids)

        x = self.interaction_1(embedded_atom_ids, distances)
        x = self.interaction_2(x, distances)
        x = self.interaction_3(x, distances)
        x = self.atomwise_1(x)
        x = self.ssp(x)
        x = self.atomwise_2(x)
        return torch.sum(x, 1)


if __name__ == '__main__':
    x = torch.rand(4, 19, 19)
    schnet = SchNet()
    schnet(x)