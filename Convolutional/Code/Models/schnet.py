import torch
import torch.nn as nn

from Convolutional.Code.Models.interaction import Interaction
from Convolutional.Code.Models.activation_functions import SSP


class SchNetFeature(nn.Module):

    def __init__(self):
        super(SchNetFeature, self).__init__()
        self.interaction_1 = Interaction()
        self.interaction_2 = Interaction()
        self.interaction_3 = Interaction()
        self.atomwise_1 = nn.Linear(64, 32)
        self.ssp = SSP()
        self.atomwise_2 = nn.Linear(32, 1)

    def forward(self, embedding, distances):
        x = self.interaction_1(embedding, distances)
        x = self.interaction_2(x, distances)
        x = self.interaction_3(x, distances)
        x = self.atomwise_1(x)
        x = self.ssp(x)
        return self.atomwise_2(x)


class SchNet(nn.Module):

    def __init__(self, n_atoms=19, n_features=64):
        super(SchNet, self).__init__()
        self.n_atoms = n_atoms
        self.embedding = nn.Embedding(n_atoms, n_features)
        self.schnetfeat = SchNetFeature()

    def forward(self, x_data):
        distances, atom_ids = x_data
        embedded_atom_ids = self.embedding(atom_ids)
        return torch.sum(torch.stack([self.schnetfeat(embedded_atom_ids[i],
                                                      distances[i]) for i in range(self.n_atoms)]))