import torch
import torch.nn as nn

from Code.Models.interaction import Interaction
from Code.Models.activation_functions import SSP
from Code.Models.base_model import BaseNet


class SchNetFeature(nn.Module):

    def __init__(self, mean, std, use_cuda=False):
        super(SchNetFeature, self).__init__()
        self.mean = mean
        self.std = std
        self.interaction_1 = Interaction(use_cuda=use_cuda)
        self.interaction_2 = Interaction(use_cuda=use_cuda)
        self.interaction_3 = Interaction(use_cuda=use_cuda)
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


class SchNet(BaseNet):

    def __init__(self, use_cuda, eval_path, comment, abc_scheme=None, n_atoms=19, n_features=64, mean=0, std=1):
        self.n_atoms = n_atoms
        self.n_features = n_features
        self.mean = mean
        self.std = std
        super(SchNet, self).__init__(use_cuda, eval_path, comment, abc_scheme=abc_scheme)

    def _setup(self):
        self.embedding = nn.Embedding(self.n_atoms, self.n_features)
        self.schnetfeat = SchNetFeature(self.mean, self.std)

    def forward(self, x_data):
        atom_ids = x_data[:, :, -1].long()
        distances = x_data[:, :, :-1]
        embedded_atom_ids = self.embedding(atom_ids)
        return torch.sum(self.schnetfeat(embedded_atom_ids, distances), 1)