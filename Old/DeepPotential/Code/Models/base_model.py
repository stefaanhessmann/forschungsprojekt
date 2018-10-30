import torch
from torch import nn as _nn
from torch import no_grad as _no_grad
from torch import cat as _cat
from Code.DataGeneration.saver import create_path
from Code.Models.nn_extentions import AbcExponentialLR
import time


class BaseNet(_nn.Module):
    def __init__(self, use_cuda, eval_path, comment, abc_scheme=(0.005, 0.96, 3.0)):
        super().__init__()
        self._setup()
        self.abc_scheme = abc_scheme
        self.optim = torch.optim.Adam(self.parameters(), lr=self.abc_scheme[0])
        self.loss_fn = _nn.MSELoss()
        self.lr_scheduler = AbcExponentialLR(self.optim, self.abc_scheme[1], self.abc_scheme[2])
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()
        self.double()
        self.comment = comment
        self.checkpoint_path = eval_path + '/ModelCheckpoints/{}/'.format(self.comment)
        create_path(self.checkpoint_path)
        self.start_fit = 0
        self.epoch = -1

    def _setup(self):
        """
        To be implemented in deriving classes.
        Define the layers here.
        """

    def forward(self, X):
        """
        To be implemented in deriving classes.
        Define the forward pass through the layers here.
        """

