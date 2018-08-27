import torch
from torch import nn as _nn
from torch import no_grad as _no_grad
from torch import cat as _cat
from Code.DataGeneration.saver import create_path
from Code.Models.nn_extentions import AbcExponentialLR
import time


class BaseNet(_nn.Module):
    def __init__(self, use_cuda, checkpoint_path, arch, abc_scheme=[0.01, 0.96, 1.5]):
        super().__init__()
        #self.arch = arch
        self._setup()
        #import pdb; pdb.set_trace()
        self.abc_scheme = abc_scheme
        self.optim = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss_fn = _nn.MSELoss()
        self.lr_scheduler = AbcExponentialLR(self.optim, self.abc_scheme[1], self.abc_scheme[2])
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()
        self.double()
        self.checkpoint_path = checkpoint_path
        create_path(checkpoint_path)
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

    def _print_progress(self, n_epochs, train_loss, test_loss):
        """
        Function to call after every epoch to check the progress.
        """
        progress = round(self.epoch / n_epochs, 2)
        time_estim = round((time.time()-self.start_fit)/self.epoch
                * (n_epochs - self.epoch)/60, 2)
        status = 'epoch: {}\tprogress: {}\ttime estimate: {}\ttrain loss: {}\ttest loss: {}'.format(
            self.epoch, progress, time_estim, round(train_loss, 6), round(test_loss, 6))
        print(status)

    def _param_save(self):
        checkpoint_file = self.checkpoint_path + 'epoch_{}'.format(self.epoch)
        torch.save(self.state_dict(), checkpoint_file)

    def fit(self, train_loader, n_epochs, test_loader=None):
        """
        Train the model with the provided data-loader.
        """
        self.start_fit = time.time()
        train_loss, test_loss = [], []
        for epoch in range(n_epochs):
            self.epoch = epoch + 1
            train_loss.append(
                self.train_step(train_loader))
            if test_loader:
                with _no_grad():
                    test_loss.append(
                        self.test_step(test_loader))
            else:
                test_loss.append(0)
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self._param_save()
            self._print_progress(n_epochs, train_loss[-1], test_loss[-1])
        return train_loss, test_loss


    def train_step(self, loader):
        """
        Run a single training epoch and do the back-propagation.
        """
        self.train()
        train_loss = 0
        for x, y in loader:
            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()
            self.optim.zero_grad()
            loss = self.forward_and_apply_loss_function(x, y)
            loss.backward()
            train_loss += loss.item()
            self.optim.step()
            self.update_momentum()
        return train_loss / float(len(loader))

    def test_step(self, loader):
        """
        Run a single validation epoch.
        """
        self.eval()
        test_loss = 0
        if loader is None:
            return None
        for x, y in loader:
            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()
            test_loss += self.forward_and_apply_loss_function(x, y).item()
        return test_loss / float(len(loader))

    def forward_and_apply_loss_function(self, x, y):
        return self.loss_fn(self(x).view(-1), y)

    def _transform(self, loader, with_label=False):
        """
        Apply the model on the data-loader.
        """
        self.eval()
        latent, labels = [], []
        for x, y in loader:
            pred = self(x)
            if self.use_cuda:
                pred = pred.cpu()
                y.cpu()
            latent.append(pred)
            labels.append(y)
        if with_label:
            return _cat(latent).data, _cat(labels).data
        return _cat(latent).data

    def transform(self, loader):
        """
        Wrapper to transform the data without labels.
        """
        return self._transform(loader=loader)

    def transform_with_label(self, loader):
        """
        Wrapper to transform the data without labels.
        """
        return self._transform(loader=loader, with_label=True)

    def update_momentum(self):
        a, b, c = self.abc_scheme
        for subnetwork in self.children():
            for layer in subnetwork.children():
                if type(layer) == torch.nn.BatchNorm1d:
                    layer.momentum = a * b**(0.6*self.epoch/c)
        return
