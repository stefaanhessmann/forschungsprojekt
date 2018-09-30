import torch
from torch import nn as _nn
from torch import no_grad as _no_grad
from torch import cat as _cat
from Code.Models.nn_extentions import AbcExponentialLR
import time
import os

def create_path(path):
    part_path = path.split('/')
    new_path = ''
    for folder in part_path:
        new_path = os.path.join(new_path, folder)
        if not os.path.exists(new_path):
            os.mkdir(new_path)

class BaseNet(_nn.Module):
    def __init__(self, use_cuda, eval_path, comment, abc_scheme=(0.001, 0.96, 1), lr_step_every='n steps'):
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
        self.eval_path = eval_path
        self.checkpoint_path = eval_path + '/ModelCheckpoints/{}/'.format(self.comment)
        create_path(self.checkpoint_path)
        self.start_fit = 0
        self.epoch = -1
        self.l_steps = 0
        self.lr_step_every = lr_step_every

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
            if self.lr_scheduler and self.lr_step_every == 'epoch':
                self.lr_scheduler.step()
                print('update the lr at step {}'.format(self.l_steps))
            self._param_save()
            self._print_progress(n_epochs, train_loss[-1], test_loss[-1])
        return train_loss, test_loss


    def train_step(self, loader):
        """
        Run a single training epoch and do the back-propagation.
        """
        #print('train step')
        self.train()
        #self.update_momentum()
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
            self.l_steps += 1
            if self.lr_scheduler and self.lr_step_every == 'n steps':
                if self.l_steps % 100000 == 0:
                    self.lr_scheduler.step()
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
        loss = self.loss_fn(self(x), y)
        return loss

    def _transform(self, loader, with_label=False, in_train_mode=False):
        """
        Apply the model on the data-loader.
        """
        if in_train_mode:
            self.train()
        else:
            self.eval()
        latent, labels = [], []
        for x, y in loader:
            if self.use_cuda:
                x = x.cuda()
            pred = self(x)
            latent.append(pred.detach().cpu())
            labels.append(y)
        if with_label:
            return _cat(latent).data, _cat(labels).data
        return _cat(latent).data

    def transform(self, loader, in_train_mode=False):
        """
        Wrapper to transform the data without labels.
        """
        return self._transform(loader=loader, in_train_mode=in_train_mode)

    def transform_with_label(self, loader, in_train_mode=False):
        """
        Wrapper to transform the data without labels.
        """
        return self._transform(loader=loader, with_label=True, in_train_mode=in_train_mode)

    def update_momentum(self):
        a, b, c = self.abc_scheme
        for subnetwork in self.children():
            for layer in subnetwork.children():
                if type(layer) == torch.nn.BatchNorm1d:
                    layer.momentum = 1 - (a * b**(0.6*self.epoch/c))
        return
