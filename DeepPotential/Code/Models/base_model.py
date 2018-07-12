import torch
from torch import nn as _nn
from torch import no_grad as _no_grad
from torch import cat as _cat
from Code.DataGeneration.saver import create_path
import time


class BaseNet(_nn.Modules):
    def __init__(self, optim, loss, use_cuda, checkpoint_path, lr_scheduler=None):
        super().__init__()
        self._setup()
        self.optim = optim
        self.loss = loss
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()
        self.checkpoint_path = checkpoint_path
        create_path(checkpoint_path)
        self.lr_scheduler = lr_scheduler
        self.start_fit = 0
        self.epoch = -1

    def _setup(self):
        """
        To be implemented in deriving classes.
        Define the layers here.
        """

    def _forward(self):
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
                           * (n_epochs - self.epoch), 2)
        status = 'epoch: {}\tprogress: {}\ttime estimate: {}\ttrain loss: {}\ttest loss: {}'.format(
            self.epoch, progress, time_estim, train_loss, test_loss)
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
            self.epoch = epoch
            train_loss.append(
                self.train_step(train_loader))
            with _no_grad():
                test_loss.append(
                    self.test_step(test_loader))
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
                x = x.cuda(async=self.async)
                y = y.cuda(async=self.async)
            self.optim.zero_grad()
            loss = self.forward_and_apply_loss_function(x, y)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        return train_loss / float(len(loader))

    def val_step(self, loader):
        """
        Run a single validation epoch.
        """
        self.eval()
        test_loss = 0
        if loader is None:
            return None
        for x, y in loader:
            if self.use_cuda:
                x = x.cuda(async=self.async)
                y = y.cuda(async=self.async)
            test_loss += self.forward_and_apply_loss_function(x, y).item()
        return test_loss / float(len(loader))

    def forward_and_apply_loss_function(self, x, y):
        return self.loss_fn(self.forward(x), y)

    def transform(self, loader):
        """
        Apply the model on the data-loader.
        """
        self.eval()
        latent = []
        for x, _ in loader:
            pred = self.forward(x)
            if self.use_cuda:
                pred = pred.cpu()
            latent.append(pred)
        return _cat(latent).data
