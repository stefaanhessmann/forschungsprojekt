import numpy as np
import torch
import os
import time
from torch import no_grad
from torch import cat
from torch.utils import data as data_utils
from matplotlib import pyplot as plt


def normalize(Y):
    Y_min = Y.min()
    Y_max = Y.max()
    return (Y - Y_min) / (Y_max - Y_min), Y_min, Y_max


def backtransform(Y_normed, Y_min, Y_max):
    return Y_normed * (Y_max - Y_min) + Y_min


def create_path(path):
    part_path = path.split('/')
    new_path = ''
    for folder in part_path:
        new_path = os.path.join(new_path, folder)
        if not os.path.exists(new_path):
            os.mkdir(new_path)


class Trainer(object):

    def __init__(self, model, optimizer, loss_fn, eval_path, comment, lr_scheduler=None, abc_schedule=None,
                 use_cuda=False, momentum_scheme=False, lr_step='e1'):
        # related to network
        self.model = model
        self.optim = optimizer(self.model.parameters(), lr=abc_schedule[0])
        self.loss_fn = loss_fn()
        self.lr_scheduler = lr_scheduler(self.optim, *abc_schedule[1:]) if not None else None
        self.abc_schedule = abc_schedule
        self.momentum_scheme = momentum_scheme
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()
        # related to data
        self.y_min, self.y_max = None, None
        self.train_loader = None
        self.test_loader = None
        # related to evaluation
        self.train_losses = []
        self.test_losses = []
        self.loss_figure = None
        self.comment = comment
        self.eval_path = eval_path
        self.checkpoint_path = eval_path + '/ModelCheckpoints/{}/'.format(self.comment)
        create_path(self.checkpoint_path)
        # related to training
        self.lr_step = lr_step
        self.start_fit = 0
        self.epoch = 0
        self.n_steps = 0

    def create_dataloaders(self, x_train, y_train, x_test, y_test, batch_size=128,
                           standardize_X=False, normalize_X=False, pin_memory=True, num_workers=2):
        assert normalize_X + standardize_X <= 1, 'Can not normalize and standardize X data!'
        # stack for normalization and standardization
        n_train = x_train.shape[0]
        x = np.vstack((x_train, x_test))
        y = np.hstack((y_train, y_test))
        # normalize y
        y, self.y_min, self.y_max = normalize(y)
        # standardize x
        if standardize_X:
            for column in range(x.shape[-1] - 1):
                col_mean = x[:, :, column].mean()
                col_std = x[:, :, column].std()
                x[:, :, column] = (x[:, :, column] - col_mean) / col_std
        if normalize_X:
            for column in range(x.shape[-1] - 1):
                x[:, :, column] = normalize(x[:, :, column])[0]
        y = y[:, np.newaxis]
        # split into test and train again:
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # create DataLoaders
        train_dataset = data_utils.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        self.train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                                  pin_memory=pin_memory, num_workers=num_workers)
        test_dataset = data_utils.TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, drop_last=True,
                                                 pin_memory=pin_memory, num_workers=num_workers)

    def _print_progress(self, n_epochs, train_loss, test_loss):
        """
        Function to call after every epoch to check the progress.
        """
        progress = round(self.epoch / n_epochs, 2)
        time_estim = round((time.time()-self.start_fit)/self.epoch * (n_epochs - self.epoch)/60, 2)
        status = 'epoch: {}\tprogress: {}\ttime estimate: {}\ttrain loss: {}\ttest loss: {}'.format(
            self.epoch, progress, time_estim, round(train_loss, 6), round(test_loss, 6))
        print(status)

    def _param_save(self):
        checkpoint_file = self.checkpoint_path + 'epoch_{}'.format(self.epoch)
        torch.save(self.model.state_dict(), checkpoint_file)

    def fit(self, n_epochs):
        """
        Train the model with the provided data-loader.
        """
        assert (self.train_loader is not None)
        self.start_fit = time.time()
        while self.epoch < n_epochs:
            self.train_losses.append(
                self.train_step(self.train_loader))
            if self.test_loader:
                with no_grad():
                    self.test_losses.append(
                        self.test_step(self.test_loader))
            else:
                self.test_losses.append(0)
            if self.lr_scheduler:
                if (self.lr_step[0] == 'e' and self.epoch % int(self.lr_step[1:])) == 0 or \
                        (self.lr_step[0] == 's' and self.n_steps % int(self.lr_step[1:]) == 0):
                    self.lr_scheduler.step()
            self._param_save()
            self.epoch += 1
            self._print_progress(n_epochs, self.train_losses[-1], self.test_losses[-1])

    def train_step(self, loader):
        """
        Run a single training epoch and do the back-propagation.
        """
        self.model.train()
        if self.momentum_scheme:
            self.update_momentum()
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
            self.n_steps += 1
        return train_loss / float(len(loader))

    def test_step(self, loader):
        """
        Run a single validation epoch.
        """
        self.model.eval()
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
        return self.loss_fn(self.model(x), y)

    def update_momentum(self):
        a, b, c = self.abc_schedule
        for subnetwork in self.model.children():
            for layer in subnetwork.children():
                if type(layer) == torch.nn.BatchNorm1d:
                    layer.momentum = 1 - (a * b**(0.6*self.epoch/c))

    def transform(self, loader, return_labels, in_train_mode=False):
        """
        Apply the model on the data-loader.
        """
        if in_train_mode:
            self.model.train()
        else:
            self.model.eval()
        latent, labels = [], []
        for x, y in loader:
            if self.use_cuda:
                x = x.cuda()
            pred = self.model(x)
            latent.append(pred.detach().cpu())
            labels.append(y)
        if return_labels:
            return cat(latent).data, cat(labels).data
        return cat(latent).data

    def transform_train(self, return_label=False, in_train_mode=False):
        """
        Wrapper to transform the train data.
        """
        return self.transform(self.train_loader, return_label, in_train_mode)

    def transform_test(self, return_label=False, in_train_mode=False):
        """
        Wrapper to transform the train data.
        """
        return self.transform(self.test_loader, return_label, in_train_mode)

    def create_loss_plot(self):
        assert (self.train_losses != [])
        self.loss_figure = plt.figure()
        plt.plot(self.train_losses, label='train')
        if not self.test_losses:
            return
        plt.plot(self.test_losses, label='test')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()

    def save_loss_plot(self):
        if not self.loss_figure:
            self.create_loss_plot()
        figure_path = self.eval_path + '/loss_plots/{}/'.format(self.comment)
        create_path(figure_path)
        self.loss_figure.savefig(figure_path+'loss_plot')

    def show_loss_plot(self):
        if not self.loss_figure:
            self.create_loss_plot()
        self.loss_figure.show()

    def _calculate_mae(self, loader):
        pred, y = self.transform(loader, return_labels=True)
        pred = backtransform(pred, self.y_min, self.y_max).squeeze()
        y = backtransform(y, self.y_min, self.y_max).squeeze()
        assert (pred.shape == y.shape)
        return abs(pred - y).mean().numpy().item()

    def calculate_mae(self):
        return self._calculate_mae(self.test_loader)

    def _calculate_train_mae(self):
        return self._calculate_mae(self.train_loader)

    def load_network_parameters(self, epoch=-1):
        checkpoint_path = self.eval_path + '/ModelCheckpoints/{}/'.format(self.comment)
        assert os.path.exists(checkpoint_path), 'No model parameters for this comment!'
        files = os.listdir(checkpoint_path)
        epochs = [int(filename.lstrip('epoch_')) for filename in files]
        assert epochs != [], 'No model parameters for this comment!'
        if epoch == -1:
            epoch = max(epochs)
        assert epoch in epochs, 'This epoch was not saved!'
        params_file = checkpoint_path + 'epoch_{}'.format(epoch)
        self.model = torch.load(params_file)
        self.epoch = epoch
        print('loading parameter file {} successful'.format(params_file))
