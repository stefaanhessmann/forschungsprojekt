import numpy as np
import torch
import os
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

class Network(object):

    def __init__(self, model):
        self.model = model
        self.eval_path = model.eval_path
        self.comment = model.comment
        self.train_losses = []
        self.test_losses = []
        self.y_min, self.y_max = None, None
        self.train_loader = None
        self.test_loader = None
        self.excluded_data = None
        self.loss_figure = None

    def create_dataloaders(self, x_train, y_train, x_test, y_test, batch_size=128, use_for_train=0.8, exclude_ids=None,
                           standardize_X=False, normalize_X=False):
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
        # exclude validation data
        if exclude_ids:
            self.excluded_data = [x_train[exclude_ids], y_train[exclude_ids]]
            use_ids = [value for value in range(len(y_train)) if value not in exclude_ids]
            x_train, y_train = x_train[use_ids], y_train[use_ids]

        # create Dataloaders
        print(x.shape, y.shape)
        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)
        train_dataset = data_utils.TensorDataset(torch.DoubleTensor(x_train), torch.DoubleTensor(y_train))
        self.train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                             pin_memory=True)
        test_dataset = data_utils.TensorDataset(torch.DoubleTensor(x_test), torch.DoubleTensor(y_test))
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, drop_last=True, pin_memory=True)

        return None

    def fit(self, n_epochs):
        assert (self.train_loader is not None)
        self.train_losses, self.test_losses = self.model.fit(self.train_loader, n_epochs, self.test_loader)

        return None

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
        return

    def show_loss_plot(self):
        if not self.loss_figure:
            self.create_loss_plot()
        self.loss_figure.show()
        return

    def transform(self, loader):
        return self.model.transform(loader)

    def calculate_mae(self, loader):
        pred, y = self.model.transform_with_label(loader)
        pred = backtransform(pred, self.y_min, self.y_max).squeeze()
        y = backtransform(y, self.y_min, self.y_max).squeeze()
        assert (pred.shape == y.shape)
        return abs(pred - y).mean().numpy().item()

    def calculate_test_mae(self):
        return self.calculate_mae(self.test_loader)

    def calculate_train_mae(self):
        return self.calculate_mae(self.train_loader)

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
        self.model.epoch = epoch
        print('loading parameter file {} successful'.format(params_file))
