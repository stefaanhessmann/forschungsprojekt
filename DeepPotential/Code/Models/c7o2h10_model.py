import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time
from Code.DataGeneration.saver import create_path


class SubNetwork(nn.Module):
    def __init__(self, input_dim):
        # 600-400-200-100-80-40-20
        super(SubNetwork, self).__init__()
        self.h1 = nn.Linear(input_dim, 600)
        self.h1_bn = nn.BatchNorm1d(600)
        self.h2 = nn.Linear(600, 400)
        self.h2_bn = nn.BatchNorm1d(400)
        self.h3 = nn.Linear(400, 200)
        self.h3_bn = nn.BatchNorm1d(200)
        self.h4 = nn.Linear(200, 100)
        self.h4_bn = nn.BatchNorm1d(100)
        self.h5 = nn.Linear(100, 80)
        self.h5_bn = nn.BatchNorm1d(80)
        self.h6 = nn.Linear(80, 40)
        self.h6_bn = nn.BatchNorm1d(40)
        self.h7 = nn.Linear(40, 20)
        self.h7_bn = nn.BatchNorm1d(20)
        self.h8 = nn.Linear(20, 1)

    def forward(self, X):
        a1 = F.relu(self.h1_bn(self.h1(X)))
        a2 = F.relu(self.h2_bn(self.h2(a1)))
        a3 = F.relu(self.h3_bn(self.h3(a2)))
        a4 = F.relu(self.h4_bn(self.h4(a3)))
        a5 = F.relu(self.h5_bn(self.h5(a4)))
        a6 = F.relu(self.h6_bn(self.h6(a5)))
        a7 = F.relu(self.h7_bn(self.h7(a6)))
        out = self.h8(a7)
        return out


class DeepPotential(nn.Module):
    def __init__(self):
        super(DeepPotential, self).__init__()
        # one subnetwork for every layer:
        sub_dim = 18 * 4
        self.h_net = SubNetwork(sub_dim)
        self.c_net = SubNetwork(sub_dim)
        self.o_net = SubNetwork(sub_dim)

    def forward(self, X):
        a1 = F.relu(self.h_net.forward(X[:, 0]))
        a2 = F.relu(self.h_net.forward(X[:, 1]))
        a3 = F.relu(self.h_net.forward(X[:, 2]))
        a4 = F.relu(self.h_net.forward(X[:, 3]))
        a5 = F.relu(self.h_net.forward(X[:, 4]))
        a6 = F.relu(self.h_net.forward(X[:, 5]))
        a7 = F.relu(self.h_net.forward(X[:, 6]))
        a8 = F.relu(self.h_net.forward(X[:, 7]))
        a9 = F.relu(self.h_net.forward(X[:, 8]))
        a10 = F.relu(self.h_net.forward(X[:, 9]))
        a11 = F.relu(self.c_net.forward(X[:, 10]))
        a12 = F.relu(self.c_net.forward(X[:, 11]))
        a13 = F.relu(self.c_net.forward(X[:, 12]))
        a14 = F.relu(self.c_net.forward(X[:, 13]))
        a15 = F.relu(self.c_net.forward(X[:, 14]))
        a16 = F.relu(self.c_net.forward(X[:, 15]))
        a17 = F.relu(self.c_net.forward(X[:, 16]))
        a18 = F.relu(self.o_net.forward(X[:, 17]))
        a19 = F.relu(self.o_net.forward(X[:, 18]))
        out = a1 + a2 + a3 + a4 + a5 + a6 \
              + a7 + a8 + a9 + a10 + a11 \
              + a12 + a13 + a14 + a15 + a16 \
              + a17 + a18 + a19
        return out


def update_lr(abc, x):
    a, b, c = abc
    return a * b ** (x / c)


def train(Model, optim, X_data, Y_data, n_epochs, batchsize, abc, use_for_train=0.8, print_every=100, n_calc_test=100,
          checkpoint_path='ModelCheckpoints/'):
    # empty lists to store data for plotting
    epochs = []
    losses = []
    tests = []
    time_per_loop = []
    # Create path for storing checkpoints:
    create_path(checkpoint_path)
    # time measurements
    start_time = time.time()
    total_time = time.time()
    # normalize Y
    Y_data, Y_min, Y_max = normalize(Y_data)
    # split dataset into test an train
    n_batches = int(X_data.shape[0] / batchsize)
    split = int(n_batches * batchsize * use_for_train)
    X_train = X_data[:split]
    Y_train = Y_data[:split]
    X_test = X_data[split:]
    Y_test = Y_data[split:]
    # define network
    model = Model
    loss_fn = nn.MSELoss()
    optim.lr = abc[0]

    n_batches = X_train.shape[0] // batchsize
    for epoch in range(n_epochs):
        # import pdb; pdb.set_trace()
        optim.lr = update_lr(abc, epoch)
        print('\nEpoch: {}\tlearning rate: {}\n---'.format(epoch, np.round(optim.lr, 5)))
        for batch_id in range(n_batches - 1):
            # forward pass
            X_batch = X_train[batch_id * batchsize:(batch_id + 1) * batchsize]
            Y_batch = Y_train[batch_id * batchsize:(batch_id + 1) * batchsize]
            Y_pred = model.forward(X_batch)
            Y_pred = Y_pred.reshape(Y_pred.shape[0])
            loss = loss_fn(Y_pred, Y_batch)
            # update parameters
            optim.zero_grad()
            loss.backward()
            optim.step()

            # for plots
            test_ids = np.random.randint(0, X_test.shape[0], min(X_test.shape[0], n_calc_test)).tolist()
            Y_pred_test = model.forward(X_test[test_ids])
            Y_pred_test = Y_pred_test.reshape(Y_pred_test.shape[0])
            mean_err = loss_fn(Y_pred_test, Y_test[test_ids]).item()
            tests.append(mean_err)
            epochs.append(epoch)
            losses.append(loss.item())

            # prints
            if batch_id % print_every == 0 and batch_id != 0:
                total_duration = time.time() - total_time
                mean_loss = np.round(np.mean(losses[-print_every:]), 6)
                total_iterations = n_epochs * n_batches
                done = epoch * n_batches + batch_id
                to_do = total_iterations - done
                av_itertime = total_duration / done
                time_estimate = av_itertime * to_do
                progress = done / total_iterations
                print('total: {} %\tcurrent epoch: {} %\tloss: {}\ttime estimate: {} min'.format(
                    np.round(progress * 100, 1),
                    np.round(batch_id / n_batches * 100, 1),
                    mean_loss,
                    np.round(time_estimate / 60, 1)))
                time_per_loop.append(av_itertime)
        #mae = np.abs(
        #    backtransform(model.forward(X_test), Y_min, Y_max).data.numpy().reshape(len(X_test))
        #    - backtransform(Y_test, Y_min, Y_max).data.numpy().reshape(len(Y_test))
        #    ).mean()
        #print('The nural network reaches a mean absolute error of {} eV'.format(mae))
        checkpoint_file = checkpoint_path + 'epoch_{}'.format(epoch)
        torch.save(model.state_dict(), checkpoint_file)
        print('saved checkpoint at: {}\n'.format(checkpoint_file))
    # create plots
    f_time, ax_time = plt.subplots()
    ax_time.plot(range(len(time_per_loop)), time_per_loop)
    ax_time.set_title("Time per loop")
    f_loss, ax_loss = plt.subplots()
    ax_loss.semilogy(epochs, losses, label='train')
    ax_loss.set_title("Loss")
    ax_loss.semilogy(epochs, tests, label='test')

    return model, optim


def normalize(Y):
    Y_min = Y.min()
    Y_max = Y.max()
    return (Y - Y_min) / (Y_max - Y_min), Y_min, Y_max

def backtransform(Y_normed, Y_min, Y_max):
    return Y_normed * (Y_max - Y_min) + Y_min
