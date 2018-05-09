import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time


class SubNetwork(nn.Module):
    
    def __init__(self, input_dim):
        #600-400-200-100-80-40-20
        super(SubNetwork, self).__init__()
        self.h1 = nn.Linear(input_dim, 600)
        self.h2 = nn.Linear(600, 400)
        self.h3 = nn.Linear(400, 200)
        self.h4 = nn.Linear(200, 100)
        self.h5 = nn.Linear(100, 80)
        self.h6 = nn.Linear(80, 40)
        self.h7 = nn.Linear(40, 20)
        self.h8 = nn.Linear(20, 1)
    
    def forward(self, X):
        a1 = F.relu(self.h1(X))
        a2 = F.relu(self.h2(a1))
        a3 = F.relu(self.h3(a2))
        a4 = F.relu(self.h4(a3))
        a5 = F.relu(self.h5(a4))
        a6 = F.relu(self.h6(a5))
        a7 = F.relu(self.h7(a6))
        out = self.h8(a7)
        return out


class DeepPotential(nn.Module):
    
    def __init__(self):
        super(DeepPotential, self).__init__()
        # one subnetwork for every layer:
        sub_dim = 18*4
        self.h_net = SubNetwork(sub_dim)
        self.c_net = SubNetwork(sub_dim)
        self.o_net = SubNetwork(sub_dim)


    def forward(self, X):
        a1 = F.relu(self.c_net.forward(X[:, 0]))
        a2 = F.relu(self.c_net.forward(X[:, 1]))
        a3 = F.relu(self.c_net.forward(X[:, 2]))
        a4 = F.relu(self.c_net.forward(X[:, 3]))
        a5 = F.relu(self.c_net.forward(X[:, 4]))
        a6 = F.relu(self.c_net.forward(X[:, 5]))
        a7 = F.relu(self.c_net.forward(X[:, 6]))
        a8 = F.relu(self.o_net.forward(X[:, 7]))
        a9 = F.relu(self.o_net.forward(X[:, 8]))
        a10 = F.relu(self.h_net.forward(X[:, 9]))
        a11 = F.relu(self.h_net.forward(X[:, 10]))
        a12 = F.relu(self.h_net.forward(X[:, 11]))
        a13 = F.relu(self.h_net.forward(X[:, 12]))
        a14 = F.relu(self.h_net.forward(X[:, 13]))
        a15 = F.relu(self.h_net.forward(X[:, 14]))
        a16 = F.relu(self.h_net.forward(X[:, 15]))
        a17 = F.relu(self.h_net.forward(X[:, 16]))
        a18 = F.relu(self.h_net.forward(X[:, 17]))
        a19 = F.relu(self.h_net.forward(X[:, 18]))
        out = a1 + a2 + a3 + a4 + a5 + a6 \
                + a7 + a8 + a9 + a10 + a11 \
                + a12 + a13 + a14 + a15 + a16 \
                + a17 + a18 + a19
        return out


def update_lr(abc, x):
    a, b, c = abc
    return a*b**(x/c)


def train(Model, optim, X_data, Y_data, n_epochs, batchsize, abc, use_for_train=0.8, print_every=100):
    # empty lists to store data for plotting
    epochs = []
    losses = []
    tests = []
    time_per_loop = []
    # time measurements
    start_time = time.time()
    total_time = time.time()

    # split dataset into test an train
    split = int(X_data.shape[0] * 0.8)
    X_train = X_data[:split]
    Y_train = Y_data[:split]
    X_test = X_data[split:]
    Y_test = Y_data[split:]
    # define network
    model = Model
    loss_fn = nn.MSELoss()
    optim.lr = abc[0]
    # learning loop:
    for epoch in range(n_epochs):
        #import pdb; pdb.set_trace()
        # forward pass
        ids = np.random.randint(0, X_train.shape[0], batchsize).tolist()
        test_ids = np.random.randint(0, X_test.shape[0], batchsize).tolist()
        Y_pred = model.forward(X_train[ids])
        loss = loss_fn(Y_pred, Y_train[ids])
        # update parameters
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        # prints
        if epoch % print_every == 0 and epoch != 0:
            duration = time.time() - start_time
            print('{}%: {} --- time estimate: {} min'.format(np.round(epoch/n_epochs*100, 1),
                                                         np.round(loss.data[0], 6),
                                                         np.round((time.time()-total_time)/epoch*(n_epochs-epoch)/60, 1)))
            time_per_loop.append(duration)
            start_time = time.time()
        # for plots
        Y_pred_test = model.forward(X_test[test_ids])
        mean_err = loss_fn(Y_pred_test, Y_test[test_ids]).data[0]
        tests.append(mean_err)
        epochs.append(epoch)
        losses.append(loss.data[0])
        optim.lr = update_lr(abc, epoch)

    # create plots
    f_time, ax_time = plt.subplots()
    ax_time.plot(range(len(time_per_loop)), time_per_loop)
    f_loss, ax_loss = plt.subplots()
    ax_loss.semilogy(epochs, losses, label='train')
    ax_loss.set_title("Loss")
    ax_loss.semilogy(epochs, tests, label='test')
    
    return model, optim