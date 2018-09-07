import numpy as np
import torch
import torch.nn as nn
from Convolutional.Code.Models.schnet import SchNet


dists = torch.tensor(np.random.random((19, 18)))
atom_ids = torch.tensor(np.random.randint(19, size=19))
energies = torch.tensor(np.random.random(1))

x_data = [dists, atom_ids]
y_data = energies.double()

schnet = SchNet()

x = schnet(x_data)

losses = []
loss = nn.MSELoss()
optimizer = torch.optim.Adam(params=schnet.parameters(), lr=0.0005)
for i in range(100):
    print(i)
    optimizer.zero_grad()
    loss_val = loss(y_data, schnet(x_data).double())
    loss_val.backward()
    optimizer.step()
    print(loss_val.data)
    losses.append(loss_val)

breakpoint = ''