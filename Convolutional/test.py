import numpy as np
import torch
import torch.nn as nn
from Convolutional.Code.Models.schnet import SchNet
from Convolutional.Code.Models.api import Network

data_path = './Dataset/c702h10_X.npy'
label_path = './Dataset/c702h10_Y.npy'
comment = 'schnet_1'
eval_path = './evaluation/iso17'

x_data = np.load(data_path)
y_data = np.load(label_path) * -1

schnet = SchNet(use_cuda=False, comment=comment, eval_path=eval_path)#mean=y_mean, std=y_std)
network = Network(schnet)
network.create_dataloaders(x_data, y_data, 2)
network.fit(2)