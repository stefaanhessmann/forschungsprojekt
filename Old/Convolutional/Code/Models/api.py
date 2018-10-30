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

