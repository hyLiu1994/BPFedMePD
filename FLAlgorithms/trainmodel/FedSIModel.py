import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from FLAlgorithms.trainmodel import bayes 
from collections import OrderedDict
from FLAlgorithms.trainmodel.layers.misc import FlattenLayer, ModuleWrapper

class DNNFedSI(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(DNNFedSI, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.linear = nn.Linear(mid_dim, output_dim)
        
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x

    def get_mark_personlized_module(self, mark_personal_layer = 0):
        if (mark_personal_layer == 0):
            return []
        if (abs(mark_personal_layer) >= 2):
            return [1, 1, 1, 1]
        if (mark_personal_layer == -1):
            return [0, 0, 1, 1]
        if (mark_personal_layer == 1):
            return [1, 1, 0, 0]

class CIFARNetFedSI(nn.Module):
    def __init__(self, n_class):
        super(CIFARNetFedSI, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.linear = nn.Linear(84, n_class)
        self.mark_list = [2, 2, 2, 2, 2]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

    def get_mark_personlized_module(self, mark_personal_layer = 0):
        if (mark_personal_layer == 0):
            return []
        module_mark_list = []
        for idx, param_num in enumerate(self.mark_list):
            padding = 0
            if (idx < mark_personal_layer or idx >= len(self.mark_list) + mark_personal_layer):
                padding = 1
            for i in range(param_num):
                module_mark_list.append(padding)
        return module_mark_list
