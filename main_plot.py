#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from utils.plot_utils import *
import torch
torch.manual_seed(0)

dataset = "Mnist" 
datasize = "large" 

if (dataset == "Mnist" and datasize == "small"): # plot for MNIST convex 
    numusers = 10 
    num_glob_iters = 800 

    local_ep = [20, 20, 20] 
    ocal_ep = [20, 20, 20] 
    lamda = [15, 15, 15] 
    learning_rate = [0.001, 0.001, 0.01] 
    beta =  [1.0, 0.1, 1.0] 
    batch_size = [50, 50, 50] 
    K = [5, 5, 5] 
    personal_learning_rate = [0.001, 0.001, 0.01] 
    algorithms = ["FedAvg", "PerAvg_p", "pFedMe_p"] 

if (dataset == "Mnist" and datasize == "large"): # plot for MNIST convex 
    numusers = 10 
    num_glob_iters = 800 
    local_ep = [20, 20, 20] 
    ocal_ep = [20, 20, 20] 
    lamda = [15, 15, 15] 
    learning_rate = [0.001, 0.001, 0.01] 
    beta =  [1.0, 0.1, 1.0] 
    batch_size = [50, 50, 50] 
    K = [5, 5, 5] 
    personal_learning_rate = [0.001, 0.001, 0.01] 
    algorithms = ["FedAvg", "PerAvg_p", "pFedMe_p"] 

plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, 
                                        Numb_Glob_Iters=num_glob_iters, lamb=lamda, 
                                        learning_rate=learning_rate, 
                                        beta = beta, algorithms_list=algorithms, 
                                        batch_size=batch_size, dataset=dataset, 
                                        datasize= datasize, k = K, personal_learning_rate = personal_learning_rate) 