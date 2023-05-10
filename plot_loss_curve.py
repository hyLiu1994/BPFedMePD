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

dataset = "FMnist" 
datasize = "small" 
for dataset in ["Mnist", "FMnist", "Cifar10"]:
    for datasize in ["small", "large"]:
        if (dataset == "Mnist" and datasize == "small"): # plot for MNIST convex 
            numusers = 10 
            num_glob_iters = 800 
            algorithms = ["FedAvg", "PerAvg_p", "pFedMe_p", 
                        "FedPer_p", "LGFedAvg_p", "FedRep_p",
                        "FedSOUL_p", "pFedBayes_p", "BPFedPD_p"] 
            local_ep = [20, 20, 20, 20, 20, 20, 20, 20, 20] 
            lamda =    [15, 15, 15, 15, 15, 15, 15, 15, 15] 
            learning_rate = [0.001, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            personal_learning_rate = [0.001, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            beta =  [1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            beta =  [1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            beta =  [1.0, 0.12, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

            # beta =  [1.0, 0.14, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            # beta =  [1.0, 0.16, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            # beta =  [1.0, 0.18, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            batch_size = [50, 50, 50, 50, 50, 50, 50, 100, 100] 
            K = [5, 5, 5, 5, 5, 5, 5, 5, 5] 

        if (dataset == "Mnist" and datasize == "large"): # plot for MNIST convex 
            numusers = 10 
            num_glob_iters = 800 
            algorithms = ["FedAvg", "PerAvg_p", "pFedMe_p", 
                        "FedPer_p", "LGFedAvg_p", "FedRep_p",
                        "FedSOUL_p", "pFedBayes_p", "BPFedPD_p"] 
            local_ep = [20, 20, 20, 20, 20, 20, 20, 20, 20] 
            lamda =    [15, 15, 15, 15, 15, 15, 15, 15, 15] 
            learning_rate = [0.001, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            personal_learning_rate = [0.001, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            beta =  [1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            batch_size = [50, 50, 50, 50, 50, 50, 50, 100, 100] 
            K = [5, 5, 5, 5, 5, 5, 5, 5, 5] 

        if (dataset == "FMnist" and datasize == "small"): # plot for MNIST convex 
            numusers = 10 
            num_glob_iters = 800 
            algorithms = ["FedAvg", "PerAvg_p", "pFedMe_p", 
                        "FedPer_p", "LGFedAvg_p", "FedRep_p",
                        "FedSOUL_p", "pFedBayes_p", "BPFedPD_p"] 
            local_ep = [20, 20, 20, 20, 20, 20, 20, 20, 20] 
            lamda =    [15, 15, 15, 15, 15, 15, 15, 15, 15] 
            learning_rate = [0.001, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            personal_learning_rate = [0.001, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            beta =  [1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            beta =  [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            # beta =  [1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            batch_size = [50, 50, 50, 50, 50, 50, 50, 100, 100] 
            K = [5, 5, 5, 5, 5, 5, 5, 5, 5] 

        if (dataset == "FMnist" and datasize == "large"): # plot for MNIST convex 
            numusers = 10 
            num_glob_iters = 800 
            algorithms = ["FedAvg", "PerAvg_p", "pFedMe_p", 
                        "FedPer_p", "LGFedAvg_p", "FedRep_p",
                        "FedSOUL_p", "pFedBayes_p", "BPFedPD_p"] 
            local_ep = [20, 20, 20, 20, 20, 20, 20, 20, 20] 
            lamda =    [15, 15, 15, 15, 15, 15, 15, 15, 15] 
            learning_rate = [0.001, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            personal_learning_rate = [0.001, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            beta =  [1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            # beta =  [1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            batch_size = [50, 50, 50, 50, 50, 50, 50, 100, 100] 
            K = [5, 5, 5, 5, 5, 5, 5, 5, 5] 

        if (dataset == "Cifar10" and datasize == "small"):
            numusers = 10 
            num_glob_iters = 800 
            algorithms = ["FedAvg", "PerAvg_p", "pFedMe_p", 
                        "FedPer_p", "LGFedAvg_p", "FedRep_p",
                        "FedSOUL_p", "pFedBayes_p", "BPFedPD_p"] 
            local_ep = [20, 20, 20, 20, 20, 20, 20, 20, 20] 
            lamda =    [15, 15, 15, 15, 15, 15, 15, 15, 15] 
            learning_rate = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            personal_learning_rate = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            beta =  [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            beta =  [1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

            # beta =  [1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            # beta =  [1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            batch_size = [50, 50, 50, 50, 50, 50, 50, 100, 100] 
            K = [5, 5, 5, 5, 5, 5, 5, 5, 5] 

        if (dataset == "Cifar10" and datasize == "large"):
            numusers = 10 
            num_glob_iters = 800 
            algorithms = ["FedAvg", "PerAvg_p", "pFedMe_p", 
                        "FedPer_p", "LGFedAvg_p", "FedRep_p",
                        "FedSOUL_p", "pFedBayes_p", "BPFedPD_p"] 
            local_ep = [20, 20, 20, 20, 20, 20, 20, 20, 20] 
            lamda =    [15, 15, 15, 15, 15, 15, 15, 15, 15] 
            learning_rate = [0.01, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            learning_rate = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            personal_learning_rate = [0.01, 0.001, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            personal_learning_rate = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
            beta =  [1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            beta =  [1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

            # beta =  [1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            # beta =  [1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
            batch_size = [50, 50, 50, 50, 50, 50, 50, 100, 100] 
            K = [5, 5, 5, 5, 5, 5, 5, 5, 5] 

        plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, 
                                                Numb_Glob_Iters=num_glob_iters, lamb=lamda, 
                                                learning_rate=learning_rate, 
                                                beta = beta, algorithms_list=algorithms, 
                                                batch_size=batch_size, dataset=dataset, 
                                                datasize= datasize, k = K, personal_learning_rate = personal_learning_rate) 