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
from utils.main_utils import load_hypermater
from utils.loadresult_utils import *
torch.manual_seed(0)

numusers = 10 
num_glob_iters = 800 
algorithm_list = ["LocalOnly", "FedAvg", "FedAvgFT", "FedBABU", "FedPer", "FedRep", "LGFedAvg", "pFedBayes", "BPFedPD","FedSOUL", "FedSIFac", "FedSI"]

args = load_hypermater()
for args.dataset in ["Mnist", "FMnist", "Cifar10"]:
    for args.datasize in ["small", "large"]:
        plot_summary_one_figure_mnist_Compare(num_users=numusers, Numb_Glob_Iters=num_glob_iters, 
                                              algorithms_list=algorithm_list, args=args) 