import h5py
import os
import glob
import copy
import numpy as np
from utils.main_utils import load_hypermater
from utils.loadresult_utils import *

# dataset_list: "Mnist", "FMnist", "Cifar10"
# datasize_list: "small", "large"
dataset_list = ["Mnist"]
datasize_list = ["small"]
algorithm_list = ["FedAvg", "FedBABU", "LGFedAvg", "BPFedPD", "FedSIFac", "FedSI"]

args = load_hypermater()
for run_idx in [0]:
    for args.algorithm in algorithm_list:
        print("\"%10s\""%(args.algorithm),  end=" ")
        for args.local_epochs in [5,10,15,20]:
            args_new = change_avg(args)
            file_path = get_file_path(args_new, loadP=False, run_idx=run_idx)[0]
            with h5py.File(file_path , 'r') as f: 
                print("", "%.2f"%(max(f['rs_glob_acc'][:] )*100), ",", end="") 
        print()
    print()