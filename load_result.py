import h5py
import os
import glob
import copy
import numpy as np
from utils.main_utils import load_hypermater
from utils.loadresult_utils import *

# dataset_list: "Mnist", "FMnist", "Cifar10"
# datasize_list: "small", "large"
dataset_list = ["Mnist", "FMnist", "Cifar10"]
#dataset_list = ["Mnist","FMnist"]
#datasize_list = ["small"]
#dataset_list = ["Mnist", "FMnist"]
datasize_list = ["small","large"]
# datasize_list = ["large"]
# algorithm list: "FedPer", "LGFedAvg", "FedRep", "FedSOUL", "BPFedPD"
#algorithm_list = ["LocalOnly", "FedAvg", "FedAvgFT", "FedBABU", "FedPer", "FedRep", "LGFedAvg", "pFedBayes", "BPFedPD","FedSOUL", "FedSIFac", "FedSI"]
#algorithm_list=["FedPer", "FedRep", "FedBABU", "LGFedAvg", "FedSOUL", "BPFedPD", "FedSI"]
algorithm_list = ["FedSI"]
# output_style: 0, 1
output_style = 1
loadP = False


def print_head(args):
    print("           ", end="")
    for args.dataset in dataset_list:
        print("%13s"%args.dataset, end="")
        print("       ", end="")
    print()

    print("           ", end="")
    for args.dataset in dataset_list:
        for args.datasize in datasize_list:
            print("%7s"%args.datasize, end="")
            print("  ", end="")
    print()

args = load_hypermater()
print(args)
print_head(args)

for run_idx in [0, 1]:
    for args.algorithm in algorithm_list:
        for args.add_new_client in [100]:
            if (output_style == 1):
                print("\"%10s\""%(args.algorithm + " " + str(args.add_new_client)), end=" ")
                print(":[", end="")
            else:
                print("%10s"%args.algorithm, args.add_new_client, end=" ")

            for args.dataset in dataset_list:
                for args.datasize in datasize_list:
                    if (args.add_new_client == 2):
                        args.only_one_local = 1
                    else:
                        args.only_one_local = 0
                    args_new = change_avg(args)
                    file_path = get_file_path(args_new, loadP=loadP, run_idx=run_idx)[0]
                    with h5py.File(file_path , 'r') as f: 
                        # print(file_path)
                        if (args.add_new_client == 1):
                            if (output_style == 1):
                                print("", "%.2f"%(max(f['rs_glob_acc'][:800] )*100), ",", end="") 
                            else:
                                print("", "%.2f"%(max(f['rs_glob_acc'][:800] )*100), "&", end="") 
                            # print(" ", "%.2f"%(max(f['rs_glob_acc'][:] )*100), "|", "%3d"%np.argmax(f['rs_glob_acc'][:]) , "&", end=" ") 
                            # print(f['rs_glob_acc'][:]) 
                        elif (args.add_new_client == 2):
                            if (output_style == 1):
                                print("", "%.2f"%(max(f['rs_glob_acc'][-100:] )*100), ",", end="") 
                            else:
                                print("", "%.2f"%(max(f['rs_glob_acc'][-100:] )*100), "&", end="") 
                        else:
                            if (output_style == 1):
                                print("", "%.2f"%(max(f['rs_glob_acc'][:] )*100), ",", end="") 
                            else:
                                print("", "%.2f"%(max(f['rs_glob_acc'][:] )*100), "&", end="") 
            if (output_style == 1):
                print("],")
            else:
                print()
    print()