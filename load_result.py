import h5py
import os
import glob
import copy
import numpy as np
from utils.main_utils import load_hypermater

def get_file_path(args, loadP = False, run_idx = 0):
    alg = args.dataset + "_" + args.datasize + "_" + args.algorithm
    if (loadP):
        alg += "_p"
    alg = alg + "_" + str(args.learning_rate) + "_" + str(args.beta) + "_" + str(args.lamda) + "_" + str(args.numusers) + "u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    if(args.algorithm == "pFedMe" or args.algorithm == "pFedMe_p"):
        alg = alg + "_" + str(args.K) + "_" + str(args.personal_learning_rate)
    alg = alg + "_" + str(run_idx)
    if (args.only_one_local):
            alg = alg + "_only_one_local"
    return "./results/"+'{}.h5'.format(alg, args.local_epochs)    

def change_avg(args_pre):
    args = copy.deepcopy(args_pre)
    if (args.algorithm == 'PerAvg'):
        args.beta = 0.1
    if (args.algorithm == 'pFedMe'):
        args.learning_rate=0.01 
        args.personal_learning_rate=0.01
    if (args.algorithm == 'pFedBayes'):
        args.batch_size=100 
    if (args.algorithm == 'BPFedPD'):
        args.batch_size=100 
    if (args.algorithm == 'FedSOUL'):
        args.beta = 1.0
    return args

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

# dataset_list: "Mnist", "FMnist", "Cifar10"
# datasize_list: "small", "large"
dataset_list = ["Mnist", "FMnist", "Cifar10"]
datasize_list = ["small", "large"]
# algorithm list: "FedPer", "LGFedAvg", "FedRep", "FedSOUL", "BPFedPD"
algorithm_list = ["FedPer", "LGFedAvg", "FedRep", "FedSOUL", "BPFedPD"]
# output_style: 0,1
output_style = 1

print_head(args)
for args.algorithm in algorithm_list:
    for args.add_new_client in [0, 1, 2]:
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
                file_path = get_file_path(args_new, loadP=True)
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