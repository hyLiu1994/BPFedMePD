import h5py
import os
import glob
from utils.main_utils import load_hypermater

def get_file_path(args, loadP = False, run_idx = 0):
    alg = args.dataset + "_" + args.datasize + "_" + args.algorithm
    if (loadP):
        alg += "_p"
    alg = alg + "_" + str(args.learning_rate) + "_" + str(args.beta) + "_" + str(args.lamda) + "_" + str(args.numusers) + "u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    if(args.algorithm == "pFedMe" or args.algorithm == "pFedMe_p"):
        alg = alg + "_" + str(args.K) + "_" + str(args.personal_learning_rate)
    alg = alg + "_" + str(run_idx)
    return "./results/"+'{}.h5'.format(alg, args.local_epochs)    

args = load_hypermater()

dataset_list = ["Mnist", "FMnist", "Cifar10"]
datasize_list = ["small", "large"]
algorithm_list = ["pFedBayes"]

args.batch_size = 100

for args.dataset in dataset_list:
    print("%13s"%args.dataset, end="")
    print("       ", end="")
print()

for args.dataset in dataset_list:
    for args.datasize in datasize_list:
        print("%8s"%args.datasize, end="")
        print("  ", end="")
print()


for args.algorithm in algorithm_list:
    for args.dataset in dataset_list:
        for args.datasize in datasize_list:
            file_path = get_file_path(args, loadP=True)
            with h5py.File(file_path , 'r') as f: 
                print(" ", "%.2f"%(max(f['rs_glob_acc'][:] )*100), " &") 