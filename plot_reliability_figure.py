import torch
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import h5py
import os
import glob
import copy
import logging
import numpy as np
from utils.main_utils import load_hypermater
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from reliability_diagrams.reliability_diagrams import reliability_diagram

def get_file_path(args, loadP = False, run_idx = 0):
    alg = args.dataset + "_" + args.datasize + "_" + args.algorithm
    if (loadP):
        alg += "_p"
    alg = alg + "_" + str(args.learning_rate) + "_" + str(args.beta) + "_" + str(args.lamda) + "_" + str(args.numusers) + "u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    if(args.algorithm == "pFedMe" or args.algorithm == "pFedMe_p"):
        alg = alg + "_" + str(args.K) + "_" + str(args.personal_learning_rate)
    alg = alg + "_" + str(run_idx)
    return "./results/"+'{}.h5'.format(alg, args.local_epochs)    

def to_one_hot(y, dtype=torch.double):
    # convert a single label into a one-hot vector
    y_output_onehot = torch.zeros((y.shape[0], y.max().type(torch.IntTensor) + 1), dtype=dtype, device=y.device)
    return y_output_onehot.scatter_(1, y.unsqueeze(1), 1)

def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def plot_calibration_error(probs, targets, path, color='darkblue'):

    targets = torch.tensor(targets)
    probs = torch.tensor(probs)
    if (probs.min() < 0 or probs.max() > 1):
        probs = F.softmax(probs, dim=-1)
    print("probs.min()", probs.min())
    # print(targets)
    # print(probs)
    confidences = probs.max(-1).values.detach().numpy()
    accuracies = probs.argmax(-1).eq(targets).numpy()

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    max_err = 0.0

    targets = targets.long()
    y_one_hot = to_one_hot(targets)
    bri = (torch.norm(probs - y_one_hot, dim=1) ** 2).mean()

    plot_acc = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # print(bin_lower, bin_upper)
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(np.float32).mean()
        # print("prop_in_bin:", prop_in_bin, in_bin.sum())

        if prop_in_bin > 0.0:
            accuracy_in_bin = accuracies[in_bin].astype(np.float32).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if np.abs(avg_confidence_in_bin - accuracy_in_bin) > max_err:
                max_err = np.abs(avg_confidence_in_bin - accuracy_in_bin)

            plot_acc.append(accuracy_in_bin)
        else:
            plot_acc.append(0.0)

    plt.figure(figsize=(4, 4))
    plt.rcParams.update({'font.size': 18})

    plt.xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=15)
    plt.yticks(ticks=[0.2, 0.4, 0.6, 0.8, 1.0], size=15)

    props = dict(boxstyle='round', facecolor='white', alpha=1.)

    plt.bar(
        bin_lowers, plot_acc, bin_uppers[0], align="edge", linewidth=1, edgecolor='k', color=color
    )
    plt.plot([0.0, 1.0], [0.0, 1.0], c="orange", lw=2)
    plt.text(
        0.05,
        0.73,
        "ECE:  {:0.3f}\nMCE: {:0.3f}\nBRI:  {:0.3f}".format(
            ece, max_err, detach_to_numpy(bri).astype(np.float32).item()
        ),
        fontsize=16,
        bbox=props
    )

    plt.xlim((0, 1.))
    plt.ylim((0, 1.))
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")

    logging.info(path)
    logging.info(bin_uppers)
    logging.info(bin_lowers)
    logging.info(plot_acc)

    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()

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

args = load_hypermater()

# dataset_list = ["Mnist", "FMnist", "Cifar10"]
# datasize_list = ["small", "large"]

dataset_list = ["Cifar10"]
datasize_list = ["small"]

# algorithm_list = ["BPFedPD", "pFedBayes"]
algorithm_list = ["FedAvg", "PerAvg", "pFedMe", "FedPer", "LGFedAvg", "FedRep", "FedSOUL", "pFedBayes", "BPFedPD"]
# algorithm_list = ["BPFedPD"]

for args.algorithm in algorithm_list:
    for args.dataset in dataset_list:
        for args.datasize in datasize_list:
            args_now = change_avg(args)
            file_path = get_file_path(args_now, loadP=True)
            label_list = np.load(file_path[:-3] + "_y.npy")
            output_list = np.load(file_path[:-3] + "_output.npy")
            plot_calibration_error(output_list, label_list, file_path[:-2] + "png")
            # compute_calibration(true_labels, pred_labels, confidences, num_bins=10)
            if (output_list.min() < 0 or output_list.max() > 1):
                output_list = F.softmax(torch.tensor(output_list), dim=-1).numpy()
            accuracies = torch.tensor(output_list).argmax(-1).numpy()
            output_list = torch.tensor(output_list).max(-1)[0].numpy()
            # reliability_diagram(label_list, accuracies, output_list , file_path[:-2] + "png")
            reliability_diagram(label_list, accuracies, output_list , file_path[:-2] + "pdf")
