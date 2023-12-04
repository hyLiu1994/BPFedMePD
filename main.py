#!/usr/bin/env python
from utils.main_utils import server_select, model_select, load_hypermater
from FLAlgorithms.trainmodel.OModels import *
from FLAlgorithms.trainmodel.FedSIModel import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)

def main(args):
    # Get device status: Check GPU or CPU
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")

    for i in range(args.times):
        print("---------------Running time:", i, "------------")
        # Generate model
        model = model_select(args)
        server = server_select(model, i, args) 

        server.train(args.add_new_client)
        if (isinstance(model[0], pBNN)):
            server.testpFedbayes()
        else:
            server.test()

    # Average data 
    # if(args.algorithm == "PerAvg"):
    #     args.algorithm == "PerAvg_p"

    # if(args.algorithm == "pFedMe"):
    #     average_data(num_users=args.numusers, loc_ep1=args.local_epochs, Numb_Glob_Iters=args.num_glob_iters, lamb=args.lamda,learning_rate=args.learning_rate, beta = args.beta, algorithms="pFedMe_p", batch_size=args.batch_size, dataset=args.dataset, datasize=args.datasize, k = args.K, personal_learning_rate = args.personal_learning_rate,times = args.times)

    # average_data(num_users=args.numusers, loc_ep1=args.local_epochs, Numb_Glob_Iters=args.num_glob_iters, lamb=args.lamda,learning_rate=args.learning_rate, beta = args.beta, algorithms=args.algorithm, batch_size=args.batch_size, dataset=args.dataset, datasize=args.datasize, k = args.K, personal_learning_rate = args.personal_learning_rate,times = args.times)

if __name__ == "__main__":
    args = load_hypermater()
    main(args)
