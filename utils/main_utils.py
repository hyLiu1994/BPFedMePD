import argparse
from FLAlgorithms.trainmodel.OModels import *
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.servers.serverpFedbayes import pFedBayes
from FLAlgorithms.servers.serverBPFedPD import BPFedPD

def load_hypermater():
    # Setting hyperpameter 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "FMnist", "Cifar10"])
    parser.add_argument("--datasize", type=str, default="small", choices=["small", "large"])
    parser.add_argument("--model_name", type=str, default="pcnn", choices=["dnn", "cnn", 'pbnn', 'pcnn'])
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_glob_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="BPFedPD",choices=["pFedMe", "PerAvg", "FedAvg", "pFedBayes", "BPFedPD"]) 
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")

    parser.add_argument("--weight_scale", type=float, default=0.1)
    parser.add_argument("--rho_offset", type=int, default=-3)
    parser.add_argument("--zeta", type=int, default=10)
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm                    : {}".format(args.algorithm))
    print("Batch size                   : {}".format(args.batch_size))
    print("Learing rate                 : {}".format(args.learning_rate))
    print("Average Moving               : {}".format(args.beta))
    print("Subset of users              : {}".format(args.numusers))
    print("Number of global rounds      : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset                      : {}".format(args.dataset))
    print("Datatype                     : {}".format(args.datasize))
    print("Local Model                  : {}".format(args.model_name))
    print("=" * 80)

    return args

def model_select(model, args):
    if(model == "dnn"):
        model = DNN(784, 100, 10).to(args.device), model

    if model == "pbnn":
        model = pBNN(784, 100, 10, args.device, args.weight_scale, args.rho_offset, args.zeta).to(args.device), model

    if(model == "cnn"):
        model = CifarNet().to(args.device), model 

    if model == "pcnn":
        model = pCIFARNet(10).to(args.device), model

    return model

def server_select(algorithm, model, exp_idx, args):
    # select algorithm
    if(algorithm == "FedAvg"):
        server = FedAvg(args.device, args.dataset, args.datasize, algorithm, model, args.batch_size, args.learning_rate, args.beta, args.lamda, args.num_glob_iters, args.local_epochs, args.optimizer, args.numusers, exp_idx)
    
    if(algorithm == "pFedMe"):
        server = pFedMe(args.device, args.dataset, args.datasize, algorithm, model, args.batch_size, args.learning_rate, args.beta, args.lamda, args.num_glob_iters, args.local_epochs, args.optimizer, args.numusers, args.K, args.personal_learning_rate, exp_idx)

    if(algorithm == "PerAvg"):
        server = PerAvg(args.device, args.dataset, args.datasize, algorithm, model, args.batch_size, args.learning_rate, args.beta, args.lamda, args.num_glob_iters, args.local_epochs, args.optimizer, args.numusers, exp_idx)

    if (algorithm == "pFedBayes"):
        server = pFedBayes(args.dataset, args.datasize, algorithm, model, args.batch_size, args.learning_rate, args.beta, args.lamda, args.num_glob_iters, args.local_epochs, args.optimizer, args.numusers, exp_idx, args.device, args.personal_learning_rate)
    
    if (algorithm == "BPFedPD"):
        server = BPFedPD(args.dataset, args.datasize, algorithm, model, args.batch_size, args.learning_rate, args.beta, args.lamda, args.num_glob_iters, args.local_epochs, args.optimizer, args.numusers, exp_idx, args.device, args.personal_learning_rate)

    return server 
