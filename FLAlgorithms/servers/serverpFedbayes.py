import torch
from tqdm import tqdm

from FLAlgorithms.users.userpFedbayes import UserpFedBayes
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.trainmodel.OModels import *
from utils.model_utils import read_data, read_user_data
import numpy as np


# Implementation for FedAvg Server
class pFedBayes(Server):
    def __init__(self, model, times, args, output_dim=10):
        super().__init__(model[0], times, args)

        # Initialize data for all  users
        self.mark_personalized_module = model[0].get_mark_personlized_module(0)
        data = read_data(args.dataset, args.datasize)
        self.personal_learning_rate = args.personal_learning_rate
        total_users = len(data[0])
        print('clients initializting...')
        for i in tqdm(range(total_users), total=total_users):
            id, train, test = read_user_data(i, data, args.dataset)
            user = UserpFedBayes(id, train, test, model, args, output_dim=output_dim)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:", args.numusers, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self, AddNewClient = False):
        loss = []
        acc = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            self.send_parameters()
            # Evaluate model each interation
            if (isinstance(self.users[0].model, pBNN)):
                self.evaluate_bayes(False)
            else:
                self.evaluate_bayes(True)

            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs)
            self.aggregate_parameters()

        self.save_results()
        return self.save_model()

    def evaluate_bayes(self, newVersion = False):
        if (newVersion): 
            stats = self.testBayesV2() 
            stats_train = self.train_error_and_loss_cifar10() 
        else:
            stats = self.testpFedbayes() 
            stats_train = self.train_error_and_loss_pFedbayes() 

        per_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        glob_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]) / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(per_acc)
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        if (self.max_acc < per_acc):
            self.max_acc = per_acc
            self.output_list = stats[-2]
            self.y_list = stats[-1]

        print("Average personal Accurancy: ", per_acc)
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ", train_loss)