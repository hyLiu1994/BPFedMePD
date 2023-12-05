import torch
import os

from FLAlgorithms.users.userFedBABU import UserFedBABU
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
 
class FedBABU(Server):
    def __init__(self, model, times, args):
        super().__init__(model[0], times, args)

        # Initialize data for all  users
        self.mark_personalized_module = model[0].get_mark_personlized_module(-1)
        print("mark_personalized_module", self.mark_personalized_module)
        data = read_data(args.dataset, args.datasize)
        self.fineturn_iters = args.num_fineturn_iters

        if (args.only_one_local):
            self.total_users = 1
        else:
            self.total_users = len(data[0])

        self.K = args.K
        self.personal_learning_rate = args.personal_learning_rate
        for i in range(self.total_users):
            id, train , test = read_user_data(i, data, args.dataset)
            user = UserFedBABU(id, train, test, model, args)
            self.users.append(user)
            self.total_train_samples += user.train_samples

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
        for glob_iter in range(self.num_glob_iters):
            if (glob_iter > self.num_glob_iters - self.fineturn_iters):
                print("-------------Fine Turn Round number: ",glob_iter, self.num_glob_iters, " -------------")
            else:
                print("-------------Round number: ",glob_iter, self.num_glob_iters, " -------------")
            # send all parameter for users 
            self.send_parameters(personalized = self.mark_personalized_module)

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                if (glob_iter > self.num_glob_iters - self.fineturn_iters):
                    user.train(self.local_epochs, only_train_personal=True) #* user.train_samples
                else:
                    user.train(self.local_epochs) #* user.train_samples
            
            self.selected_users = self.select_users(glob_iter,self.num_users)

            self.evaluate_personalized_model(hasPMB=False)
            self.aggregate_parameters()

        self.save_results()
        self.save_model()
    
  
