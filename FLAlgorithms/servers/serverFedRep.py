import torch
import os

from FLAlgorithms.users.userFedRep import UserFedRep
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
 
class FedRep(Server):
    def __init__(self, device,  dataset, datasize, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times, only_one_local = False):
        super().__init__(device, dataset, datasize, algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times, only_one_local)

        # Initialize data for all  users
        self.mark_personalized_module = model[0].get_mark_personlized_module(-1)
        print("mark_personalized_module", self.mark_personalized_module)
        data = read_data(dataset, datasize)

        if (only_one_local):
            self.total_users = 1
        else:
            self.total_users = len(data[0])

        self.K = K
        self.personal_learning_rate = personal_learning_rate
        for i in range(self.total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserFedRep(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate)
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
        if (AddNewClient):
            self.users_copy = self.users
            self.users = self.users[1:]
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, self.num_glob_iters, " -------------")
            # send all parameter for users 
            self.send_parameters(personalized = self.mark_personalized_module)

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            self.selected_users = self.select_users(glob_iter,self.num_users)

            self.evaluate_personalized_model(hasPMB=False)
            self.aggregate_parameters()

        if (AddNewClient):
            loss = []
            self.users = self.users_copy[0:1]
            self.mark_personalized_module[-1] = self.mark_personalized_module[-2] = 1
            for glob_iter in range(AddNewClient):
                print("-------------Add New Client Round number: ",glob_iter, AddNewClient, " -------------")
                # send all parameter for users 
                self.send_parameters(personalized = self.mark_personalized_module)

                # Evaluate gloal model on user for each interation
                print("Evaluate global model")
                print("")
                self.evaluate()

                # do update for all users not only selected users
                for user in self.users:
                    user.train(self.local_epochs, only_train_personal=True) #* user.train_samples
                
                self.selected_users = self.select_users(glob_iter,self.num_users)

                self.evaluate_personalized_model(hasPMB=False)
                self.aggregate_parameters()


        self.save_results()
        self.save_model()
    
  
