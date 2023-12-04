import torch
import os

from FLAlgorithms.users.userFedSI import UserFedSI
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

from torch.nn.utils import parameters_to_vector

# Implementation for TCYB Server

class FedSI(Server):
    def __init__(self, device, dataset, datasize, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times, args, only_one_local = False):
        super().__init__(device, dataset, datasize, algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times, only_one_local)

        # Initialize data for all  users
        self.mark_personalized_module = model[0].get_mark_personlized_module(-1)
        print("mark_personalized_module", self.mark_personalized_module)

        data = read_data(dataset, datasize)
        self.device = device
        self.subnetwork_rate = args.subnetwork_rate
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserFedSI(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, args)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        # MNIST 
        # self.lamda = 1e-4
        # Cifar10
        self.lamda = 1e-4
        self.n_params = len(parameters_to_vector(self.model.parameters()).detach())
        self.global_sigma = torch.zeros(parameters_to_vector(self.model.parameters()).shape).to(device)
        self.global_sigma += 1 / self.lamda
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedSI server.")

    def send_grads(self, AddNewClient = False):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def send_parameters(self, personalized = []):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model, personalized)
            user.sigma = self.global_sigma

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        self.global_sigma = torch.zeros(parameters_to_vector(self.model.parameters()).shape).to(self.device)
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            self.global_sigma += user.sigma * user.train_samples / total_train
        print("before_global_sigma", self.global_sigma.sum()) 
        # MNIST
        zero_elements = (torch.abs(self.global_sigma) <= 1e-10)
        self.global_sigma[zero_elements] = 1 / self.lamda
        inf_elements = (torch.abs(self.global_sigma) <= 1e-3) * (torch.abs(self.global_sigma) > 1e-10)
        self.global_sigma[inf_elements] = 1e-3
        # Cifar10
        # zero_elements = (torch.abs(self.global_sigma) <= 1e-10)
        # self.global_sigma[zero_elements] = 1 / self.lamda
        # inf_elements = (torch.abs(self.global_sigma) <= 1e-1) * (torch.abs(self.global_sigma) > 1e-10)
        # self.global_sigma[inf_elements] = 1e-1

        print("after_global_sigma", self.global_sigma.mean())

    def train(self, add_new_client = False):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters(personalized = self.mark_personalized_module)
            # Evaluate model each interation
            self.evaluate()
            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs, self.mark_personalized_module) #* user.train_samples
            self.evaluate_personalized_model(hasPMB=False)
            self.aggregate_parameters()

        self.save_results()
        self.save_model()