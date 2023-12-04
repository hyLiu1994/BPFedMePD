import torch
import os

from FLAlgorithms.users.userperavg import UserPerAvg
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data

# Implementation for per-FedAvg Server

class PerAvg(Server):
    def __init__(self, model, times, args):
        super().__init__(model[0], times, args)

        # Initialize data for all  users
        self.mark_personalized_module = model[0].get_mark_personlized_module(-1)
        data = read_data(args.dataset, args.datasize)
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, args.dataset)
            user = UserPerAvg(id, train, test, model, args, total_users, args.numusers)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:", args.numusers, " / " ,total_users)
        print("Finished creating Local Per-Avg.")

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
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model with one step update")
            print("")
            self.evaluate_one_step()

            # choose several users to send back upated model to server
            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs) # * user.train_samples
                # user.train_one_step(self.local_epochs) # * user.train_samples
                
            self.aggregate_parameters()

        self.save_results()
        self.save_model()
