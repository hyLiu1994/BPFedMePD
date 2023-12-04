import torch
import os

from FLAlgorithms.users.userLocalOnly import userLocalOnly
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class LocalOnly(Server):
    def __init__(self, model, exp_idx, args):
        super().__init__(args.device, args.dataset, args.datasize, args.algorithm, model[0], args.batch_size, args.learning_rate, args.beta, args.lamda, args.num_glob_iters,
                         args.local_epochs, args.optimizer, args.numusers, exp_idx, args.only_one_local)

        # Initialize data for all  users
        self.mark_personalized_module = model[0].get_mark_personlized_module(0)

        data = read_data(args.dataset, args.datasize)
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, args.dataset)
            user = userLocalOnly(id, train, test, model, args)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:", args.numusers, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def train(self, add_new_client = False):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")

            # Evaluate model each interation
            self.evaluate()
            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples 
            self.evaluate_personalized_model(hasPMB=False)

        self.save_results()
        self.save_model()