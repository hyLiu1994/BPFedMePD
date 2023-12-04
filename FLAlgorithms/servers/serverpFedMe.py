import torch
import os

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
 
# Implementation for pFedMe Server

class pFedMe(Server):
    def __init__(self, model, times, args):
        super().__init__(model[0], times, args)

        # Initialize data for all  users
        self.mark_personalized_module = model[0].get_mark_personlized_module(0)
        data = read_data(args.dataset, args.datasize)
        total_users = len(data[0])
        self.K = args.K
        self.personal_learning_rate = args.personal_learning_rate
        for i in range(total_users):
            id, train , test = read_user_data(i, data, args.dataset)
            user = UserpFedMe(id, train, test, model, args)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        # print("Number of users / total users:",num_users, " / " ,total_users)
        # print("Finished creating pFedMe server.")

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
            print("-------------Round number: ",glob_iter, self.num_glob_iters, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            #self.aggregate_parameters()
            self.persionalized_aggregate_parameters()


        #print(loss)
        self.save_results()
        self.save_model()
    
  
