import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from FLAlgorithms.trainmodel import bayes 
from collections import OrderedDict
from FLAlgorithms.trainmodel.layers.misc import FlattenLayer, ModuleWrapper

class DNN(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.linear = nn.Linear(mid_dim, output_dim)
        
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def get_mark_personlized_module(self, mark_personal_layer = 0):
        if (mark_personal_layer == 0):
            return []
        if (abs(mark_personal_layer) >= 2):
            return [1, 1, 1, 1]
        if (mark_personal_layer == -1):
            return [0, 0, 1, 1]
        if (mark_personal_layer == 1):
            return [1, 1, 0, 0]

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.linear = nn.Linear(84, 10)
        self.mark_list = [2, 2, 2, 2, 2]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

    def get_mark_personlized_module(self, mark_personal_layer = 0):
        if (mark_personal_layer == 0):
            return []
        module_mark_list = []
        for idx, param_num in enumerate(self.mark_list):
            padding = 0
            if (idx < mark_personal_layer or idx >= len(self.mark_list) + mark_personal_layer):
                padding = 1
            for i in range(param_num):
                module_mark_list.append(padding)
        return module_mark_list

class pBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device=torch.device('cpu'),
                 weight_scale=0.1, rho_offset=-3, zeta=10):
        super(pBNN, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = 1
        self.mean_prior = 10
        self.sigma_prior = 5
        self.layer_param_shapes = self.get_layer_param_shapes()
        self.mus = nn.ParameterList()
        self.rhos = nn.ParameterList()
        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        self.zeta = torch.tensor(zeta, device=self.device)
        self.sigmas = torch.tensor([1.] * len(self.layer_param_shapes), device=self.device)
        self.mark_list = [2, 2]

        for shape in self.layer_param_shapes:
            mu = nn.Parameter(torch.normal(mean=torch.zeros(shape), std=self.weight_scale * torch.ones(shape)))
            rho = nn.Parameter(self.rho_offset + torch.zeros(shape))
            self.mus.append(mu)
            self.rhos.append(rho)

    def get_layer_param_shapes(self):
        layer_param_shapes = []
        for i in range(self.num_layers + 1):
            if i == 0:
                W_shape = (self.input_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            elif i == self.num_layers:
                W_shape = (self.hidden_dim, self.output_dim)
                b_shape = (self.output_dim,)
            else:
                W_shape = (self.hidden_dim, self.hidden_dim)
                b_shape = (self.hidden_dim,)
            layer_param_shapes.extend([W_shape, b_shape])
        return layer_param_shapes

    def transform_rhos(self, rhos):
        return [F.softplus(rho) for rho in rhos]

    def transform_gaussian_samples(self, mus, rhos, epsilons):
        # compute softplus for variance
        self.sigmas = self.transform_rhos(rhos)
        samples = []
        for j in range(len(mus)): samples.append(mus[j] + self.sigmas[j] * epsilons[j])
        return samples

    def sample_epsilons(self, param_shapes):
        epsilons = [torch.normal(mean=torch.zeros(shape), std=0.001 * torch.ones(shape)).to(self.device) for shape in
                    param_shapes]
        return epsilons

    def net(self, X, layer_params):
        layer_input = X
        for i in range(len(layer_params) // 2 - 1):
            h_linear = torch.mm(layer_input, layer_params[2 * i]) + layer_params[2 * i + 1]
            layer_input = F.relu(h_linear)
        output = torch.mm(layer_input, layer_params[-2]) + layer_params[-1]
        return output

    def log_softmax_likelihood(self, yhat_linear, y):
        return torch.nansum(y * F.log_softmax(yhat_linear), dim=0)

    def combined_loss_personal(self, output, label_one_hot, params, mus, sigmas, mus_local, sigmas_local, num_batches):
        # Calculate data likelihood
        log_likelihood_sum = torch.sum(self.log_softmax_likelihood(output, label_one_hot))
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i], sigmas[i]),
                            Normal(mus_local[i].detach(), sigmas_local[i].detach())))  for i in range(len(params))])
        return 1.0 / num_batches * (self.zeta * KL_q_w) - log_likelihood_sum

    def combined_loss_local(self, params, mus, sigmas, mus_local, sigmas_local, num_batches):
        KL_q_w = sum([torch.sum(kl_divergence(Normal(mus[i].detach(), sigmas[i].detach()), 
                                              Normal(mus_local[i], sigmas_local[i]))) for i in range(len(params))])
        return 1.0 / num_batches * (self.zeta * KL_q_w)

    def get_mark_personlized_module(self, mark_personal_layer = 0):
        if (mark_personal_layer == 0):
            return []
        module_mark_list = []
        for i in range(2):
            for idx, param_num in enumerate(self.mark_list):
                padding = 0
                if (idx < mark_personal_layer or idx >= len(self.mark_list) + mark_personal_layer):
                    padding = 1
                for k in range(param_num):
                    module_mark_list.append(padding)
        return module_mark_list

class pBNN_v2(ModuleWrapper):
    def __init__(self, n_classes=10, k=1., **kwargs):
        super(pBNN_v2, self).__init__(**kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.features = nn.Sequential(OrderedDict([
            ('flatten', Flatten()), 
        ])) 
        self.classifier = nn.Sequential(OrderedDict([ 
            ('fc1', bayes.FFGLinear(784, 100)), 
            ('relu', nn.ReLU()), 
            ('linear', bayes.FFGLinear(100, n_classes)), 
            ('logsoftmax', nn.LogSoftmax(dim=1)), 
        ])) 
        self.mark_list = [4, 4]
        if self.device:
            self.to(self.device)
        
    def forward(self, input):
        return self.classifier(self.features(input))

    def get_dist(self):
        modules = [self.classifier.fc1.dist(), self.classifier.linear.dist()]
        return modules
    
    def get_parameter(self):
        modules = [self.classifier.fc1.q_params(), self.classifier.linear.q_params()]
        return modules
    
    def get_mark_personlized_module(self, mark_personal_layer = 0):
        if (mark_personal_layer == 0):
            return []
        module_mark_list = []
        for idx, param_num in enumerate(self.mark_list):
            padding = 0
            if (idx < mark_personal_layer or idx >= len(self.mark_list) + mark_personal_layer):
                padding = 1
            for i in range(param_num):
                module_mark_list.append(padding)
        return module_mark_list

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class pCIFARNet(ModuleWrapper):
    def __init__(self, n_classes=10, k=1., **kwargs):
        super(pCIFARNet, self).__init__(**kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.features = nn.Sequential(OrderedDict([
            ('conv1', bayes.BayesConv2d(3, 6, 5)), 
            ('relu1', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(2, 2)), 
            ('conv2', bayes.BayesConv2d(6, 16, 5)), 
            ('relu2', nn.ReLU()), 
            ('flatten', Flatten()), 
        ])) 
        self.classifier = nn.Sequential(OrderedDict([ 
            ('fc1', bayes.FFGLinear(4 * 16 * 25, 120)), 
            ('relu3', nn.ReLU()), 
            ('fc2', bayes.FFGLinear(120, 84)), 
            ('relu4', nn.ReLU()), 
            ('linear', bayes.FFGLinear(84, n_classes)), 
            ('logsoftmax', nn.LogSoftmax(dim=1)), 
        ])) 
        self.mark_list = [4, 4, 4, 4, 4]
        if self.device:
            self.to(self.device)
        
    def forward(self, input):
        return self.classifier(self.features(input))

    def get_dist(self):
        modules = [self.features.conv1.dist(), self.features.conv2.dist(), self.classifier.fc1.dist(), self.classifier.fc2.dist(), self.classifier.linear.dist()]
        return modules
    
    def get_parameter(self):
        modules = [self.features.conv1.q_params(), self.features.conv2.q_params(), self.classifier.fc1.q_params(), self.classifier.fc2.q_params(), self.classifier.linear.q_params()]
        return modules
    
    def get_mark_personlized_module(self, mark_personal_layer = 0):
        if (mark_personal_layer == 0):
            return []
        module_mark_list = []
        for idx, param_num in enumerate(self.mark_list):
            padding = 0
            if (idx < mark_personal_layer or idx >= len(self.mark_list) + mark_personal_layer):
                padding = 1
            for i in range(param_num):
                module_mark_list.append(padding)
        return module_mark_list
    
class DNNSoul(ModuleWrapper):
    def __init__(self, n_classes=10, k=1., **kwargs):
        super(DNNSoul, self).__init__(**kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.features = nn.Sequential(OrderedDict([
            ('flatten', Flatten()), 
        ])) 
        self.classifier = nn.Sequential(OrderedDict([ 
            ('fc1', nn.Linear(784, 100)), 
            ('relu', nn.ReLU()), 
            ('linear', bayes.FFGLinear(100, n_classes)), 
            ('logsoftmax', nn.LogSoftmax(dim=1)), 
        ])) 
        self.mark_list = [2, 4]
        if self.device:
            self.to(self.device)

    def forward(self, input):
        return self.classifier(self.features(input))

    def get_dist(self):
        modules = [self.classifier.linear.dist()]
        return modules
    
    def get_parameter(self):
        modules = [self.classifier.linear.q_params()]
        return modules
    
    def get_mark_personlized_module(self, mark_personal_layer = 0):
        if (mark_personal_layer == 0):
            return []
        module_mark_list = []
        for idx, param_num in enumerate(self.mark_list):
            padding = 0
            if (idx < mark_personal_layer or idx >= len(self.mark_list) + mark_personal_layer):
                padding = 1
            for i in range(param_num):
                module_mark_list.append(padding)
        return module_mark_list

class CIFARNetSoul(ModuleWrapper):
    def __init__(self, n_classes=10, k=1., **kwargs):
        super(CIFARNetSoul, self).__init__(**kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 6, 5)), 
            ('relu1', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(2, 2)), 
            ('conv2', nn.Conv2d(6, 16, 5)), 
            ('relu2', nn.ReLU()), 
            ('flatten', Flatten()), 
        ])) 
        self.classifier = nn.Sequential(OrderedDict([ 
            ('fc1', nn.Linear(4 * 16 * 25, 120)), 
            ('relu3', nn.ReLU()), 
            ('fc2', nn.Linear(120, 84)), 
            ('relu4', nn.ReLU()), 
            ('linear', bayes.FFGLinear(84, n_classes)), 
            ('logsoftmax', nn.LogSoftmax(dim=1)), 
        ])) 
        self.mark_list = [2, 2, 2, 2, 4]
        if self.device:
            self.to(self.device)
        
    def forward(self, input):
        return self.classifier(self.features(input))

    def get_dist(self):
        modules = [self.classifier.linear.dist()]
        return modules
    
    def get_parameter(self):
        modules = [self.classifier.linear.q_params()]
        return modules
    
    def get_mark_personlized_module(self, mark_personal_layer = 0):
        if (mark_personal_layer == 0):
            return []
        module_mark_list = []
        for idx, param_num in enumerate(self.mark_list):
            padding = 0
            if (idx < mark_personal_layer or idx >= len(self.mark_list) + mark_personal_layer):
                padding = 1
            for i in range(param_num):
                module_mark_list.append(padding)
        return module_mark_list