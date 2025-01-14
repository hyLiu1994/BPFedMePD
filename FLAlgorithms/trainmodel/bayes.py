from torch import distributions as dist
import torch
from torch import nn
import torch.nn.functional as F
import utils
import yaml
from torch.nn import Parameter

from .layers.misc import ModuleWrapper

global_eps = 1
class BayesConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size, device,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):

        super(BayesConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = device

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))
        self.W_rho = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 0.001).to(self.device)
            # self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            self.W_sigma = F.softplus(self.W_rho)
            weight = self.W_mu + W_eps * self.W_sigma * global_eps

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 0.001).to(self.device)
                # self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                self.bias_sigma = F.softplus(self.bias_rho)
                bias = self.bias_mu + bias_eps * self.bias_sigma * global_eps
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def dist(self):
        # self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        self.W_sigma = F.softplus(self.W_rho)
        return dist.Normal(self.W_mu, self.W_sigma * global_eps)

    def q_params(self):
        return self.W_mu, self.W_rho

class FFGLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, device, bias=True, priors=None):
        super(FFGLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = device

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_rho = Parameter(torch.empty((out_features, in_features), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_features), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_features), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 0.001).to(self.device)
            # self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            self.W_sigma = F.softplus(self.W_rho)
            weight = self.W_mu + W_eps * self.W_sigma * global_eps 

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 0.001).to(self.device)
                # self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                self.bias_sigma = F.softplus(self.bias_rho)
                bias = self.bias_mu + bias_eps * self.bias_sigma * global_eps
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(input, weight, bias)
    
    def dist(self):
        # self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        self.W_sigma = F.softplus(self.W_rho)
        # self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        self.bias_sigma = F.softplus(self.bias_rho)         
        if (self.use_bias):
            return [dist.Normal(self.W_mu, self.W_sigma * global_eps), dist.Normal(self.bias_mu, self.bias_sigma * global_eps)]
        else:
            return [dist.Normal(self.W_mu, self.W_sigma * global_eps)]

    def q_params(self):
        if (self.use_bias):
            return [self.W_mu, self.W_rho, self.bias_mu, self.bias_rho]
        else:
            return [self.W_mu, self.W_rho]


