import pytest
from itertools import product

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import wide_resnet50_2

from laplace import Laplace, SubnetLaplace, FullSubnetLaplace, DiagSubnetLaplace
from laplace.baselaplace import DiagLaplace
from laplace.utils import (SubnetMask, RandomSubnetMask, LargestMagnitudeSubnetMask,
                           LargestVarianceDiagLaplaceSubnetMask, LargestVarianceSWAGSubnetMask,
                           ParamNameSubnetMask, ModuleNameSubnetMask, LastLayerSubnetMask)

import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
from torch.utils.data import DataLoader

subset_indices = list(range(128))
# train_dataloader = DataLoader(Subset(training_data, subset_indices), batch_size=64, shuffle=True)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits 

from laplace import Laplace
# Pre-trained model
model = NeuralNetwork().to(device)

# Examples of different ways to specify the subnetwork
# via indices of the vectorized model parameters
#
# Example 1: select the 128 parameters with the largest magnitude
# from laplace.utils import LargestMagnitudeSubnetMask
# subnetwork_mask = LargestMagnitudeSubnetMask(model, n_params_subnet=128)
# subnetwork_indices = subnetwork_mask.select()

# # Example 2: specify the layers that define the subnetwork
# from laplace.utils import ModuleNameSubnetMask
# subnetwork_mask = ModuleNameSubnetMask(model, module_names=['layer.1', 'layer.3'])
# subnetwork_mask.select()
# subnetwork_indices = subnetwork_mask.indices

# Example 3: manually define the subnetwork via custom subnetwork indices
print("Begin Select!")
import torch
from laplace.utils import LargestVarianceDiagLaplaceSubnetMask
diag_la = Laplace(model, 'classification', 
                  subset_of_weights='all', 
                  hessian_structure='diag')
subnetwork_mask = LargestVarianceDiagLaplaceSubnetMask(model, n_params_subnet=128, diag_laplace_model = diag_la)
subnetwork_indices = subnetwork_mask.select(train_dataloader)
print(subnetwork_indices)
print("subnetwork_indices!")

# Define and fit subnetwork LA using the specified subnetwork indices
la = Laplace(model, 'classification',
             subset_of_weights='subnetwork',
             hessian_structure='full',
             subnetwork_indices=subnetwork_indices)

print("Begin!")
print(la.fit(train_dataloader))
