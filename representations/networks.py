
# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
  
  ----------------------------------------- 
  Construction of feature representations
  ----------------------------------------- 
  
  + build_network:
    --------------
            |
            +--------> feedforward_network:
            |
            +--------> recurrent_network:
            |
            +--------> MNIST_network: 

"""

# TODO: add arguments details 


from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import torch
from torch.autograd import Variable 
import torch.nn.functional as nnf
from torch.utils.data import random_split
from torch.optim import SGD 
from torch.distributions import constraints
import torchvision as torchv
import torchvision.transforms as torchvt
from torch import nn
import torchvision.transforms as transforms
from torch.autograd import grad
import scipy.stats as st

from copy import deepcopy
import time

torch.manual_seed(1) 


ACTIVATION_DICT = {"ReLU": torch.nn.ReLU(), "Hardtanh": torch.nn.Hardtanh(),
                   "ReLU6": torch.nn.ReLU6(), "Sigmoid": torch.nn.Sigmoid(),
                   "Tanh": torch.nn.Tanh(), "ELU": torch.nn.ELU(),
                   "CELU": torch.nn.CELU(), "SELU": torch.nn.SELU(), 
                   "GLU": torch.nn.GLU(), "LeakyReLU": torch.nn.LeakyReLU(),
                   "LogSigmoid": torch.nn.LogSigmoid(), "Softplus": torch.nn.Softplus()}


def build_network(network_name, params):

    if network_name=="feedforward":
        
        net = feedforward_network(params)

    return net


def feedforward_network(params):

    modules          = []

    if params["dropout_active"]: 

        modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

    modules.append(torch.nn.Linear(params["input_dim"], params["num_hidden"]))
    modules.append(ACTIVATION_DICT[params["activation"]])

    for u in range(params["num_layers"] - 1):

        if params["dropout_active"]:

            modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

        modules.append(torch.nn.Linear(params["num_hidden"], params["num_hidden"]))
        modules.append(ACTIVATION_DICT[params["activation"]])

    modules.append(torch.nn.Linear(params["num_hidden"], params["rep_dim"]))

    _architecture    = nn.Sequential(*modules)

    return _architecture
