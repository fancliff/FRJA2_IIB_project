import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

import torch
# torch.set_num_threads(4)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit
import csv
import os
import time
from typing import List
import h5py
import datetime
import generators as gen
import models as md
import routines as rt 

out_channels_list = [
    [4,6,4],
    [4,8,4],
    [4,8,8,4],
    [4,6,8,6,4],
    [4,8,12,8,4],
    [4,4,8,8,4,4,4], # BSL
    [4,6,8,12,8,6,4],
    [4,6,6,8,8,6,6,4],
    [4,4,8,8,12,12,8,8,4],
    [4,4,8,8,12,16,12,8,8,4],
    [4,4,6,6,6,8,8,6,6,6,4,4],
]

kernel_size_list = [
    [5],
    [7],
    [9], # BSL
    [11],
    [13],
]


params_sum = 0
max_params = 0
for oc in out_channels_list:
    for ks in kernel_size_list:
        model = md.RegressionModel1(
            data_channels=2,
            out_channels=oc,
            kernel_size=ks,
            batch_norm=True,
            P_dropout=0.0,
            max_pool=False,
        )
        params = rt.count_parameters(model)
        if params > max_params:
            max_params = params
        params_sum += params

print(params_sum)
print(max_params)