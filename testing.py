import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit

from generators import n_channels_gen, n_channels_triangle_gen
import models as md
import routines as rt 

import time

#                      mag, real, imag, phase, log_mag
inputs1 = np.array([True, False, False, False, False])
inputs2 = np.array([False, True, True, False, False])

data, labels, params = n_channels_triangle_gen(
    num_samples=2, 
    signal_length=1024, 
    sigma_min=0.01, 
    sigma_max=0.1, 
    zeta_max=0.1,
    zeta_min=0.01,
    min_max=True, 
    enabled_inputs=inputs1,
    params_out=True,
    pulse_width=0.1,
    )
val_dataset_1 = md.n_channel_dataset(data, labels, params)
val_dataloader_1 = DataLoader(val_dataset_1, batch_size=32, shuffle=True)

rt.plot_samples(val_dataloader_1, 2, binary_labels=False)