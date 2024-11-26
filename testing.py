import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit

from generators import n_channels_gen
import models as md
import routines as rt 

import time

#                      mag, real, imag, phase, log_mag
outputs1 = np.array([True, False, False, False, False])
outputs2 = np.array([False, True, True, False, False])

data, labels = n_channels_gen(num_samples=32, signal_length=1024, enabled_outputs=outputs2, noise=True, sigma_max=0.1, min_max=True)
val_dataset_1 = md.n_channel_dataset(data, labels)
val_dataloader_1 = DataLoader(val_dataset_1, batch_size=32, shuffle=True)

rt.plot_samples(val_dataloader_1, 5)