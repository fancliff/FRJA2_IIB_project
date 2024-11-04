import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit

from generators import mag_1D_no_noise as generate_data
from models import simple_dataset, PeakMag1
from routines import train_model_binary, plot_losses, compare_models, load_model

data, labels = generate_data(num_samples=1000, signal_length=1024)
train_dataset = simple_dataset(data, labels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

data, labels = generate_data(num_samples=100, signal_length=1024)
val_dataset = simple_dataset(data, labels)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

model1 = PeakMag1()

plot_during = False
plot_after = True

result_dict = train_model_binary(
                model1, 
                train_dataloader, 
                val_dataloader, 
                save_name='PeakMag1_1', #None if no save required
                num_epochs = 10, 
                acceptance=0.5, 
                plotting=plot_during
                )

if plot_after:
    plot_losses(result_dict,log_scale=True)

#load models from save files or train above
#model1 = load_model('PeakMag1_1')
#model2 = load_model('PeakMag1_2')

compare_models(model1, model1, val_dataloader, acceptance=0.5)