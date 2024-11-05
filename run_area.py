import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit

from generators import mag_1D_noise as generate_data
from generators import mag_1D_no_noise as generate_data_no_noise
from models import simple_dataset, PeakMag1, PeakMag2
from routines import train_model_binary, compare_models, load_model, plot_loss_history, plot_precision_history, plot_recall_history, plot_samples

data, labels = generate_data(num_samples=4000, signal_length=1024, sigma_max=0.1)
train_dataset = simple_dataset(data, labels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

data, labels = generate_data(num_samples=400, signal_length=1024, sigma_max=0.1)
val_dataset = simple_dataset(data, labels)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

data, labels = generate_data_no_noise(num_samples=4000, signal_length=1024)
train_dataset_no_noise = simple_dataset(data, labels)
train_dataloader_no_noise = DataLoader(train_dataset_no_noise, batch_size=32, shuffle=True)


plot_samples(train_dataloader, 3)

model1 = PeakMag2()
model2 = PeakMag2()

plot_during = False

'''

result_dict1 = train_model_binary(
                model1, 
                train_dataloader, 
                val_dataloader, 
                save_name='PeakMag1_2', #None if no save required
                num_epochs = 6, 
                acceptance=0.5, 
                plotting=plot_during
                )

result_dict2 = train_model_binary(
                model2, 
                train_dataloader, 
                val_dataloader, 
                save_name='PeakMag2_1', #None if no save required
                num_epochs = 6, 
                acceptance=0.5, 
                plotting=plot_during
                )

plot_loss_history([result_dict1,result_dict2],log_scale=True)
plot_precision_history([result_dict1,result_dict2],log_scale=True)
plot_recall_history([result_dict1,result_dict2],log_scale=True)

#load models from save files or train above
#model1 = load_model('PeakMag1_1')
#model2 = load_model('PeakMag1_2')

criterion=nn.BCEWithLogitsLoss()
compare_models(model1, model2, val_dataloader, criterion, acceptance1=0.5, acceptance2=0.5)

'''