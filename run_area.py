import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit

from generators import mag_1D_noise_normalised as generate_data
from models import simple_dataset, PeakMag1, PeakMag2, PeakMag3
import routines as rt 

data, labels = generate_data(num_samples=4000, signal_length=1024, sigma_max=0.1, min_max=True)
train_dataset = simple_dataset(data, labels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

data, labels = generate_data(num_samples=400, signal_length=1024, sigma_max=0.1, min_max=True)
val_dataset = simple_dataset(data, labels)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

data, labels = generate_data(num_samples=400, signal_length=1024, sigma_max=0.1, min_max=True)
test_dataset = simple_dataset(data, labels)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#plot_samples(train_dataloader, 20)

#plot_samples(val_dataloader, 20)

#model1 = rt.load_model('PeakMagNoNoise1')
#print(rt.count_parameters(model1))

model1 = PeakMag3()
#model2 = PeakMag3()

results = []

plot_during = True

result_dict1 = rt.train_model_binary(
                model1, 
                train_dataloader, 
                val_dataloader, 
                save_name='PeakMag3_4_030', #None if no save required
                num_epochs = 20, 
                acceptance=0.5, 
                plotting=plot_during
                )
results.append(result_dict1)


'''
result_dict2 = rt.train_model_binary(
                model2, 
                train_dataloader, 
                val_dataloader, 
                save_name='PeakMag3_2', #None if no save required
                num_epochs = 20, 
                acceptance=0.5, 
                plotting=plot_during
                )
results.append(result_dict2)
'''

'''
rt.plot_loss_history(results,log_scale=True)
rt.plot_precision_history(results,log_scale=True)
rt.plot_recall_history(results,log_scale=True)
'''

#load models from save files or train above
#model1 = rt.load_model('PeakMag1_1')
model2 = rt.load_model('PeakMag3_3_025')

criterion=nn.BCEWithLogitsLoss()
rt.compare_models(model1, model2, val_dataloader, criterion, acceptance1=0.5, acceptance2=0.5)

