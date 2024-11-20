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
outputs1 = np.array([False, True, True, False, False])
outputs2 = np.array([False, True, True, False, False])

data, labels = n_channels_gen(num_samples=4000, signal_length=1024, sigma_max=0.1, min_max=True, enabled_outputs=outputs1)
train_dataset_1 = md.n_channel_dataset(data, labels)
train_dataloader_1 = DataLoader(train_dataset_1, batch_size=32, shuffle=True)

data, labels = n_channels_gen(num_samples=400, signal_length=1024, sigma_max=0.1, min_max=True, enabled_outputs=outputs1)
val_dataset_1 = md.n_channel_dataset(data, labels)
val_dataloader_1 = DataLoader(val_dataset_1, batch_size=32, shuffle=True)



data, labels = n_channels_gen(num_samples=4000, signal_length=1024, sigma_max=0.1, min_max=True, enabled_outputs=outputs2)
train_dataset_2 = md.n_channel_dataset(data, labels)
train_dataloader_2 = DataLoader(train_dataset_2, batch_size=32, shuffle=True)

data, labels = n_channels_gen(num_samples=400, signal_length=1024, sigma_max=0.1, min_max=True, enabled_outputs=outputs2)
val_dataset_2 = md.n_channel_dataset(data, labels)
val_dataloader_2 = DataLoader(val_dataset_2, batch_size=32, shuffle=True)



#rt.plot_samples(train_dataloader, 20)
#rt.plot_samples(val_dataloader, 20)

#model1 = rt.load_model('PeakMag4_1')
#print(rt.count_parameters(model1))

'''

model1 = md.PeakMag6(data_channels=np.sum(outputs1))
print(f'Model 1 trainable parameters: {rt.count_parameters(model1)}')

model2 = md.PeakMag8(data_channels=np.sum(outputs2))
print(f'Model 2 trainable parameters: {rt.count_parameters(model2)}')

results = []

plot_during = False


start1 = time.time()
result_dict1,_ = rt.train_model_binary(
                model1, 
                train_dataloader_1, 
                val_dataloader_1, 
                save_name=None, #None if no save required
                num_epochs = 100, 
                acceptance=0.5, 
                plotting=plot_during,
                patience = 10,
                )
results.append(result_dict1)
end1 = time.time()




start2 = time.time()
result_dict2,_ = rt.train_model_binary(
                model2, 
                train_dataloader_2, 
                val_dataloader_2, 
                save_name='PeakMag8_1', #None if no save required
                num_epochs = 100, 
                acceptance=0.5, 
                plotting=plot_during,
                patience = 10,
                )
results.append(result_dict2)
end2 = time.time()



rt.plot_loss_history(results,log_scale=True)
rt.plot_precision_history(results,log_scale=False)
rt.plot_recall_history(results,log_scale=False)

print('Time taken for model 1 training: ', end1-start1)
print('Time taken for model 2 training: ', end2-start2)

'''

#load models from save files or train above
model1 = rt.load_model('PeakMag6_1')
model2 = rt.load_model('PeakMag8_1')

criterion=nn.BCELoss()
rt.compare_models(model1, model2, val_dataloader_1, criterion, acceptance1=0.5, acceptance2=0.5)

rt.plot_predictions(model1, val_dataloader_1, 10, acceptance=0.5)
rt.plot_predictions(model2, val_dataloader_1, 10, acceptance=0.5)


