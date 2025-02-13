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

from typing import List

#                      mag, real, imag, phase, log_mag
outputs1 = np.array([False, True, True, False, False])
outputs2 = np.array([False, True, True, False, False])

data, labels = n_channels_gen(
    num_samples=2000, 
    signal_length=1024, 
    sigma_min=0.01, 
    sigma_max=0.1, 
    zeta_max=0.1,
    zeta_min=0.01,
    three_db_bandwidth=False,
    fixed_bandwidth=0.02,
    min_max=True, 
    enabled_outputs=outputs1
    )
train_dataset_1 = md.n_channel_dataset(data, labels)
train_dataloader_1 = DataLoader(train_dataset_1, batch_size=32, shuffle=True)

data, labels = n_channels_gen(
    num_samples=2000, 
    signal_length=1024, 
    sigma_min=0.01, 
    sigma_max=0.1, 
    zeta_max=0.1,
    zeta_min=0.01,
    three_db_bandwidth=False,
    fixed_bandwidth=0.02,
    min_max=True, 
    enabled_outputs=outputs1
    )
val_dataset_1 = md.n_channel_dataset(data, labels)
val_dataloader_1 = DataLoader(val_dataset_1, batch_size=32, shuffle=True)

#rt.plot_samples(train_dataloader_1, 5)
#rt.plot_samples(val_dataloader_1, 5)

'''

model1 = md.NewModelGeneral(data_channels=np.sum(outputs1), 
                            out_channels=[4,4,8,8,4,2,1],
                            kernel_size=9,
                            )
print(f'Model 1 trainable parameters: {rt.count_parameters(model1)}')
print(f'Model 1 receptive field: {rt.calculate_total_receptive_field(model1)}')



model2 = md.NewModelGeneral(data_channels=np.sum(outputs2),
                            out_channels=[4,4,8,8,4,2,1],
                            kernel_size=7,
                            )
print(f'Model 2 trainable parameters: {rt.count_parameters(model2)}')
print(f'Model 2 receptive field: {rt.calculate_total_receptive_field(model2)}')



results = []

plot_during = False



start1 = time.time()
result_dict1,_ = rt.train_model_binary(
                model1, 
                train_dataloader_1, 
                val_dataloader_1, 
                save_name='kernel_9_fixed_03', #None if no save required
                num_epochs = 200, 
                acceptance=0.5, 
                plotting=plot_during,
                patience = 40,
                )
results.append(result_dict1)
end1 = time.time()



start2 = time.time()
result_dict2,_ = rt.train_model_binary(
                model2, 
                train_dataloader_1, 
                val_dataloader_1, 
                save_name='kernel_9_fixed_03', #None if no save required
                num_epochs = 200, 
                acceptance=0.5, 
                plotting=plot_during,
                patience = 40,
                )
results.append(result_dict2)
end2 = time.time()



rt.plot_loss_history(results, log_scale=True, show=False)
rt.plot_precision_history(results, log_scale=False, show=False)
rt.plot_recall_history(results, log_scale=False, show=False)

print('Time taken for model 1 training: ', end1-start1)
print('Time taken for model 2 training: ', end2-start2)


'''

#load models from save files or train above
model1 = rt.load_model('kernel_9_fixed_02')
model2 = rt.load_model('kernel_9_fixed_02')

#rt.visualise_activations(model1, val_dataloader_1, 3)
#rt.visualise_activations_with_signal(model1, val_dataloader_1, 3)
#rt.compare_activations(model1, model2, val_dataloader_1, 3)



criterion=nn.BCELoss()
rt.compare_models(
    model1, 
    model2,
    val_dataloader_1,
    val_dataloader_1,
    criterion,
    acceptance1=0.5,
    acceptance2=0.5,
)



#rt.plot_predictions([model1, model2], val_dataloader_1, 5, acceptance=0.5)



