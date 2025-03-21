import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit

from generators import n_channels_gen, n_channels_stepped_gen, n_channels_triangle_gen, n_channels_multi_labels_gen
import models as md
import routines as rt 

import time

from typing import List

import h5py

#                    real, imag, phase, mag, log_mag
inputs1 = np.array([True, True, False, False, True])

#                 modes, a mag, a phase, zeta 
labels1 = np.array([True, True, True, True])


project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/datasets/'
data_name = 'data_all_labels_0_alpha_phase.h5'
data_file = project_path + data_name

# data, labels, params = n_channels_multi_labels_gen(
#     num_samples=100000, 
#     signal_length=1024, 
#     sigma_min=0.01, 
#     sigma_max=0.1, 
#     zeta_max=0.1,
#     zeta_min=0.01,
#     alpha_phase_std_dev=np.pi/6,
#     min_max=True, 
#     enabled_inputs=inputs1,
#     label_outputs=labels1,
#     params_out=True,
#     pulse_width=0.05,
#     )

# with h5py.File(data_file, 'w') as f:
#     f.create_dataset('data', data=data)
#     f.create_dataset('labels', data=labels)
#     f.create_dataset('params', data=params)

with h5py.File(data_file, 'r') as f:
    val_data = f['data'][:5000]
    val_labels = f['labels'][:5000]
    val_params = f['params'][:5000]
    train_data = f['data'][5000:8000]
    train_labels = f['labels'][5000:8000]
    train_params = f['params'][5000:8000]

train_dataset_1 = md.n_channel_dataset(train_data, train_labels, train_params)
train_dataloader_1 = DataLoader(train_dataset_1, batch_size=32, shuffle=True)
val_dataset_1 = md.n_channel_dataset(val_data, val_labels, val_params)
val_dataloader_1 = DataLoader(val_dataset_1, batch_size=32, shuffle=True)

# rt.plot_predictions_all_labels(None, val_dataloader_1, 5, labels1)

# model1 = md.DenseNet1(data_channels=np.sum(outputs1),
#                     out_channels=1,
#                     growth_rate=8,
#                     block_config=[6,6,6,6],
#                     transition_channels=16, 
#                     # Change this to be a constant factor e.g divide by 2 or 3 instead of constant number?????
#                     kernel_size=3,
# )
# print(f'Model 1 trainable parameters: {rt.count_parameters(model1)}')
# print(f'Model 1 receptive field: {rt.calculate_total_receptive_field(model1)}')

# model1 = md.ResNet1(data_channels=np.sum(inputs1), 
#                     out_channels=[4,4,8,8,6,6,4],
#                     kernel_size=[13]
#                     )
# print(f'Model 1 trainable parameters: {rt.count_parameters(model1)}')
# print(f'Model 1 receptive field: {rt.calculate_total_receptive_field(model1)}')

# model2 = md.ResNet1(data_channels=np.sum(inputs1), 
#                     out_channels=[4,6,6,8,8,12,8,8,6,6,4],
#                     kernel_size=[13]
#                     )
# print(f'Model 2 trainable parameters: {rt.count_parameters(model2)}')
# print(f'Model 2 receptive field: {rt.calculate_total_receptive_field(model2)}')

# model1 = md.RegressionModel1(data_channels=np.sum(inputs1), 
#                             out_channels=[4,4,8,12,12,8,4,2,1],
#                             kernel_size=[13],
#                             batch_norm=True,
#                             P_dropout=0.0,
#                             max_pool=False,
#                             )
# print(f'Model 1 trainable parameters: {rt.count_parameters(model1)}')
# print(f'Model 1 receptive field: {rt.calculate_total_receptive_field(model1)}')


# results = []

# plot_during = False

# Save names for ResNet
# save1 = '_' + '_'.join([''.join(map(str, model1.out_channels)), ''.join(map(str, model1.kernel_size)), str(model1.__class__.__name__)])
# save2 = '_' + '_'.join([''.join(map(str, model2.out_channels)), ''.join(map(str, model2.kernel_size)), str(model2.__class__.__name__)])

# Save names for DenseNet
# save1 = '_' + '_'.join([str(model1.out_channels), str(model1.kernel_size), '.'.join(map(str, model1.block_config)), str(model1.growth_rate), str(model1.transition_channels), str(model1.__class__.__name__)])


# start1 = time.time()
# result_dict1,_ = rt.train_model_regression(
#                 model1, 
#                 train_dataloader_1, 
#                 val_dataloader_1,
#                 save_suffix = save1 + '_a_phase_w_phase',
#                 num_epochs = 200, 
#                 plotting=plot_during,
#                 patience = 20,
#                 )
# results.append(result_dict1)
# end1 = time.time()

# start2 = time.time()
# result_dict2,_ = rt.train_model_regression(
#                 model2, 
#                 train_dataloader_1, 
#                 val_dataloader_1,
#                 save_suffix = save2 + '_a_phase',
#                 num_epochs = 200, 
#                 plotting=plot_during,
#                 patience = 20,
#                 )
# results.append(result_dict2)
# end2 = time.time()

# print('Time taken for model 1 training: ', end1-start1)
# print('Time taken for model 2 training: ', end2-start2)



# models from save files or train above
# model1 = rt.load_model('03_21_10_17_4488664_13_ResNet1_a_phase_w_phase.pth')
model2 = rt.load_model('03_12_19_10_4488664_13_ResNet1_0_a_phase.pth')



# print('\nModel 1 mean frequency error:', rt.calculate_mean_frequency_error_triangle(model1, val_dataloader_1, labels1))
# print('Model 2 mean frequency error:', rt.calculate_mean_frequency_error_triangle(model2, val_dataloader_1, labels1))

# rt.compare_models_regression([model1, model2], val_dataloader_1)

# rt.plot_predictions_all_labels([model1, model2], val_dataloader_1, 5, labels1)

rt.plot_FRF_comparison(model2, val_dataloader_1, num_samples=5, FRF_type=0)
rt.plot_FRF_comparison(model2, val_dataloader_1, num_samples=5, FRF_type=1)