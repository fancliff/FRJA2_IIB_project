import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

import torch
torch.set_num_threads(4)

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

print('\nUsing', torch.get_num_threads(), 'threads\n')

#                    real, imag, phase, mag, log_mag
inputs1 = np.array([True, True, False, False, False])

#                 modes, a mag, a phase, zeta 
labels1 = np.array([True, True, True, True])

project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/datasets/'
data_name = 'final_real_imag.h5'
data_file = os.path.join(project_path, data_name)

# save first 20,000 samples for validation if needed
with h5py.File(data_file, 'r') as f:
    val_data = f['data'][:5000]
    val_labels = f['labels'][:5000]
    val_params = f['params'][:5000]
    train_data = f['data'][20000:26000]
    train_labels = f['labels'][20000:26000]
    train_params = f['params'][20000:26000]
    scale_factors = f['scale_factors'][:]

train_dataset_1 = md.n_channel_dataset(train_data, train_labels, train_params)
train_dataloader_1 = DataLoader(train_dataset_1, batch_size=32, shuffle=True)
val_dataset_1 = md.n_channel_dataset(val_data, val_labels, val_params)
val_dataloader_1 = DataLoader(val_dataset_1, batch_size=32, shuffle=True)

out_channels_list = [
    [4,6,4],
    [4,8,4],
    [4,6,8,4],
    [4,6,8,6,4],
    [4,8,12,8,4],
    [4,6,6,8,6,6,4],
    [4,6,6,8,8,6,6,4],
    # [4,8,12,16,12,8,4],
]

kernel_size_list = [
    [5],
    [7],
    [9],
    [11],
    [13],
]

######### EXPECTED TRAINING TIME #########
params_sum = 0
max_params = 0
num_models = 0
for oc in out_channels_list:
    for ks in kernel_size_list:
        # model = md.RegressionModel1(
        #     data_channels=2,
        #     out_channels=oc,
        #     kernel_size=ks,
        #     batch_norm=True,
        #     P_dropout=0.0,
        #     max_pool=False,
        # )
        model = md.ResNet1(
            data_channels=2,
            out_channels=oc,
            kernel_size=ks,
        )
        num_models += 1
        params = rt.count_parameters(model)
        if params > max_params:
            max_params = params
        params_sum += params

est_time = (params_sum/1700)*(200/150)*(len(train_data)/3000)*695/3600
avg_params = params_sum/num_models

print(f'\nTotal Trainable Parameters: {params_sum}')
print(f'Total Number of models: {num_models}')
print(f'Average parameters per model: {avg_params:.0f}')
print(f'Max trainable parameters single model: {max_params}')
print(f'Estimated Training time: {est_time:.1f}hrs\n')

######### AUTOMATIC TRAINING #########

save_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/'
output_csv = os.path.join(save_path,'model_training_results.csv')

error_log = os.path.join(save_path, "training_failures.log")

# Write header if file doesn't exist
if not os.path.exists(output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Model ID', 'Trainable Parameters', 'Receptive Field', 'Mean FRF Error', 'Training Time (s)'])


for oc in out_channels_list:
    for ks in kernel_size_list:
        try:
            if len(ks) != 1 and len(ks) != len(oc):
                print(f'Skipping: kernel sizes {ks} incompatible with out_channels {oc}')
                continue

            # model = md.RegressionModel1(
            #     data_channels=int(np.sum(inputs1)),
            #     out_channels=oc,
            #     kernel_size=ks,
            #     batch_norm=True,
            #     P_dropout=0.0,
            #     max_pool=False,
            # )
            
            model = md.ResNet1(
                data_channels=int(np.sum(inputs1)),
                out_channels=oc,
                kernel_size=ks,
            )

            save_suffix = '_' + '_'.join([
                ''.join(map(str, model.out_channels)),
                ''.join(map(str, model.kernel_size)),
                str(model.__class__.__name__)
            ])

            print(f'Starting: {save_suffix}')

            params = rt.count_parameters(model)
            receptive_field = rt.calculate_total_receptive_field(model)

            start = time.process_time()
            _, _ = rt.train_model_regression(
                model,
                train_dataloader_1,
                val_dataloader_1,
                save_suffix=save_suffix,
                num_epochs=200,
                plotting=False,
                patience=20,
                printing=False,
            )
            end = time.process_time()
            training_time = end - start

            frf_error = rt.calculate_mean_FRF_error(
                model, val_dataloader_1, scale_factors,
                FRF_type=1, signal_length=1024, norm=True
            )

            print(f'Finished: {save_suffix}\n')

            # Append to CSV after each run
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    timestamp,
                    save_suffix,
                    params,
                    receptive_field,
                    frf_error,
                    training_time
                ])

        except Exception as e:
            print(f'Error for ks={ks}, oc={oc}: {e}\n')
            with open(error_log, "a") as log:
                log.write(f"Error for ks={ks}, oc={oc} — {str(e)}\n")
            continue

print(f'\nAll available results have been appended to: {output_csv}')
