import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit
import pydvma
import h5py
from scipy.signal import resample

from generators import n_channels_gen, n_channels_triangle_gen
import models as md
import routines as rt 
import real_data_routines as rdrt

import time

import numpy as np
from scipy.signal import resample
import torch

# Load and process TF data
data = pydvma.load_data()
# print(data)


########### 3C6 ###########
# 3C6 2s - 1 TF
# 3C6 30s - 1 TF
# 3C6 impulse - 1 TF

tf_data = data.tf_data_list[0]
tf_arr = np.array(tf_data.tf_data) 
tf_arr = tf_arr.squeeze(-1)

########### 4C6 ###########
# 4C6 - 12 TFs

# a = 1 # 1 - 12 for which TF
# tf_data = data.tf_data_list[a-1]
# tf_arr = np.array(tf_data.tf_data)
# tf_arr = tf_arr.squeeze(-1)
# x = np.linspace(0,1,len(tf_arr))
# x = x[1:] # remove 0 frequency
# tf_arr = tf_arr[1:] # remove 0 frequency
# tf_arr = tf_arr / (1j * x) # convert from acceleration to velocity (divide by iw)

###########     ###########

if np.isnan(tf_arr).any():
    print('\nNAN in tf_arr\n')

# Extract real and imaginary parts
tf_real = np.real(tf_arr)
tf_imag = np.imag(tf_arr)

x = np.linspace(0,1,len(tf_real))
plt.plot(x,tf_real,label='Real')
plt.plot(x,tf_imag,label='Imag')
plt.legend()
plt.show()

# Frequency cutoff
cut_off_f = float(input('Enter cut-off frequency between 0 and 1: '))
if not 0 <= cut_off_f <= 1:
    raise ValueError("Cut-off frequency must be between 0 and 1.")
cut_off_idx = int(cut_off_f * len(tf_real))
print(f"Cut-off index: {cut_off_idx}")

# Apply cutoff
tf_real = tf_real[:cut_off_idx]
tf_imag = tf_imag[:cut_off_idx]

# Resample to 1024 points 

# fourier - introduces wiggles - bad
# tf_real_resampled = resample(tf_real, 1024)
# tf_imag_resampled = resample(tf_imag, 1024)

# linear resampling, no wiggles
tf_real_resampled = rdrt.resample_linear_scipy(tf_real, 1024)
tf_imag_resampled = rdrt.resample_linear_scipy(tf_imag, 1024)

x = np.linspace(0,1,1024)
plt.plot(x,tf_real_resampled,label='Real')
plt.plot(x,tf_imag_resampled,label='Imag')
plt.legend()
plt.show()


# Create magnitude and normalise
tf_mag = np.abs(tf_real_resampled + 1j * tf_imag_resampled)
tf_mag_normed = (tf_mag - np.min(tf_mag)) / (np.max(tf_mag) - np.min(tf_mag))

# Apply magnitude normalisation to real/imag components
tf_real_normalised = tf_real_resampled * (tf_mag_normed / tf_mag)
tf_imag_normalised = tf_imag_resampled * (tf_mag_normed / tf_mag)
tf_phase = np.angle(tf_real_normalised+1j*tf_imag_normalised)
tf_log_mag = np.log10(tf_mag_normed)

# Combine into final array and convert to tensor
tf_processed = np.stack([
    tf_real_normalised, 
    tf_imag_normalised,
    # tf_phase,
    # tf_mag_normed,
    # tf_log_mag,
    ])  
# Shape (n, 1024)
tf_processed_no_norm = np.stack([
    tf_real_resampled,
    tf_imag_resampled,
])
tf_tensor = torch.tensor(tf_processed, dtype=torch.float32)  # Convert to tensor
tf_tensor_no_norm = torch.tensor(tf_processed_no_norm, dtype=torch.float32)

# Add batch dimension (model expects [batch_size, channels, sequence_length])
tf_tensor = tf_tensor.unsqueeze(0)  # Now shape (1, n, 1024)
tf_tensor_no_norm = tf_tensor_no_norm.unsqueeze(0)

# Verify the shape matches model expectations
print("Final tensor shape:", tf_tensor.shape)  # Should be (1, n, 1024)


########## dataset testing #########

# model = rt.load_model('05_19_12_56_4488444_9_RegressionModel1.pth')
# model = rt.load_model('05_19_17_30_4488444_9_RegressionModel1_001_min_zeta.pth')
# model = rt.load_model('05_20_11_34_4488444_9_RegressionModel1_pi_12_phase_001_min_zeta_real_imag.pth')
# model = rt.load_model('05_19_15_22_4488444_9_RegressionModel1_10_modes.pth')
# model = rt.load_model('05_20_17_18_4488444_9_RegressionModel1_pi_12_phase_001_min_zeta_10_modes_real_imag.pth')
# model = rt.load_model('05_20_17_37_4488444_9_RegressionModel1_pi_12_phase_0005_min_zeta_10_modes_real_imag.pth')
# model = rt.load_model('05_21_11_06_4488444_9_RegressionModel1_pi_12_phase_001_min_zeta_10_modes_10_max_amplitude_real_imag.pth')

########### input testing ###########

# model = rt.load_model('05_20_17_18_4488444_9_RegressionModel1_pi_12_phase_001_min_zeta_10_modes_real_imag.pth')
# model = rt.load_model('05_20_19_41_4488444_9_RegressionModel1_pi_10_phase_001_min_zeta_10_modes_real_imag.pth')
# model = rt.load_model('05_20_19_53_4488444_9_RegressionModel1_pi_8_phase_001_min_zeta_10_modes_real_imag.pth')
# model = rt.load_model('05_20_20_40_4488444_9_RegressionModel1_pi_6_phase_001_min_zeta_10_modes_real_imag.pth')

#                 modes, a mag, a phase, zeta 
labels1 = np.array([True, True, True, True])

project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/datasets/'
data_name = 'final_real_imag.h5'

######## dataset testing #########
# data_name = 'real_imag_all_labels_pi_12_alpha_phase_scaled_001_min_zeta.h5'
# data_name = 'real_imag_all_labels_0_alpha_phase_scaled_001_min_zeta.h5'
# data_name = 'real_imag_all_labels_0_alpha_phase_scaled.h5'
# data_name = 'real_imag_all_labels_pi_12_alpha_phase_scaled_0005_min_zeta_10_modes.h5'
# data_name = 'real_imag_all_labels_pi_12_alpha_phase_scaled_001_min_zeta_10_modes_5_max_amplitude.h5'
# data_name = 'real_imag_all_labels_pi_12_alpha_phase_scaled_001_min_zeta_10_modes_10_max_amplitude.h5'

########## phase testing ##########
# data_name = 'real_imag_all_labels_pi_12_alpha_phase_scaled_001_min_zeta_10_modes.h5'
# data_name = 'real_imag_all_labels_pi_10_alpha_phase_scaled_001_min_zeta_10_modes.h5'
# data_name = 'real_imag_all_labels_pi_8_alpha_phase_scaled_001_min_zeta_10_modes.h5'
# data_name = 'real_imag_all_labels_pi_6_alpha_phase_scaled_001_min_zeta_10_modes.h5'

data_file = project_path + data_name

with h5py.File(data_file, 'r') as f:
    scale_factors = f['scale_factors'][:]

model = rt.load_model('05_22_02_34_446668866644_11_RegressionModel1.pth')

# rdrt.plot_predictions_all_labels(model, tf_tensor, labels1, scale_factors, N=2, Wn=0.1, plot_phase=True)
# rdrt.plot_FRF_comparison(model, tf_tensor, scale_factors, FRF_type=1, norm=True, plot_phase=True, q=0)
rdrt.plot_FRF_cloud_single_sample(model, tf_tensor, 100, scale_factors, 0.05, FRF_type=1, q=0, window_scale=0.6)
rdrt.optimiser_handler(model, tf_tensor, tf_tensor_no_norm, scale_factors, omega_weight=0, plot=True, q=0, window_scale=0.6)
# omega_weight can be helpful for stabilising the model, 
# it's a bit of a hack but the model natural frequency estimation 
# is so much better than the other parameters so it works fine
# If natural frequencies moving around way more than expected try omega_weight = 1 or more