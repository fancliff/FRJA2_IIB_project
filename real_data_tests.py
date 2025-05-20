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

from generators import n_channels_gen, n_channels_triangle_gen
import models as md
import routines as rt 
import real_data_routines as rdrt

import time

data = pydvma.load_data()
tf_data = data.tf_data_list[0]
tf_arr = np.array(tf_data.tf_data)

N = len(tf_arr)
x = np.linspace(0,1,N)

plt.plot(x, np.real(tf_arr), label = 'Real part')
plt.plot(x, np.imag(tf_arr), label = 'Imag part')
plt.plot(x, np.abs(tf_arr), label = 'Magnitude')
plt.legend()
plt.show()


tf = np.array([
    np.real(tf_arr),
    np.imag(tf_arr),
    # np.abs(tf_arr),
])

tf = np.squeeze(tf,-1)
print(np.shape(tf))


cut_off_f = float(input('Enter cut-off frequency between 0 and 1: '))
if not 0 <= cut_off_f <= 1:
    raise ValueError("Cut-off frequency must be between 0 and 1.")
cut_off_idx = int(cut_off_f * tf.shape[1])

print(f"Cut-off index: {cut_off_idx}")

tf = tf[:, :cut_off_idx]
x = np.linspace(0,1,cut_off_idx)

tf_mag = np.abs(tf[0]+1j*tf[1])
tf_mag_min = np.min(tf_mag)
tf_mag_max = np.max(tf_mag)

tf_mag_normed = (tf_mag-tf_mag_min)/(tf_mag_max-tf_mag_min)
tf[0] = tf[0] * tf_mag_normed/tf_mag
tf[1] = tf[1] * tf_mag_normed/tf_mag
# tf[2] = tf_mag_normed

plt.plot(x, tf[0], label = 'Real part')
plt.plot(x, tf[1], label = 'Imag part')
# plt.plot(x, tf[2], label = 'Magnitude')
plt.legend()
plt.show()

tf_tensor = torch.tensor(tf, dtype=torch.float32).unsqueeze(0)


##########

model = rt.load_model('05_19_12_56_4488444_9_RegressionModel1.pth')

#                 modes, a mag, a phase, zeta 
labels1 = np.array([True, True, True, True])

project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/datasets/'
data_name = 'real_imag_all_labels_pi_12_alpha_phase_scaled.h5'
data_file = project_path + data_name

with h5py.File(data_file, 'r') as f:
    scale_factors = f['scale_factors'][:]

rdrt.plot_predictions_all_labels(model, tf_tensor, labels1, scale_factors, N=2, Wn=0.1, plot_phase=True)