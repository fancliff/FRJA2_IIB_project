import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
import timeit
from numba import jit

#@jit(nopython=True)
def generate_data(num_samples=1000, signal_length=400):
    
    data = np.empty((num_samples,signal_length),dtype=np.float64)
    labels = np.empty((num_samples,signal_length),dtype=np.int32)

    frequencies = np.linspace(0,1,signal_length)
    for i in range(num_samples):
        
        H_v = np.zeros(signal_length, dtype=np.complex128)
        label = np.zeros(signal_length, dtype=np.int32)
        
        #max 5 modes
        num_modes = np.random.randint(1,6)
        #when noise is added change num_modes to include 0
        # to improve model add a random sign to alpha_j 
        alphas = np.random.uniform(1, 2, size=5)
        zetas = np.exp(np.random.uniform(np.log(0.01), np.log(0.1), size=5))
        #no modes at very edge of frequency range
        omegas = np.random.uniform(0.001, 0.999, size=5)
        
        for j, w in enumerate(frequencies):
            H_f = 0.0j
            for n in range(num_modes):
                # to improve model add a random sign to alpha_j 
                alpha_n = alphas[n]
                zeta_n = zetas[n]
                omega_n = omegas[n]
                
                denominator = omega_n**2 - w**2 + 2j * zeta_n * w
                numerator = 1j*w*alpha_n
                
                H_f += numerator/denominator
                #set label bandwidth to 3db or arbitrary value
                #bandwidth = 0.02 #(8 frequency points wide if 400 points)
                bandwidth = 2*omega_n*zeta_n
                if omega_n - bandwidth <= w <= omega_n + bandwidth:
                    label[j] = 1
            
            H_v[j] = H_f
        
        data[i, :] = np.abs(H_v) #use signal magnitude for now
        labels[i, :] = label
        
    return data, labels

'''
data, labels = generate_data(num_samples=1000, signal_length=400)

# Visualize a sample signal with labels
plt.figure(figsize=(12, 6))
plt.plot(data[0], label='Signal')
plt.plot(labels[0] * 5, label='Labels (scaled)', linestyle='--')  # Scale labels for visibility
plt.title('Sample Vibration Signal with Labels')
plt.xlabel('Normalised Frequency')
plt.ylabel('Amplitude / Label')
plt.legend()
plt.show()

'''

print(timeit.timeit(generate_data,number=100))
#60s without jit, 1000 signals, 400points, 10 repeats
#12s with jit, 1000 signals, 400 points, 1000 repeats
#improvements of ~500x