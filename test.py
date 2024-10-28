import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt

num_samples=30
signal_length=400

frequencies = np.linspace(0,1,signal_length)
for _ in range(num_samples):
    alphas = []
    zetas = []
    omegas = []
    H_v = np.zeros(signal_length,dtype=np.complex128)
    
    num_modes = np.random.randint(1,5)
    #when noise is added change num_modes to include 0
    for n in range(num_modes):
        # to improve model add a random sign to alpha_j 
        alphas.append(np.random.uniform(1,2))
        zetas.append(scipy.stats.loguniform.rvs(0.01,0.2))
        #no modes at very edge of frequency range
        omegas.append(np.random.uniform(0.001,0.999))
        
    
    for i, w in enumerate(frequencies):
        H_f = 0.0j
        for n in range(num_modes):
            # to improve model add a random sign to alpha_j 
            alpha_n = alphas[n]
            zeta_n = zetas[n]
            omega_n = omegas[n]
            
            denominator = omega_n**2 - w**2 + 2j * zeta_n * w
            numerator = 1j*w*alpha_n
            
            H_f += numerator/denominator
        
        H_v[i] = H_f
        
    plt.figure(figsize=(10, 6))

    plt.plot(frequencies, np.abs(H_v), label='|H_v(f)|', color='blue')
    plt.plot(frequencies, np.real(H_v), label='Re[H_v(f)]', linestyle='--', color='orange')
    plt.plot(frequencies, np.imag(H_v), label='Im[H_v(f)]', linestyle=':', color='green')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Velocity Transfer Function')
    plt.title('Modal Sum Velocity Transfer Function')
    plt.grid(True)
    plt.legend()
    plt.show()