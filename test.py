import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
import timeit
from numba import jit, types

@jit(nopython=True)
def jit_signal_gen():
    num_samples=1000
    signal_length=400

    frequencies = np.linspace(0,1,signal_length)
    for _ in range(num_samples):
        alphas = []
        zetas = []
        omegas = []
        H_v = np.zeros(signal_length,dtype=types.complex128)
        label = np.zeros(signal_length)
        
        num_modes = np.random.randint(1,5)
        #when noise is added change num_modes to include 0
        for n in range(num_modes):
            # to improve model add a random sign to alpha_j 
            alphas.append(np.random.uniform(1,2))
            zetas.append(np.exp(np.random.uniform(np.log(0.01), np.log(0.1))))
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
                #set label bandwidth to 3db or arbitrary value
                #bandwidth = 0.02 #(8 frequency points wide if 400 points)
                bandwidth = 2*omega_n*zeta_n
                label[(frequencies >= omega_n - bandwidth) & (frequencies <= omega_n + bandwidth)] = 1
            
            H_v[i] = H_f
        
        '''
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, np.abs(H_v), label='|H_v(f)|', color='blue')
        plt.plot(frequencies, np.real(H_v), label='Re[H_v(f)]', linestyle='--', color='orange')
        plt.plot(frequencies, np.imag(H_v), label='Im[H_v(f)]', linestyle=':', color='green')
        plt.scatter(frequencies, label, label='Label')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Velocity Transfer Function')
        plt.title('Modal Sum Velocity Transfer Function')
        plt.grid(True)
        plt.legend()
        plt.show()
        '''



def no_jit_signal_gen():
    num_samples=10
    signal_length=400

    frequencies = np.linspace(0,1,signal_length)
    for _ in range(num_samples):
        alphas = []
        zetas = []
        omegas = []
        H_v = np.zeros(signal_length,dtype=np.complex128)
        label = np.zeros(signal_length,dtype=int)
        
        num_modes = np.random.randint(1,5)
        #when noise is added change num_modes to include 0
        for n in range(num_modes):
            # to improve model add a random sign to alpha_j 
            alphas.append(np.random.uniform(1,2))
            zetas.append(scipy.stats.loguniform.rvs(0.01,0.1))
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
                #set label bandwidth to 3db or arbitrary value
                #bandwidth = 0.02 #(8 frequency points wide if 400 points)
                bandwidth = 2*omega_n*zeta_n
                label[(frequencies >= omega_n - bandwidth) & (frequencies <= omega_n + bandwidth)] = 1
            
            H_v[i] = H_f
            


print(timeit.timeit(jit_signal_gen,number=100))
#60s without jit, 1000 signals, 400points, 10 repeats
#27s with jit, 1000 signals, 400points, 100 repeats
#improvements of ~ 20x

