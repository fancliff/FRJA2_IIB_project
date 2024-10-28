import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy

class simple_dataset(Dataset):
    def __init__(self, num_samples=100, signal_length=400):
        self.num_samples = num_samples
        self.signal_length = signal_length
        self.data = []
        self.labels = []
        
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
                    
                    #set label bandwidth to 3db or arbitrary value
                    #bandwidth = 0.02 #(8 frequency points wide if 400 points)
                    bandwidth = 2*omega_n*zeta_n
                    label[(frequencies >= omega_n - bandwidth) & (frequencies <= omega_n + bandwidth)] = 1
                    
                H_v[i] = H_f
                
            self.data.append(np.abs(H_v)) #use signal magnitude for now
            self.labels.append(label)