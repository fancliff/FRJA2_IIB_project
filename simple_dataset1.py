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
        
        for _ in range(num_samples):
            num_modes = np.random.randint(0,5)
            H_v = np.zeros(signal_length,dtype=np.complex128)
            for _ in range(num_modes):
                # to improve model add a random sign to alpha_j 
                alpha_j = np.random.uniform(1,2)
                zeta_j = scipy.stats.loguniform(0.01,0.2)
                omega_j = np.random.uniform(0,1)
                for w in range(1/signal_length):
                    print(w)
            