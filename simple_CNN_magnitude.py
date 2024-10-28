import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
from numba import jit, types

@jit(nopython=True)
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
                
            self.data.append(np.abs(H_v)) #use signal magnitude for now
            self.labels.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
class PeakMagCNN(nn.module):
    def __init__(self):
        super(PeakMagCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64*50,128) #adjust based on output size after convolution and pooling
        self.fc2 = nn.Linear(128,400) #output size for a signal length of 400
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
dataset = simple_dataset(num_samples=1000, signal_length=400)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PeakMagCNN()
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for signals,labels in dataloader:
        optimiser.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Finished Training')

