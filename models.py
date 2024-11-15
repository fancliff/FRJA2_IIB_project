import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit

class two_channel_dataset(Dataset):
    """
    data: numpy array or similar of shape (num_samples, 2, signal_length).
        2 represents two channels (real and imaginary parts by default).
    labels: numpy array or similar of shape (num_samples, signal_length).
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        #convert signal to tensor with shape (channels, signal_length)
        signal = torch.tensor(self.data[idx], dtype=torch.float32)
        #convert label to tensor with shape (signal_length)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return signal, label


class one_channel_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        #convert to tensor and add channel dimension to signal
        #convert both signal and label to float 32 datatypes
        signal = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return signal, label


class EarlyStopping:
    def __init__(self, 
                patience=4, 
                verbose=False, 
                delta=0, 
                path='C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/checkpoint.pth'
                ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, val_loss)
        elif val_loss < self.best_loss + self.delta:
            self.save_checkpoint(model, val_loss)
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def save_checkpoint(self, model, val_loss):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
    
    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))


#as PeakMag3 but with 2 input channels
class PeakMag4(nn.Module):
    def __init__(self, input_length: int = 1024):
        super(PeakMag4, self).__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=21, padding=10)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=21, padding=10)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64*128,2048) 
        self.fc2 = nn.Linear(2048,input_length) 
        self.dropout = nn.Dropout(0.3)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        x = torch.sigmoid(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    #more thought on this required
    #def compute_fc_input_size(self, input_length):


#wider kernel size than PeakMag1
#dropout of p=0.2 at each layer
class PeakMag3(nn.Module):
    def __init__(self):
        super(PeakMag3, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=21, padding=10)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=21, padding=10)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64*128,2048) 
        self.fc2 = nn.Linear(2048,1024) 
        self.dropout = nn.Dropout(0.3)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        x = torch.sigmoid(x)
        #convert to probabilities for binary classfication
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PeakMag2(nn.Module):
    #wider kernel size than PeakMag1
    def __init__(self):
        super(PeakMag2, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=21, padding=10)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=21, padding=10)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64*128,2048) 
        self.fc2 = nn.Linear(2048,1024) 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc1(x))
        
        #x = self.dropout(x)
        
        x = self.fc2(x)
        
        x = torch.sigmoid(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PeakMag1(nn.Module):
    def __init__(self):
        super(PeakMag1, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64*128,2048) #adjust based on output size after convolution and pooling
        self.fc2 = nn.Linear(2048,1024) #output size for a signal length of 400
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc1(x))
        
        #x = self.dropout(x)
        
        x = self.fc2(x)
        
        x = torch.sigmoid(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

