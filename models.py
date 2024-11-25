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


#can be used for one, two or more channel datasets
class n_channel_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        #convert signal to tensor with shape (channels, signal_length)
        signal = torch.tensor(self.data[idx], dtype=torch.float32)
        
        # If there's only one channel, unsqueeze to add a channel dimension
        if signal.ndim == 1:  # Check if signal is 1D
            signal = signal.unsqueeze(0)  # Add a channel dimension
        
        #convert label to tensor with shape (signal_length)
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


#New structure with up convolutional layers and no pooling
#very similar trainable parameters to PeakMag6 but more convolutional layers
#slightly narrower kernel size than PeakMag6
#No output FC layer
#Test performance of adding that layer on top of this model
#num parameters: 120,657

#Model currently does not train at all. why?
class NewModel1(nn.Module):
    def __init__(self, data_channels: int, input_length: int = 1024):
        super(NewModel1, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, 16, kernel_size=13, padding=6)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=13, padding=6)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=13, padding=6)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=13, padding=6)
        self.conv5 = nn.Conv1d(64, 32, kernel_size=13, padding=6)
        self.conv6 = nn.Conv1d(32, 16, kernel_size=13, padding=6)
        self.conv7 = nn.Conv1d(16, 1, kernel_size=13, padding=6)
        self.dropout = nn.Dropout(0.2) # 0.2 sensible default
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv7(x))
        #x = self.dropout(x) # no dropout on final layer
        
        x = torch.sigmoid(x)
        
        return x


#as PeakMag7 but with added convolutional layer
#num parameters: 358,800
class PeakMag8(nn.Module):
    def __init__(self, data_channels: int, input_length: int = 1024):
        super(PeakMag8, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, 16, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=21, padding=10)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=21, padding=10)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=21, padding=10)
        self.pool = nn.MaxPool1d(kernel_size=2) # default stride = kernel_size
        self.global_pool = nn.AdaptiveAvgPool1d(1) # Global average pooling
        self.fc1 = nn.Linear(128,input_length)
        self.dropout = nn.Dropout(0.2) # 0.2 sensible default
        
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
        
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = self.global_pool(x) # Output shape: [batch_size, 128, 1]
        x = x.squeeze(-1) # Remove the last dimension: [batch_size, 128, 1] -> [batch_size, 128]
        
        x = self.fc1(x)
        
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


#as PeakMag6 but with 2 FC layers to test impact of GAP vs PeakMag5
#num parameters: 2,285,840
class PeakMag7(nn.Module):
    def __init__(self, data_channels: int, input_length: int = 1024):
        super(PeakMag7, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, 16, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=21, padding=10,)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=21, padding=10)
        self.pool = nn.MaxPool1d(kernel_size=2) # default stride = kernel_size
        self.global_pool = nn.AdaptiveAvgPool1d(1) # Global average pooling
        self.fc1 = nn.Linear(64,2048)
        self.fc2 = nn.Linear(2048,input_length)
        self.dropout = nn.Dropout(0.2) # 0.2 sensible default
        
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
        
        x = self.global_pool(x) # Output shape: [batch_size, 64, 1]
        x = x.squeeze(-1) # Remove the last dimension: [batch_size, 64, 1] -> [batch_size, 64]
        
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


#big cut on number of parameters using GAP and 1FC layer
#num parameters: 121,104
class PeakMag6(nn.Module):
    def __init__(self, data_channels: int, input_length: int = 1024):
        super(PeakMag6, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, 16, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=21, padding=10,)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=21, padding=10)
        self.pool = nn.MaxPool1d(kernel_size=2) # default stride = kernel_size
        self.global_pool = nn.AdaptiveAvgPool1d(1) # Global average pooling
        self.fc1 = nn.Linear(64,input_length)
        self.dropout = nn.Dropout(0.2) # 0.2 sensible default
        
    def forward(self,x):
        #Input shape: [batch_size, data_channels, input_length]
        x = F.relu(self.conv1(x)) # Output shape: [batch_size, 16, input_length]
        x = self.dropout(x)
        x = self.pool(x) # Output shape: [batch_size, 16, input_length/2]
        
        x = F.relu(self.conv2(x)) # Output shape: [batch_size, 32, input_length/2]
        x = self.dropout(x)
        x = self.pool(x) # Output shape: [batch_size, 32, input_length/4]
        
        x = F.relu(self.conv3(x)) # Output shape: [batch_size, 64, input_length/4]
        x = self.dropout(x)
        x = self.pool(x) # Output shape: [batch_size, 64, input_length/8]
        
        x = self.global_pool(x) # Output shape: [batch_size, 64, 1]
        x = x.squeeze(-1) # Remove the last dimension: [batch_size, 64, 1] -> [batch_size, 64]
        
        x = self.fc1(x) # Output shape: [batch_size, input_length]
        
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


#as PeakMag4 but with flexible number of input channels
#num parameters:
class PeakMag5_small_kernel(nn.Module):
    def __init__(self, data_channels: int, input_length: int = 1024):
        super(PeakMag5_small_kernel, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64*128,2048) 
        self.fc2 = nn.Linear(2048,input_length) 
        self.dropout = nn.Dropout(0.3)
        
    def forward(self,x):
        #Input shape: [batch_size, data_channels, input_length]
        x = F.relu(self.conv1(x)) # Output shape: [batch_size, 16, input_length]
        x = self.dropout(x)
        x = self.pool(x) # Output shape: [batch_size, 16, input_length/2]
        
        x = F.relu(self.conv2(x)) # Output shape: [batch_size, 32, input_length/2]
        x = self.dropout(x)
        x = self.pool(x) # Output shape: [batch_size, 32, input_length/4]
        
        x = F.relu(self.conv3(x)) # Output shape: [batch_size, 64, input_length/4]
        x = self.dropout(x)
        x = self.pool(x) # Output shape: [batch_size, 64, input_length/8]
        
        x = x.view(-1, self.num_flat_features(x)) # Output shape: [batch_size, 64*input_length/8]
        
        x = F.relu(self.fc1(x)) # Output shape: [batch_size, 2048]
        
        x = self.dropout(x)
        
        x = self.fc2(x) # Output shape: [batch_size, input_length]
        
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


#as PeakMag4 but with flexible number of input channels
#num parameters:
class PeakMag5(nn.Module):
    def __init__(self, data_channels: int, input_length: int = 1024):
        super(PeakMag5, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, 16, kernel_size=21, padding=10)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=21, padding=10)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=21, padding=10)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64*128,2048) 
        self.fc2 = nn.Linear(2048,input_length) 
        self.dropout = nn.Dropout(0.3)
        
    def forward(self,x):
        #Input shape: [batch_size, data_channels, input_length]
        x = F.relu(self.conv1(x)) # Output shape: [batch_size, 16, input_length]
        x = self.dropout(x)
        x = self.pool(x) # Output shape: [batch_size, 16, input_length/2]
        
        x = F.relu(self.conv2(x)) # Output shape: [batch_size, 32, input_length/2]
        x = self.dropout(x)
        x = self.pool(x) # Output shape: [batch_size, 32, input_length/4]
        
        x = F.relu(self.conv3(x)) # Output shape: [batch_size, 64, input_length/4]
        x = self.dropout(x)
        x = self.pool(x) # Output shape: [batch_size, 64, input_length/8]
        
        x = x.view(-1, self.num_flat_features(x)) # Output shape: [batch_size, 64*input_length/8]
        
        x = F.relu(self.fc1(x)) # Output shape: [batch_size, 2048]
        
        x = self.dropout(x)
        
        x = self.fc2(x) # Output shape: [batch_size, input_length]
        
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

