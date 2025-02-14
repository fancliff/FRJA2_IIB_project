import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit
from typing import List, Union

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



class NewModelGeneral(nn.Module):
    def __init__(self, 
                data_channels: int, 
                out_channels: List[int], 
                kernel_size: List[int] = [13], 
                input_length: int = 1024,
                batch_norm: bool = True,
                P_dropout: float = 0.0,
                max_pool: bool = False # Max pooling after every layer
                ):
        """
        A CNN model with flexible output channels, batch normalization, 
        and kernel size/padding customization.
        
        :param data_channels: The number of input channels.
        :param input_length: The length of the input sequence.
        :param out_channels: List of output channel sizes for each Conv1d layer.
        :param kernel_size: Either an integer kernel size or a list of kernel sizes for each Conv1d layer.
        """
        super(NewModelGeneral, self).__init__()
        self.data_channels = data_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_length = input_length
        self.batch_norm = batch_norm
        self.P_dropout = P_dropout
        self.max_pool= max_pool

        if len(kernel_size) == 1:
            kernel_size = kernel_size * len(out_channels)
        else:
            assert len(kernel_size) == len(out_channels), \
                "The length of kernel_size list must be 1 or match the length of out_channels list."

        # Default to calculating padding as (kernel_size - 1) / 2 for each layer
        padding = [(k - 1) // 2 for k in kernel_size]

        # Define Conv1d and BatchNorm1d layers dynamically using the out_channels and kernel_size list
        self.convs = nn.ModuleList()  # to hold all convolution layers
        self.regs = nn.ModuleList()   # to hold all BatchNorm or dropout layers (regularisers)
        self.pools = nn.ModuleList()  # to hold all max pooling layers
        
        in_channels = data_channels  # The initial input channel is the number of input channels (data_channels)
        
        if max_pool:
            no_pool_idx = -1 # all layers have pooling if len out_channels is even
            if len(out_channels) % 2 == 1:
                no_pool_idx = len(out_channels) // 2 # middle layer no pooling if len out_channels is odd
        
        for i, out_channel in enumerate(out_channels):
            kernel = kernel_size[i]  # Get the kernel size for the current layer
            pad = padding[i]  # Get the padding value for the current layer
            
            if max_pool:
                if i == no_pool_idx:
                    self.pools.append(nn.Identity())  # No pooling for this layer
                else:
                    if i >= len(out_channels) // 2:
                        self.pools.append(nn.Upsample(scale_factor=2))  # Add an upsampling layer
                    else:
                        self.pools.append(nn.MaxPool1d(kernel_size=2))  # Add a max pooling layer
            else:
                self.pools.append(nn.Identity()) # No pooling for all layers

            self.convs.append(nn.Conv1d(in_channels, out_channel, kernel_size=kernel, padding=pad))
            
            if batch_norm:
                self.regs.append(nn.BatchNorm1d(out_channel))
            elif P_dropout > 0:
                self.regs.append(nn.Dropout(P_dropout))
            else:
                self.regs.append(nn.Identity())  # No regularisation
            
            in_channels = out_channel  # Update the input channels for the next convolution layer

    def forward(self, x):
        for conv, reg, pool in zip(self.convs[:-1], self.regs, self.pools):
            x = F.relu(reg(conv(x))) # Convolution -> Regularisation -> ReLU activation
            x = pool(x)  # Max pooling & Upsampling if max_pool is True
        
        x = self.convs[-1](x)  # Final conv layer without BN or ReLU
        x = torch.sigmoid(x)  # Sigmoid activation for output

        return x



#New structure with up convolutional layers and no pooling
#No output FC layer
#num parameters: 2,429

# Too deep? Not training at all? Vanishing Gradient?
# Try batch normalisation then try adding skip connections
class NewModel4(nn.Module):
    def __init__(self, data_channels: int, input_length: int = 1024):
        super(NewModel4, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, 4, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(4, 4, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(4, 6, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(6, 6, kernel_size=7, padding=3)
        self.conv5 = nn.Conv1d(6, 8, kernel_size=7, padding=3)
        self.conv6 = nn.Conv1d(8, 8, kernel_size=7, padding=3)
        self.conv7 = nn.Conv1d(8, 6, kernel_size=7, padding=3)
        self.conv8 = nn.Conv1d(6, 6, kernel_size=7, padding=3)
        self.conv9 = nn.Conv1d(6, 4, kernel_size=7, padding=3)
        self.conv10 = nn.Conv1d(4, 4, kernel_size=7, padding=3)
        self.conv11 = nn.Conv1d(4, 2, kernel_size=7, padding=3)
        self.conv12 = nn.Conv1d(2, 2, kernel_size=7, padding=3)
        self.conv13 = nn.Conv1d(2, 2, kernel_size=7, padding=3)
        self.conv14 = nn.Conv1d(2, 1, kernel_size=7, padding=3)
        self.dropout = nn.Dropout(0.1) # 0.1 so as not to reduce model power
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        
        x = torch.sigmoid(x)
        
        return x


#As NewModel3 but with batch normalisation
#num parameters: 2,197
class NewModel3b(nn.Module):
    def __init__(self, data_channels: int, input_length: int = 1024):
        super(NewModel3b, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, 4, kernel_size=13, padding=6)
        self.conv2 = nn.Conv1d(4, 4, kernel_size=13, padding=6)
        self.conv3 = nn.Conv1d(4, 8, kernel_size=13, padding=6)
        self.conv4 = nn.Conv1d(8, 8, kernel_size=13, padding=6)
        self.conv5 = nn.Conv1d(8, 4, kernel_size=13, padding=6)
        self.conv6 = nn.Conv1d(4, 2, kernel_size=13, padding=6)
        self.conv7 = nn.Conv1d(2, 1, kernel_size=13, padding=6)
        self.dropout = nn.Dropout(0.1) # 0.1 so as not to reduce model power
        self.bn1 = nn.BatchNorm1d(4)
        self.bn2 = nn.BatchNorm1d(4)
        self.bn3 = nn.BatchNorm1d(8)
        self.bn4 = nn.BatchNorm1d(8)
        self.bn5 = nn.BatchNorm1d(4)
        self.bn6 = nn.BatchNorm1d(2)
        
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        
        x = self.conv7(x)
        x = torch.sigmoid(x)
        
        return x



#New structure with up convolutional layers and no pooling
#No output FC layer
#num parameters: 2,137
class NewModel3(nn.Module):
    def __init__(self, data_channels: int, input_length: int = 1024):
        super(NewModel3, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, 4, kernel_size=13, padding=6)
        self.conv2 = nn.Conv1d(4, 4, kernel_size=13, padding=6)
        self.conv3 = nn.Conv1d(4, 8, kernel_size=13, padding=6)
        self.conv4 = nn.Conv1d(8, 8, kernel_size=13, padding=6)
        self.conv5 = nn.Conv1d(8, 4, kernel_size=13, padding=6)
        self.conv6 = nn.Conv1d(4, 2, kernel_size=13, padding=6)
        self.conv7 = nn.Conv1d(2, 1, kernel_size=13, padding=6)
        self.dropout = nn.Dropout(0.1) # 0.1 so as not to reduce model power
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv6(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = self.conv7(x)
        #x = self.dropout(x) # no dropout on final layer
        
        x = torch.sigmoid(x)
        
        return x


#New structure with up convolutional layers and no pooling
#No output FC layer
#num parameters: 7,701
class NewModel2(nn.Module):
    def __init__(self, data_channels: int, input_length: int = 1024):
        super(NewModel2, self).__init__()
        self.conv1 = nn.Conv1d(data_channels, 4, kernel_size=13, padding=6)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=13, padding=6)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=13, padding=6)
        self.conv4 = nn.Conv1d(16, 16, kernel_size=13, padding=6)
        self.conv5 = nn.Conv1d(16, 8, kernel_size=13, padding=6)
        self.conv6 = nn.Conv1d(8, 4, kernel_size=13, padding=6)
        self.conv7 = nn.Conv1d(4, 1, kernel_size=13, padding=6)
        self.dropout = nn.Dropout(0.1) # 0.1 so as not to reduce model power
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv6(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = self.conv7(x)
        #x = self.dropout(x) # no dropout on final layer
        
        x = torch.sigmoid(x)
        
        return x


#New structure with up convolutional layers and no pooling
#very similar trainable parameters to PeakMag6 but more convolutional layers
#slightly narrower kernel size than PeakMag6
#No output FC layer
#Test performance of adding that layer on top of this model
#num parameters: 120,657
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
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = F.relu(self.conv6(x))
        #x = self.dropout(x)
        #x = self.pool(x)
        
        x = self.conv7(x)
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

