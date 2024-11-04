import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit

from data_gen import mag_1D_no_noise as generate_data
from models import simple_dataset, PeakMagCNN
from routines import train_model_binary, evaluate_model, plot_predictions, save_model, load_model

data, labels = generate_data(num_samples=1000, signal_length=1024)
dataset = simple_dataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

data, labels = generate_data(num_samples=100, signal_length=1024)
val_dataset = simple_dataset(data, labels)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

model = PeakMagCNN()

train_model_binary(model, dataloader, num_epochs=10)