import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython=True)
def generate_data(num_samples=1000, signal_length=1024):
    
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
        
        for n in range(num_modes):
            # to improve model add a random sign to alpha_j 
            alpha_n = alphas[n]
            zeta_n = zetas[n]
            omega_n = omegas[n]
            
            for j, w in enumerate(frequencies):
                H_f = 0.0j
                
                denominator = omega_n**2 - w**2 + 2j * zeta_n * w
                numerator = 1j*w*alpha_n
                
                H_f += numerator/denominator
                H_v[j] += H_f
            
            #set label bandwidth to 3db or arbitrary value
            #bandwidth = 0.02 #(8 frequency points wide if 400 points)
            bandwidth = 2*omega_n*zeta_n
            label[(frequencies >= omega_n - bandwidth) & (frequencies <= omega_n + bandwidth)] = 1
        
        data[i, :] = np.abs(H_v) #use signal magnitude for now
        labels[i, :] = label
        
    return data, labels

class simple_dataset(Dataset):
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

class PeakMagCNN(nn.Module):
    def __init__(self):
        super(PeakMagCNN, self).__init__()
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
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Generate vibration data
data, labels = generate_data(num_samples=1000, signal_length=1024)

# Visualize a sample signal with labels
plt.figure(figsize=(12, 6))
plt.plot(data[0], label='Signal')
plt.plot(labels[0] * 5, label='Labels (scaled)', linestyle='--')  # Scale labels for visibility
plt.title('Sample Vibration Signal with Labels')
plt.xlabel('Normalised Frequency')
plt.ylabel('Amplitude / Label')
plt.legend()
plt.show()

dataset = simple_dataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PeakMagCNN()
criterion = nn.BCEWithLogitsLoss() #combines sigmoid and BCE loss
optimiser = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, dataloader, num_epochs = 10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data,labels in dataloader:
            optimiser.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
        
        avg_loss = running_loss/len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
        evaluate_model(model, dataloader, acceptance=0.5)
    print('Finished Training')
    
def evaluate_model(model, dataloader, acceptance=0.5):
    model.eval()
    total_correct = 0
    total_labels = 0
    with torch.no_grad():
        for data, labels in dataloader:
            predictions = model(data).squeeze()
            probabilities = torch.sigmoid(predictions)
            predicted_labels = (probabilities >= acceptance).float()
            correct = (predicted_labels == labels).float().sum()
            total_correct += correct.item()
            total_labels += labels.numel()
    accuracy = total_correct/total_labels
    print(f'Accuracy: {accuracy:.4f}')
    
def plot_predictions(model, dataloader, num_samples=5, acceptance=0.5):
    model.eval()
    samples_plotted = 0
    with torch.no_grad():
        for data, labels in dataloader:
            predictions = model(data).squeeze()
            probabilities = torch.sigmoid(predictions)
            predicted_labels = (probabilities >= acceptance).float()

            for i in range(min(num_samples, len(data))):
                plt.figure(figsize=(12, 6))
                
                plt.plot(data[i].squeeze().cpu().numpy(), label="Signal", color="blue")
                plt.plot(labels[i].cpu().numpy() * 5, label="Actual Labels (scaled)", linestyle="--", color="green")
                plt.plot(probabilities[i].cpu().numpy() * 10, label="Prediction Probability (scaled)", color="orange")
                plt.plot(predicted_labels[i].cpu().numpy() * 5, label="Predicted Labels (scaled)", linestyle=":", color="red")

                plt.title(f'Sample {i+1}')
                plt.xlabel('Frequency (Normalized)')
                plt.ylabel('Amplitude / Label')
                plt.legend()
                plt.show()

                samples_plotted += 1
                if samples_plotted >= num_samples:
                    return  # Exit after plotting specified number of samples

train_model(model, dataloader, num_epochs=10)

project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project'
save_path = '/Models/simple_CNN_magnitude_' + str(1) + '.pth'
torch.save(model.state_dict(), project_path+save_path)
print(f'Model saved to {save_path}')

#load an old model for evaluation 
#can create a new model then load to compare 2 versions
#load_path = '/Models/simple_CNN_magnitude_1.pth'
#model.load_state_dict(torch.load(project_path+load_path,weights_only=True))

#generate new random data for model evaluation
data, labels = generate_data(num_samples=1000, signal_length=1024)
dataset = simple_dataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#evaluate_model(model, data, labels, acceptance=0.5)
plot_predictions(model, dataloader, num_samples=10, acceptance=0.5)