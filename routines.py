import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit



def train_model_binary(model, dataloader, num_epochs = 10):
    criterion = nn.BCEWithLogitsLoss() #combines sigmoid and BCE loss
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    
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
            output1 = model(data).squeeze()
            probabilities = torch.sigmoid(output1)
            calculate_precision_recall_binary(probabilities, labels, acceptance)
    accuracy = total_correct/total_labels
    print(f'Recall: {recall:.4f}')
    
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

def load_model(model_path):
    model = torch.load(f'{model_path}.pth')
    return model

def save_model(model, save_name):
    project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/Models'
    model_path = project_path + save_name
    torch.save(model, f'{model_path}.pth')
    print(f'Model saved to {model_path}.pth')
    return

def compare_models(model1, model2, dataloader, acceptance):
    
    
    return

def calculate_precision_recall_binary(outputs, labels, acceptance):
    #only works if labels are binary
    
    predicted = (outputs > acceptance).float()
    true_positives = (predicted * labels).sum().item()
    #false_positives = (predicted * (1 - labels)).sum().item()
    #false_negatives = ((1 - predicted) * labels).sum().item()
    total_predictions = predicted.sum().item() # Total positive predictions
    total_labels = labels.sum().item() # Total positive labels
    precision = ( true_positives / total_predictions
        if total_predictions > 0
        else 0)
    recall = ( true_positives / total_labels
        if total_labels > 0
        else 0 )
    return precision, recall