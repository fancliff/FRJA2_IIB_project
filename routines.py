import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit



def train_model_binary(model, train_dataloader, val_dataloader, num_epochs, acceptance):
    
    criterion = nn.BCEWithLogitsLoss() #combines sigmoid and BCE loss
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    
    result_dict = {
        "training_loss": [],
        "training_precision": [],
        "training_recall": [],
        "validation_loss": [],
        "validation_precision": [],
        "validation_recall": [],
        "epochs": []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_recall, train_precision = training_step(model, train_dataloader, criterion, optimiser, acceptance)
        
        model.eval()
        val_loss, val_recall, val_precision = validation_loss_recall_precision(model, val_dataloader, criterion, acceptance)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Training Precision: {train_precision:.4f}, Training Recall: {val_recall:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}')
        
        result_dict["training_loss"].append(train_loss)
        result_dict["training_precision"].append(train_precision)
        result_dict["training_recall"].append(train_recall)
        result_dict["validation_loss"].append(val_loss)
        result_dict["validation_precision"].append(val_precision)
        result_dict["validation_recall"].append(val_recall)
        result_dict["epochs"].append(epoch + 1)
        
    print('Finished Training')
    
    save_model(model, 'PeakMag1')
    
    return result_dict



def training_step(model, dataloader, criterion, optimiser, acceptance):
    #remember to set model to eval mode 
    #before running this function IF using validation data
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_samples = len(dataloader)
    
    for data, labels in dataloader:
        optimiser.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        
        outputs = outputs.squeeze()
        probabilities = torch.sigmoid(outputs)
        batch_precision, batch_recall = calculate_precision_recall_binary(probabilities, labels, acceptance)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * len(data)
        total_precision += batch_precision * len(data)
        total_recall += batch_recall * len(data)
    avg_loss = total_loss / total_samples
    avg_recall = total_recall / total_samples
    avg_precision = total_precision / total_samples
    
    #recall is total correct positive predictions/total positive labels
    #precision is total correct positive predictions/total positive predictions
    return avg_loss, avg_recall, avg_precision

def validation_loss_recall_precision(model, dataloader, criterion, acceptance):
    #remember to set model to eval mode 
    #before running this function IF using validation data
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_samples = len(dataloader)
    
    with torch.no_grad():
        for data, labels in dataloader:
            outputs = model(data).squeeze()
            probabilities = torch.sigmoid(outputs)
            batch_precision, batch_recall = calculate_precision_recall_binary(probabilities, labels, acceptance)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(data)
            total_precision += batch_precision * len(data)
            total_recall += batch_recall * len(data)
    avg_loss = total_loss / total_samples
    avg_recall = total_recall / total_samples
    avg_precision = total_precision / total_samples
    
    #recall is total correct positive predictions/total positive labels
    #precision is total correct positive predictions/total positive predictions
    return avg_loss, avg_recall, avg_precision


def plot_predictions(model, dataloader, num_samples, acceptance):
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

def compare_models(model1, model2, dataloader, acceptance):
    model1.eval()
    model2.eval()
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_precision1 = 0.0
    total_precision2 = 0.0
    total_recall1 = 0.0
    total_recall2 = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, labels in dataloader:
            output1 = model1(data).squeeze()
            output2 = model2(data).squeeze()
            probabilities1 = torch.sigmoid(output1)
            probabilities2 = torch.sigmoid(output2)
            batch_precision, batch_recall = calculate_precision_recall_binary(probabilities1, labels, acceptance)
            batch_precision, batch_recall = calculate_precision_recall_binary(probabilities2, labels, acceptance)
            
            total_loss += loss.item() * len(data)
            total_precision += batch_precision * len(data)
            total_recall += batch_recall * len(data)
            total_samples += len(data)
    accuracy1 = true_positives1/total_labels
    accuracy2 = true_positives2/total_labels
    print(f'Model 1 Accuracy: {accuracy1:.4f}')
    print(f'Model 2 Accuracy: {accuracy2:.4f}')

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