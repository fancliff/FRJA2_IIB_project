import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit


def train_model_binary(model, train_dataloader, val_dataloader, save_name, num_epochs, acceptance, plotting=True):
    
    criterion = nn.BCELoss() 
    #No longer need BCE with logits.
    #Applying sigmoid at the end of the model instead.
    #BCE with logits may be more numerically stable
    #if issues arise can switch back easily
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    
    if plotting:
        # Initialize the plot
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.set_yscale('log')
        train_loss_line, = ax.plot([], [], label='Training Loss', color='blue')
        val_loss_line, = ax.plot([], [], label='Validation Loss', color='orange')
        ax.legend()
    
    result_dict = {
        "training_loss": [],
        "training_precision": [],
        "training_recall": [],
        "validation_loss": [],
        "validation_precision": [],
        "validation_recall": [],
        "epochs": []}
    
    for epoch in range(num_epochs):
        model.train() #!!!
        train_loss, train_recall, train_precision = training_step(model, train_dataloader, criterion, optimiser, acceptance)
        
        model.eval() #!!!
        val_loss, val_recall, val_precision = validation_loss_recall_precision(model, val_dataloader, criterion, acceptance)
        
        print()
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
        
        if plotting:
            # Update the plot
            train_loss_line.set_xdata(result_dict["epochs"])
            train_loss_line.set_ydata(result_dict["training_loss"])
            val_loss_line.set_xdata(result_dict["epochs"])
            val_loss_line.set_ydata(result_dict["validation_loss"])
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
    
    print()
    print('Finished Training')
    
    #save the model if a save name is provided
    save_name is not None and save_model(model, save_name)
    
    if plotting:
        plt.ioff()
        plt.show()
    
    return result_dict


def training_step(model, dataloader, criterion, optimiser, acceptance):
    #remember to set model to training mode 
    #before running this function
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_samples = 0
    
    for data, labels in dataloader:
        optimiser.zero_grad()
        outputs = model(data).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimiser.step()
        
        batch_precision, batch_recall = calculate_precision_recall_binary(outputs, labels, acceptance)
        loss = criterion(outputs, labels.float())
        total_loss += loss.item() * len(data)
        total_precision += batch_precision * len(data)
        total_recall += batch_recall * len(data)
        total_samples += len(data)
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
    total_samples = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            outputs = model(data).squeeze()
            batch_precision, batch_recall = calculate_precision_recall_binary(outputs, labels, acceptance)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item() * len(data)
            total_precision += batch_precision * len(data)
            total_recall += batch_recall * len(data)
            total_samples += len(data)
    avg_loss = total_loss / total_samples
    avg_recall = total_recall / total_samples
    avg_precision = total_precision / total_samples
    
    #recall is total correct positive predictions/total positive labels
    #precision is total correct positive predictions/total positive predictions
    return avg_loss, avg_recall, avg_precision


def plot_loss_history(results, log_scale=True):
    plt.figure(figsize=(8, 4))
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if log_scale:
        plt.yscale('log')
    
    for i, result_dict in enumerate(results):
        plt.plot(result_dict["epochs"], result_dict["training_loss"], label=f"Model {i+1}: Training Loss")
        plt.plot(result_dict["epochs"], result_dict["validation_loss"], label=f"Model {i+1} Validation Loss")

    plt.legend()
    plt.show()


def plot_precision_history(results, log_scale=True):
    plt.figure(figsize=(8, 4))
    plt.title('Training and Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    if log_scale:
        plt.yscale('log')
    
    for i, result_dict in enumerate(results):
        plt.plot(result_dict["epochs"], result_dict["training_precision"], label=f"Model {i+1}: Training Precision")
        plt.plot(result_dict["epochs"], result_dict["validation_precision"], label=f"Model {i+1} Validation Precision")

    plt.legend()
    plt.show()


def plot_recall_history(results, log_scale=True):
    plt.figure(figsize=(8, 4))
    plt.title('Training and Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    if log_scale:
        plt.yscale('log')
    
    for i, result_dict in enumerate(results):
        plt.plot(result_dict["epochs"], result_dict["training_recall"], label=f"Model {i+1}: Training Recall")
        plt.plot(result_dict["epochs"], result_dict["validation_recall"], label=f"Model {i+1} Validation Recall")

    plt.legend()
    plt.show()


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


def plot_samples(dataloader, num_samples):
    samples_plotted = 0
    for data, labels in dataloader:
        for i in range(min(num_samples, len(data))):
            plt.figure(figsize=(12, 6))
            plt.plot(data[i].squeeze().cpu().numpy(), label="Signal", color="blue")
            plt.plot(labels[i].cpu().numpy() * 2, label="Labels (scaled)", linestyle="--", color="green")
            plt.title(f'Sample {i+1}')
            plt.xlabel('Frequency (Normalized)')
            plt.ylabel('Amplitude / Label')
            plt.legend()
            plt.show()
            samples_plotted += 1
            if samples_plotted >= num_samples:
                return  # Exit after plotting specified number of samples


def load_model(save_name):
    project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/Models/'
    model_path = project_path + save_name
    model = torch.load(f'{model_path}.pth')
    return model


def save_model(model, save_name):
    project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/Models/'
    model_path = project_path + save_name
    torch.save(model, f'{model_path}.pth')
    print(f'Model saved to {model_path}.pth')


def compare_models(model1, model2, dataloader, criterion, acceptance1, acceptance2):
    model1.eval()
    model2.eval()
    loss1, recall1, precision1 = validation_loss_recall_precision(model1, dataloader, criterion, acceptance1)
    loss2, recall2, precision2 = validation_loss_recall_precision(model2, dataloader, criterion, acceptance2)
    print('Model 1:')
    print(f'Loss: {loss1:.4f}, Precision: {precision1:.4f}, Recall: {recall1:.4f}\n')
    print('Model 2:')
    print(f'Loss: {loss2:.4f}, Precision: {precision2:.4f}, Recall: {recall2:.4f}\n')


def calculate_precision_recall_binary(outputs, labels, acceptance):
    #only works if labels are binary

    predicted = (outputs > acceptance).float()
    true_positives = (predicted * labels).sum().item()

    total_predictions = predicted.sum().item() # Total positive predictions
    total_labels = labels.sum().item() # Total positive labels

    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_labels if total_labels > 0 else 0

    return precision, recall


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



