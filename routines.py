import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit
from models import EarlyStopping
import matplotlib.patches as mpatches #for legend using masks
import matplotlib.colors as mcolors #for custom colormap
import datetime #for timestamping save files


def train_model_binary(model, train_dataloader, val_dataloader, save_suffix, num_epochs, acceptance, plotting=True, patience=4):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    criterion = nn.BCELoss() 
    #No longer need BCE with logits.
    #Applying sigmoid at the end of the model instead.
    #BCE with logits may be more numerically stable
    #if issues arise can switch back easily
    optimiser = optim.Adam(model.parameters(), lr=0.001) 
    #Consider L2 regularisation with e.g weight_decay=1e-5
    
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
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print('Early stopping triggered, stopping training...')
            #early_stopping.load_checkpoint(model)
            break
    
    print()
    print('Finished Training')
    
    #load the best model using EarlyStopping class 
    #do this regardless of whether training was stopped early
    early_stopping.load_checkpoint(model)
    
    #save the model if a save name is provided
    save_suffix is not None and save_model(model, save_suffix)
    
    if plotting:
        plt.ioff()
        plt.show()
    
    return result_dict, model


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


def plot_loss_history(results, log_scale=True, show=False):
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
    plt.savefig('C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/Figures/current/loss_plot.png')
    if show: plt.show()


def plot_precision_history(results, log_scale=False, show=False):
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
    plt.savefig('C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/Figures/current/precision_plot.png')
    if show: plt.show()


def plot_recall_history(results, log_scale=False, show=False):
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
    plt.savefig('C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/Figures/current/recall_plot.png')
    if show: plt.show()


def plot_predictions(models, dataloader, num_samples, acceptance):
    samples_plotted = 0
    with torch.no_grad():
        for data, labels in dataloader:
            batch_size = len(data)
            for i in range(min(num_samples,batch_size)):
                x = np.linspace(0, 1, len(data[i][0]))
                for model_idx, model in enumerate(models):
                    model.eval()
                    probabilities = model(data).squeeze()
                    predicted_labels = (probabilities >= acceptance).float()
                    num_channels = data[i].shape[0]
                    fig, axes = plt.subplots(num_channels, 1, figsize=(8, 4), sharex=True)
                    if num_channels == 1:
                        axes = [axes]  # Convert single axis to list for iteration

                    for j in range(num_channels):
                        labels_arr = labels[i].cpu().numpy()
                        signal_arr = data[i][j].cpu().numpy()
                        prob_arr = probabilities[i].cpu().numpy()
                        predictions_arr = predicted_labels[i].cpu().numpy()

                        axes[j].plot(x, signal_arr, label="Signal", color="blue")
                        axes[j].plot(x, prob_arr, label="Prediction Probability", color="orange")

                        # Masked signal line for predicted labels
                        masked_labels = np.where(predictions_arr == 1, signal_arr, np.nan)
                        axes[j].plot(
                            x, masked_labels,
                            color='red',
                            label='Predicted Labels',
                            alpha=0.5,
                            linewidth=5,
                        )

                        # Plot the actual labels as a semi-transparent mask
                        mask = np.zeros_like(labels_arr)
                        mask[labels_arr == 1] = 1
                        axes[j].imshow(
                            mask.reshape(1, -1),
                            cmap='Greys',
                            extent=(0, 1, axes[j].get_ylim()[0], axes[j].get_ylim()[1]),
                            aspect='auto',
                            alpha=0.15,
                        )
                    
                    plt.tight_layout()
                    plt.show()
                
                samples_plotted += 1
                if samples_plotted >= num_samples:
                    return # Exit if enough samples are plotted


def plot_samples(dataloader, num_samples):
    samples_plotted = 0
    for data, labels in dataloader:
        for i in range(min(num_samples, len(data))):
            num_channels = data[i].shape[0]
            fig, axes = plt.subplots(num_channels, 1, figsize=(12, 6), sharex=True)
            if num_channels == 1:
                axes = [axes] # Convert single axis to list for iteration
            
            x = np.linspace(0,1,len(data[i][0]))
            
            for j in range(num_channels):
                labels_arr = labels[i].cpu().numpy()
                axes[j].plot(x,data[i][j].cpu().numpy(), label="Signal", color="blue")
                #axes[j].plot(labels_arr, label="Actual Labels", linestyle="--", color="green")
                #axes[j].set_ylabel('Amplitude / Label')
                
                # Define explicit colors for each label
                label_colors = {
                    1: (0.1, 0.6, 0.1, 0.4),  # Green for label 1
                    2: (0.6, 0.1, 0.1, 0.4),  # Red for label 2
                    3: (0.1, 0.1, 0.6, 0.4),  # Blue for label 3
                    4: (0.6, 0.6, 0.1, 0.4),  # Yellow for label 4
                    5: (0.6, 0.1, 0.6, 0.4),  # Purple for label 5
                }
                mask_patches = []
                # Overlay the masks
                for class_label in np.unique(labels_arr):
                    if class_label == 0:
                        continue
                    mask = np.zeros_like(labels_arr)
                    mask[labels_arr == class_label] = 1
                    label_alpha = label_colors[class_label][3]  # Use transparency from the RGBA tuple
                    label_color = label_colors[class_label]  # Set explicit color
                    label_cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", label_color])
                    
                    axes[j].imshow(
                        mask.reshape(1, -1),
                        cmap=label_cmap,  # Disable automatic colormap
                        extent=(0, 1, axes[j].get_ylim()[0], axes[j].get_ylim()[1]),
                        aspect='auto',
                        alpha=label_alpha,  # Use transparency from the RGBA tuple
                    )
                    
                    #add a proxy legend element for the mask for labels other than 0
                    if class_label != 0:
                        mask_patch = mpatches.Patch(color=label_color, alpha=label_alpha, label=f'Modes: {class_label}')
                        mask_patches.append(mask_patch)
                
                #axes[j].set_title(f'Channel {j+1}')
                #axes[j].set_ylabel('Signal Amplitude')
                axes[j].legend(handles=axes[j].get_legend_handles_labels()[0] + mask_patches)

            #plt.suptitle(f'Sample {i+1}')
            #plt.xlabel('Frequency (Normalized)')
            plt.tight_layout()
            #plt.subplots_adjust(top=0.9) # Adjust suptitle position
            plt.show()
            
            samples_plotted += 1
            if samples_plotted >= num_samples:
                return  # Exit after plotting specified number of samples


def load_model(save_name):
    project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/Models/'
    model_path = project_path + save_name
    model = torch.load(f'{model_path}.pth')
    return model


def save_model(model, save_suffix):
    project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/Models/'
    now = datetime.datetime.now()
    model_path = project_path + now.strftime("%m_%d_%H_%M_%S") + save_suffix
    torch.save(model, f'{model_path}.pth')
    print(f'Model saved to {model_path}.pth')


def compare_models(model1, model2, dataloader1, dataloader2, criterion, acceptance1, acceptance2):
    model1.eval()
    model2.eval()
    loss1, recall1, precision1 = validation_loss_recall_precision(model1, dataloader1, criterion, acceptance1)
    loss2, recall2, precision2 = validation_loss_recall_precision(model2, dataloader2, criterion, acceptance2)
    fscore1 = 2 * (precision1 * recall1) / (precision1 + recall1)
    fscore2 = 2 * (precision2 * recall2) / (precision2 + recall2)
    print('Model 1:')
    print(f'Loss: {loss1:.4f}, Precision: {precision1:.4f}, Recall: {recall1:.4f}, F-score: {fscore1:.4f}\n')
    print('Model 2:')
    print(f'Loss: {loss2:.4f}, Precision: {precision2:.4f}, Recall: {recall2:.4f}, F-score: {fscore2:.4f}\n')


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


def calculate_total_receptive_field(model):
    #Assumes model is a sequence of layers
    total_receptive_field = 1
    for layer in model.modules():
        if isinstance(layer, nn.Conv1d):
            kernel_size = layer.kernel_size[0]
            stride = layer.stride[0]
            dilation = layer.dilation[0]
            total_receptive_field += (kernel_size - 1) * dilation
        if isinstance(layer, nn.MaxPool1d):
            kernel_size = layer.kernel_size
            stride = layer.stride
            total_receptive_field += kernel_size - 1
    return total_receptive_field


def visualise_activations(model, dataloader, num_samples):
    #Dictionary to store activations of each layer
    activations = {}
    
    #Register hooks for all layers to store activations
    for name, layer in model.named_modules():
        #Only register hooks for Conv1d layers (adjust as needed)
        if isinstance(layer, nn.Conv1d): 
            layer.register_forward_hook(
                lambda self, input, output, name=name: activations.update({name: output})
            )
    
    samples_plotted = 0
    for data, _ in dataloader:
        for i in range(min(num_samples, len(data))):
            signal = data[i].unsqueeze(0) # Add batch dimension
            model.eval()
            model(signal) # Forward pass to populate activations dictionary
            
            for name, activation in activations.items():
                activation = activation[0].detach().cpu().numpy() #First batch element
                num_channels = activation.shape[0]
                
                plt.figure(figsize=(10,5))
                for channel in range(num_channels):
                    plt.plot(activation[channel], label=f'Channel {channel+1}')
                    
                plt.title(f'Activations of {name} - Sample {samples_plotted+1}')
                plt.xlabel('Position')
                plt.ylabel('Activation')
                #plt.legend() #legend is quite large for many channels in conv layers
                plt.show()
            
            samples_plotted += 1
            if samples_plotted >= num_samples:
                return  # Exit after plotting specified number of samples


def visualise_activations_with_signal(model, dataloader, num_samples):
    #Dictionary to store activations of each layer
    activations = {}
    
    #Register hooks for all layers to store activations
    for name, layer in model.named_modules():
        #Only register hooks for Conv1d layers (adjust as needed)
        if isinstance(layer, nn.Conv1d): 
            layer.register_forward_hook(
                lambda self, input, output, name=name: activations.update({name: output})
            )
    
    samples_plotted = 0
    for data, labels in dataloader:
        for i in range(min(num_samples, len(data))):
            signal = data[i].unsqueeze(0) # Add batch dimension
            labels_arr = labels[i].cpu().numpy()
            model.eval()
            model(signal) # Forward pass to populate activations dictionary
            
            num_input_channels = data[i].shape[0]
            
            for name, activation in activations.items():
                activation = activation[0].detach().cpu().numpy() #First batch element
                num_layers = activation.shape[0]
                len_activation = activation.shape[1]
                num_subplots = num_input_channels + 1 # Add one for activations
                
                fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 3*num_subplots), sharex=True)
                if num_subplots == 1:
                    axes = [axes] # Convert single axis to list for iteration
                
                
                for k in range(num_layers):
                    #plot the activations of the current layer at the bottom
                    axes[-1].plot(activation[k])
                
                axes[-1].set_title(f'Activations of {name}')
                axes[-1].set_xlabel('Position')
                axes[-1].set_ylabel('Activation')
                
                for j in range(num_input_channels):
                    #plot the signal for the current input channel
                    #resized to match the length of the activations
                    signal_channel = data[i][j].cpu().numpy()
                    signal_channel = scipy.ndimage.zoom(signal_channel, len_activation/len(signal_channel))
                    axes[j].plot(signal_channel)
                    axes[j].set_title(f'Input channel {j+1}')
                    axes[j].set_xlabel('Position')
                    axes[j].set_ylabel('Amplitude')
                
                #plot labels as semi transparent mask
                labels_arr_resized = scipy.ndimage.zoom(labels_arr, len_activation/len(labels_arr))
                mask = np.ones_like(labels_arr_resized)
                mask[labels_arr_resized == 1] = 0
                for q in range(num_subplots): #exclude the activations plot
                    axes[q].imshow(
                        mask.reshape(1, -1),
                        cmap='Greys',
                        extent=(0, len_activation, axes[q].get_ylim()[0], axes[q].get_ylim()[1]),
                        aspect='auto',
                        alpha=0.15,
                    )
                
                plt.suptitle(f'Sample {samples_plotted+1}')
                plt.tight_layout()
                plt.subplots_adjust(top=0.9) # Adjust suptitle position
                plt.show()
            
            samples_plotted += 1
            if samples_plotted >= num_samples:
                return  # Exit after plotting specified number of samples


def compare_activations(model1, model2, dataloader, num_samples):
    #input models should have same structure
    
    #Dictionary to store activations of each layer
    activations1 = {}
    activations2 = {}
    
    #Register hooks for all layers to store activations
    for name, layer in model1.named_modules():
        #Only register hooks for Conv1d layers (adjust as needed)
        if isinstance(layer, nn.Conv1d): 
            layer.register_forward_hook(
                lambda self, input, output, name=name: activations1.update({name: output})
            )
    
    for name, layer in model2.named_modules():
        #Only register hooks for Conv1d layers (adjust as needed)
        if isinstance(layer, nn.Conv1d): 
            layer.register_forward_hook(
                lambda self, input, output, name=name: activations2.update({name: output})
            )
    
    samples_plotted = 0
    model1.eval()
    model2.eval()
    for data, _ in dataloader:
        for i in range(min(num_samples, len(data))):
            signal = data[i].unsqueeze(0) # Add batch dimension
            model1(signal) # Forward pass to populate activations dictionary
            model2(signal)
            
            for (name1, activation1), (name2, activation2) in zip(activations1.items(), activations2.items()):
                activation1 = activation1[0].detach().cpu().numpy() #First batch element
                activation2 = activation2[0].detach().cpu().numpy()
                num_channels1 = activation1.shape[0]
                num_channels2 = activation2.shape[0]
                
                fig,axes = plt.subplots(2,1,figsize=(12,6),sharex=True)
                for j in range(2):
                    if j==0:
                        axes[j].set_title(f'Model 1: Activations of {name1}')
                        num_channels = num_channels1
                        activations = activation1
                    else:
                        axes[j].set_title(f'Model 2: Activations of {name2}')
                        num_channels = num_channels2
                        activations = activation2
                    
                    for channel in range(num_channels):
                        axes[j].plot(activations[channel], label=f'Channel {channel+1}')
                    
                    axes[j].set_xlabel('Position')
                    axes[j].set_ylabel('Activation')
                plt.suptitle(f'Sample {samples_plotted+1}')
                #plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                plt.show()
            
            samples_plotted += 1
            if samples_plotted >= num_samples:
                return  # Exit after plotting specified number of samples


#add compare activations with signal function



