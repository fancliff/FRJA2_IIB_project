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
        train_loss, train_recall, train_precision = training_step_binary(model, train_dataloader, criterion, optimiser, acceptance)
        
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


def training_step_binary(model, dataloader, criterion, optimiser, acceptance):
    #remember to set model to training mode 
    #before running this function
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_samples = 0
    
    for data, labels, _ in dataloader:
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
        for data, labels, _ in dataloader:
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


def train_model_regression(model, train_dataloader, val_dataloader, save_suffix, num_epochs, criterion = nn.MSELoss(), plotting=True, patience=4):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    criterion = criterion
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
        "training_mse": [],
        "training_mae": [],
        "training_r2": [],
        "validation_loss": [],
        "validation_mse": [],
        "validation_mae": [],
        "validation_r2": [],
        "epochs": []}
    
    for epoch in range(num_epochs):
        model.train() #!!!
        train_loss, train_mse, train_mae, train_r2 = training_step_regression(model, train_dataloader, criterion, optimiser)
        
        model.eval() #!!!
        val_loss, val_mse, val_mae, val_r2 = validation_loss_regression(model, val_dataloader, criterion)
        
        print()
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Training MSE: {train_mse:.4f}, Training MAE: {train_mae:.4f}, Training R²: {train_r2:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation MSE: {val_mse:.4f}, Validation MAE: {val_mae:.4f}, Validation R²: {val_r2:.4f}')
        
        result_dict["training_loss"].append(train_loss)
        result_dict["training_mse"].append(train_mse)
        result_dict["training_mae"].append(train_mae)
        result_dict["training_r2"].append(train_r2)
        result_dict["validation_loss"].append(val_loss)
        result_dict["validation_mse"].append(val_mse)
        result_dict["validation_mae"].append(val_mae)
        result_dict["validation_r2"].append(val_r2)
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
    if save_suffix is not None:
        save_model(model, save_suffix)
    
    if plotting:
        plt.ioff()
        plt.show()
    
    return result_dict, model


def training_step_regression(model, dataloader, criterion, optimiser):
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    total_samples = 0
    
    for data, labels, _ in dataloader:
        optimiser.zero_grad()
        outputs = model(data).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimiser.step()
        mse, mae, r2 = calculate_regression_metrics(outputs, labels)
        total_loss += loss.item() * len(data)
        total_mse += mse * len(data)
        total_mae += mae * len(data)
        total_r2 += r2 * len(data)
        total_samples += len(data)
    
    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_r2 = total_r2 / total_samples
    
    return avg_loss, avg_mse, avg_mae, avg_r2


def validation_loss_regression(model, dataloader, criterion):
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data, labels, _ in dataloader:
            outputs = model(data).squeeze()
            loss = criterion(outputs, labels.float())
            mse, mae, r2 = calculate_regression_metrics(outputs, labels)
            total_loss += loss.item() * len(data)
            total_mse += mse * len(data)
            total_mae += mae * len(data)
            total_r2 += r2 * len(data)
            total_samples += len(data)
    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_r2 = total_r2 / total_samples
    
    return avg_loss, avg_mse, avg_mae, avg_r2


def calculate_regression_metrics(outputs, labels):
    outputs = outputs.float()
    labels = labels.float()
    
    mse = F.mse_loss(outputs, labels).item()
    mae = F.l1_loss(outputs, labels).item()
    var_labels = torch.var(labels).item()
    if var_labels == 0:
        r2 = 0
    else:
        r2 = 1 - mse / var_labels
    
    return mse, mae, r2


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
        for data, labels, params in dataloader:
            batch_size = len(data)
            for i in range(min(num_samples,batch_size)):
                x = np.linspace(0, 1, len(data[i][0]))
                omegas = params[i, :, 0].cpu().numpy() if params is not None else []
                omegas = omegas[~np.isnan(omegas)] # Remove NaN values
                
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
                        
                        # Plot the true omegas as vertical dashed lines
                        for k, omega in enumerate(omegas):
                            axes[j].axvline(x=omega, color='black', linestyle='--', label='Mode Frequency' if k==0 else '')
                        
                        # Plot predicted omegas as vertical dotted lines
                        predicted_omegas = est_nat_freq_binary(prob_arr, 
                                                            acceptance=acceptance,
                                                            method='midpoint',
                                                            bandwidth=0.04,
                                                            overlap_threshold=0.5)
                        for k, omega in enumerate(predicted_omegas):
                            axes[j].axvline(x=omega, color='cyan', linestyle=':', label='Predicted Mode Frequency' if k==0 else '')
                    
                    plt.tight_layout()
                    plt.show()
                
                samples_plotted += 1
                if samples_plotted >= num_samples:
                    return # Exit if enough samples are plotted


def plot_samples(dataloader, num_samples, binary_labels=True):
    samples_plotted = 0
    for data, labels, params in dataloader:
        for i in range(min(num_samples, len(data))):
            num_channels = data[i].shape[0]
            fig, axes = plt.subplots(num_channels, 1, figsize=(12, 6), sharex=True)
            if num_channels == 1:
                axes = [axes] # Convert single axis to list for iteration
            
            x = np.linspace(0,1,len(data[i][0]))
            omegas = params[i, :, 0].cpu().numpy() if params is not None else []
            omegas = omegas[~np.isnan(omegas)] # Remove NaN values
            
            for j in range(num_channels):
                labels_arr = labels[i].cpu().numpy()
                axes[j].plot(x,data[i][j].cpu().numpy(), label="Signal", color="blue")
                axes[j].plot(x,labels_arr, label="Actual Labels", linestyle="--", color="green")
                #axes[j].set_ylabel('Amplitude / Label')
                
                mask_patches = []
                if binary_labels:
                    # Define explicit colors for each label
                    label_colors = {
                        1: (0.1, 0.6, 0.1, 0.4),  # Green for label 1
                        2: (0.6, 0.1, 0.1, 0.4),  # Red for label 2
                        3: (0.1, 0.1, 0.6, 0.4),  # Blue for label 3
                        4: (0.6, 0.6, 0.1, 0.4),  # Yellow for label 4
                        5: (0.6, 0.1, 0.6, 0.4),  # Purple for label 5
                    }
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
                
                #Plot the omegas as vertical dashed lines
                for k, omega in enumerate(omegas):
                    axes[j].axvline(x=omega, color='black', linestyle='--', label='Mode Frequency' if k==0 else '')
                
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


def plot_step_predictions(models, dataloader, num_samples, threshold=0.5):
    samples_plotted = 0
    with torch.no_grad():
        for data, labels, params in dataloader:
            batch_size = len(data)
            for i in range(min(num_samples,batch_size)):
                x = np.linspace(0, 1, len(data[i][0]))
                omegas = params[i, :, 0].cpu().numpy() if params is not None else []
                omegas = omegas[~np.isnan(omegas)] # Remove NaN values
                
                for model_idx, model in enumerate(models):
                    model.eval()
                    model_output = model(data).squeeze()
                    num_channels = data[i].shape[0]
                    fig, axes = plt.subplots(num_channels, 1, figsize=(8, 4), sharex=True)
                    if num_channels == 1:
                        axes = [axes]  # Convert single axis to list for iteration

                    for j in range(num_channels):
                        labels_arr = labels[i].cpu().numpy()
                        signal_arr = data[i][j].cpu().numpy()
                        fitted_curve = model_output[i].cpu().numpy()
                        floored_curve = (np.floor(fitted_curve + threshold)).astype(int)
                        predicted_omegas = []
                        for step_level in range(1, max(floored_curve)+1):
                            for k in range(len(floored_curve)):
                                if floored_curve[k] == step_level:
                                    predicted_omegas.append(x[k])
                                    break

                        axes[j].plot(x, signal_arr, label="Signal", color="blue")
                        axes[j].plot(x, fitted_curve, label="Predicted curve", color="orange")
                        axes[j].plot(x, labels_arr, label="Actual Labels", linestyle="--", color="green")
                        
                        # Plot the true omegas as vertical dashed lines
                        for k, omega in enumerate(omegas):
                            axes[j].axvline(x=omega, color='black', linestyle='--', label='Mode Frequency' if k==0 else '')
                        
                        for k, omega in enumerate(predicted_omegas):
                            axes[j].axvline(x=omega, color='cyan', linestyle=':', label='Predicted Mode Frequency' if k==0 else '')
                        
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.show()
                
                samples_plotted += 1
                if samples_plotted >= num_samples:
                    return # Exit if enough samples are plotted


def plot_triangle_predictions(models, dataloader, num_samples, N=2, Wn=0.2):
    samples_plotted = 0
    with torch.no_grad():
        for data, labels, params in dataloader:
            batch_size = len(data)
            for i in range(min(num_samples,batch_size)):
                x = np.linspace(0, 1, len(data[i][0]))
                omegas = params[i, :, 0].cpu().numpy() if params is not None else []
                omegas = omegas[~np.isnan(omegas)] # Remove NaN values
                
                for model_idx, model in enumerate(models):
                    model.eval()
                    model_output = model(data).squeeze()
                    num_channels = data[i].shape[0]
                    fig, axes = plt.subplots(num_channels, 1, figsize=(8, 4), sharex=True)
                    if num_channels == 1:
                        axes = [axes]  # Convert single axis to list for iteration

                    for j in range(num_channels):
                        labels_arr = labels[i].cpu().numpy()
                        signal_arr = data[i][j].cpu().numpy()
                        fitted_curve = model_output[i].cpu().numpy()
                        b,a = scipy.signal.butter(N,Wn)
                        smoothed_curve = scipy.signal.filtfilt(b,a,fitted_curve)

                        axes[j].plot(x, signal_arr, label="Signal", color="blue")
                        axes[j].plot(x, fitted_curve, label="Predicted curve", color="orange")
                        axes[j].plot(x, smoothed_curve, label="Smoothed curve", color="red")
                        axes[j].plot(x, labels_arr, label="Actual Labels", linestyle="--", color="green")
                        
                        # Plot the true omegas as vertical dashed lines
                        for k, omega in enumerate(omegas):
                            axes[j].axvline(x=omega, color='black', linestyle='--', label='Mode Frequency' if k==0 else '')
                        
                        predicted_omegas = est_nat_freq_triangle_rise(smoothed_curve, up_inc=0.5)
                        for k, omega in enumerate(predicted_omegas):
                            axes[j].axvline(x=omega, color='cyan', linestyle=':', label='Predicted Mode Frequency' if k==0 else '')
                        
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.show()
                
                samples_plotted += 1
                if samples_plotted >= num_samples:
                    return # Exit if enough samples are plotted


def plot_predictions_all_labels(models, dataloader, num_samples, label_defs, N=2, Wn=0.2):
    samples_plotted = 0
    with torch.no_grad():
        for data, labels, params in dataloader:
            assert len(label_defs) == len(labels[0]), "Number of label definitions must match number of labels"
            batch_size = len(data)
            for i in range(min(num_samples,batch_size)):
                x = np.linspace(0, 1, len(data[i][0]))
                omegas = params[i, :, 0].cpu().numpy() if params is not None else []
                omegas = omegas[~np.isnan(omegas)] # Remove NaN values
                
                if models is None: # Plot just sample and labels
                    num_data_channels = data[i].shape[0] 
                    num_label_channels = labels[i].shape[0] # Add subfigure for each label
                    num_channels = num_data_channels + num_label_channels
                    fig, axes = plt.subplots(num_channels, 1, figsize=(8, 4), sharex=True)

                    for j in range(num_data_channels):
                        signal_arr = data[i][j].cpu().numpy()
                        axes[j].plot(x, signal_arr, label="Signal", color="blue")
                        
                        # Plot the true omegas as vertical dashed lines
                        for k, omega in enumerate(omegas):
                            axes[j].axvline(x=omega, color='black', linestyle='--', label='Mode Frequency' if k==0 else '')
                        
                        axes[j].legend()
                    
                    j = 0 # Reset channel index for labels
                    if label_defs[0]: # Mode triangle labelling
                        subplot_labels(axes[j+num_data_channels], x, None, labels[i][j], "modes", N, Wn)
                        j += 1
                    if label_defs[1]: # Amplitude
                        subplot_labels(axes[j+num_data_channels], x, None, labels[i][j], "amplitude", N, Wn)
                        j += 1
                    if label_defs[2]: # Log10_zeta
                        subplot_labels(axes[j+num_data_channels], x, None, labels[i][j], "log10_zeta", N, Wn)
                        j += 1
                    if label_defs[3]: # omega_n
                        subplot_labels(axes[j+num_data_channels], x, None, labels[i][j], "omega_n", N, Wn)
                        j += 1
                    
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.show()
                
                else:
                    for model_idx, model in enumerate(models): # Plot predictions
                        model.eval()
                        model_output = model(data).squeeze()
                        num_data_channels = data[i].shape[0] 
                        num_label_channels = labels[i].shape[0] # Add subfigure for each label
                        num_channels = num_data_channels + num_label_channels
                        fig, axes = plt.subplots(num_channels, 1, figsize=(8, 4), sharex=True)

                        for j in range(num_data_channels):
                            signal_arr = data[i][j].cpu().numpy()
                            axes[j].plot(x, signal_arr, label="Signal", color="blue")
                            
                            # Plot the true omegas as vertical dashed lines
                            for k, omega in enumerate(omegas):
                                axes[j].axvline(x=omega, color='black', linestyle='--', label='Mode Frequency' if k==0 else '')
                            
                            axes[j].legend()
                        
                        j = 0 # Reset channel index for labels
                        if label_defs[0]: # Mode triangle labelling
                            subplot_labels(axes[j+num_data_channels], x, model_output[i][j], labels[i][j], "modes", N, Wn)
                            j += 1
                        if label_defs[1]: # Amplitude
                            subplot_labels(axes[j+num_data_channels], x, model_output[i][j], labels[i][j], "amplitude", N, Wn)
                            j += 1
                        if label_defs[2]: # Log10_zeta
                            subplot_labels(axes[j+num_data_channels], x, model_output[i][j], labels[i][j], "log10_zeta", N, Wn)
                            j += 1
                        if label_defs[3]: # omega_n
                            subplot_labels(axes[j+num_data_channels], x, model_output[i][j], labels[i][j], "omega_n", N, Wn)
                            j += 1
                        
                        plt.legend()
                        
                        plt.tight_layout()
                        plt.show()
                
                samples_plotted += 1
                if samples_plotted >= num_samples:
                    return # Exit if enough samples are plotted


def subplot_labels(axes_j, x, model_output_i_j, labels_arr_i_j, name, N, Wn):
    handles, labels = axes_j.get_legend_handles_labels() # Get existing legend handles and labels

    if model_output_i_j is not None:
        fitted_curve = model_output_i_j.cpu().numpy()
        b,a = scipy.signal.butter(N,Wn)
        smoothed_curve = scipy.signal.filtfilt(b,a,fitted_curve)
        
        h1, = axes_j.plot(x, fitted_curve, label=f"Predicted {name}", color="orange")
        h2, = axes_j.plot(x, smoothed_curve, label=f"Smoothed {name}", color="red")
        handles.extend([h1, h2])
        labels.extend([f"Predicted {name}", f"Smoothed {name}"])
        
        if name == 'modes':
            predicted_omegas = est_nat_freq_triangle_rise(smoothed_curve, up_inc=0.5)
            for k, omega in enumerate(predicted_omegas):
                h3 = axes_j.axvline(x=omega, color='cyan', linestyle=':', label='Predicted Mode Frequency' if k == 0 else '')
                if k == 0:  # Only add one label to avoid duplicates
                    handles.append(h3)
                    labels.append('Predicted Mode Frequency')

    h4, = axes_j.plot(x, labels_arr_i_j, label=f"Actual {name} Labels", linestyle="--", color="green")
    handles.append(h4)
    labels.append(f"Actual {name} Labels")

    axes_j.legend(handles, labels) # Update the legend


def est_nat_freq_binary(prob_arr, acceptance=0.5, method='midpoint', bandwidth=None, overlap_threshold=0.5):
    """
    Estimate natural frequencies from a PyTorch CNN model.

    Args:
        prob_arr (np.array): Array of predicted probabilities for a single signal.
        acceptance (float): Probability threshold for accepting a frequency.
        method (str): Method for estimating frequencies ('midpoint', 'peak', or 'confidence').
        bandwidth (float or None): Fixed bandwidth around each mode. If None, no bandwidth-based overlap detection is performed.
        overlap_threshold (float): Percentage of the bandwidth that must be exceeded to assume modal overlap.
    Returns:
        np.array: numpy array containing estimated natural frequencies for each batch.
    """
    freq_estimates = []

    if method not in ['midpoint', 'peak', 'confidence']:
        raise ValueError("Invalid method. Choose from 'midpoint', 'peak', or 'confidence'.")

    signal_length = len(prob_arr)
    x = np.linspace(0, 1, signal_length)  # Frequency values
    # Extend frequency range to include the artificial signal edges added later
    x = np.concatenate(([2*x[0] - x[1]], x, [2*x[-1] - x[-2]]))

    predicted_labels = prob_arr >= acceptance

    if not np.any(predicted_labels):
        # No frequencies predicted for this sample
        return np.array([])

    # Find start and end indices of each mode
    # Add False values at the start and end to detect mode boundaries at the edges
    # Use np.diff to find the indices where the predicted labels change
    # Use np.where to find the indices of the changes - extracts non zero indices
    # np.where with no condition returns indices of non-zero elements because zero is False
    mode_indices = np.where(np.diff(np.concatenate(([False], predicted_labels, [False]))))[0] 
    mode_indices = mode_indices.reshape(-1, 2) # reshape into pairs of start and end indices

    for start, end in mode_indices:
        mode_width = x[end] - x[start]
        
        if bandwidth is not None and mode_width > bandwidth * (1+overlap_threshold):
            num_modes = int(np.ceil(mode_width / bandwidth))
            first_mode = x[start] + bandwidth / 2
            last_mode = x[end] - bandwidth / 2
            
            freq_estimates.extend(np.linspace(first_mode, last_mode, num_modes))
        else:
            if method == 'midpoint':
                # Find the midpoint of each mode
                freq_estimates.append((x[start] + x[end - 1]) / 2)

            elif method == 'peak':
                # Find the peak probability of each mode
                freq_estimates.append(x[np.argmax(prob_arr[start:end]) + start])

            elif method == 'confidence':
                # Weighted average of frequencies within each mode
                mode_probs = prob_arr[start:end]
                mode_freqs = x[start:end]
                if np.sum(mode_probs) > 0:  # Avoid division by zero
                    weighted_avg = np.sum(mode_freqs * mode_probs) / np.sum(mode_probs)
                    freq_estimates.append(weighted_avg)

    return np.array(freq_estimates)


def est_nat_freq_triangle_rise(curve, up_inc=0.4):
    length = len(curve)
    x = np.linspace(0, 1, length)
    dY = np.gradient(curve, x)
    d2Y = np.gradient(dY, x)
    
    zero_crossings = np.where(np.diff(np.sign(dY)))[0]
    peak_indices = [idx for idx in zero_crossings if d2Y[idx] < 0]  # find all peaks
    peak_indices.append(length-1)  
    # add the last point as a peak in case the regression curve has not rounded over yet
    # if there is no mode present it will be removed by the next line anyway
    peak_indices = [idx for idx in peak_indices if curve[idx] >= up_inc * 0.95]  # remove peaks below up_inc (and a small factor)
    trough_indices = [idx for idx in zero_crossings if d2Y[idx] > 0]  # find all troughs
    
    max_dy_indices = []
    prev_peak_idx = 0
    for peak_idx in peak_indices:
        min_left_value = curve[prev_peak_idx:peak_idx].min()
        # Check if the drop to the left is greater than up_inc
        if curve[peak_idx] - min_left_value > up_inc:
            nearest_left_trough_idx = max([idx for idx in trough_indices if idx < peak_idx], default=0) 
            # find nearest trough to left, if no trough found set to default 0
            max_dy_indices.append(np.argmax(dY[nearest_left_trough_idx:peak_idx]) + nearest_left_trough_idx)
            # Update prev_peak_idx only for detected peaks
            prev_peak_idx = peak_idx
    
    freq_estimates = x[max_dy_indices]
    return np.array(freq_estimates)


def calculate_frequency_error(predicted_frequencies, true_frequencies, max_error=1.0):
    """
    Calculate the error between predicted and true natural frequencies.

    Args:
        predicted_frequencies (np.array): Array of predicted frequencies.
        true_frequencies (np.array): Array of true frequencies.
        max_error (float): Penalty for unmatched predictions or true frequencies.

    Returns:
        float: Mean absolute error between predicted and true frequencies.
    """

    # Handle empty cases
    if len(predicted_frequencies) == 0 and len(true_frequencies) == 0:
        return 0.0  # No error if both are empty
    elif len(predicted_frequencies) == 0:
        return max_error * len(true_frequencies)  # All true frequencies are unmatched
    elif len(true_frequencies) == 0:
        return max_error * len(predicted_frequencies)  # All predictions are unmatched

    # Create a cost matrix for bipartite matching
    cost_matrix = np.abs(predicted_frequencies[:, None] - true_frequencies[None, :])

    # Use the Hungarian algorithm to find the optimal matching
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

    # Calculate the total error for matched pairs
    total_error = cost_matrix[row_ind, col_ind].sum()

    # Add penalties for unmatched predictions and true frequencies
    num_unmatched_pred = len(predicted_frequencies) - len(row_ind)
    num_unmatched_true = len(true_frequencies) - len(col_ind)
    total_error += max_error * (num_unmatched_pred + num_unmatched_true)

    # Calculate the mean error
    mean_error = total_error / max(len(predicted_frequencies), len(true_frequencies))

    return mean_error


def calculate_mean_frequency_error(model, dataloader, acceptance=0.5, method='midpoint', max_error=1.0, bandwidth=None, overlap_threshold=0.5):
    total_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, labels, params in dataloader:
            batch_size = len(data)
            for i in range(batch_size):
                probabilities = model(data).squeeze()
                probabilities = probabilities[i].cpu().numpy()
                predicted_frequencies = est_nat_freq_binary(probabilities, 
                                                            acceptance, 
                                                            method,
                                                            bandwidth,
                                                            overlap_threshold
                                                            )
                true_frequencies = params[i, :, 0].cpu().numpy()
                true_frequencies = true_frequencies[~np.isnan(true_frequencies)]  # Remove NaN values
                error = calculate_frequency_error(predicted_frequencies, true_frequencies, max_error=max_error)
                total_error += error
                total_samples += 1
    mean_error = total_error / total_samples
    return mean_error


def calculate_mean_frequency_error_triangle(model, dataloader, up_inc=0.4, N=2, Wn=0.2, max_error=1.0):
    total_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, labels, params in dataloader:
            batch_size = len(data)
            for i in range(batch_size):
                model.eval()
                model_output = model(data).squeeze()
                fitted_curve = model_output[i].cpu().numpy()
                b,a = scipy.signal.butter(N,Wn)
                smoothed_curve = scipy.signal.filtfilt(b,a,fitted_curve)
                predicted_frequencies = est_nat_freq_triangle_rise(smoothed_curve, up_inc=up_inc)
                true_frequencies = params[i, :, 0].cpu().numpy()
                true_frequencies = true_frequencies[~np.isnan(true_frequencies)]  # Remove NaN values
                error = calculate_frequency_error(predicted_frequencies, true_frequencies, max_error=max_error)
                total_error += error
                total_samples += 1
    mean_error = total_error / total_samples
    return mean_error


def load_model(save_name):
    project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/Models/'
    model_path = project_path + save_name
    model = torch.load(f'{model_path}')
    return model


def save_model(model, save_suffix):
    project_path = 'C:/Users/Freddie/Documents/IIB project repository/myenv/FRJA2_IIB_project/Models/'
    now = datetime.datetime.now()
    model_path = project_path + now.strftime("%m_%d_%H_%M") + save_suffix
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
    for data, _, _ in dataloader:
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
    for data, labels, _ in dataloader:
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
    for data, _, _ in dataloader:
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



