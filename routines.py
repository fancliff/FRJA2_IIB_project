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
    try:
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
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    except Exception as e:
        print(f'\nTraining failed with error: {e}')
    finally: # Always save the model regardless of error e.g out of memory etc.
        print()
        print('Finished Training. Loading best model...')
        
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
        outputs = model(data)
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
            outputs = model(data)
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
    # print("Output shape:", outputs.shape)
    # print("Label shape:", labels.shape)
    
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
                        # axes[j].plot(x, prob_arr, label="Output", color="orange")

                        # Masked signal line for predicted labels
                        masked_labels = np.where(predictions_arr == 1, signal_arr, np.nan)
                        axes[j].plot(
                            x, masked_labels,
                            color='red',
                            label='Model Predictions',
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
                            axes[j].axvline(x=omega, color='black', linestyle='--', label=r'True $\omega_n$' if k==0 else '')
                        
                        # Plot predicted omegas as vertical dotted lines
                        predicted_omegas = est_nat_freq_binary(prob_arr, 
                                                            acceptance=acceptance,
                                                            method='midpoint',
                                                            bandwidth=0.04,
                                                            overlap_threshold=0.5)
                        for k, omega in enumerate(predicted_omegas):
                            axes[j].axvline(x=omega, color='cyan', linestyle=':', label=r'Predicted $\omega_n$' if k==0 else '')
                        mask_patch = [mpatches.Patch(color='grey', alpha=0.5, label = 'Mode labels')]
                        if j == 0:
                            axes[j].legend(
                                handles=axes[j].get_legend_handles_labels()[0] + mask_patch,
                                loc = 'upper center',
                                bbox_to_anchor=(0.5, 1.3),
                                ncol = 5
                            )
                    axes[0].set_ylabel('Real Part')
                    axes[1].set_ylabel('Imaginary Part')
                    axes[1].set_xlabel('Normalised Frequency')
                    
                    plt.tight_layout()
                    plt.show()
                
                samples_plotted += 1
                if samples_plotted >= num_samples:
                    return # Exit if enough samples are plotted

# New for final report plotting
def plot_samples_report(dataloader, num_samples):
    samples_plotted = 0
    for data, labels, params in dataloader:
        for i in range(min(num_samples, len(data))):
            num_channels = data[i].shape[0]
            fig, axes = plt.subplots(num_channels, 1, figsize=(8, 4), sharex=True)
            if num_channels == 1:
                axes = [axes] # Convert single axis to list for iteration
            
            x = np.linspace(0,1,len(data[i][0]))
            omegas = params[i, :, 0].cpu().numpy() if params is not None else []
            omegas = omegas[~np.isnan(omegas)] # Remove NaN values
            
            for j in range(num_channels):
                labels_arr = labels[i].cpu().numpy()
                axes[j].plot(x,data[i][j].cpu().numpy(), label="Signal", color="blue")
                # axes[j].plot(x,labels_arr, label="Actual Labels", linestyle="--", color="green")
                # axes[j].set_ylabel('Amplitude / Label')
                
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
                
                mask_patch = [mpatches.Patch(color='grey', alpha=0.5, label = 'Mode labels')]
                
                #Plot the omegas as vertical dashed lines
                for k, omega in enumerate(omegas):
                    axes[j].axvline(x=omega, color='black', linestyle='--', label='Mode Frequency' if k==0 else '')
                
                #axes[j].set_title(f'Channel {j+1}')
                #axes[j].set_ylabel('Signal Amplitude')
                if j == 0:
                    axes[j].legend(
                        handles=axes[j].get_legend_handles_labels()[0] + mask_patch,
                        loc = 'upper center',
                        bbox_to_anchor=(0.5, 1.3),
                        ncol = 3
                    )
                    
            
            axes[0].set_ylabel('Real Part')
            axes[1].set_ylabel('Imaginary Part')
            axes[1].set_xlabel('Normalised Frequency')

            #plt.suptitle(f'Sample {i+1}')
            #plt.xlabel('Frequency (Normalized)')
            plt.tight_layout()
            #plt.subplots_adjust(top=0.9) # Adjust suptitle position
            plt.show()
            
            samples_plotted += 1
            if samples_plotted >= num_samples:
                return  # Exit after plotting specified number of samples


def plot_samples_old(dataloader, num_samples, binary_labels=True):
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
                        
                        predicted_omegas,_ = est_nat_freq_triangle_rise(smoothed_curve, up_inc=0.5)
                        for k, omega in enumerate(predicted_omegas):
                            axes[j].axvline(x=omega, color='cyan', linestyle=':', label='Predicted Mode Frequency' if k==0 else '')
                        
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.show()
                
                samples_plotted += 1
                if samples_plotted >= num_samples:
                    return # Exit if enough samples are plotted


def plot_predictions_all_labels(models, dataloader, num_samples, label_defs, scale_factors = None, N=2, Wn=0.2):
    samples_plotted = 0
    with torch.no_grad():
        for data, labels, params in dataloader:
            batch_size = len(data)
            for i in range(min(num_samples, batch_size)):
                x = np.linspace(0, 1, len(data[i][0]))
                omegas = params[i, :, 0].cpu().numpy() if params is not None else []
                omegas = omegas[~np.isnan(omegas)]

                if models is None:
                    continue  # Only supporting model output plotting for now
                else:
                    for model in models:
                        model.eval()
                        model_output = model(data)
                        num_data_channels = data[i].shape[0]
                        num_label_channels = labels[i].shape[0]
                        num_channels = num_data_channels + num_label_channels
                        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels), sharex=True)

                        # Ensure axes is always iterable
                        if num_channels == 1:
                            axes = [axes]

                        legend_handles = []
                        legend_labels = []

                        # Plot signal
                        for j in range(num_data_channels):
                            signal_arr = data[i][j].cpu().numpy()
                            h_signal, = axes[j].plot(x, signal_arr, color="blue", label="Signal")
                            for k, omega in enumerate(omegas):
                                h_true = axes[j].axvline(x=omega, color='black', linestyle='--',
                                                        label=r'True $\omega_n$' if k == 0 else '')
                                if k == 0:
                                    legend_handles.append(h_true)
                                    legend_labels.append(r'True $\omega_n$')
                            axes[j].set_ylabel("Real part" if j == 0 else "Imaginary part")
                            legend_handles.append(h_signal)
                            legend_labels.append("Signal")

                        j = 0
                        label_names = ["Modes", r"$|\alpha_n|$", r"$\angle \alpha_n$", r"$\log_{10}(\zeta_n)$"]
                        label_keys = [0, 1, 2, 3]
                        
                        modes_curve = model_output[i][0].cpu().numpy()
                        b, a = scipy.signal.butter(N, Wn)
                        smoothed_modes = scipy.signal.filtfilt(b, a, modes_curve)
                        predicted_omegas, _ = est_nat_freq_triangle_rise(smoothed_modes, up_inc=0.5)

                        for idx, label_name in zip(label_keys, label_names):
                            if label_defs[idx]:
                                ax = axes[num_data_channels + j]
                                if scale_factors is None or idx == 0: # no scaling for mode labels
                                    # ax.set_ylabel(f"{label_name}: Target")
                                    ax.set_ylabel(f"{label_name}")
                                else:
                                    scale = scale_factors[j-1]
                                    # ax.set_ylabel(f"{label_name} / {scale:.2f} : Target")
                                    ax.set_ylabel(f"{label_name} / {scale:.2f}")
                                h, l = subplot_labels(ax, x, model_output[i][j], labels[i][j], label_name, N, Wn, predicted_omegas)
                                legend_handles.extend(h)
                                legend_labels.extend(l)
                                j += 1

                        axes[-1].set_xlabel("Normalised Frequency")
                        # Remove duplicate labels
                        unique_legend = dict(zip(legend_labels, legend_handles))
                        fig.legend(unique_legend.values(), unique_legend.keys(),
                                loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
                        plt.tight_layout(rect=[0, 0, 1, 0.92])
                        plt.show()

                samples_plotted += 1
                if samples_plotted >= num_samples:
                    return


def subplot_labels(axes_j, x, model_output_i_j, labels_arr_i_j, name, N, Wn, predicted_omegas):
    handles, labels = axes_j.get_legend_handles_labels()

    if model_output_i_j is not None:
        fitted_curve = model_output_i_j.cpu().numpy()
        b, a = scipy.signal.butter(N, Wn)
        smoothed_curve = scipy.signal.filtfilt(b, a, fitted_curve)

        h1, = axes_j.plot(x, fitted_curve, color="orange", label="Output")
        h2, = axes_j.plot(x, smoothed_curve, color="red", label="Smoothed output")
        handles.extend([h1, h2])
        labels.extend(["Output", "Smoothed output"])
        
        for k, omega in enumerate(predicted_omegas):
            h3 = axes_j.axvline(x=omega, color='cyan', linestyle=':', label=r'Predicted $\omega_n$' if k == 0 else '')
            if k == 0:
                if "Modes" in name:
                    handles.append(h3)
                    labels.append(r'Predicted $\omega_n$')

    h4, = axes_j.plot(x, labels_arr_i_j, linestyle="--", color="green", label="Target")
    handles.append(h4)
    labels.append("Target")

    return handles, labels



# def plot_predictions_all_labels(models, dataloader, num_samples, label_defs, N=2, Wn=0.2):
#     samples_plotted = 0
#     with torch.no_grad():
#         for data, labels, params in dataloader:
#             batch_size = len(data)
#             for i in range(min(num_samples,batch_size)):
#                 x = np.linspace(0, 1, len(data[i][0]))
#                 omegas = params[i, :, 0].cpu().numpy() if params is not None else []
#                 omegas = omegas[~np.isnan(omegas)] # Remove NaN values
                
#                 if models is None: # Plot just sample and labels
#                     num_data_channels = data[i].shape[0] 
#                     num_label_channels = labels[i].shape[0] # Add subfigure for each label
#                     num_channels = num_data_channels + num_label_channels
#                     fig, axes = plt.subplots(num_channels, 1, figsize=(8, 4), sharex=True)

#                     for j in range(num_data_channels):
#                         signal_arr = data[i][j].cpu().numpy()
#                         axes[j].plot(x, signal_arr, label="Signal", color="blue")
                        
#                         # Plot the true omegas as vertical dashed lines
#                         for k, omega in enumerate(omegas):
#                             axes[j].axvline(x=omega, color='black', linestyle='--', label='Mode Frequency' if k==0 else '')
                        
#                         axes[j].legend()
                    
#                     j = 0 # Reset channel index for labels
#                     if label_defs[0]: # Mode triangle labelling
#                         subplot_labels(axes[j+num_data_channels], x, None, labels[i][j], "modes", N, Wn)
#                         j += 1
#                     if label_defs[1]: # Amplitude magnitude
#                         subplot_labels(axes[j+num_data_channels], x, None, labels[i][j], "amplitude magnitude", N, Wn)
#                         j += 1
#                     if label_defs[2]: # Amplitude phase
#                         subplot_labels(axes[j+num_data_channels], x, None, labels[i][j], "amplitude phase", N, Wn)
#                         j += 1
#                     if label_defs[3]: # Log10_zeta
#                         subplot_labels(axes[j+num_data_channels], x, None, labels[i][j], "log10_zeta", N, Wn)
#                         j += 1
                    
#                     plt.legend()
                    
#                     plt.tight_layout()
#                     plt.show()
                
#                 else:
#                     for model_idx, model in enumerate(models): # Plot predictions
#                         model.eval()
#                         model_output = model(data)
#                         num_data_channels = data[i].shape[0] 
#                         num_label_channels = labels[i].shape[0] # Add subfigure for each label
#                         num_channels = num_data_channels + num_label_channels
#                         fig, axes = plt.subplots(num_channels, 1, figsize=(8, 4), sharex=True)

#                         for j in range(num_data_channels):
#                             signal_arr = data[i][j].cpu().numpy()
#                             axes[j].plot(x, signal_arr, label="Signal", color="blue")
                            
#                             # Plot the true omegas as vertical dashed lines
#                             for k, omega in enumerate(omegas):
#                                 axes[j].axvline(x=omega, color='black', linestyle='--', label='Mode Frequency' if k==0 else '')
#                             if j ==0:
                                
#                                 axes[j].legend(                       
#                                     loc = 'upper center',
#                                     bbox_to_anchor=(0.5, 1.7),
#                                     ncol = 2
#                                     )
                        
#                         j = 0 # Reset channel index for labels
#                         if label_defs[0]: # Mode triangle labelling
#                             subplot_labels(axes[j+num_data_channels], x, model_output[i][j], labels[i][j], "Modes", N, Wn)
#                             j += 1
#                         if label_defs[1]: # Amplitude magnitude
#                             subplot_labels(axes[j+num_data_channels], x, model_output[i][j], labels[i][j], "Amplitude magnitude", N, Wn)
#                             j += 1
#                         if label_defs[2]: # Amplitude phase
#                             subplot_labels(axes[j+num_data_channels], x, model_output[i][j], labels[i][j], "Amplitude phase", N, Wn)
#                             j += 1
#                         if label_defs[3]: # Log10 zeta
#                             subplot_labels(axes[j+num_data_channels], x, model_output[i][j], labels[i][j], "log10_zeta", N, Wn)
#                             j += 1
                        
#                         plt.legend(
#                             loc = 'upper center',
#                             bbox_to_anchor=(0.5,1.7),
#                             ncol = 4
#                         )
                        
#                         plt.tight_layout()
#                         plt.show()
                
#                 samples_plotted += 1
#                 if samples_plotted >= num_samples:
#                     return # Exit if enough samples are plotted


# def subplot_labels(axes_j, x, model_output_i_j, labels_arr_i_j, name, N, Wn):
#     handles, labels = axes_j.get_legend_handles_labels() # Get existing legend handles and labels

#     if model_output_i_j is not None:
#         fitted_curve = model_output_i_j.cpu().numpy()
#         b,a = scipy.signal.butter(N,Wn)
#         smoothed_curve = scipy.signal.filtfilt(b,a,fitted_curve)
        
#         h1, = axes_j.plot(x, fitted_curve, label=f"Output", color="orange")
#         h2, = axes_j.plot(x, smoothed_curve, label=f"Smoothed output", color="red")
#         handles.extend([h1, h2])
#         labels.extend([f"Output", f"Smoothed output"])
        
#         if name == 'modes':
#             predicted_omegas,_ = est_nat_freq_triangle_rise(smoothed_curve, up_inc=0.5)
#             for k, omega in enumerate(predicted_omegas):
#                 h3 = axes_j.axvline(x=omega, color='cyan', linestyle=':', label=r'Predicted $\omega_n$' if k == 0 else '')
#                 if k == 0:  # Only add one label to avoid duplicates
#                     handles.append(h3)
#                     labels.append(r'Predicted $\omega_n$')

#     h4, = axes_j.plot(x, labels_arr_i_j, label=f"{name} target output", linestyle="--", color="green")
#     handles.append(h4)
#     labels.append(f"{name} target output")

#     # axes_j.legend(handles, labels) # Update the legend


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


# incompatible with jit so replaced for 25% speed increase
def est_nat_freq_triangle_rise_old(curve, up_inc):
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
    freq_est_idxs = max_dy_indices
    return np.array(freq_estimates), freq_est_idxs

@jit(nopython=True)
def est_nat_freq_triangle_rise(curve, up_inc):
    length = len(curve)
    x = np.linspace(0, 1, length)

    # Finite difference approximation of gradient
    dY = np.zeros(length)
    d2Y = np.zeros(length)
    for i in range(1, length-1):
        dY[i] = (curve[i+1] - curve[i-1]) / (x[i+1] - x[i-1])
    dY[0] = dY[1]
    dY[-1] = dY[-2]

    for i in range(1, length-1):
        d2Y[i] = (dY[i+1] - dY[i-1]) / (x[i+1] - x[i-1])
    d2Y[0] = d2Y[1]
    d2Y[-1] = d2Y[-2]

    # Find zero crossings in dY
    zero_crossings = []
    for i in range(1, length):
        if dY[i-1] * dY[i] < 0:
            zero_crossings.append(i-1)

    # Peaks and troughs
    peak_indices = []
    trough_indices = []
    for i in zero_crossings:
        if d2Y[i] < 0:
            peak_indices.append(i)
        elif d2Y[i] > 0:
            trough_indices.append(i)

    # Always include last index
    peak_indices.append(length - 1)

    # Filter out small peaks
    filtered_peak_indices = []
    for i in peak_indices:
        if curve[i] >= up_inc * 0.95:
            filtered_peak_indices.append(i)
    peak_indices = filtered_peak_indices

    # Extract mode frequency estimates
    max_dy_indices = []
    prev_peak_idx = 0
    for peak_idx in peak_indices:
        # Min value to the left of this peak
        min_left_value = curve[prev_peak_idx]
        for i in range(prev_peak_idx, peak_idx):
            if curve[i] < min_left_value:
                min_left_value = curve[i]

        if curve[peak_idx] - min_left_value > up_inc:
            # Find nearest left trough
            nearest_left_trough_idx = 0
            for idx in trough_indices:
                if idx < peak_idx and idx > nearest_left_trough_idx:
                    nearest_left_trough_idx = idx

            # Find index of max dY in region
            max_dy = dY[nearest_left_trough_idx]
            max_idx = nearest_left_trough_idx
            for i in range(nearest_left_trough_idx, peak_idx):
                if dY[i] > max_dy:
                    max_dy = dY[i]
                    max_idx = i

            max_dy_indices.append(max_idx)
            prev_peak_idx = peak_idx

    freq_estimates = np.zeros(len(max_dy_indices))
    for i in range(len(max_dy_indices)):
        freq_estimates[i] = x[max_dy_indices[i]]

    return freq_estimates, np.array(max_dy_indices)


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


# @jit(nopython=True)
def calculate_mean_frequency_error_triangle(model, dataloader, label_defs, up_inc=0.20, N=2, Wn=0.2, max_error=1.0):
    total_error = 0.0
    total_samples = 0
    if not label_defs[0]: # Mode triangle labelling
        print('\n Error calculation requires mode triangle labelling.')
        return 0.0
    with torch.no_grad():
        for data, labels, params in dataloader:
            batch_size = len(data)
            for i in range(batch_size):
                model.eval()
                model_output = model(data)
                if model_output.shape[0] > 1: # If more than one label channel just extract mode triangle labels
                    fitted_curve = model_output[i][0].cpu().numpy()
                else:
                    fitted_curve = model_output[i].cpu().numpy()
                b,a = scipy.signal.butter(N,Wn)
                smoothed_curve = scipy.signal.filtfilt(b,a,fitted_curve)
                predicted_frequencies,_ = est_nat_freq_triangle_rise(smoothed_curve, up_inc=up_inc)
                true_frequencies = params[i, :, 0].cpu().numpy()
                true_frequencies = true_frequencies[~np.isnan(true_frequencies)]  # Remove NaN values
                error = calculate_frequency_error(predicted_frequencies, true_frequencies, max_error=max_error)
                total_error += error
                total_samples += 1
    mean_error = total_error / total_samples
    return mean_error



def estimate_parameter(output, predicted_freq_idxs, label_halfwidth=0.02, window_scale=0.6, N=2, Wn=0.2):
    # Smooth output signal
    b,a = scipy.signal.butter(N,Wn)
    output = scipy.signal.filtfilt(b,a,output)
    
    # Convert window_halfwidth to index space
    window_halfwidth_idx = int(label_halfwidth * window_scale * len(output))
    
    # Extract point estimates
    point_estimates = output[predicted_freq_idxs]
    
    # Initialize lists for means and variances
    means = []
    variances = []
    
    # Calculate means and variances using index windows
    for freq_idx in predicted_freq_idxs:
        start_idx = max(0, freq_idx - window_halfwidth_idx)
        end_idx = min(len(output), freq_idx + window_halfwidth_idx + 1)
        window = output[start_idx:end_idx]
        
        means.append(np.mean(window))
        variances.append(np.var(window))
    
    return np.array([np.array(point_estimates), np.array(means), np.array(variances)])



def compare_FRF(input_signal, all_outputs, scale_factors, FRF_type = 0, signal_length = 1024, norm = True):
    # Assumes model ouputs are modes, a mag, a phase, log10_zeta
    # FRF type: 0 for just magnitude, 1 for real and imaginary
    
    # Extract the predicted frequencies
    mode_channel = all_outputs[0]
    predicted_freqs,predicted_freq_idxs = est_nat_freq_triangle_rise(mode_channel, up_inc=0.35)
    
    # point estimate [0], mean [1], variance [2]
    x = 1
    a_mag = estimate_parameter(all_outputs[1], predicted_freq_idxs)[x]
    a_phase = estimate_parameter(all_outputs[2], predicted_freq_idxs)[x]
    log10_zeta = estimate_parameter(all_outputs[3], predicted_freq_idxs)[x]
    
    a_mag_scale = scale_factors[0]
    a_phase_scale = scale_factors[1]
    log10_zeta_scale = scale_factors[2]
    
    # Reconstruct FRF
    H_v = construct_FRF(
        predicted_freqs, 
        a_mag*a_mag_scale, 
        a_phase*a_phase_scale,
        10**(log10_zeta*log10_zeta_scale),
        signal_length,
        min_max=norm
    )

    # Compare the FRFs
    if FRF_type == 0:
        H_v = np.abs(H_v)
        MSE_mag = quick_ms_diff(input_signal,H_v,signal_length)
        return MSE_mag, H_v
    else:
        H_v_real = np.real(H_v)
        H_v_imag = np.imag(H_v)
        MSE_real = quick_ms_diff(input_signal[0],H_v_real,signal_length)
        MSE_imag = quick_ms_diff(input_signal[1],H_v_imag,signal_length)
        return (MSE_real+MSE_imag)/2, np.array([H_v_real, H_v_imag])


@jit(nopython=True)
def quick_ms_diff(a, b, signal_length):
    output = np.zeros(signal_length, dtype = np.float64)
    output = np.mean((a-b)**2)
    return output


@jit(nopython=True)
def construct_FRF(omegas, alpha_mags, alpha_phases, zetas, signal_length, min_max: bool = True):
    H_v = np.zeros(signal_length, dtype=np.complex128)
    frequencies = np.linspace(0, 1, signal_length)
    for n in range(len(omegas)):
        alpha_n = alpha_mags[n]
        zeta_n = zetas[n]
        omega_n = omegas[n]
        alpha_phase_n = alpha_phases[n]
        
        for j, w in enumerate(frequencies):
            H_f = 0.0j
            
            denominator = omega_n**2 - w**2 + 2j * zeta_n * w
            numerator = 1j*w*alpha_n*np.exp(1j*alpha_phase_n)
            
            H_f += numerator/denominator
            
            H_v[j] += H_f
    
    if min_max:
        mag_no_norm = np.abs(H_v)
        if not np.all(mag_no_norm==0):
            min_mag = np.min(mag_no_norm)
            max_mag = np.max(mag_no_norm)
            range_mag = max_mag-min_mag
            if range_mag > 0:
                mag = (mag_no_norm - min_mag)/range_mag
                H_v = H_v * (mag/np.maximum(mag_no_norm, 1e-12))
    return H_v



def calculate_mean_FRF_error(model, dataloader, scale_factors, FRF_type, signal_length, norm):
    total_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, labels, params in dataloader:
            batch_size = len(data)
            for i in range(batch_size):
                output = model(data).squeeze()
                # input_signal = data[i].cpu().numpy()
                true_omegas = params[i, :, 0].cpu().numpy()
                mask = ~np.isnan(true_omegas)
                true_omegas = true_omegas[mask]
                true_alpha_mags = params[i, :, 1].cpu().numpy()[mask]
                true_alpha_phases = params[i, :, 2].cpu().numpy()[mask]
                true_zetas = params[i, :, 3].cpu().numpy()[mask]
                
                reconstructed_input = construct_FRF(true_omegas, true_alpha_mags, true_alpha_phases, true_zetas, signal_length, min_max=norm)
                if FRF_type == 0:
                    reconstructed_input = np.abs(reconstructed_input)
                else:
                    reconstructed_input = np.array([np.real(reconstructed_input), np.imag(reconstructed_input)])
                error, _ = compare_FRF(reconstructed_input, output[i].cpu().numpy(), scale_factors, FRF_type, signal_length, norm)
                total_error += error
                total_samples += 1
    mean_error = total_error / total_samples
    return mean_error



def plot_FRF_comparison(model, dataloader, num_samples, scale_factors, FRF_type=0, signal_length=1024, norm=True):
    samples_plotted = 0
    with torch.no_grad():
        for data, labels, params in dataloader:
            batch_size = len(data)
            for i in range(min(num_samples,batch_size)):
                output = model(data).squeeze()
                # input_signal = data[i].cpu().numpy()
                true_omegas = params[i, :, 0].cpu().numpy()
                mask = ~np.isnan(true_omegas)
                true_omegas = true_omegas[mask]
                true_alpha_mags = params[i, :, 1].cpu().numpy()[mask]
                true_alpha_phases = params[i, :, 2].cpu().numpy()[mask]
                true_zetas = params[i, :, 3].cpu().numpy()[mask]
                
                reconstructed_input = construct_FRF(true_omegas, true_alpha_mags, true_alpha_phases, true_zetas, signal_length, min_max=norm)
                if FRF_type == 0:
                    reconstructed_input = np.abs(reconstructed_input)
                else:
                    reconstructed_input = np.array([np.real(reconstructed_input), np.imag(reconstructed_input)])
                error, H_v = compare_FRF(reconstructed_input, output[i].cpu().numpy(), scale_factors,FRF_type, signal_length, norm)
                
                modes_output = output[i][0].cpu().numpy()
                b, a = scipy.signal.butter(2,0.2)
                smoothed_modes = scipy.signal.filtfilt(b,a,modes_output)
                predicted_omegas, _ = est_nat_freq_triangle_rise(smoothed_modes, up_inc=0.5)
                
                frequencies = np.linspace(0, 1, signal_length)
                if FRF_type == 0:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                    ax.plot(frequencies, H_v, label='Predicted FRF', color='orange')
                    ax.plot(frequencies, reconstructed_input, label='True FRF (w/o noise)', color='blue')
                    for k, omega in enumerate(true_omegas):
                        ax.axvline(x=omega, color='black', linestyle='--',
                                                label=r'True $\omega_n$' if k == 0 else '')
                    for k, omega in enumerate(predicted_omegas):
                        ax.axvline(x=omega, color='cyan', linestyle=':', 
                                        label=r'Predicted $\omega_n$' if k == 0 else '')
                    fig.suptitle('Magnitude FRF Comparison: MSE = {:.4f}'.format(error))
                    ax.legend(
                        loc='upper center',
                        ncol = 4,
                        bbox_to_anchor = (0.5,1.15)
                    )
                    plt.tight_layout(rect=[0, 0, 1, 1.05])
                else:
                    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                    ax[0].plot(frequencies, H_v[0], label='Predicted FRF', color='orange')
                    ax[0].plot(frequencies, reconstructed_input[0], label='True FRF', color='blue')
                    ax[1].plot(frequencies, H_v[1], label='Predicted FRF', color='orange')
                    ax[1].plot(frequencies, reconstructed_input[1], label='True FRF', color='blue')
                    for k, omega in enumerate(true_omegas):
                        ax[0].axvline(x=omega, color='black', linestyle='--',
                                                label=r'True $\omega_n$' if k == 0 else '')
                        ax[1].axvline(x=omega, color='black', linestyle='--')
                    for k, omega in enumerate(predicted_omegas):
                        ax[0].axvline(x=omega, color='cyan', linestyle=':', 
                                            label=r'Predicted $\omega_n$' if k == 0 else '')
                        ax[1].axvline(x=omega, color='cyan', linestyle=':')
                    fig.suptitle('Complex FRF Comparison: MSE = {:.4f}'.format(error))
                    ax[0].legend(
                        loc='upper center',
                        ncol = 4,
                        bbox_to_anchor = (0.5,1.15),
                    )
                    plt.tight_layout(rect=[0, 0, 1, 1.02])
                
                
                plt.show(block=False)
                
                plot_model_predictions_single_sample(
                    model, 
                    data[i], 
                    labels[i], 
                    params[i], 
                    label_defs=[True, True, True, True],
                    scale_factors=scale_factors
                )
                
                samples_plotted += 1
                if samples_plotted >= num_samples:
                    return



def plot_model_predictions_single_sample(model, data, labels, params, label_defs, scale_factors=None, N=2, Wn=0.2):
    x = np.linspace(0, 1, len(data[0]))
    omegas = params[:, 0].cpu().numpy()
    omegas = omegas[~np.isnan(omegas)]

    model.eval()
    model_output = model(data.unsqueeze(0))  # Add batch dimension
    num_data_channels = data.shape[0]
    num_label_channels = labels.shape[0]
    num_channels = num_data_channels + num_label_channels
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels), sharex=True)

    if num_channels == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    for j in range(num_data_channels):
        signal_arr = data[j].cpu().numpy()
        h_signal, = axes[j].plot(x, signal_arr, color="blue", label="Signal")
        for k, omega in enumerate(omegas):
            h_true = axes[j].axvline(x=omega, color='black', linestyle='--',
                                        label=r'True $\omega_n$' if k == 0 else '')
            if k == 0:
                legend_handles.append(h_true)
                legend_labels.append(r'True $\omega_n$')
        axes[j].set_ylabel("Real part" if j == 0 else "Imaginary part")
        legend_handles.append(h_signal)
        legend_labels.append("Signal")

    j = 0
    label_names = ["Modes", r"$|\alpha_n|$", r"$\angle \alpha_n$", r"$\log_{10}(\zeta_n)$"]
    label_keys = [0, 1, 2, 3]
    
    modes_curve = model_output[0][0].cpu().numpy() # shape has batch size
    b, a = scipy.signal.butter(N, Wn)
    smoothed_modes = scipy.signal.filtfilt(b, a, modes_curve)
    predicted_omegas, _ = est_nat_freq_triangle_rise(smoothed_modes, up_inc=0.5)

    for idx, label_name in zip(label_keys, label_names):
        if label_defs[idx]:
            ax = axes[num_data_channels + j]
            if scale_factors is None or idx == 0:
                ax.set_ylabel(f"{label_name}")
            else:
                scale = scale_factors[j - 1]
                ax.set_ylabel(f"{label_name} / {scale:.2f}")
            h, l = subplot_labels(ax, x, model_output[0][j], labels[j], label_name, N, Wn, predicted_omegas)
            legend_handles.extend(h)
            legend_labels.extend(l)
            j += 1

    axes[-1].set_xlabel("Normalised Frequency")
    unique_legend = dict(zip(legend_labels, legend_handles))
    fig.legend(unique_legend.values(), unique_legend.keys(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def plot_FRF_cloud():
    pass


# @jit(nopython=True)
def generate_random_FRFs():
    pass


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


def compare_models_regression(models, dataloader1, criterion=nn.MSELoss()):
    # single dataloader for all models
    for i, model in enumerate(models):
        model.eval()
        loss, mse, mae, r2 = validation_loss_regression(model, dataloader1, criterion)
        print(f'\nModel {i+1}:')
        print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}')


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



