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
from models import ResidualBlock
import matplotlib.patches as mpatches #for legend using masks
import matplotlib.colors as mcolors #for custom colormap
import datetime #for timestamping save files
import routines as rt

def plot_predictions_all_labels(model, data, label_defs, scale_factors = None, N=2, Wn=0.1, plot_phase: bool = True):
    with torch.no_grad():
        model.eval()
        model_output = model(data).squeeze(0) # remove batch dimension for single sample
        data = data.squeeze(0) # remove batch dimension 
        x = np.linspace(0, 1, data.shape[-1])
        num_data_channels = data.shape[0]
        num_label_channels = np.sum(label_defs)
        num_channels = num_data_channels + num_label_channels
        
        if plot_phase:
            fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels), sharex=True)
        else:
            fig, axes = plt.subplots(num_channels - 1, 1, figsize=(10, 2 * num_channels), sharex=True)

        # Ensure axes is always iterable
        if num_channels == 1:
            axes = [axes]

        legend_handles = []
        legend_labels = []

        # Plot signal
        for j in range(num_data_channels):
            signal_arr = data[j].cpu().numpy()
            h_signal, = axes[j].plot(x, signal_arr, color="blue", label="Signal")
            legend_handles.append(h_signal)
            legend_labels.append("Signal")
        axes[0].set_ylabel('Real Part')
        axes[1].set_ylabel('Imaginary Part')

        j = 0
        label_names = ["Modes", r"$|\alpha_n|$", r"$\angle \alpha_n$", r"$\log_{10}(\zeta_n)$"]
        label_keys = [0, 1, 2, 3]
        
        modes_curve = model_output[0].cpu().numpy()
        b, a = scipy.signal.butter(2, 0.3) # only light smoothing for mode curve
        smoothed_modes = scipy.signal.filtfilt(b, a, modes_curve)
        predicted_omegas, _ = rt.est_nat_freq_triangle_rise(smoothed_modes)

        if plot_phase:
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
                    h, l = subplot_labels(ax, x, model_output[j], label_name, N, Wn, predicted_omegas)
                    legend_handles.extend(h)
                    legend_labels.extend(l)
                    j += 1
        else:
            ax = axes[2]
            ax.set_ylabel(f"{label_names[0]}")
            h,l = subplot_labels(ax,x,model_output[0], label_names[0], N, Wn, predicted_omegas)
            legend_handles.extend(h)
            legend_labels.extend(l)
            
            ax = axes[3]
            ax.set_ylabel(f"{label_names[1]} / {scale_factors[0]:.2f}")
            h,l = subplot_labels(ax,x,model_output[1], label_names[1], N, Wn, predicted_omegas)
            legend_handles.extend(h)
            legend_labels.extend(l)
            
            ax = axes[4]
            ax.set_ylabel(f"{label_names[3]} / {scale_factors[2]:.2f}")
            h,l = subplot_labels(ax,x,model_output[3], label_names[3], N, Wn, predicted_omegas)
            legend_handles.extend(h)
            legend_labels.extend(l)

        axes[-1].set_xlabel("Normalised Frequency")
        # Remove duplicate labels
        unique_legend = dict(zip(legend_labels, legend_handles))
        fig.legend(unique_legend.values(), unique_legend.keys(),
                loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()
        return

def subplot_labels(axes_j, x, model_output_i_j, name, N, Wn, predicted_omegas):
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
    return handles, labels



