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
from scipy.interpolate import interp1d

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
        b, a = scipy.signal.butter(2, 0.25) # only light smoothing for mode curve
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
                loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
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


def compare_FRF(input_signal, all_outputs, scale_factors, FRF_type = 0, norm = True, q = 1):
    # Assumes model ouputs are modes, a mag, a phase, log10_zeta
    # FRF type: 0 for just magnitude, 1 for real and imaginary
    
    # Extract the predicted frequencies
    mode_channel = all_outputs[0]
    # only lightly smooth mode channel before passing to predict frequncies
    b, a = scipy.signal.butter(2,0.25)
    mode_channel = scipy.signal.filtfilt(b,a,mode_channel)
    predicted_freqs,predicted_freq_idxs = rt.est_nat_freq_triangle_rise(mode_channel)
    
    # point estimate [0], mean [1], variance [2]
    x = q # point estimate may be better when many modes which overlap
    a_mag = rt.estimate_parameter(all_outputs[1], predicted_freq_idxs)[x]
    a_phase = rt.estimate_parameter(all_outputs[2], predicted_freq_idxs)[x]
    log10_zeta = rt.estimate_parameter(all_outputs[3], predicted_freq_idxs)[x]
    
    a_mag_scale = scale_factors[0]
    a_phase_scale = scale_factors[1]
    log10_zeta_scale = scale_factors[2]
    
    # Reconstruct FRF
    signal_length = input_signal.shape[-1]
    try:
        H_v = rt.construct_FRF(
            predicted_freqs, 
            a_mag*a_mag_scale, 
            a_phase*a_phase_scale,
            10**(log10_zeta*log10_zeta_scale),
            signal_length,
            min_max=norm
        )
    except Exception as e:
        print('Exception: ', e)
        print(
            predicted_freqs, 
            a_mag*a_mag_scale, 
            a_phase*a_phase_scale,
            10**(log10_zeta*log10_zeta_scale)
        )

    # Compare the FRFs
    input_signal = input_signal.cpu().numpy()
    if FRF_type == 0:
        output = np.abs(H_v)
        return rt.quick_ms_diff(input_signal, output), output
    else:
        output_signal = np.array([np.real(H_v),np.imag(H_v)])
        MSE_real = rt.quick_ms_diff(input_signal, output_signal[0], signal_length)
        MSE_imag = rt.quick_ms_diff(input_signal, output_signal[1], signal_length)
        MSE = (MSE_real + MSE_imag)/2
        return MSE, output_signal


def plot_FRF_comparison(model, data, scale_factors, FRF_type=1, norm=True, plot_phase = True, q = 1):
    with torch.no_grad():
        model.eval()
        output = model(data).squeeze(0)
        data_copy = data
        data = data.squeeze(0)
        
        if FRF_type == 0:
            data = np.abs(data[0:2])
        else:
            data = data[0:2]
        error, H_v = compare_FRF(data, output.cpu().numpy(), scale_factors, FRF_type, norm, q=q)
        print('\n', error)
        
        modes_output = output[0].cpu().numpy()
        b, a = scipy.signal.butter(2,0.25) # only light smoothing for modes
        smoothed_modes = scipy.signal.filtfilt(b,a,modes_output)
        predicted_omegas, _ = rt.est_nat_freq_triangle_rise(smoothed_modes)
        
        frequencies = np.linspace(0, 1, data.shape[-1])
        if FRF_type == 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(frequencies, H_v, label='Predicted FRF', color='orange')
            ax.plot(frequencies, data, label='True FRF', color='blue')
            for k, omega in enumerate(predicted_omegas):
                ax.axvline(x=omega, color='cyan', linestyle=':', 
                                label=r'Predicted $\omega_n$' if k == 0 else '')
            fig.suptitle('Magnitude FRF Comparison: MSE = {:.4f}'.format(error))
            ax.legend(
                loc='upper center',
                ncol = 3,
                bbox_to_anchor = (0.5,1.15)
            )
            plt.tight_layout(rect=[0, 0, 1, 1.05])
        else:
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax[0].plot(frequencies, H_v[0], label='Predicted FRF', color='orange')
            ax[0].plot(frequencies, data[0], label='True FRF', color='blue')
            ax[1].plot(frequencies, H_v[1], label='Predicted FRF', color='orange')
            ax[1].plot(frequencies, data[1], label='True FRF', color='blue')
            for k, omega in enumerate(predicted_omegas):
                ax[0].axvline(x=omega, color='cyan', linestyle=':', 
                                    label=r'Predicted $\omega_n$' if k == 0 else '')
                ax[1].axvline(x=omega, color='cyan', linestyle=':')
            fig.suptitle('Complex FRF Comparison: MSE = {:.4f}'.format(error))
            ax[0].legend(
                loc='upper center',
                ncol = 3,
                bbox_to_anchor = (0.5,1.15),
            )
            plt.tight_layout(rect=[0, 0, 1, 1.02])
        
        
        plt.show(block=False)
        
        plot_predictions_all_labels(
            model, 
            data_copy, 
            label_defs=[True, True, True, True],
            scale_factors=scale_factors,
            plot_phase=plot_phase
        )


def resample_linear_scipy(data, new_length):
    original_length = len(data)
    x_original = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, new_length)
    
    # Create interpolation function
    interp_func = interp1d(x_original, data, kind='linear', fill_value='extrapolate')
    
    # Apply interpolation
    return interp_func(x_new)


def resample_linear_numpy(data, new_length):
    original_length = data.shape[-1]
    x_old = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, new_length)

    return np.interp(x_new, x_old, data)


def plot_FRF_cloud_single_sample(model,data,num_cloud_samples,scale_factors,FRF_type=1,signal_length=1024,q=1):
    ''' q=0 for point estimate or 1 for mean estimate for parameter estimation'''
    with torch.no_grad():
        model.eval()
        output = model(data).squeeze(0)
        data_copy = data
        data = data.squeeze(0)
        
        if FRF_type == 0:
            data = np.abs(data[0:2])
        else:
            data = data[0:2]
        error, H_v = compare_FRF(data, output.cpu().numpy(), scale_factors,FRF_type, norm=True, q=q)
        print('\n', error)
        
        modes_output = output[0].cpu().numpy()
        b, a = scipy.signal.butter(2,0.25) # only light smoothing for modes
        smoothed_modes = scipy.signal.filtfilt(b,a,modes_output)
        predicted_omegas, predicted_freq_idxs = rt.est_nat_freq_triangle_rise(smoothed_modes)
        
        window_scale = 0.6
        q = q # 0 for point estimate, 1 for mean
        a_mag = rt.estimate_parameter(output[1].cpu().numpy(), predicted_freq_idxs, window_scale=window_scale)
        a_phase = rt.estimate_parameter(output[2].cpu().numpy(), predicted_freq_idxs, window_scale=window_scale)
        log10_zeta = rt.estimate_parameter(output[3].cpu().numpy(), predicted_freq_idxs, window_scale=window_scale)
        
        a_mag_scale = scale_factors[0]
        a_phase_scale = scale_factors[1]
        log10_zeta_scale = scale_factors[2]
        
        num_predicted_modes = len(predicted_omegas)
        a_means = a_mag[q]*a_mag_scale
        a_vars = a_mag[2]*(a_mag_scale**2) # variance times by scale factor^2
        phi_means = a_phase[q]*a_phase_scale
        phi_vars = a_phase[2]*(a_phase_scale**2)
        log10_zeta_means = log10_zeta[q]*log10_zeta_scale
        log10_zeta_vars = log10_zeta[2]*(log10_zeta_scale**2)
        
        param_means = [a_means, phi_means, log10_zeta_means]
        param_vars = [a_vars, phi_vars, log10_zeta_vars]
        
        FRF_clouds = generate_random_FRFs(
            num_samples=num_cloud_samples,
            predicted_omegas=predicted_omegas,
            param_means=param_means,
            param_vars=param_vars,
            signal_length=signal_length,
            FRF_type=FRF_type,
        )
        
        frequencies = np.linspace(0, 1, data.shape[-1])
        if FRF_type == 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(frequencies, H_v, label='Predicted FRF', color='orange')
            ax.plot(frequencies, data, label='True FRF', color='blue')
            for k, omega in enumerate(predicted_omegas):
                ax.axvline(x=omega, color='cyan', linestyle=':', 
                                label=r'Predicted $\omega_n$' if k == 0 else '')
            fig.suptitle('Magnitude FRF Comparison: MSE = {:.4f}'.format(error))
            ax.legend(
                loc='upper center',
                ncol = 3,
                bbox_to_anchor = (0.5,1.15)
            )
            plt.tight_layout(rect=[0, 0, 1, 1.05])
        else:
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax[0].plot(frequencies, H_v[0], label='Predicted FRF', color='orange')
            ax[0].plot(frequencies, data[0], label='True FRF', color='blue')
            ax[1].plot(frequencies, H_v[1], label='Predicted FRF', color='orange')
            ax[1].plot(frequencies, data[1], label='True FRF', color='blue')
            for k, omega in enumerate(predicted_omegas):
                ax[0].axvline(x=omega, color='cyan', linestyle=':', 
                                    label=r'Predicted $\omega_n$' if k == 0 else '')
                ax[1].axvline(x=omega, color='cyan', linestyle=':')
            fig.suptitle('Complex FRF Comparison: MSE = {:.4f}'.format(error))
            ax[0].legend(
                loc='upper center',
                ncol = 3,
                bbox_to_anchor = (0.5,1.15),
            )
            plt.tight_layout(rect=[0, 0, 1, 1.02])
        
        plt.show(block=False)
        
        plot_predictions_all_labels(
            model, 
            data_copy, 
            label_defs=[True, True, True, True],
            scale_factors=scale_factors,
            plot_phase=True
        )


@jit(nopython=True)
def generate_random_FRFs(num_samples,predicted_omegas,param_means,param_vars,signal_length,FRF_type=1):
    # params in order |a_n|, phase a_n, log10(zeta_n)
    alpha_means = param_means[0]
    phi_means = param_means[1]
    log10zeta_means = param_means[2]
    alpha_vars = param_vars[0]
    phi_vars = param_vars[1]
    log10zeta_vars = param_vars[2]
    alpha_sigmas = np.sqrt(alpha_vars)
    phi_sigmas = np.sqrt(phi_vars)
    log10zeta_sigmas = np.sqrt(log10zeta_vars)
    if FRF_type == 0:
        data = np.empty((num_samples, 1, signal_length),dtype=np.float64)
    else:
        data = np.empty((num_samples, 2, signal_length),dtype=np.float64)
    
    frequencies = np.linspace(0,1,signal_length)
    num_modes = len(predicted_omegas)
    for i in range(num_samples):
        H_v = np.zeros(signal_length, dtype=np.complex128)
        for n in range(num_modes):
            omega_n = predicted_omegas[n]
            alpha_n = np.random.normal(alpha_means[n],alpha_sigmas[n])
            phi_n = np.random.normal(phi_means[n],phi_sigmas[n])
            zeta_n = 10**np.random.normal(log10zeta_means[n],log10zeta_sigmas[n])
            
            for j, w in enumerate(frequencies):
                H_f = 0.0j
                denominator = omega_n**2 - w**2 + 2j * zeta_n * w
                numerator = 1j*w*alpha_n*np.exp(1j*phi_n)
                H_f += numerator/denominator
                H_v[j] += H_f

        mag_no_norm = np.abs(H_v)
        mag = mag_no_norm
        real_pt = np.real(H_v)
        imag_pt = np.imag(H_v)
        max_mag = np.max(mag_no_norm)
        min_mag = np.min(mag_no_norm)
        mag = (mag_no_norm - min_mag)/(max_mag - min_mag)
        real_pt = real_pt * mag/mag_no_norm
        imag_pt = imag_pt * mag/mag_no_norm
        
        if FRF_type == 0:
            data[i, 0, :] = mag
        else:
            data[i, 0, :] = real_pt
            data[i, 1, :] = imag_pt
    return data
