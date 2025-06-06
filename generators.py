import numpy as np
import matplotlib.pyplot as plt
from numba import jit



@jit(nopython=True)
def n_channels_multi_labels_gen_scaled(
    num_samples: int,
    signal_length: int,
    enabled_inputs: np.ndarray,
    label_outputs: np.ndarray,
    noise: bool = True,
    alpha_min: float = 1,
    alpha_max: float = 2,
    sigma_min: float = 0.01,
    sigma_max: float = 0.1,
    zeta_min: float = 0.01,
    zeta_max: float = 0.1,
    alpha_phase_std_dev: float = np.pi/6,
    max_modes: int = 5,
    min_max: bool = True,
    params_out: bool = True,
    triangle_width: float = 0.05,
    square_width: float = 0.04,
    ):
    
    num_outs = np.sum(enabled_inputs)
    num_labels = np.sum(label_outputs)

    data = np.empty((num_samples, num_outs, signal_length),dtype=np.float64)
    labels = np.empty((num_samples, num_labels, signal_length),dtype=np.float64)
    if params_out:
        params = np.full((num_samples, max_modes, 4), np.nan, dtype=np.float64)
    scale_factors = np.empty(np.sum(label_outputs[1:]),dtype=np.float64) # mode labelling is not scaled
    target_A = triangle_width/3
    a,b = alpha_min, alpha_max # uniform distribution of alpha
    scale_alpha_mag = np.sqrt((a**2 + a*b + b**2)*square_width/(3*target_A))
    a,b = np.log10(zeta_max), np.log10(zeta_min) # unfiform distribution of log10(zeta)
    scale_log10_zeta = np.sqrt((a**2 + a*b + b**2)*square_width/(3*target_A))
    if alpha_phase_std_dev == 0:
        scale_alpha_phase = 1
    else:
        scale_alpha_phase = np.sqrt(square_width/target_A)*alpha_phase_std_dev
    c = 0
    if label_outputs[1]: # alpha mag
        scale_factors[c] = scale_alpha_mag
        c+=1
    if label_outputs[2]: #alpha phase
        scale_factors[c] = scale_alpha_phase
        c+=1
    if label_outputs[3]: #log10zeta
        scale_factors[c] = scale_log10_zeta
        c+=1

    frequencies = np.linspace(0,1,signal_length)
    
    for i in range(num_samples):
        
        H_v = np.zeros(signal_length, dtype=np.complex128)
        
        if label_outputs[0]: # mode triangles
            triangle_label = np.zeros(signal_length, dtype=np.float64)
        
        num_modes = np.random.randint(0,max_modes+1)
        
        alphas = np.random.choice(np.array([-1,1]),size=num_modes)*np.random.uniform(alpha_min, alpha_max, size=num_modes)
        
        alphas_phase = np.random.normal(0, alpha_phase_std_dev, size=num_modes)
        alphas_phase = np.clip(alphas_phase, -np.pi/2, np.pi/2)
        # restrict phase to be between -pi/2 and pi/2 so that sign of alpha does not flip
        
        zetas = 10**(np.random.uniform(np.log10(zeta_min), np.log10(zeta_max), size=num_modes))
        # no modes at very edge of frequency range
        omegas = np.random.uniform(0.01, 0.99, size=num_modes)
        
        if params_out:
            params[i, :num_modes, 0] = omegas
            params[i, :num_modes, 1] = alphas
            params[i, :num_modes, 2] = alphas_phase
            params[i, :num_modes, 3] = zetas
        
        if noise:
            # noise for each sample is different and random
            # should sigma be log uniform like zetas?
            sigma = np.random.uniform(sigma_min, sigma_max)
            
            # add noise to real and imaginary parts
            noise_arr = np.random.normal(0, np.exp(sigma), signal_length) + 1j*np.random.normal(0, np.exp(sigma), signal_length)
            H_v += noise_arr
        
        for n in range(num_modes):
            # to improve model add a random sign to alpha_j 
            alpha_n = alphas[n]
            zeta_n = zetas[n]
            omega_n = omegas[n]
            alpha_phase_n = alphas_phase[n]
            
            for j, w in enumerate(frequencies):
                H_f = 0.0j
                
                denominator = omega_n**2 - w**2 + 2j * zeta_n * w
                numerator = 1j*w*alpha_n*np.exp(1j*alpha_phase_n)
                
                H_f += numerator/denominator
                
                H_v[j] += H_f
            
            if label_outputs[0]: # mode triangles
                pulse = np.zeros(signal_length, dtype=np.float64)
                idx = np.argmin(np.abs(frequencies - omega_n))
                pulse[idx] = 1.0
            
                for j in range(idx + 1, signal_length):
                    if frequencies[j] > omega_n + triangle_width:
                        break
                    pulse[j] = 1.0 - (frequencies[j] - omega_n)/triangle_width
            
                triangle_label += pulse
        
        k = 0
        if label_outputs[0]: # mode triangles
            labels[i, k, :] = triangle_label
            k += 1
        
        if label_outputs[1]: # amplitude magnitude step function
            labels[i, k, :] = make_step_func_labels(omegas, alphas/scale_alpha_mag, frequencies, signal_length)
            k += 1
        
        if label_outputs[2]: # amplitude phase step function
            labels[i, k, :] = make_step_func_labels(omegas, alphas_phase/scale_alpha_phase, frequencies, signal_length)
            k += 1
        
        if label_outputs[3]: # damping ratio step function (log 10 scale)
            labels[i, k, :] = make_step_func_labels(omegas, np.log10(zetas)/scale_log10_zeta, frequencies, signal_length)
            k += 1


        mag_no_norm = np.abs(H_v)
        mag = mag_no_norm
        
        real_pt = np.real(H_v)
        imag_pt = np.imag(H_v)
        
        phase = np.angle(H_v) 
        #phase is by definition normalised between -pi and pi
        
        log10_mag = np.log10(mag_no_norm)
        #separate normalisation. Still min max normalisation
        
        if min_max:
            #aim is to maintain the phase of out if using real and imaginary parts
            max_mag = np.max(mag_no_norm)
            min_mag = np.min(mag_no_norm)
            mag = (mag_no_norm - min_mag)/(max_mag - min_mag)
            real_pt = real_pt * mag/mag_no_norm
            imag_pt = imag_pt * mag/mag_no_norm
            # log10_mag = (log10_mag - np.min(log10_mag))/(np.max(log10_mag) - np.min(log10_mag))
        
        # Populate output data based on enabled_inputs
        j = 0
        if enabled_inputs[0]:  # real
            data[i, j, :] = real_pt
            j += 1
        if enabled_inputs[1]:  # imag
            data[i, j, :] = imag_pt
            j += 1
        if enabled_inputs[2]:  # phase
            data[i, j, :] = phase
            j += 1
        if enabled_inputs[3]:  # mag
            data[i, j, :] = mag
            j += 1
        if enabled_inputs[4]:  # log_mag
            data[i, j, :] = log10_mag
            j += 1

    return data, labels, params if params_out else None, scale_factors






@jit(nopython=True)
def make_step_func_labels(
    natural_frequencies: np.ndarray,
    values: np.ndarray,
    frequencies: np.ndarray,
    signal_length: int,
    label_halfwidth: float = 0.02,
    ) -> np.ndarray:
    
    label = np.zeros(signal_length, dtype=np.float64)
    
    for i, freq in enumerate(frequencies):
        within_bandwidth = np.abs(natural_frequencies - freq) <= label_halfwidth
        if np.any(within_bandwidth):
            overlapping_indices = np.where(within_bandwidth)[0]
            distances = np.abs(natural_frequencies[overlapping_indices] - freq)
            nearest_idx = overlapping_indices[np.argmin(distances)]
            label[i] = values[nearest_idx]
        else:
            label[i] = 0.0
    return label


@jit(nopython=True)
def n_channels_multi_labels_gen(
    num_samples: int,
    signal_length: int,
    enabled_inputs: np.ndarray,
    label_outputs: np.ndarray,
    noise: bool = True,
    sigma_min: float = 0.01,
    sigma_max: float = 0.1,
    zeta_min: float = 0.0001,
    zeta_max: float = 0.5,
    alpha_phase_std_dev: float = np.pi/6,
    max_modes: int = 5,
    min_max: bool = False,
    params_out: bool = True,
    pulse_width: float = 0.02,
    ):
    
    num_outs = np.sum(enabled_inputs)
    num_labels = np.sum(label_outputs)

    data = np.empty((num_samples, num_outs, signal_length),dtype=np.float64)
    labels = np.empty((num_samples, num_labels, signal_length),dtype=np.float64)
    if params_out:
        params = np.full((num_samples, max_modes, 4), np.nan, dtype=np.float64)

    frequencies = np.linspace(0,1,signal_length)
    
    for i in range(num_samples):
        
        H_v = np.zeros(signal_length, dtype=np.complex128)
        
        if label_outputs[0]: # mode triangles
            triangle_label = np.zeros(signal_length, dtype=np.float64)
        
        num_modes = np.random.randint(0,max_modes+1)
        
        alphas = np.random.choice(np.array([-1,1]),size=num_modes)*np.random.uniform(1, 2, size=num_modes)
        
        alphas_phase = np.random.normal(0, alpha_phase_std_dev, size=num_modes)
        alphas_phase = np.clip(alphas_phase, -np.pi/2, np.pi/2)
        # restrict phase to be between -pi/2 and pi/2 so that sign of alpha does not flip
        
        zetas = 10**(np.random.uniform(np.log10(zeta_min), np.log10(zeta_max), size=num_modes))
        # no modes at very edge of frequency range
        omegas = np.random.uniform(0.001, 0.999, size=num_modes)
        
        if params_out:
            params[i, :num_modes, 0] = omegas
            params[i, :num_modes, 1] = alphas
            params[i, :num_modes, 2] = alphas_phase
            params[i, :num_modes, 3] = zetas
        
        if noise:
            # noise for each sample is different and random
            # should sigma be log uniform like zetas?
            sigma = np.random.uniform(sigma_min, sigma_max)
            
            # add noise to real and imaginary parts
            noise_arr = np.random.normal(0, np.exp(sigma), signal_length) + 1j*np.random.normal(0, np.exp(sigma), signal_length)
            H_v += noise_arr
        
        for n in range(num_modes):
            # to improve model add a random sign to alpha_j 
            alpha_n = alphas[n]
            zeta_n = zetas[n]
            omega_n = omegas[n]
            alpha_phase_n = alphas_phase[n]
            
            for j, w in enumerate(frequencies):
                H_f = 0.0j
                
                denominator = omega_n**2 - w**2 + 2j * zeta_n * w
                numerator = 1j*w*alpha_n*np.exp(1j*alpha_phase_n)
                
                H_f += numerator/denominator
                
                H_v[j] += H_f
            
            if label_outputs[0]: # mode triangles
                pulse = np.zeros(signal_length, dtype=np.float64)
                idx = np.argmin(np.abs(frequencies - omega_n))
                pulse[idx] = 1.0
            
                for j in range(idx + 1, signal_length):
                    if frequencies[j] > omega_n + pulse_width:
                        break
                    pulse[j] = 1.0 - (frequencies[j] - omega_n)/pulse_width
            
                triangle_label += pulse
        
        k = 0
        if label_outputs[0]: # mode triangles
            labels[i, k, :] = triangle_label
            k += 1
        
        if label_outputs[1]: # amplitude magnitude step function
            labels[i, k, :] = make_step_func_labels(omegas, alphas, frequencies, signal_length)
            k += 1
        
        if label_outputs[2]: # amplitude phase step function
            labels[i, k, :] = make_step_func_labels(omegas, alphas_phase, frequencies, signal_length)
            k += 1
        
        if label_outputs[3]: # damping ratio step function (log 10 scale)
            labels[i, k, :] = make_step_func_labels(omegas, np.log10(zetas), frequencies, signal_length)
            k += 1


        mag_no_norm = np.abs(H_v)
        mag = mag_no_norm
        
        real_pt = np.real(H_v)
        imag_pt = np.imag(H_v)
        
        phase = np.angle(H_v) 
        #phase is by definition normalised between -pi and pi
        
        log10_mag = np.log10(mag_no_norm)
        #separate normalisation. Still min max normalisation
        
        if min_max:
            #aim is to maintain the phase of out if using real and imaginary parts
            max_mag = np.max(mag_no_norm)
            min_mag = np.min(mag_no_norm)
            mag = (mag_no_norm - min_mag)/(max_mag - min_mag)
            real_pt = real_pt * mag/mag_no_norm
            imag_pt = imag_pt * mag/mag_no_norm
            # log10_mag = (log10_mag - np.min(log10_mag))/(np.max(log10_mag) - np.min(log10_mag))
        
        # Populate output data based on enabled_inputs
        j = 0
        if enabled_inputs[0]:  # real
            data[i, j, :] = real_pt
            j += 1
        if enabled_inputs[1]:  # imag
            data[i, j, :] = imag_pt
            j += 1
        if enabled_inputs[2]:  # phase
            data[i, j, :] = phase
            j += 1
        if enabled_inputs[3]:  # mag
            data[i, j, :] = mag
            j += 1
        if enabled_inputs[4]:  # log_mag
            data[i, j, :] = log10_mag
            j += 1

    return data, labels, params if params_out else None



@jit(nopython=True)
def n_channels_multi_labels_gen_old(
    num_samples: int,
    signal_length: int,
    enabled_inputs: np.ndarray,
    label_outputs: np.ndarray,
    noise: bool = True,
    sigma_min: float = 0.01,
    sigma_max: float = 0.1,
    zeta_min: float = 0.0001,
    zeta_max: float = 0.5,
    max_modes: int = 5,
    min_max: bool = False,
    params_out: bool = True,
    pulse_width: float = 0.02,
    ):
    
    num_outs = np.sum(enabled_inputs)
    num_labels = np.sum(label_outputs)

    data = np.empty((num_samples, num_outs, signal_length),dtype=np.float64)
    labels = np.empty((num_samples, num_labels, signal_length),dtype=np.float64)
    if params_out:
        params = np.full((num_samples, max_modes, 3), np.nan, dtype=np.float64)

    frequencies = np.linspace(0,1,signal_length)
    
    for i in range(num_samples):
        
        H_v = np.zeros(signal_length, dtype=np.complex128)
        
        if label_outputs[0]: # mode triangles
            triangle_label = np.zeros(signal_length, dtype=np.float64)
        
        num_modes = np.random.randint(0,max_modes+1)
        
        alphas = np.random.choice(np.array([-1,1]),size=num_modes)*np.random.uniform(1, 2, size=num_modes)
        
        zetas = 10**(np.random.uniform(np.log10(zeta_min), np.log10(zeta_max), size=num_modes))
        # no modes at very edge of frequency range
        omegas = np.random.uniform(0.001, 0.999, size=num_modes)
        
        if params_out:
            params[i, :num_modes, 0] = omegas
            params[i, :num_modes, 1] = alphas
            params[i, :num_modes, 2] = zetas
        
        if noise:
            # noise for each sample is different and random
            # should sigma be log uniform like zetas?
            sigma = np.random.uniform(sigma_min, sigma_max)
            
            # add noise to real and imaginary parts
            noise_arr = np.random.normal(0, np.exp(sigma), signal_length) + 1j*np.random.normal(0, np.exp(sigma), signal_length)
            H_v += noise_arr
        
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
            
            if label_outputs[0]: # mode triangles
                pulse = np.zeros(signal_length, dtype=np.float64)
                idx = np.argmin(np.abs(frequencies - omega_n))
                pulse[idx] = 1.0
            
                for j in range(idx + 1, signal_length):
                    if frequencies[j] > omega_n + pulse_width:
                        break
                    pulse[j] = 1.0 - (frequencies[j] - omega_n)/pulse_width
            
                triangle_label += pulse
        
        k = 0
        if label_outputs[0]: # mode triangles
            labels[i, k, :] = triangle_label
            k += 1
        
        if label_outputs[1]: # amplitude step function
            labels[i, k, :] = make_step_func_labels(omegas, alphas, frequencies, signal_length)
            k += 1
        
        if label_outputs[2]: # damping ratio step function (log 10 scale)
            labels[i, k, :] = make_step_func_labels(omegas, np.log10(zetas), frequencies, signal_length)
            k += 1
        
        if label_outputs[3]: # omega step function
            labels[i, k, :] = make_step_func_labels(omegas, omegas, frequencies, signal_length)
            k += 1


        mag_no_norm = np.abs(H_v)
        mag = mag_no_norm
        
        real_pt = np.real(H_v)
        imag_pt = np.imag(H_v)
        
        phase = np.angle(H_v) 
        #phase is by definition normalised between -pi and pi
        
        log10_mag = np.log10(mag_no_norm)
        #separate normalisation. Still min max normalisation
        
        if min_max:
            #aim is to maintain the phase of out if using real and imaginary parts
            max_mag = np.max(mag_no_norm)
            min_mag = np.min(mag_no_norm)
            mag = (mag_no_norm - min_mag)/(max_mag - min_mag)
            real_pt = real_pt * mag/mag_no_norm
            imag_pt = imag_pt * mag/mag_no_norm
            log10_mag = (log10_mag - np.min(log10_mag))/(np.max(log10_mag) - np.min(log10_mag))
        
        # Populate output data based on enabled_inputs
        j = 0
        if enabled_inputs[0]:  # mag
            data[i, j, :] = mag
            j += 1
        if enabled_inputs[1]:  # real
            data[i, j, :] = real_pt
            j += 1
        if enabled_inputs[2]:  # imag
            data[i, j, :] = imag_pt
            j += 1
        if enabled_inputs[3]:  # phase
            data[i, j, :] = phase
            j += 1
        if enabled_inputs[4]:  # log_mag
            data[i, j, :] = log10_mag
            j += 1

    return data, labels, params if params_out else None



#with noise and normalisation
@jit(nopython=True)
def n_channels_triangle_gen(
    num_samples: int,
    signal_length: int,
    enabled_inputs: np.ndarray,
    noise: bool = True,
    sigma_min: float = 0.01,
    sigma_max: float = 0.1,
    zeta_min: float = 0.0001,
    zeta_max: float = 0.5,
    max_modes: int = 5,
    min_max: bool = False,
    params_out: bool = True,
    pulse_width: float = 0.02,
    ):
    
    num_outs = np.sum(enabled_inputs)

    data = np.empty((num_samples, num_outs, signal_length),dtype=np.float64)
    labels = np.empty((num_samples,signal_length),dtype=np.float64)
    if params_out:
        params = np.full((num_samples, max_modes, 3), np.nan, dtype=np.float64)

    frequencies = np.linspace(0,1,signal_length)
    
    for i in range(num_samples):
        
        H_v = np.zeros(signal_length, dtype=np.complex128)
        label = np.zeros(signal_length, dtype=np.float64)
        
        num_modes = np.random.randint(0,max_modes+1)
        
        alphas = np.random.choice(np.array([-1,1]),size=num_modes)*np.random.uniform(1, 2, size=num_modes)
        
        zetas = 10**(np.random.uniform(np.log10(zeta_min), np.log10(zeta_max), size=num_modes))
        # no modes at very edge of frequency range
        omegas = np.random.uniform(0.001, 0.999, size=num_modes)
        
        if params_out:
            params[i, :num_modes, 0] = omegas
            params[i, :num_modes, 1] = alphas
            params[i, :num_modes, 2] = zetas
        
        if noise:
            # noise for each sample is different and random
            # should sigma be log uniform like zetas?
            sigma = np.random.uniform(sigma_min, sigma_max)
            
            # add noise to real and imaginary parts
            noise_arr = np.random.normal(0, np.exp(sigma), signal_length) + 1j*np.random.normal(0, np.exp(sigma), signal_length)
            H_v += noise_arr
        
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
            
            pulse = np.zeros(signal_length, dtype=np.float64)
            idx = np.argmin(np.abs(frequencies - omega_n))
            pulse[idx] = 1.0
            
            for j in range(idx + 1, signal_length):
                if frequencies[j] > omega_n + pulse_width:
                    break
                pulse[j] = 1.0 - (frequencies[j] - omega_n)/pulse_width
            
            label = label + pulse

        mag_no_norm = np.abs(H_v)
        mag = mag_no_norm
        
        real_pt = np.real(H_v)
        imag_pt = np.imag(H_v)
        
        phase = np.angle(H_v) 
        #phase is by definition normalised between -pi and pi
        
        log10_mag = np.log10(mag_no_norm)
        #separate normalisation. Still min max normalisation
        
        if min_max:
            #aim is to maintain the phase of out if using real and imaginary parts
            max_mag = np.max(mag_no_norm)
            min_mag = np.min(mag_no_norm)
            mag = (mag_no_norm - min_mag)/(max_mag - min_mag)
            real_pt = real_pt * mag/mag_no_norm
            imag_pt = imag_pt * mag/mag_no_norm
            log10_mag = (log10_mag - np.min(log10_mag))/(np.max(log10_mag) - np.min(log10_mag))
        
        # Populate output data based on enabled_inputs
        j = 0
        if enabled_inputs[0]:  # mag
            data[i, j, :] = mag
            j += 1
        if enabled_inputs[1]:  # real
            data[i, j, :] = real_pt
            j += 1
        if enabled_inputs[2]:  # imag
            data[i, j, :] = imag_pt
            j += 1
        if enabled_inputs[3]:  # phase
            data[i, j, :] = phase
            j += 1
        if enabled_inputs[4]:  # log_mag
            data[i, j, :] = log10_mag
            j += 1

        labels[i, :] = label

    return data, labels, params if params_out else None



#with noise and normalisation
@jit(nopython=True)
def n_channels_gen(
    num_samples: int,
    signal_length: int,
    enabled_inputs: np.ndarray,
    noise: bool = True,
    sigma_min: float = 0.01,
    sigma_max: float = 0.1,
    zeta_min: float = 0.0001,
    zeta_max: float = 0.5,
    three_db_bandwidth: bool = True,
    fixed_bandwidth: float = 0.02,
    max_modes: int = 5,
    min_max: bool = False,
    multiclass: int = 0,
    params_out: bool = True,
    ):
    
    num_outs = np.sum(enabled_inputs)

    data = np.empty((num_samples, num_outs, signal_length),dtype=np.float64)
    labels = np.empty((num_samples,signal_length),dtype=np.int32)
    if params_out:
        params = np.full((num_samples, max_modes, 3), np.nan, dtype=np.float64)

    frequencies = np.linspace(0,1,signal_length)
    for i in range(num_samples):
        
        H_v = np.zeros(signal_length, dtype=np.complex128)
        label = np.zeros(signal_length, dtype=np.int32)
        
        num_modes = np.random.randint(0,max_modes+1)
        #when noise is added change num_modes to include 0
        # to improve model generalisation add a random sign to alpha_j 
        
        alphas = np.random.choice(np.array([-1,1]),size=num_modes)*np.random.uniform(1, 2, size=num_modes)
        #alphas = np.random.uniform(1, 2, size=5)
        
        zetas = 10**(np.random.uniform(np.log10(zeta_min), np.log10(zeta_max), size=num_modes))
        #no modes at very edge of frequency range
        omegas = np.random.uniform(0.001, 0.999, size=num_modes)
        
        if params_out:
            params[i, :num_modes, 0] = omegas
            params[i, :num_modes, 1] = alphas
            params[i, :num_modes, 2] = zetas
        
        if noise:
            #noise for each sample is different and random
            #should sigma be log uniform like zetas?
            sigma = np.random.uniform(sigma_min, sigma_max)
            
            #add noise to real and imaginary parts
            noise_arr = np.random.normal(0, np.exp(sigma), signal_length) + 1j*np.random.normal(0, np.exp(sigma), signal_length)
            H_v += noise_arr
        
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
            
            if three_db_bandwidth:
                bandwidth = 2*omega_n*zeta_n
            else:
                bandwidth = fixed_bandwidth
            #multiclass = 0 is binary classification
            #multiclass = 1 is multiclass classification with max label value of 2
            #multiclass = 2 is multiclass classification with max label value of max_modes
            if multiclass == 0:
                label[(frequencies >= omega_n - bandwidth/2) & (frequencies <= omega_n + bandwidth/2)] = 1
            elif multiclass == 1 or multiclass == 2:
                label[(frequencies >= omega_n - bandwidth/2) & (frequencies <= omega_n + bandwidth/2)] += 1
        
        #restrict multiclass labels to 2 if multiclass == 1
        if multiclass == 1:
            label = np.minimum(label, 2).astype(np.int32) #label must remain as a numpy array for JIT w numba

        mag_no_norm = np.abs(H_v)
        mag = mag_no_norm
        
        real_pt = np.real(H_v)
        imag_pt = np.imag(H_v)
        
        phase = np.angle(H_v) 
        #phase is by definition normalised between -pi and pi
        
        log10_mag = np.log10(mag_no_norm)
        #separate normalisation. Still min max normalisation
        
        if min_max:
            #aim is to maintain the phase of out if using real and imaginary parts
            max_mag = np.max(mag_no_norm)
            min_mag = np.min(mag_no_norm)
            mag = (mag_no_norm - min_mag)/(max_mag - min_mag)
            real_pt = real_pt * mag/mag_no_norm
            imag_pt = imag_pt * mag/mag_no_norm
            log10_mag = (log10_mag - np.min(log10_mag))/(np.max(log10_mag) - np.min(log10_mag))
        
        # Populate output data based on enabled_inputs
        j = 0
        if enabled_inputs[0]:  # mag
            data[i, j, :] = mag
            j += 1
        if enabled_inputs[1]:  # real
            data[i, j, :] = real_pt
            j += 1
        if enabled_inputs[2]:  # imag
            data[i, j, :] = imag_pt
            j += 1
        if enabled_inputs[3]:  # phase
            data[i, j, :] = phase
            j += 1
        if enabled_inputs[4]:  # log_mag
            data[i, j, :] = log10_mag
            j += 1

        labels[i, :] = label

    return data, labels, params if params_out else None



#with noise and normalisation
@jit(nopython=True)
def n_channels_stepped_gen(
    num_samples: int,
    signal_length: int,
    enabled_inputs: np.ndarray,
    noise: bool = True,
    sigma_min: float = 0.01,
    sigma_max: float = 0.1,
    zeta_min: float = 0.0001,
    zeta_max: float = 0.5,
    max_modes: int = 5,
    min_max: bool = False,
    params_out: bool = True,
    ):
    
    num_outs = np.sum(enabled_inputs)

    data = np.empty((num_samples, num_outs, signal_length),dtype=np.float64)
    labels = np.empty((num_samples,signal_length),dtype=np.int32)
    if params_out:
        params = np.full((num_samples, max_modes, 3), np.nan, dtype=np.float64)

    frequencies = np.linspace(0,1,signal_length)
    for i in range(num_samples):
        
        H_v = np.zeros(signal_length, dtype=np.complex128)
        label = np.zeros(signal_length, dtype=np.int32)
        
        num_modes = np.random.randint(0,max_modes+1)
        #when noise is added change num_modes to include 0
        # to improve model generalisation add a random sign to alpha_j 
        
        alphas = np.random.choice(np.array([-1,1]),size=num_modes)*np.random.uniform(1, 2, size=num_modes)
        #alphas = np.random.uniform(1, 2, size=5)
        
        zetas = 10**(np.random.uniform(np.log10(zeta_min), np.log10(zeta_max), size=num_modes))
        #no modes at very edge of frequency range
        omegas = np.random.uniform(0.001, 0.999, size=num_modes)
        
        if params_out:
            params[i, :num_modes, 0] = omegas
            params[i, :num_modes, 1] = alphas
            params[i, :num_modes, 2] = zetas
        
        if noise:
            #noise for each sample is different and random
            #should sigma be log uniform like zetas?
            sigma = np.random.uniform(sigma_min, sigma_max)
            
            #add noise to real and imaginary parts
            noise_arr = np.random.normal(0, np.exp(sigma), signal_length) + 1j*np.random.normal(0, np.exp(sigma), signal_length)
            H_v += noise_arr
        
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
            
            label[(frequencies >= omega_n)] += 1 # Step function labelling

        mag_no_norm = np.abs(H_v)
        mag = mag_no_norm
        
        real_pt = np.real(H_v)
        imag_pt = np.imag(H_v)
        
        phase = np.angle(H_v) 
        #phase is by definition normalised between -pi and pi
        
        log10_mag = np.log10(mag_no_norm)
        #separate normalisation. Still min max normalisation
        
        if min_max:
            #aim is to maintain the phase of out if using real and imaginary parts
            max_mag = np.max(mag_no_norm)
            min_mag = np.min(mag_no_norm)
            mag = (mag_no_norm - min_mag)/(max_mag - min_mag)
            real_pt = real_pt * mag/mag_no_norm
            imag_pt = imag_pt * mag/mag_no_norm
            log10_mag = (log10_mag - np.min(log10_mag))/(np.max(log10_mag) - np.min(log10_mag))
        
        # Populate output data based on enabled_inputs
        j = 0
        if enabled_inputs[0]:  # mag
            data[i, j, :] = mag
            j += 1
        if enabled_inputs[1]:  # real
            data[i, j, :] = real_pt
            j += 1
        if enabled_inputs[2]:  # imag
            data[i, j, :] = imag_pt
            j += 1
        if enabled_inputs[3]:  # phase
            data[i, j, :] = phase
            j += 1
        if enabled_inputs[4]:  # log_mag
            data[i, j, :] = log10_mag
            j += 1

        labels[i, :] = label

    return data, labels, params if params_out else None



#with noise and normalisation
@jit(nopython=True)
def real_imag(
    num_samples: int,
    signal_length: int,
    sigma_max: float = 0.1,
    max_modes: int = 5,
    normalise = None,
    norm_95: bool = False,
    min_max: bool = False,
    ):
    
    data = np.empty((num_samples, 2, signal_length),dtype=np.float64)
    labels = np.empty((num_samples,signal_length),dtype=np.int32)

    frequencies = np.linspace(0,1,signal_length)
    for i in range(num_samples):
        
        H_v = np.zeros(signal_length, dtype=np.complex128)
        label = np.zeros(signal_length, dtype=np.int32)
        
        num_modes = np.random.randint(0,max_modes+1)
        #when noise is added change num_modes to include 0
        # to improve model generalisation add a random sign to alpha_j 
        
        alphas = np.random.choice(np.array([-1,1]),size=max_modes)*np.random.uniform(1, 2, size=5)
        #alphas = np.random.uniform(1, 2, size=5)
        
        zetas = 10**(np.random.uniform(-2, -1, size=max_modes))
        #no modes at very edge of frequency range
        omegas = np.random.uniform(0.001, 0.999, size=max_modes)
        
        #noise for each sample is different and random
        sigma = np.random.uniform(0.01, sigma_max)
        
        #add noise to real and imaginary parts
        noise = np.random.normal(0, np.exp(sigma), signal_length) + 1j*np.random.normal(0, np.exp(sigma), signal_length)
        H_v += noise
        
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
        
        mag = np.abs(H_v)
        
        real_pt = np.real(H_v)
        imag_pt = np.imag(H_v)
        
        if normalise is not None:
            real_pt = normalise(real_pt)
            imag_pt = normalise(imag_pt)
        
        if norm_95:
            max = np.max(mag)
            mag = mag/(0.95*max)
            real_pt = real_pt/(0.95*max)
            imag_pt = imag_pt/(0.95*max)
        
        if min_max:
            #aim is to maintain the phase of out if using real and imaginary parts
            max_mag = np.max(mag)
            min_mag = np.min(mag)
            mag_normed = (mag - min_mag)/(max_mag - min_mag)
            real_pt = real_pt * mag_normed/mag
            imag_pt = imag_pt * mag_normed/mag

        data[i, 0, :] = real_pt
        data[i, 1, :] = imag_pt
        labels[i, :] = label

    return data, labels



@jit(nopython=True)
def mag_1D_noise_normalised(
    num_samples: int,
    signal_length: int,
    sigma_max: float = 0.1,
    normalise = None,
    norm_95: bool = False,
    min_max: bool = False,
    ):
    
    data = np.empty((num_samples,signal_length),dtype=np.float64)
    labels = np.empty((num_samples,signal_length),dtype=np.int32)
    ws = np.empty((num_samples),dtype=np.float64)
    zs = np.empty((num_samples),dtype=np.float64)
    a_s = np.empty((num_samples),dtype=np.float64)

    frequencies = np.linspace(0,1,signal_length)
    for i in range(num_samples):
        
        H_v = np.zeros(signal_length, dtype=np.complex128)
        label = np.zeros(signal_length, dtype=np.int32)
        
        #max 5 modes
        num_modes = np.random.randint(0,6)
        #when noise is added change num_modes to include 0
        # to improve model generalisation add a random sign to alpha_j 
        
        alphas = np.random.choice(np.array([-1,1]),size=5)*np.random.uniform(1, 2, size=5)
        #alphas = np.random.uniform(1, 2, size=5)
        
        zetas = 10**(np.random.uniform(-2, -1, size=5))
        #no modes at very edge of frequency range
        omegas = np.random.uniform(0.001, 0.999, size=5)
        
        #noise for each sample is different and random
        sigma = np.random.uniform(0.01, sigma_max)
        
        #add noise to real and imaginary parts
        noise = np.random.normal(0, np.exp(sigma), signal_length) + 1j*np.random.normal(0, np.exp(sigma), signal_length)
        H_v += noise
        
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
        
        mag = np.abs(H_v)
        out = np.abs(H_v)
        
        if normalise is not None:
            out = normalise(out)
        
        if norm_95:
            max = np.max(mag)
            mag = mag/(0.95*max)
            out = out/(0.95*max)
        
        if min_max:
            #aim is to maintain the phase of out if using real and imaginary parts
            max_mag = np.max(mag)
            min_mag = np.min(mag)
            mag_normed = (mag - min_mag)/(max_mag - min_mag)
            out = out * mag_normed/mag 
        
        #ws[i] = omegas[:num_modes]
        #zs[i] = zetas[:num_modes]
        #a_s[i] = alphas[:num_modes]
        data[i, :] = out 
        labels[i, :] = label

    return data, labels



@jit(nopython=True)
def mag_1D_noise(num_samples=1000, signal_length=1000, sigma_max=0.1):
    
    data = np.empty((num_samples,signal_length),dtype=np.float64)
    labels = np.empty((num_samples,signal_length),dtype=np.int32)
    ws = np.empty((num_samples),dtype=np.float64)
    zs = np.empty((num_samples),dtype=np.float64)
    a_s = np.empty((num_samples),dtype=np.float64)

    frequencies = np.linspace(0,1,signal_length)
    for i in range(num_samples):
        
        H_v = np.zeros(signal_length, dtype=np.complex128)
        label = np.zeros(signal_length, dtype=np.int32)
        
        #max 5 modes
        num_modes = np.random.randint(0,6)
        #when noise is added change num_modes to include 0
        # to improve model add a random sign to alpha_j 
        alphas = np.random.uniform(1, 2, size=5)
        zetas = 10**(np.random.uniform(-2, -1, size=5))
        #no modes at very edge of frequency range
        omegas = np.random.uniform(0.001, 0.999, size=5)
        
        #noise for each sample is different and random
        sigma = np.random.uniform(0.01, sigma_max)
        
        #add noise to real and imaginary parts
        noise = np.random.normal(0, np.exp(sigma), signal_length) + 1j*np.random.normal(0, np.exp(sigma), signal_length)
        H_v += noise
        
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
        
        #ws[i] = omegas[:num_modes]
        #zs[i] = zetas[:num_modes]
        #a_s[i] = alphas[:num_modes]
        data[i, :] = np.abs(H_v) #use signal magnitude for now
        labels[i, :] = label

    return data, labels



#debugging of below method required, setting of ws, zs, a_s not working with jit
#aim of method is to return the parameters of the modes in the signal for better plotting
@jit(nopython=True)
def mag_1D_no_noise_new(num_samples=1000, signal_length=1000):
    
    data = np.empty((num_samples,signal_length),dtype=np.float64)
    labels = np.empty((num_samples,signal_length),dtype=np.int32)
    ws = np.empty((num_samples),dtype=np.float64)
    zs = np.empty((num_samples),dtype=np.float64)
    a_s = np.empty((num_samples),dtype=np.float64)

    frequencies = np.linspace(0,1,signal_length)
    for i in range(num_samples):
        
        H_v = np.zeros(signal_length, dtype=np.complex128)
        label = np.zeros(signal_length, dtype=np.int32)
        
        #max 5 modes
        num_modes = np.random.randint(1,6)
        #when noise is added change num_modes to include 0
        # to improve model add a random sign to alpha_j 
        alphas = np.random.uniform(1, 2, size=5)
        zetas = 10**(np.random.uniform(-2, -1, size=5))
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
        
        ws[i] = omegas[:num_modes]
        zs[i] = zetas[:num_modes]
        a_s[i] = alphas[:num_modes]
        data[i, :] = np.abs(H_v) #use signal magnitude for now
        labels[i, :] = label

    return data, labels, ws, zs, a_s



@jit(nopython=True)
def mag_1D_no_noise(num_samples=1000, signal_length=1000):
    
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
        zetas = 10**(np.random.uniform(-2, -1, size=5))
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




