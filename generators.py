import numpy as np
import matplotlib.pyplot as plt
from numba import jit


#finish adding normalisation function
#see vaibhavs report for best normalisation function
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
        alphas = np.random.choice([-1,1],size=5)*np.random.uniform(1, 2, size=5)
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