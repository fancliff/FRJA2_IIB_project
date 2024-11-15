import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import routines as rt

num_samples: int = 5
signal_length: int = 1024
sigma_max: float = 0.1

data1 = np.empty((num_samples,signal_length),dtype=np.float64)
data2 = np.empty((num_samples,signal_length),dtype=np.float64)

frequencies = np.linspace(0,1,signal_length)
for i in range(num_samples):
    
    H_v = np.zeros(signal_length, dtype=np.complex128)
    H_v_2 = np.zeros(signal_length, dtype=np.complex128)
    label = np.zeros(signal_length, dtype=np.int32)
    
    #max 5 modes
    num_modes = np.random.randint(0,6)
    #when noise is added change num_modes to include 0
    # to improve model generalisation add a random sign to alpha_j 
    
    alphas = np.random.choice(np.array([-1,1]),size=5)*np.random.uniform(1, 2, size=5)
    
    zetas = 10**(np.random.uniform(-4, -2, size=5))
    #no modes at very edge of frequency range
    omegas = np.random.uniform(0.001, 0.999, size=5)
    
    #noise for each sample is different and random
    sigma = np.random.uniform(0.01, sigma_max)
    

    #add noise to real and imaginary parts
    noise = np.random.normal(0, np.exp(sigma), signal_length) + 1j*np.random.normal(0, np.exp(sigma), signal_length)
    H_v += noise
    H_v_2 += noise

    
    for n in range(num_modes):
        # to improve model add a random sign to alpha_j 
        alpha_n = alphas[n]
        zeta_n = zetas[n]
        omega_n = omegas[n]
        
        for j, w in enumerate(frequencies):
            H_f = 0.0j
            H_f_2 = 0.0j
            
            denominator = omega_n**2 - w**2 + 2j * zeta_n * w
            numerator = 1j*w*alpha_n
            
            H_f += numerator/denominator
            H_f_2 += abs(numerator)/denominator
            
            H_v[j] += H_f
            H_v_2[j] += H_f_2
        
        #set label bandwidth to 3db or arbitrary value
        #bandwidth = 0.02 #(8 frequency points wide if 400 points)
        bandwidth = 2*omega_n*zeta_n
        label[(frequencies >= omega_n - bandwidth) & (frequencies <= omega_n + bandwidth)] = 1
    
    plt.figure(figsize=(12, 6))
    #plt.plot(frequencies, np.abs(H_v), label='|H_v(f)|', color='blue')
    #plt.plot(frequencies, np.abs(H_v_2), label='|H_v_2(f)|', color='red')
    plt.plot(frequencies, 20*np.log10(np.abs(H_v)), label='20log10|H_v(f)|', color='blue', linestyle='--')
    plt.plot(frequencies, 20*np.log10(np.abs(H_v_2)), label='20log10|H_v_2(f)|', color='red', linestyle='--')
    #plt.plot(frequencies, np.real(H_v), label='Re[H_v(f)]', linestyle='--', color='orange')
    #plt.plot(frequencies, np.imag(H_v), label='Im[H_v(f)]', linestyle=':', color='green')
    #plt.plot(frequencies, labels[i] * 5, label='Labels (scaled)', linestyle='--')
    plt.title('Sample Vibration Signal with Labels')
    plt.xlabel('Normalised Frequency')
    plt.ylabel('Amplitude / Label')
    plt.legend()
    plt.show()