import numpy as np
import matplotlib.pyplot as plt

num_samples: int = 5
signal_length: int = 1024
sigma_max: float = 0.1

data1 = np.empty((num_samples,signal_length),dtype=np.float64)
labels = np.empty((num_samples,signal_length),dtype=np.int32)
zeros = np.zeros(signal_length)
ones = zeros + 1
minusones = zeros - 1

frequencies = np.linspace(0,1,signal_length)
for i in range(num_samples):
    
    H_v = np.zeros(signal_length, dtype=np.complex128)
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
    #H_v += noise

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
        #bandwidth = 2*omega_n*zeta_n
        bandwidth = 0.005
        label[(frequencies >= omega_n - bandwidth) & (frequencies <= omega_n + bandwidth)] = 1
        
        #label[(frequencies >= omega_n - bandwidth) & (frequencies <= omega_n + bandwidth)] += 1
    
    mag = np.abs(H_v)
    phase_no_norm = np.angle(H_v)
    
    out_95 = [np.real(H_v),np.imag(H_v)]
    out_min_max = [np.real(H_v),np.imag(H_v)]
    
    
    max_mag = np.max(mag)
    min_mag  = np.min(mag)
    
    mag_95 = mag/(0.95*max_mag)
    out_95 = out_95/(0.95*max_mag)
    
    #aim is to maintain the phase of out if using real and imaginary parts
    mag_min_max = (mag - min_mag)/(max_mag - min_mag)
    out_min_max = out_min_max * mag_min_max/mag 

    #data1[i, :] = out
    labels[i, :] = label
    
    phase_min_max = np.angle(out_min_max[0] + 1j*out_min_max[1])
    phase_95 = np.angle(out_95[0] + 1j*out_95[1])
    
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, zeros, color='black')
    plt.plot(frequencies, ones, color='black')
    plt.plot(frequencies, minusones, color='black')
    
    #plt.plot(frequencies, mag_95, label='|H_v(f)| 0.95', color='blue')
    #plt.plot(frequencies, mag_min_max, label='|H_v(f)| min_max', color='red')
    
    #plt.plot(frequencies, phase_min_max, label='Phase min_max', color='blue', linestyle='--')
    #plt.plot(frequencies, phase_95, label='Phase 95', color='red', linestyle='--')
    #plt.plot(frequencies, phase_no_norm, label='Phase no norm', color='green', linestyle='--')
    
    #plt.plot(frequencies, out_min_max[0], label='Re[H_v(f)] min_max', linestyle='--', color='orange')
    #plt.plot(frequencies, out_min_max[1], label='Im[H_v(f)] min_max', linestyle=':', color='green')
    
    #plt.plot(frequencies, out_95[0], label='Re[H_v(f)] 0.95', linestyle='--', color='blue')
    #plt.plot(frequencies, out_95[1], label='Im[H_v(f)] 0.95', linestyle=':', color='red')
    
    plt.plot(frequencies, labels[i] * 0.5, label='Labels (scaled)', linestyle='--')
    
    plt.title('Sample Vibration Signal with Labels')
    plt.xlabel('Normalised Frequency')
    plt.ylabel('Amplitude / Label')
    plt.legend()
    plt.show()