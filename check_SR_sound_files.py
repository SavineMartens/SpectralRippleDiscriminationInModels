import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
import os

data_dir = 'C:\\Users\\ssmmartens\\OneDrive - LUMC\\Sounds\\Ripple' 
# List your .wav files here
# file_paths = {
#     "1": "i_1.000_1.wav",
#     "3": "i_1.000_3.wav",
#     "8": "i_1.000_8.wav",
#     "10": "i_1.000_10.wav"
# }

ripple_to_plot = '1.000'

for phase_shift in range(1,10):

    file_paths = {
    # "i": "i_"+ ripple_to_plot + "_" + str(phase_shift) + ".wav",
    "i1" : "i1_"+ ripple_to_plot + "_" + str(phase_shift) + ".wav",
    "i2" : "i2_"+ ripple_to_plot + "_" + str(phase_shift) + ".wav",
    "s" : "s_"+ ripple_to_plot + "_" + str(phase_shift) + ".wav",
}

    # Container for FFT results
    fft_data = {}

    # Read each file, perform FFT, and store magnitude and phase
    for label, path in file_paths.items():
        print(path)
        sr, signal = wavfile.read(os.path.join(data_dir, path))
        
        # Use only one channel if stereo
        if signal.ndim > 1:
            signal = signal[:, 0]
        
        # Normalize and take FFT
        signal = signal / np.max(np.abs(signal))
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sr)

        # Store magnitude and phase (only positive frequencies)
        pos_freqs = freqs > 0
        fft_data[label] = {
            "freqs": freqs[pos_freqs],
            "magnitude": np.abs(fft_result)[pos_freqs],
            "phase": np.angle(fft_result)[pos_freqs],
            "label": path
        }

    # Plot magnitude spectra
    plt.figure(num=1, figsize=(12, 6))
    for label, data in fft_data.items():
        plt.plot(data["freqs"], 20 * np.log10(data["magnitude"] + 1e-12), label=data["label"])
    plt.title("Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xscale('log', base=2)
    plt.xlim(100, 5000)  
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    # Plot phase spectra
    # plt.figure(figsize=(12, 6))
    # for label, data in fft_data.items():
    #     plt.plot(data["freqs"], data["phase"], label=data['label'])
    # plt.title("Phase Spectrum")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Phase (radians)")
    # plt.xscale('log', base=2)
    # plt.xlim(100, 5000)  
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    
plt.show()