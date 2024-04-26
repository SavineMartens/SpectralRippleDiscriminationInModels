import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from scipy.io import wavfile
from scipy.signal import hilbert
from scipy import signal
from utilities import *
from SMRTvsSR import get_spectrum


####################### 
# Spectral Ripple TRUE SOUND
SR_sound_dir = './sounds/spectral ripple/'
name_list = ['s_1.000_1', 'i1_1.000_1']
label_list = ['standard', 'inverted']
# fig, axes = plt.subplots(2,1, sharex=True, sharey=True, figsize=(10, 5))
fig = plt.figure(figsize=(8,3))
plt.subplots_adjust(bottom=0.179)
plt.xscale('log', base=2)
plt.xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
for n, name in enumerate(name_list):
    outline_SR, frequency = get_spectrum(glob.glob(SR_sound_dir + '*' + name + '.*')[0])
    plt.plot(frequency, outline_SR, label=label_list[n])
    plt.title('Spectral ripple by Won et al. (1 RPO)', fontsize=18)
    plt.ylabel('Normalized power', fontsize=18)
    plt.xlabel('Frequency [Hz]', fontsize=18)
    plt.xlim(100, 8e3)
    plt.ylim(0,1)
    plt.legend(fontsize=18)

fig.savefig('./sounds/SR_1RPO.png')
# plt.show()

# # FAKE SOUND
# mod_depth = 30
# phase = 0
# # log2_freq_offset = 
# ripple = (1-(mod_depth/2))+ (np.sin((((np.log(frequency,2))-log2_freq_offset)*2*np.pi*ripples_per_octave)+
#            ((phase/360)*(2*np.pi))))*(0.5*mod_depth)


########################
# # SMRT
# SMRT_sound_dir = './sounds/SMRT/'
# fname_SMRT = 'SMRT_stimuli_C_dens_100_rate_5_depth_20_width_1.wav'
# Fs, audio_signal = wavfile.read(SMRT_sound_dir + fname_SMRT)
# plt.figure()
# winlen = 1024 ;
# window=0.5*np.blackman(winlen)+0.5*np.hamming(winlen) ;
# f, t, Sxx = signal.spectrogram(audio_signal, Fs, noverlap=1000, window=window)
# plt.pcolormesh(t, f, Sxx, cmap='viridis', vmax=100)#, vmax=50
# plt.ylim([0, 8e3])
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
plt.show()