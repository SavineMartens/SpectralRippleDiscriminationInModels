import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from scipy.io import wavfile
from scipy.signal import hilbert
from scipy import signal
from utilities import *
from SMRTvsSR import get_FFT_spectrum


####################### 
# Spectral Ripple TRUE SOUND
SR_sound_dir = "C:\\Users\\ssmmartens\\OneDrive - LUMC\\Sounds\\Ripple\\" #'./sounds/spectral ripple/'
density = '1'
name_list = ['s_'+density+'.000_1', 'i1_'+density+'.000_1', 'i2_'+density+'.000_1']
label_list = ['standard', 'inverted_1', 'inverted_2']
c_list = ['blue', 'red', 'orange']
# fig, axes = plt.subplots(2,1, sharex=True, sharey=True, figsize=(10, 5))
fig = plt.figure(figsize=(8,3))
plt.subplots_adjust(bottom=0.179)
plt.xscale('log', base=2)
plt.xticks([125, 250, 500, 1000, 2000, 4000, 8000], labels=['125', '250', '500', '1000', '2000', '4000', '8000'])
for n, name in enumerate(name_list):
    # try:
    outline_SR, frequency = get_FFT_spectrum(glob.glob(SR_sound_dir + '*' + name + '.*')[0])
    # plt.plot(frequency, outline_SR, label=label_list[n], color=c_list[n])
    plt.plot(frequency, outline_SR, label=name, color=c_list[n])
    # plt.title('Spectral ripple by Won et al. (4 RPO)', fontsize=18)
    plt.ylabel('Normalized power', fontsize=16)
    plt.xlabel('Frequency [Hz]', fontsize=16)
    plt.xlim(80, 8e3)
    plt.ylim(0,1)
    plt.legend(fontsize=16)
    # except:
        # continue

fig.savefig('./sounds/SR_'+density+'RPO_'+ str(len(name_list))+'.png')
plt.show()

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