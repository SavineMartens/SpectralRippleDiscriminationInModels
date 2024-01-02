# import sys
# sys.path.append("...") # double dot when running file, single dot with debugging
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/python/temporal')
import numpy as np
import abt
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from abt.defaults import DEFAULT_BINS
import glob
import os
import re
from scipy.io import wavfile
from abt import wav_to_electrodogram
from matplotlib.patches import Rectangle

data_dir = './data/spectrum' 
file_str = 'i_'

original_Fs = True
plot_time_signal = False
plot_FFT_signal = False

edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]

name_list = ['s_0.500_1', 'i1_0.500_1', 's_1.414_1', 'i1_1.414_1', 's_2.000_1', 'i1_2.000_1', 's_4.000_1', 'i1_4.000_1']
num_rows = len(name_list)

use_windowing = True
use_preemp = True
use_zeropad = False
use_mean = False

winn_plot = True

plot_old = False
# use_electrodogram = True

if plot_old:
    fig , ax = plt.subplots(num_rows,1)

fig1 , ax1 = plt.subplots(2, int(num_rows/2))
fig1.set_size_inches(17,4)
plt.subplots_adjust(left=0.074, right=0.964)
iterator = 0


for i_n, name in enumerate(name_list):
    previous1 = 0
    previous2 = 0   
    # grey line
    if 'abt' in data_dir:
         sound_name = abt.sounds[name]
    else:
         sound_name = data_dir + '\\' + name + '.wav'
    if original_Fs:
        Fs, audio_signal_single = wavfile.read(sound_name) # Fs = 44100
        audio_signal = np.expand_dims(audio_signal_single, axis=0)
    else:
        audio_signal, _ = abt.frontend.read_wav(sound_name)
        Fs = 17400
    # repeat sound to increase number of datapoints and thus spectral resolution? No https://dsp.stackexchange.com/questions/73577/why-does-a-longer-observation-time-improve-dft-resolution-but-repeating-a-signa
    # audio_signal = np.expand_dims(np.repeat(audio_signal, 2), axis=0)
    if use_windowing:
        # windowing to reduce spectral leakage
        window = 0.5 * (np.blackman(len(audio_signal[0])) + np.hanning(len(audio_signal[0]))) 
        # window = window.reshape(1, window.size)
        # windowed_audio_signal = window*audio_signal
        audio_signal = (audio_signal * window)
        window_str = 'Y'
    else:
        window_str = 'N'

    if plot_time_signal:
        fig, axes = plt.subplots(figsize=(5, 1))
        fig.suptitle(name)
        t_vector = np.linspace(0,len(np.squeeze(audio_signal))/Fs, len(np.squeeze(audio_signal)) )
        axes.plot(t_vector, audio_signal.ravel())
        # axes.plot(t_vector, windowed_audio_signal.ravel(), '--', color='red')
        # axes.plot(t_vector, window.ravel(), color='green')
        axes.set_xlabel('Time [s]')
        axes.grid()
        # plt.show()
    # nFft = audio_signal[0].shape[1] # low time resolution --> high frequency resolution

    if use_preemp:
        # use pre-emphasis with FFT
        audio_signal = abt.frontend.td_filter(audio_signal)
        # pre_audio_signal = abt.frontend.td_filter(audio_signal)
        preemp_str = 'Y'
    else:
        preemp_str = 'N'

    if use_zeropad:
        # zero padding
        FFT = np.fft.fft(audio_signal, 2**15) # nothing
        zeropad_str = 'Y'
    else: # without zero padding
        FFT = np.fft.rfft(audio_signal) # only real output 
        zeropad_str = 'N'

    # windowed_FFT = np.fft.rfft(windowed_audio_signal) # only windowing
    # pre_FFT = np.fft.rfft(pre_audio_signal) # only pre-emphasis

    abs_fourier_transform = np.abs(FFT)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, Fs/2, abs_fourier_transform.shape[1])
    max_power = power_spectrum.max()
    normalized_power = power_spectrum/max_power
    outline = hilbert(normalized_power)
    outline_db = 10 * np.log10(power_spectrum/np.min(power_spectrum))
    duration = len(audio_signal)/Fs
    time = np.arange(0, duration, 1/Fs)

    if plot_old:
        # ax[i_n].plot(frequency, np.squeeze(normalized_power), color='gray')
        ax[i_n].plot(frequency, np.squeeze(outline), color = 'gray')
        ax[i_n].set_xscale('log', base=2)
        ax[i_n].set_xlim((200, 8700))
        ax[i_n].set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
        ax[i_n].vlines(edges, 0, 1, color='lightgray')
        RPO = re.search('_(.*)_', name)
        ax[i_n].set_ylabel(RPO.group(1) + ' RPO' )


    # winn plot
    if  name[0] == 's':
        color = 'orange'
    elif name[0] == 'i':
        color = 'black'
    if i_n !=0 and i_n%2 == 0:
        iterator += 1
    ax1[0,iterator].plot(frequency, np.squeeze(outline), color = color)
    ax1[0,iterator].set_xscale('log', base=2)
    ax1[0,iterator].set_xlim((200, 8700))
    ax1[0,iterator].set_ylim((0, 1))
    ax1[0,iterator].set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
    edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]
    for edge in edges:
        ax1[0,iterator].vlines(edge, 0, 1, color='lightgray')
    RPO = re.search('_(.*)_', name)
    ax1[0,iterator].set_title(RPO.group(1)+ ' RPO' )


    Fs = 17400
    # blue line
    # part of SCS:
    audio_signal, _ = abt.frontend.read_wav(sound_name)
    signal_emph = abt.frontend.td_filter(audio_signal)
    signal_agc, agc = abt.automatic_gain_control.dual_loop_td_agc(signal_emph)
    signal_buffered = abt.buffer.window_buffer(signal_agc)
    signal_fft = abt.filterbank.fft_filterbank(signal_buffered, nFft=256, plot_magnitude_FFT=plot_FFT_signal)
    signal_hilbert = abt.filterbank.hilbert_envelope(signal_fft) # in log2

    signal_energy = abt.filterbank.channel_energy(signal_fft, agc.smpGain) # agc.smpGain = [1 x len(sound)-15]
    # Compute channel-by-channel noise reduction gains.
    noise_reduction_gains, *_ = abt.noise_reduction.noise_reduction(signal_energy) 
                        # output size = num_chan x total_frames
    # apply gains to envelopes to reduce estimated noise
    envelope = signal_hilbert + noise_reduction_gains
            # output size = num_chan x total_frames
    signal_hilbert = envelope

    # # transform to correct unit
    if use_mean:
        averaged_log2 = np.mean(signal_hilbert, axis=1) # np.sum(signal_hilbert, axis=1) 
        averaged_power_unit = np.sqrt(2**averaged_log2)
        max_power_SCS = averaged_power_unit.max()
        normalized_bins = averaged_power_unit/max_power_SCS # (averaged_power_unit-averaged_power_unit.min())/(max_power_SCS-averaged_power_unit.min()) #
    else:
        averaged_log2 = np.sum(signal_hilbert, axis=1) 
        averaged_power_unit = averaged_log2 
        max_power_SCS = averaged_power_unit.max()
        normalized_bins = (averaged_power_unit-averaged_power_unit.min())/(max_power_SCS-averaged_power_unit.min()) #

    # if use_electrodogram:
    #     elgram, _ , _ = wav_to_electrodogram(sound_name, ramp_bool=False)
    #     bins = np.max(abs(elgram), axis=1)
    #     normalized_bins = (bins-min(bins))/(max(bins)-min(bins)) # too many, I need to get 15 not 16

    for i, bin in enumerate(normalized_bins): 
        if plot_old:
            # correction for having more bins in one frequency band
            # ax[i_n].hlines(bin/DEFAULT_BINS[i], edges[i], edges[i+1], colors='deepskyblue', linewidth= 3)
            # ax[i_n].hlines(bin/DEFAULT_BINS[i], edges[i], edges[i+1], colors='white', linewidth= 1) 
            # Using w/o bin correction does kind of reflect alternating behaviour
            ax[i_n].hlines(bin, edges[i], edges[i+1], colors='deepskyblue', linewidth= 3) # 
            ax[i_n].hlines(bin, edges[i], edges[i+1], colors='white', linewidth= 1) 

        # winn plot
        if  name[0] == 's':

            ax1[1, iterator].hlines(bin, edges[i], edges[i+1], colors='deepskyblue', linewidth= 3) # 
            ax1[1, iterator].hlines(bin, edges[i], edges[i+1], colors='white', linewidth= 1)
            ax1[1, iterator].vlines(edges[i], previous1, bin, colors='deepskyblue', linewidth= 3) # 
            previous1 = bin 
        elif  name[0] == 'i':
            ax1[1, iterator].hlines(bin, edges[i], edges[i+1], colors='black', linewidth= 3) 
            ax1[1, iterator].vlines(edges[i], previous2, bin, colors='black', linewidth= 3) # 
            previous2 = bin
            # ax[1, iterator].hlines(bin, edges[i], edges[i+1], colors='white', linewidth= 1) 
        ax1[1,iterator].set_xscale('log', base=2)
        ax1[1,iterator].set_xlim((200, 8700))
        ax1[1,iterator].set_ylim((0, 1))
        ax1[1,iterator].set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
        ax1[1,iterator].set_xlabel('Frequency [Hz]')
        # for edge in edges:
        ax1[1,iterator].vlines(edges, 0, 1, color='lightgray')

    #add rectangle to plot
    ax1[0,iterator].add_patch(Rectangle((0,0), edges[0],1,
                    edgecolor = 'grey',
                    facecolor = 'grey',
                    fill=True,
                    alpha=0.25))
    ax1[0,iterator].add_patch(Rectangle((edges[-1],0), edges[-1]+2000,1,
                    edgecolor = 'grey',
                    facecolor = 'grey',
                    fill=True,
                    alpha=0.25))
    ax1[1,iterator].add_patch(Rectangle((0,0), edges[0],1,
                edgecolor = 'grey',
                facecolor = 'grey',
                fill=True,
                alpha=0.25))
    ax1[1,iterator].add_patch(Rectangle((edges[-1],0), edges[-1]+2000,1,
                edgecolor = 'grey',
                facecolor = 'grey',
                fill=True,
                alpha=0.25))    


if plot_old:
    ax[i_n].set_xlabel('Frequency [Hz]')
    fig.supylabel('Normalized spectral power')
    fig.suptitle('windowing:' + window_str + ', Pre-emp:' + preemp_str + ', zero-padding:'+ zeropad_str)

ax1[0,0].set_ylabel('Acoustic')
ax1[1,0].set_ylabel('Electric')
fig1.savefig('./figures/spectrum/AcousticAndElectricSpectralRipples.png')

plt.show()


