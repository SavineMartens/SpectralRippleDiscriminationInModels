# import sys
# sys.path.append("...") # double dot when running file, single dot with debugging
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/python/temporal')
import numpy as np
import abt
import matplotlib.pyplot as plt
from scipy.signal import hilbert
# from abt.defaults import DEFAULT_BINS
import glob
import os
import re
from scipy.io import wavfile
# from abt import wav_to_electrodogram
from matplotlib.patches import Rectangle
from utilities import butter_lowpass_filter, transform_pulse_train_to_121_virtual


def get_acoustic_spectrum(sound_name, use_preemp):
    Fs_wav, audio_signal = wavfile.read(sound_name) # Fs = 44100
    audio_signal = np.expand_dims(audio_signal, axis=0)

    if use_preemp:
        # use pre-emphasis with FFT
        audio_signal = abt.frontend.td_filter(audio_signal)
        print('using pre-emphasis')
        pre_str = 'PreEmpOn'
    else:
        pre_str = 'PreEmpOff'
    FFT = np.fft.rfft(audio_signal) # only real output 
    abs_fourier_transform = np.abs(FFT)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, Fs_wav/2, abs_fourier_transform.shape[1])
    max_power = power_spectrum.max()
    outline = hilbert(power_spectrum)
    outline /= outline.max()
    return outline, frequency, pre_str

def get_electric_spectrum(sound_name, get_virtual_bins=False):
    # part of SCS:
    audio_signal, _ = abt.frontend.read_wav(sound_name)
    Fs = 17400
    signal_emph = abt.frontend.td_filter(audio_signal)
    signal_agc, agc = abt.automatic_gain_control.dual_loop_td_agc(signal_emph)
    signal_buffered = abt.buffer.window_buffer(signal_agc)
    signal_fft = abt.filterbank.fft_filterbank(signal_buffered, nFft=256)
    signal_hilbert = abt.filterbank.hilbert_envelope(signal_fft) # in log2
    signal_energy = abt.filterbank.channel_energy(signal_fft, agc.smpGain) # agc.smpGain = [1 x len(sound)-15]
    # Compute channel-by-channel noise reduction gains.
    noise_reduction_gains, *_ = abt.noise_reduction.noise_reduction(signal_energy) # output size = num_chan x total_frames
    # apply gains to envelopes to reduce estimated noise
    envelope = signal_hilbert + noise_reduction_gains # output size = num_chan x total_frames
    signal_hilbert = envelope

    if get_virtual_bins:
        # Find frequency band with largest amplitude of subsample (every third FFT input frame)
        peak_freq, peak_loc = abt.post_filterbank.spec_peak_locator(signal_fft[:,2::3])  # [num_chan x np.floor(total_frames)/3], [2x num_chan x np.floor(total_frames)/3]
        # upsample back to full framerate and add padding
        peak_freq = abt.post_filterbank.upsample(peak_freq, signal_fft.shape[1]) # [nChan x total_frames]
        peak_loc = abt.post_filterbank.upsample(peak_loc, signal_fft.shape[1]) # [nChan x total_frames] in values from 0 to 15
        weights = abt.post_filterbank.current_steering_weights(peak_loc) # [2*nChan x total_frames] will sum up to 1 for all channels
        # Create carrier function with period of 1/peak_freq, maximum depends on implant's maximal stimulation rate
        carrier, audio_idx = abt.post_filterbank.carrier_synthesis(peak_freq) # [nChan x np.ceil((nHop / fs) * total_frames / (2 * pulseWidth * nChan * 1e-6) - 1] = num_forward_telemetry_frames
        # maps the acoustic values per frequency band into current amplitudes that are used to modulate biphasic pulses
        signal, weights_transformed = abt.mapping.f120(carrier, envelope, weights, audio_idx) # [2*nChan x num_forward_telemetry_frames]
        elgram, weights_matrix = abt.electrodogram.f120(signal, weights_transformed=weights_transformed) #[nEl x 30* num_forward_telemetry_frames] in [uA]
        pulse_train = transform_pulse_train_to_121_virtual(elgram, weights_matrix)
        summed_PT = np.max(abs(pulse_train), axis=1)
        normalized_bins = (summed_PT-summed_PT.min())/(summed_PT.max()-summed_PT.min()) 
        return normalized_bins
    
    # transform to correct unit
    averaged_log2 = np.sum(signal_hilbert, axis=1) 
    averaged_power_unit = averaged_log2 
    max_power_SCS = averaged_power_unit.max()
    normalized_bins = (averaged_power_unit-averaged_power_unit.min())/(max_power_SCS-averaged_power_unit.min()) #
    return normalized_bins


def plot_acoustic_electric_fig(name_list, use_preemp):
    fig1 , ax1 = plt.subplots(2, int(len(name_list)/2))
    fig1.set_size_inches(17,5)
    plt.subplots_adjust(left=0.038, right=0.99, wspace=0.112, top=0.9)
    iterator = 0
    for i_n, name in enumerate(name_list):
        previous1 = 0
        previous2 = 0   

        sound_name = sound_dir + name + '.wav'
        outline, frequency, pre_str = get_acoustic_spectrum(sound_name, use_preemp)
        normalized_bins = get_electric_spectrum(sound_name) 
        
        if  name[0] == 's':
            color = color_s
        elif name[0] == 'i':
            color = color_i
        if i_n !=0 and i_n%2 == 0:
            iterator += 1
        if iterator == 3:
            if  name[0] == 's':
                label = 'standard'
            elif name[0] == 'i':
                label = 'inverted'
        else:
            label = ''
        ax1[0,iterator].plot(frequency, np.squeeze(outline), color = color, label=label)
        ax1[0,iterator].set_xscale('log', base=2)
        ax1[0,iterator].set_xlim((200, 8700))
        ax1[0,iterator].set_ylim((0, 1))
        ax1[0,iterator].set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
        edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]
        x=3 # check edges
        for edge in edges:
            ax1[0,iterator].vlines(edge, 0, 1, color='lightgray')
        RPO_re = re.search('_(.*)_', name)
        RPO = RPO_re.group(1)
        while RPO[-2:] == '00':
            RPO = RPO[:-2]
        ax1[0,iterator].set_title(RPO + ' RPO', fontsize=fontsize )
        if iterator == 3:
            ax1[0,iterator].legend(loc='upper right', fontsize=fontsize)

        for i, bin in enumerate(normalized_bins): 
            if  name[0] == 's':
                if iterator == 3 and i == 1:
                    ax1[1, iterator].hlines(bin, edges[i], edges[i+1], colors=color_s, linewidth= 3, label='standard') # 
                else:
                    ax1[1, iterator].hlines(bin, edges[i], edges[i+1], colors=color_s, linewidth= 3, label='') # 
                ax1[1, iterator].vlines(edges[i], previous1, bin, colors=color_s, linewidth= 3, label='_nolegend_') # 

                previous1 = bin 
            elif  name[0] == 'i':
                if iterator == 3 and i == 1:
                    ax1[1, iterator].hlines(bin, edges[i], edges[i+1], colors=color_i, linewidth= 3, label='inverted') # 
                else:
                    ax1[1, iterator].hlines(bin, edges[i], edges[i+1], colors=color_i, linewidth= 3, label='') 
                ax1[1, iterator].vlines(edges[i], previous2, bin, colors=color_i, linewidth= 3, label='') # 
                previous2 = bin
            ax1[1,iterator].set_xscale('log', base=2)
            ax1[1,iterator].set_xlim((200, 8700))
            ax1[1,iterator].set_ylim((0, 1))
            ax1[1,iterator].set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
            ax1[1,iterator].set_xlabel('Frequency [Hz]')
            # for edge in edges:
            ax1[1,iterator].vlines(edges, 0, 1, color='lightgray')
            if iterator == 3:
                ax1[1,iterator].legend(loc='upper right', fontsize=fontsize)

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

    ax1[0,0].set_ylabel('Acoustic', fontsize=fontsize)
    ax1[1,0].set_ylabel('Electric', fontsize=fontsize)
    return fig1, pre_str


def plot_acoustic_electric_virtual_fig(name_list, use_preemp):
    fig2 , ax2 = plt.subplots(3, 1)
    fig2.set_size_inches(16,6)
    edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]
    # plt.subplots_adjust(left=0.038, right=0.99, wspace=0.112, top=0.9)
    for i_n, name in enumerate(name_list):
        previous1 = 0
        previous2 = 0   

        sound_name = sound_dir + name + '.wav'
        outline, frequency, pre_str = get_acoustic_spectrum(sound_name, use_preemp)
        normalized_bins = get_electric_spectrum(sound_name, get_virtual_bins=False) 
        virtual_bins = get_electric_spectrum(sound_name, get_virtual_bins=True)
        
        if  name[0] == 's':
            color = color_s
            label = 'standard'
        elif name[0] == 'i':
            color = color_i
            label = 'inverted'

        ax2[0].plot(frequency, np.squeeze(outline), color = color, label=label)
        ax2[0].set_xscale('log', base=2)
        ax2[0].set_xlim((200, 8700))
        ax2[0].set_ylim((0, 1))
        ax2[0].set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
        ax2[0].vlines(edges, 0, 1, color='lightgray')
        RPO_re = re.search('_(.*)_', name)
        RPO = RPO_re.group(1)
        while RPO[-2:] == '00':
            RPO = RPO[:-2]
        ax2[0].set_title(RPO + ' RPO', fontsize=fontsize )
        ax2[0].legend(loc='upper left', fontsize=fontsize)

        for i, bin in enumerate(normalized_bins): 
            if  name[0] == 's':
                if i == 1:
                    ax2[1].hlines(bin, edges[i], edges[i+1], colors=color_s, linewidth= 3, label='standard') # 
                else:
                    ax2[1].hlines(bin, edges[i], edges[i+1], colors=color_s, linewidth= 3, label='') # 
                ax2[1].vlines(edges[i], previous1, bin, colors=color_s, linewidth= 3, label='_nolegend_') # 

                previous1 = bin 
            elif  name[0] == 'i':
                if i == 1:
                    ax2[1].hlines(bin, edges[i], edges[i+1], colors=color_i, linewidth= 3, label='inverted') # 
                else:
                    ax2[1].hlines(bin, edges[i], edges[i+1], colors=color_i, linewidth= 3, label='') 
                ax2[1].vlines(edges[i], previous2, bin, colors=color_i, linewidth= 3, label='') # 
                previous2 = bin
            ax2[1].set_xscale('log', base=2)
            ax2[1].set_xlim((200, 8700))
            ax2[1].set_ylim((0, 1))
            ax2[1].set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
            ax2[1].set_xlabel('Frequency [Hz]')
            ax2[1].vlines(edges, 0, 1, color='lightgray')
            ax2[1].legend(loc='upper left', fontsize=fontsize)

        #add rectangle to plot
        for i in range(3):
            ax2[i].add_patch(Rectangle((0,0), edges[0],1,
                            edgecolor = 'grey',
                            facecolor = 'grey',
                            fill=True,
                            alpha=0.25))
            ax2[i].add_patch(Rectangle((edges[-1],0), edges[-1]+2000,1,
                            edgecolor = 'grey',
                            facecolor = 'grey',
                            fill=True,
                            alpha=0.25))
            ax2[i].add_patch(Rectangle((0,0), edges[0],1,
                        edgecolor = 'grey',
                        facecolor = 'grey',
                        fill=True,
                        alpha=0.25))
            ax2[i].add_patch(Rectangle((edges[-1],0), edges[-1]+2000,1,
                        edgecolor = 'grey',
                        facecolor = 'grey',
                        fill=True,
                        alpha=0.25))    

        # virtual bins
        new_edges = []
        for x_i in range(1, len(edges)):
            section = (edges[x_i] - edges[x_i-1])/8
            new_edges.append(edges[x_i-1])
            for b in range(1,8):
                new_edges.append(edges[x_i-1]+b*section)
        new_edges.append(edges[-1])
        for i, bin in enumerate(virtual_bins): 
            if i == len(virtual_bins)-1:
                continue
            if  name[0] == 's':
                if i == 1:
                    ax2[2].hlines(bin, new_edges[i], new_edges[i+1], colors=color_s, linewidth= 3, label='standard') # 
                else:
                    ax2[2].hlines(bin, new_edges[i], new_edges[i+1], colors=color_s, linewidth= 3, label='') # 
                ax2[2].vlines(new_edges[i], previous1, bin, colors=color_s, linewidth= 3, label='_nolegend_') # 

                previous1 = bin 
            elif  name[0] == 'i':
                if i == 1:
                    ax2[2].hlines(bin, new_edges[i], new_edges[i+1], colors=color_i, linewidth= 3, label='inverted') # 
                else:
                    ax2[2].hlines(bin, new_edges[i], new_edges[i+1], colors=color_i, linewidth= 3, label='') 
                ax2[2].vlines(new_edges[i], previous2, bin, colors=color_i, linewidth= 3, label='') # 
                previous2 = bin
            ax2[2].set_xscale('log', base=2)
            ax2[2].set_xlim((200, 8700))
            ax2[2].set_ylim((0, 1))
            ax2[2].set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
            ax2[2].set_xlabel('Frequency [Hz]', fontsize=fontsize)
            ax2[2].vlines(new_edges, 0, 1, color='lightgray', alpha=0.02)
            ax2[0].vlines(new_edges, 0, 1, color='lightgray', alpha=0.02)
            ax2[2].legend(loc='upper left', fontsize=fontsize)

    ax2[0].set_ylabel('Acoustic', fontsize=fontsize)
    ax2[1].set_ylabel('Electric', fontsize=fontsize)
    ax2[2].set_ylabel('Virtual', fontsize=fontsize)

    return fig2, pre_str



data_dir = './data/spectrum' 
sound_dir = './sounds/spectral ripple/'

color_s = 'blue'
color_i = 'red'
use_preemp = True
fontsize = 14.5

plot_electric = False
plot_virtual = True

if plot_electric:
    get_virtual_bins = False
    name_list = ['s_0.500_1', 'i1_0.500_1', 's_1.414_1', 'i1_1.414_1', 's_2.000_1', 'i1_2.000_1', 's_4.000_1', 'i1_4.000_1']
    fig1, pre_str = plot_acoustic_electric_fig(name_list, use_preemp)
    fig1.savefig('./figures/spectrum/AcousticAndElectricSpectralRipples'+color_i + color_s + pre_str +'.png')

if plot_virtual:
    name_list = ['s_2.828_1', 'i1_2.828_1'] 
    fig2, pre_str = plot_acoustic_electric_virtual_fig(name_list, use_preemp)
    fig2.savefig('./figures/spectrum/AcousticElectricVirtualSpectralRipples'+color_i + color_s + pre_str +'.png')

plt.show()


