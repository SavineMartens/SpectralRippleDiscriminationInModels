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
import librosa


def get_acoustic_spectrum(sound_name, use_preemp):
    # Fs_wav, audio_signal = wavfile.read(sound_name) # Fs = 44100
    if sound_name[-3:] == 'wav':
        # breakpoint()
        Fs_wav, audio_signal = wavfile.read(sound_name) # Fs = 44100
    elif sound_name[-3:] == 'mp3':
        audio_signal, Fs_wav = librosa.load(sound_name, sr=None)
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


def plot_acoustic_electric_fig(name_list, use_preemp, phase='1'):
    fig1 , ax1 = plt.subplots(2, int(len(name_list)/2))
    fig1.set_size_inches(17,5)
    plt.subplots_adjust(left=0.038, right=0.99, wspace=0.112, top=0.9)
    iterator = 0
    for i_n, name in enumerate(name_list):
        previous1 = 0
        previous2 = 0   

        if name[-3:] == 'mp3':
            sound_name = sound_dir + name
            if  name[8] == 'd':
                color = color_s
            elif name[8] == 'u':    
                color = color_i
            if i_n !=0 and i_n%2 == 0:
                iterator += 1
            if iterator == int(len(name_list)/2)-1:
                if  name[8] == 'd':
                    label = 'down'
                elif name[8] == 'u':
                    label = 'up'  
            else:
                 label = ''                        
        else:
            sound_name = sound_dir + name+ phase + '.wav'
            if 'width_20' in name:
                color = color_i
            elif 'width_' in name:
                color = color_s
            elif  name[0] == 's':
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
        print(sound_name)
        outline, frequency, pre_str = get_acoustic_spectrum(sound_name, use_preemp)
        normalized_bins = get_electric_spectrum(sound_name) 
        

        ax1[0,iterator].plot(frequency, np.squeeze(outline), color = color, label=label)
        ax1[0,iterator].set_xscale('log', base=2)
        ax1[0,iterator].set_xlim((200, 8700))
        ax1[0,iterator].set_ylim((0, 1))
        ax1[0,iterator].set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
        edges = [306, 442, 578, 646, 782, 918, 1054, 1250, 1529, 1801, 2141, 2549, 3025, 3568, 4248, 8054] #[340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]
        x=3 # check edges
        for edge in edges:
            # print(edge)
            ax1[0,iterator].vlines(edge, 0, 1, color='lightgray')

        if name[-3:] == 'mp3': # STRIPES
            RPO = re.search('_(.*)_', name).group(1)
            ax1[0,iterator].set_title('Glide: ' + RPO[2:], fontsize=fontsize )
        elif 'width_' in name: # SMRT
            width = re.search('width_(.*)', name).group(1)
            if width != '20':
                ax1[0,iterator].set_title('RPO: ' + width, fontsize=fontsize )
        else: # Spectral ripple
            RPO_re = re.search('_(.*)_', name)
            RPO = RPO_re.group(1)
            while RPO[-2:] == '00':
                RPO = RPO[:-2]
            ax1[0,iterator].set_title(RPO + ' RPO', fontsize=fontsize )
            width = '20'
        if iterator == 3:
            ax1[0,iterator].legend(loc='upper right', fontsize=fontsize)

        for i, bin in enumerate(normalized_bins): 
            if  name[:2] == 's_' or name[8] == 'd' or width != '20':
                if iterator == int(len(name_list)/2)-1 and i == 1:
                    ax1[1, iterator].hlines(bin, edges[i], edges[i+1], colors=color_s, linewidth= 3, label=label) # 
                else:
                    ax1[1, iterator].hlines(bin, edges[i], edges[i+1], colors=color_s, linewidth= 3, label='') # 
                ax1[1, iterator].vlines(edges[i], previous1, bin, colors=color_s, linewidth= 3, label='_nolegend_') # 

                previous1 = bin 
            elif  name[0] == 'i'or name[8] == 'u' or 'width_20' in name: # 
                if iterator == int(len(name_list)/2)-1 and i == 1:
                    ax1[1, iterator].hlines(bin, edges[i], edges[i+1], colors=color_i, linewidth= 3, label=label) # 
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
    edges = [306, 442, 578, 646, 782, 918, 1054, 1250, 1529, 1801, 2141, 2549, 3025, 3568, 4248, 8054]    #[340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]
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


if __name__ == '__main__':  
    color_s = 'blue'
    color_i = 'red'
    fontsize = 14.5 
    test_type = 'SR' # 'SR' / 'STRIPES' / 'SMRT'

    if test_type == 'SR':
        data_dir = './data/spectrum' 
        sound_dir = 'C:\\Users\\ssmmartens\\OneDrive - LUMC\\Sounds\\Ripple\\' #'./sounds/spectral ripple/'

        # color_s = 'blue'
        # color_i = 'red'
        use_preemp = True
        # fontsize = 14.5

        plot_electric = True
        plot_virtual = False

        if plot_electric:
            get_virtual_bins = False
            name_list = ['s_0.500_', 'i1_0.500_', 's_1.414_', 'i1_1.414_', 's_2.000_', 'i1_2.000_', 's_4.000_', 'i1_4.000_']
            for phase in range(7, 30):
                phase = str(phase) # '1' / '2' / '3'    
                fig1, pre_str = plot_acoustic_electric_fig(name_list, use_preemp, phase)
                fig1.savefig('./figures/spectrum/AcousticAndElectricSpectralRipples'+color_i + color_s + pre_str +'306_8054Hz_phase'+ phase +'.png')

        if plot_virtual:
            name_list = ['s_2.828_', 'i1_2.828_'] 
            fig2, pre_str = plot_acoustic_electric_virtual_fig(name_list, use_preemp)
            fig2.savefig('./figures/spectrum/AcousticElectricVirtualSpectralRipples'+color_i + color_s + pre_str +'.png')


    if test_type == 'STRIPES':
        data_dir = './data/STRIPES/'
        sound_dir = './sounds/STRIPES/'
        use_preemp = True
        ripple_list = [3.0, 5.0, 7.0, 9.0]#np.arange(5.5, 10.5, 0.5)
        fname_list = []

        for ripple_ud in ripple_list:
            if ripple_ud == 1.0:
                ripple_ud = 1.1 
            fname_list.append('stripes_u_'+ str(ripple_ud) + '_0.mp3')
            fname_list.append('stripes_d_'+ str(ripple_ud) + '_0.mp3')
            
        fig1, pre_str = plot_acoustic_electric_fig(fname_list, use_preemp)
        fig1.savefig('./figures/STRIPES/Acoustic_Electric_'+ str(ripple_list) + '.png')

    if test_type == 'SMRT':
        data_dir = './data/SMRT/'
        sound_dir = './sounds/SMRT/'
        use_preemp = True
        RPO_list = ['_1', '_2', '_3']
        fname_list = []
        for RPO in RPO_list:
            fname_list.append('SMRT_stimuli_C_dens_100_rate_5_depth_20_width' + RPO)
            fname_list.append('SMRT_stimuli_C_dens_100_rate_5_depth_20_width_20')
        fig1, pre_str = plot_acoustic_electric_fig(fname_list, use_preemp)
        fig1.savefig('./figures/SMRT/Acoustic_Electric_'+ str(RPO_list) + '.png')

    plt.show()


