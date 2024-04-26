import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert


def get_normalized_spectrum(fname, filter_bool=True, filter_order = 4, cut_off_freq = 100):
    if '.mat' in fname:
        unfiltered, _, _, _, fiber_frequencies, _, _, _ = load_mat_structs_Hamacher(fname, unfiltered_type = 'OG')
        normal_spectrum = (np.mean(unfiltered, axis=1)-np.min(np.mean(unfiltered, axis=1)))/(np.max(np.mean(unfiltered, axis=1))-np.min(np.mean(unfiltered, axis=1)))
        return normal_spectrum, fiber_frequencies
    if '.npy' in fname:
        spike_matrix = np.load(fname, allow_pickle=True)  
        spike_vector = np.mean(spike_matrix, axis=1)
        electric_spectrum = (spike_vector-np.min(spike_vector))/(np.max(spike_vector)-np.min(spike_vector))   
        if filter_bool:
            spike_vector2 = butter_lowpass_filter(spike_vector, cut_off_freq, len(spike_vector), filter_order) # Can't LP f because the Fs is not consistent
            electric_spectrum2 = (spike_vector2-np.min(spike_vector2))/(np.max(spike_vector2)-np.min(spike_vector2))
        else:
            electric_spectrum2 = None
        return electric_spectrum, electric_spectrum2


def get_FFT_spectrum(fname):
    Fs, audio_signal = wavfile.read(fname)
    FFT = np.fft.rfft(audio_signal) 
    abs_fourier_transform = np.abs(FFT)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, Fs/2, len(abs_fourier_transform))
    max_power = power_spectrum.max()
    normalized_power = power_spectrum/max_power
    outline = hilbert(normalized_power)
    return outline, frequency


def double_sound_spectrum(RPO):
    SR_sound_dir = './sounds/spectral ripple/'
    SMRT_sound_dir = './sounds/SMRT/'

    outline_SR, frequency = get_FFT_spectrum(glob.glob(SR_sound_dir + '*i1_' + RPO + '.*')[0]) # Fs = 44100
    outline_SMRT, _ = get_FFT_spectrum(glob.glob(SMRT_sound_dir + '*width_*' + RPO + '*')[0]) # Fs = 44100

    fig, axes = plt.subplots(2,1, sharex=True, sharey=True, figsize=(10, 5))
    plt.subplots_adjust(hspace=0.274)
    plt.subplot(2,1,1)
    plt.plot(frequency, outline_SR)
    plt.title('Spectral ripple')
    plt.ylabel('Normalized power')
    plt.xscale('log', base=2)
    plt.xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
    plt.xlim(min(frequency), 1e4)
    plt.ylim(0,1)
    plt.vlines(5000, 0, 1, colors='red')
    plt.subplot(2,1,2)
    plt.plot(frequency, outline_SMRT)
    plt.title('SMRT')
    plt.ylabel('Normalized power')
    plt.vlines(6500, 0, 1, colors='red')
    plt.show()
    return fig
    

def double_spectrum_one_fig(RPO_list, NH_dB):
    fig, ax = plt.subplots(len(RPO_list), 2, figsize=(12, 4), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.079, bottom=0.11, right=0.967, top=0.929, wspace=0.105)
    bar_width = 15
    
    if NH_dB == 50:
        dB_str = '2024-04-16'
        # dB_str = '2024-04-02' # 2847
    if NH_dB == 65:
        dB_str = '2023-11-24'

    alpha = 0.2
    for RPO in RPO_list:
        # get EH
        fname_NH_SR = glob.glob(sr_data_dir + '*i1*' + RPO + '_1_*.mat')[0]
        print('SR:', fname_NH_SR)
        fname_NH_SMRT = glob.glob(SMRT_data_dir + '*' + dB_str + '*width_*' + RPO + '*.mat')[0]
        print('SMRT:', fname_NH_SR)
        normal_spectrum_SR, fiber_frequencies = get_normalized_spectrum(fname_NH_SR) 
        normal_spectrum_SMRT, _ =  get_normalized_spectrum(fname_NH_SMRT)          
       
        #Spectral ripple
        plt.subplot(len(RPO_list), 2, 1)
        plt.bar(fiber_frequencies, normal_spectrum_SR, width=bar_width, alpha=alpha, color='blue')
        filter_sig_i = butter_lowpass_filter(normal_spectrum_SR, cut_off_freq, len(normal_spectrum_SR), filter_order)
        plt.plot(fiber_frequencies, filter_sig_i, color='blue', label='inverted')
        plt.xlim(min(fiber_frequencies), 8e3)
        plt.vlines(5000, 0, 1, colors='red')
        plt.ylim((0,1))
        plt.ylabel(RPO + ' RPO \n normalized \n spiking')
        plt.title('Spectral ripple')
        plt.xlabel('Frequency [Hz]')
        
        # SMRT
        plt.subplot(len(RPO_list), 2, 2)
        plt.bar(fiber_frequencies, normal_spectrum_SMRT, width=bar_width, alpha=alpha, color='blue')
        filter_sig_s = butter_lowpass_filter(normal_spectrum_SMRT, cut_off_freq, len(normal_spectrum_SMRT), filter_order)
        plt.plot(fiber_frequencies, filter_sig_s, color='blue', label='standard')
        plt.vlines(6500, 0, 1, colors='red')
        plt.title('SMRT')
        plt.xlabel('Frequency [Hz]')
        # plt.show()
        # x=3
       
    return fig





if __name__ == "__main__":

    SMRT_data_dir = './data/SMRT/'

    filter_order = 4
    cut_off_freq = 100
    RPO = '4'
    NH_dB = 50

    sr_data_dir = './data/spectrum/'+str(NH_dB) +'dB_1903F/'

    # sound spectrum
    # fig = double_sound_spectrum(RPO)
    # fig.savefig('./figures/SRvsSMRT_audio' + '_'.join(RPO) + 'RPO.png')
    
    # neural activation
    fig = double_spectrum_one_fig(RPO, NH_dB)
    fig.savefig('./figures/SRvsSMRT_filtered' + '_'.join(RPO) + 'RPO_'+ str(NH_dB) +'dB1903Fib.png')
    plt.show()

  