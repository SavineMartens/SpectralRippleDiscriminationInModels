import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert
import matplotlib.transforms as mtransforms # labeling axes

id_peak_sr = 1801
id_peak_smrt = 1851
id_peak_sr_s = 1801
id_peak_sr_i = 1784

def get_normalized_spectrum(fname, filter_bool=True, filter_order = 4, cut_off_freq = 100):
    if '.mat' in fname:
        try:
            unfiltered, _, _, _, fiber_frequencies, _, _, _ = load_mat_structs_Hamacher(fname, unfiltered_type = 'OG')
        except:
            unfiltered, _, _, _, fiber_frequencies, _, _, _ = load_matrices_from_vectors_Bruce_struct(fname)
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

def get_spectrum(fname, filter_bool=True, filter_order = 4, cut_off_freq = 100):
    if '.mat' in fname:
        try:
            unfiltered, _, _, _, fiber_frequencies, _, _, _ = load_mat_structs_Hamacher(fname, unfiltered_type = 'OG')
        except:
            unfiltered, _, _, _, fiber_frequencies, _, _, _ = load_matrices_from_vectors_Bruce_struct(fname)
        spectrum = np.sum(unfiltered, axis=1)
        return spectrum, fiber_frequencies
    if '.npy' in fname:
        spike_matrix = np.load(fname, allow_pickle=True)  
        electric_spectrum = np.sum(spike_matrix, axis=1)
        if filter_bool:
            spike_vector2 = butter_lowpass_filter(electric_spectrum, cut_off_freq, len(electric_spectrum), filter_order) # Can't LP f because the Fs is not consistent
            electric_spectrum2 = (spike_vector2-np.min(spike_vector2))/(np.max(spike_vector2)-np.min(spike_vector2))
        else:
            electric_spectrum2 = None
        return electric_spectrum, electric_spectrum2


def get_FFT_spectrum(fname):
    if fname[-3:] == 'mp3':
        import audio2numpy as a2n
        audio_signal, Fs =a2n.audio_from_file(fname)
    elif fname[-3:] == 'wav':
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
    ax = plt.subplot(2,1,1)
    plt.plot(frequency, outline_SR)
    plt.title('Spectral ripple')
    plt.ylabel('Normalized power')
    plt.xscale('log', base=2)
    plt.xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
    plt.xlim(min(frequency), 1e4)
    plt.ylim(0,1)
    plt.vlines(5000, 0, 1, colors='red')
    ax = plt.subplot(2,1,2)
    plt.plot(frequency, outline_SMRT)
    plt.title('SMRT')
    plt.ylabel('Normalized power')
    plt.vlines(6500, 0, 1, colors='red')
    plt.show()
    return fig

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx    

def octave_from_frequency(F1,F2):
    F_lower = min(F1,F2)
    F_upper = max(F1,F2)
    Octave = np.log(F_upper/F_lower)/np.log(2)
    return Octave


def spectrum_standard_inverted(RPO, octave_spaced=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.079, bottom=0.11, right=0.967, top=0.929, wspace=0.105)
    bar_width = 15

    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    alpha = 0.2
    # get EH
    fname_NH_SR_s = glob.glob(sr_data_dir + '*_s_*' + RPO + '*_1_*.mat')[0]
    print('SR_s:', fname_NH_SR_s)
    fname_NH_SR_i = glob.glob(sr_data_dir + '*_i1_*' + RPO + '*_1_*.mat')[0]
    print('SR_i:', fname_NH_SR_i)
    
    normal_spectrum_SR_s, fiber_frequencies = get_normalized_spectrum(fname_NH_SR_s) 
    normal_spectrum_SR_i, fiber_frequencies = get_normalized_spectrum(fname_NH_SR_i)    
    id_peak_sr_s = 1801
    id_peak_sr_i = 1784

    #Spectral ripple standard
    ax = plt.subplot(1, 2, 1)
    ax.text(-0.01, 0.1, 'A', transform=ax.transAxes + trans,
        fontsize=18, verticalalignment='top', color='white')
    plt.bar(fiber_frequencies, normal_spectrum_SR_s, width=bar_width, alpha=alpha, color='blue')
    filter_sig_sr_s = butter_lowpass_filter(normal_spectrum_SR_s, cut_off_freq, len(normal_spectrum_SR_s), filter_order)
    plt.plot(fiber_frequencies, filter_sig_sr_s, color='blue')
    plt.xlim(min(fiber_frequencies), 8e3)
    plt.vlines(5000, 0, 1, colors='red')
    plt.ylim((0,1))
    plt.ylabel(RPO + ' RPO \n normalized \n spiking')
    plt.title('Spectral ripple standard')
    plt.xlabel('Frequency [Hz]')

    if octave_spaced:
        plt.xscale('log', base=2)
        plt.xticks([500, 1000, 2000, 4000, 8000], labels=['500', '1000', '2000', '4000', '8000'])   

    # Spectral ripple inverted
    ax = plt.subplot(1, 2, 2)
    ax.text(-0.01, 0.1, 'B', transform=ax.transAxes + trans,
        fontsize=18, verticalalignment='top', color='white')
    plt.bar(fiber_frequencies, normal_spectrum_SR_i, width=bar_width, alpha=alpha, color='blue')
    filter_sig_sr_i = butter_lowpass_filter(normal_spectrum_SR_i, cut_off_freq, len(normal_spectrum_SR_i), filter_order)
    plt.plot(fiber_frequencies, filter_sig_sr_i, color='blue')
    plt.vlines(5000, 0, 1, colors='red')
    plt.title('Spectral ripple inverted')   
    plt.xlabel('Frequency [Hz]')
    if octave_spaced:
        plt.xscale('log', base=2)
        plt.xticks([500, 1000, 2000, 4000, 8000], labels=['500', '1000', '2000', '4000', '8000'])   
    x=3

    dB=3
    print(dB, 'dB point')
    _, id_5000 = find_nearest(fiber_frequencies, 5000)
    # 3 dB point:
    y_val_SR_s = normal_spectrum_SR_s[id_peak_sr_s]/(10**(dB/10))
    y_val_SR_i = normal_spectrum_SR_i[id_peak_sr_i]/(10**(dB/10))
    y_val_5000_s = normal_spectrum_SR_s[id_5000]/(10**(dB/10))
    y_val_5000_i = normal_spectrum_SR_i[id_5000]/(10**(dB/10))
    # needs to be larger than the y_val_SR!!!    
    SR_s_dB, idx_SR_s_dB = find_nearest(filter_sig_sr_s[id_peak_sr_s:], y_val_SR_s)
    SR_i_dB, idx_SR_i_dB = find_nearest(filter_sig_sr_i[id_peak_sr_i:], y_val_SR_i)

    SR_s_5000_dB, idx_5000_SR_s_dB = find_nearest(filter_sig_sr_s[id_peak_sr_s:], y_val_5000_s)
    SR_i_5000_dB, idx_5000_SR_i_dB = find_nearest(filter_sig_sr_i[id_peak_sr_i:], y_val_5000_i)
    idx_SR_s_dB += id_peak_sr_s
    idx_SR_i_dB += id_peak_sr_i

    idx_5000_SR_s_dB += id_peak_sr_s
    idx_5000_SR_i_dB += id_peak_sr_i

    f_SR_s_dB = fiber_frequencies[idx_SR_s_dB] # x dB point versus peak standard
    f_SR_i_dB = fiber_frequencies[idx_SR_i_dB] # x dB point versus peak inverted

    f_5000_SR_s_dB = fiber_frequencies[idx_5000_SR_s_dB] # x dB point versus 5000 Hz
    f_5000_SR_i_dB = fiber_frequencies[idx_5000_SR_i_dB] # x dB point versus 5000 Hz

    ax = plt.subplot(1, 2, 1)
    plt.scatter(f_SR_s_dB, SR_s_dB, label= str(dB) + ' from peak')
    plt.scatter(f_5000_SR_s_dB, SR_s_5000_dB, color='r', label= str(dB) + ' from limit')
    plt.legend()

    ax = plt.subplot(1, 2, 2)
    plt.scatter(f_SR_i_dB, SR_i_dB, label= str(dB) + ' from peak')
    plt.scatter(f_5000_SR_i_dB, SR_i_5000_dB, color='r', label= str(dB) + ' from limit')
    plt.legend()

    # x peak dB point vs end spectrum
    print('S: C vs B', octave_from_frequency(f_SR_s_dB, 5000)) # frequency x dB from peak vs 5000 Hz
    print('I: C vs B', octave_from_frequency(f_SR_i_dB, 6500)) # x frequency dB from peak vs 6500 Hz

    # x peak dB point vs peak
    print('S: A vs C', octave_from_frequency(f_SR_s_dB, fiber_frequencies[id_peak_sr_s])) # x frequency dB from peak vs peak
    print('I: C vs A', octave_from_frequency(f_SR_i_dB, fiber_frequencies[id_peak_sr_i])) # frequency x dB from peak vs peak

    # x dB point of edge spectrum vs edge spectrum
    print('S: B vs D', octave_from_frequency(f_5000_SR_s_dB, 5000)) # frequency x dB from 5000 Hz vs 5000 Hz
    print('I: B vs D', octave_from_frequency(f_5000_SR_i_dB, 6500))# frequency x dB from 6500 Hz vs 6500 Hz  

    plt.show()

def double_normalized_spectrum_standard_inverted_one_fig(RPO, NH_dB, octave_spaced):
    if num_CFs == 1903:
        id_peak_smrt = 1851
        id_peak_sr_s = 1801
        id_peak_sr_i = 1784
        id_peak_smrt_R = 1852
    elif num_CFs == 2416:
        id_peak_smrt_R = 2287
        id_peak_smrt = 2281
        id_peak_sr_s = 2147
        id_peak_sr_i = 2099

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.079, bottom=0.11, right=0.967, top=0.929, wspace=0.105)
    bar_width = 15

    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    alpha = 0.2
    # get EH
    fname_NH_SR_s = glob.glob(sr_data_dir + '*_s_*' + RPO + '*_1_*.mat')[0]
    print('SR_s:', fname_NH_SR_s)
    fname_NH_SR_i = glob.glob(sr_data_dir + '*_i1_*' + RPO + '*_1_*.mat')[0]
    print('SR_i:', fname_NH_SR_i)
    
    normal_spectrum_SR_s, fiber_frequencies = get_normalized_spectrum(fname_NH_SR_s)  # 
    normal_spectrum_SR_i, fiber_frequencies = get_normalized_spectrum(fname_NH_SR_i)    


    #Spectral ripple standard
    ax = plt.subplot(1, 2, 1)
    ax.text(-0.01, 0.1, 'A', transform=ax.transAxes + trans,
        fontsize=18, verticalalignment='top', color='white')
    plt.bar(fiber_frequencies, normal_spectrum_SR_s, width=bar_width, alpha=alpha, color='blue')
    filter_sig_sr_s = butter_lowpass_filter(normal_spectrum_SR_s, cut_off_freq, len(normal_spectrum_SR_s), filter_order)
    plt.plot(fiber_frequencies, filter_sig_sr_s, color='blue', label='filtered response (standard)')
    # plt.xlim(min(fiber_frequencies), 8e3)
    plt.xlim(4e3, max(fiber_frequencies))
    plt.vlines(5000, 0, 1, colors='red', label='frequency limit')
    plt.ylim((0,1))
    plt.ylabel('Normalized \n spiking')
    plt.title('Spectral ripple: inverted versus standard (4.0 RPO)')
    plt.xlabel('Frequency [Hz]')
    # Spectral ripple inverted
    plt.bar(fiber_frequencies, normal_spectrum_SR_i, width=bar_width, alpha=alpha, color='magenta')
    filter_sig_sr_i = butter_lowpass_filter(normal_spectrum_SR_i, cut_off_freq, len(normal_spectrum_SR_i), filter_order)
    plt.plot(fiber_frequencies, filter_sig_sr_i, color='magenta', label='filtered response (inverted)')
    dB=3
    print(dB, 'dB point')
    _, id_5000 = find_nearest(fiber_frequencies, 5000)
    # 3 dB point:
    y_val_SR_s = normal_spectrum_SR_s[id_peak_sr_s]/(10**(dB/10))
    y_val_SR_i = normal_spectrum_SR_i[id_peak_sr_i]/(10**(dB/10))
    y_val_5000_s = normal_spectrum_SR_s[id_5000]/(10**(dB/10))
    y_val_5000_i = normal_spectrum_SR_i[id_5000]/(10**(dB/10))
    # needs to be larger than the y_val_SR!!!    
    SR_s_dB, idx_SR_s_dB = find_nearest(filter_sig_sr_s[id_peak_sr_s:], y_val_SR_s)
    SR_i_dB, idx_SR_i_dB = find_nearest(filter_sig_sr_i[id_peak_sr_i:], y_val_SR_i)

    SR_s_5000_dB, idx_5000_SR_s_dB = find_nearest(filter_sig_sr_s[id_peak_sr_s:], y_val_5000_s)
    SR_i_5000_dB, idx_5000_SR_i_dB = find_nearest(filter_sig_sr_i[id_peak_sr_i:], y_val_5000_i)
    idx_SR_s_dB += id_peak_sr_s
    idx_SR_i_dB += id_peak_sr_i

    idx_5000_SR_s_dB += id_peak_sr_s
    idx_5000_SR_i_dB += id_peak_sr_i

    f_SR_s_dB = fiber_frequencies[idx_SR_s_dB] # x dB point versus peak standard
    f_SR_i_dB = fiber_frequencies[idx_SR_i_dB] # x dB point versus peak inverted

    f_5000_SR_s_dB = fiber_frequencies[idx_5000_SR_s_dB] # x dB point versus 5000 Hz
    f_5000_SR_i_dB = fiber_frequencies[idx_5000_SR_i_dB] # x dB point versus 5000 Hz

    # SOMETHING IS OFF why is f_5000_SR_i_dB = 8028

    peak_marker = '*'
    peak_marker_size = 100
    ax = plt.subplot(1, 2, 1)
    plt.scatter(fiber_frequencies[id_peak_sr_s], filter_sig_sr_s[id_peak_sr_s], color='blue', s=peak_marker_size, marker=peak_marker, label= 'standard peak')
    plt.scatter(f_SR_s_dB, SR_s_dB, color='blue', label= str(dB) + ' dB (standard) from peak')
    # plt.scatter(f_5000_SR_s_dB, SR_s_5000_dB, color='r', label= str(dB) + ' dB (standard) from limit')
    plt.scatter(fiber_frequencies[id_peak_sr_i], filter_sig_sr_i[id_peak_sr_i], color='magenta', marker=peak_marker, s=peak_marker_size, label= 'inverted peak')
    plt.scatter(f_SR_i_dB, SR_i_dB, color='magenta', label= str(dB) + ' dB (inverted) from peak')
    # plt.scatter(f_5000_SR_i_dB, SR_i_5000_dB, color='r', label= str(dB) + ' dB (inverted) from limit')
    plt.legend()

    # SMRT
    _, id_6500 = find_nearest(fiber_frequencies, 6500)

    if num_CFs == 1903:
        time_str = ''
        if NH_dB == 50:
            dB_str = '2024-04-16'
        if NH_dB == 65:
            dB_str = '2023-11-24'
    if num_CFs == 2416:
        dB_str = ''
        time_str = '*2025'
        
    fname_NH_SMRT_RT = glob.glob(SMRT_data_dir + time_str +'*' + dB_str + '*100*width_*' + RPO[0] + '*.mat')[0]
    fname_NH_SMRT_R = glob.glob(SMRT_data_dir + time_str + '*100*width_20*'+ str(num_CFs) + 'CFs.mat')[0]
    # fname_NH_SMRT_RT1 = glob.glob(SMRT_data_dir + time_str + '*100*width_1_*' + str(num_CFs) +'CFs.mat')[0]
    print('SMRT RT:', fname_NH_SMRT_RT)
    print('SMRT R:', fname_NH_SMRT_R)
    normal_spectrum_SMRT_RT, _ =  get_normalized_spectrum(fname_NH_SMRT_RT)
    # normal_spectrum_SMRT_RT1, _ =  get_normalized_spectrum(fname_NH_SMRT_RT1)       
    normal_spectrum_SMRT_R, _ =  get_normalized_spectrum(fname_NH_SMRT_R)       
    ax = plt.subplot(1, 2, 2)
    ax.text(-0.01, 0.1, 'B', transform=ax.transAxes + trans,
        fontsize=18, verticalalignment='top', color='white')
    plt.bar(fiber_frequencies, normal_spectrum_SMRT_RT, width=bar_width, alpha=alpha, color='blue')
    # plt.bar(fiber_frequencies, normal_spectrum_SMRT_RT1, width=bar_width, alpha=alpha, color='blue')
    plt.bar(fiber_frequencies, normal_spectrum_SMRT_R, width=bar_width, alpha=alpha, color='magenta')
    filter_sig_smrt_RT = butter_lowpass_filter(normal_spectrum_SMRT_RT, cut_off_freq, len(normal_spectrum_SMRT_RT), filter_order)
    # filter_sig_smrt_RT1 = butter_lowpass_filter(normal_spectrum_SMRT_RT1, cut_off_freq, len(normal_spectrum_SMRT_RT1), filter_order)
    filter_sig_smrt_R = butter_lowpass_filter(normal_spectrum_SMRT_R, cut_off_freq, len(normal_spectrum_SMRT_R), filter_order)
    plt.plot(fiber_frequencies, filter_sig_smrt_RT, color='blue', label='filtered response (4.0 RPO)')
    # plt.plot(fiber_frequencies, filter_sig_smrt_RT1, color='green', label='filtered response (1.0 RPO)')
    plt.plot(fiber_frequencies, filter_sig_smrt_R, color='magenta', label='filtered response (20.0 RPO)')
    plt.vlines(6500, 0, 1, colors='red', label='frequency limit')
    plt.title('SMRT: 4.0 RPO versus 20.0 RPO')
    plt.xlabel('Frequency [Hz]')
    y_val_SMRT_RT = normal_spectrum_SMRT_RT[id_peak_smrt]/(10**(dB/10))
    y_val_6500_RT = normal_spectrum_SMRT_RT[id_6500]/(10**(dB/10)) 

    y_val_SMRT_R = normal_spectrum_SMRT_R[id_peak_smrt_R]/(10**(dB/10))
    y_val_6500_R = normal_spectrum_SMRT_R[id_6500]/(10**(dB/10)) 

    # needs to be larger than the y_val_SR!!!    
    SMRT_RT_dB, idx_SMRT_RT_dB = find_nearest(filter_sig_smrt_RT[id_peak_smrt:], y_val_SMRT_RT)
    y_SMRT_RT_6500, idx_SMRT_RT_dB = find_nearest(filter_sig_smrt_RT[id_peak_smrt:], y_val_SMRT_RT)    
    SMRT_6500_RT_dB, idx_6500_SMRT_RT_dB = find_nearest(filter_sig_smrt_RT[id_peak_smrt:], y_val_6500_RT)

    SMRT_R_dB, idx_SMRT_R_dB = find_nearest(filter_sig_smrt_R[id_peak_smrt_R:], y_val_SMRT_R)
    y_SMRT_R_6500, idx_SMRT_R_dB = find_nearest(filter_sig_smrt_R[id_peak_smrt_R:], y_val_SMRT_R)    
    SMRT_6500_R_dB, idx_6500_SMRT_R_dB = find_nearest(filter_sig_smrt_R[id_peak_smrt_R:], y_val_6500_R)

    idx_SMRT_RT_dB += id_peak_smrt
    idx_SMRT_R_dB += id_peak_smrt_R
    idx_6500_SMRT_RT_dB += id_peak_smrt
    idx_6500_SMRT_R_dB += id_peak_smrt_R
    f_SMRT_RT_dB = fiber_frequencies[idx_SMRT_RT_dB] # x dB point versus peak SMRT
    f_SMRT_R_dB = fiber_frequencies[idx_SMRT_R_dB] # x dB point versus peak SMRT
    f_6500_SMRT_RT_dB = fiber_frequencies[idx_6500_SMRT_RT_dB] # x dB point versus 6500 Hz
    f_6500_SMRT_R_dB = fiber_frequencies[idx_6500_SMRT_R_dB] # x dB point versus 6500 Hz

    plt.scatter(fiber_frequencies[id_peak_smrt], filter_sig_smrt_RT[id_peak_smrt], color='blue', s=peak_marker_size, marker=peak_marker, label= '4 RPO peak')
    plt.scatter(f_SMRT_RT_dB, SMRT_RT_dB, color='blue', label= str(dB) + ' dB (4.0 RPO) from peak')
    # plt.scatter(f_6500_SMRT_RT_dB, SMRT_6500_RT_dB, color='r', label= str(dB) + ' dB (4.0 RPO) from limit')
    plt.scatter(fiber_frequencies[id_peak_smrt_R], filter_sig_smrt_R[id_peak_smrt_R], color='magenta', s=peak_marker_size, marker=peak_marker, label= '20 RPO peak')
    plt.scatter(f_SMRT_R_dB, SMRT_R_dB, color='magenta', label= str(dB) + ' dB (20.0 RPO) from peak')
    # plt.scatter(f_6500_SMRT_R_dB, SMRT_6500_R_dB, color='r', label= str(dB) + ' dB (20.0 RPO) from limit')
    
    plt.legend(loc='center left')
    

    # print('SRs: C vs B', octave_from_frequency(f_SR_s_dB, 5000)) # frequency x dB from peak vs 5000 Hz
    # print('SRi: C vs B', octave_from_frequency(f_SR_i_dB, 5000)) # frequency x dB from peak vs 5000 Hz
    # print('SMRT: C vs B', octave_from_frequency(f_SMRT_dB, 6500)) # x frequency dB from peak vs 6500 Hz

    # x peak dB point vs peak
    print('SRs: C vs A', octave_from_frequency(f_SR_s_dB, fiber_frequencies[id_peak_sr_s])) # x frequency dB from peak vs peak
    print('SRi: C vs A', octave_from_frequency(f_SR_i_dB, fiber_frequencies[id_peak_sr_i])) # x frequency dB from peak vs peak
    print('SMRT: C vs A', octave_from_frequency(f_SMRT_RT_dB, fiber_frequencies[id_peak_smrt])) # frequency x dB from peak vs peak

    # x dB point of edge spectrum vs edge spectrum
    print('SRs: B vs D', octave_from_frequency(f_5000_SR_s_dB, 5000)) # frequency x dB from 5000 Hz vs 5000 Hz
    print('SRi: B vs D', octave_from_frequency(f_5000_SR_i_dB, 5000)) # frequency x dB from 5000 Hz vs 5000 Hz
    print('SMRT: B vs D', octave_from_frequency(f_6500_SMRT_RT_dB, 6500))# frequency x dB from 6500 Hz vs 6500 Hz   

    # 4 RPO vs 20 RPO
    print('SMRT 4: C vs A', octave_from_frequency(f_SMRT_RT_dB, fiber_frequencies[id_peak_smrt])) # frequency x dB from peak vs peak
    print('SMRT 20: C vs A', octave_from_frequency(f_SMRT_R_dB, fiber_frequencies[id_peak_smrt_R])) # frequency x dB from peak vs peak

    print('SMRT 4: B vs D', octave_from_frequency(f_6500_SMRT_RT_dB, 6500))# frequency x dB from 6500 Hz vs 6500 Hz  
    print('SMRT 20: B vs D', octave_from_frequency(f_6500_SMRT_R_dB, 6500))# frequency x dB from 6500 Hz vs 6500 Hz 

    return fig


def double_spectrum_standard_inverted_one_fig(RPO, NH_dB, octave_spaced):
    if num_CFs == 1903:
        id_peak_smrt = 1851
        id_peak_sr_s = 1801
        id_peak_sr_i = 1784
        id_peak_smrt_R = 1852
    elif num_CFs == 2416:
        id_peak_smrt_R = 2287
        id_peak_smrt = 2281
        id_peak_sr_s = 2147
        id_peak_sr_i = 2099

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.079, bottom=0.11, right=0.967, top=0.929, wspace=0.105)
    bar_width = 15

    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    alpha = 0.2
    # get EH
    fname_NH_SR_s = glob.glob(sr_data_dir + '*_s_*' + RPO + '*_1_*.mat')[0]
    print('SR_s:', fname_NH_SR_s)
    fname_NH_SR_i = glob.glob(sr_data_dir + '*_i1_*' + RPO + '*_1_*.mat')[0]
    print('SR_i:', fname_NH_SR_i)
    
    normal_spectrum_SR_s, fiber_frequencies = get_spectrum(fname_NH_SR_s)  
    normal_spectrum_SR_i, fiber_frequencies = get_spectrum(fname_NH_SR_i)    

    # get single fiber (50 total) spike rate, duration sound is =.5
    normal_spectrum_SR_s = normal_spectrum_SR_s/0.5/50
    normal_spectrum_SR_i = normal_spectrum_SR_i/0.5/50

    #Spectral ripple standard
    ax = plt.subplot(1, 2, 1)
    ax.text(-0.01, 0.1, 'A', transform=ax.transAxes + trans,
        fontsize=18, verticalalignment='top', color='white')
    plt.bar(fiber_frequencies, normal_spectrum_SR_s, width=bar_width, alpha=alpha, color='blue')
    filter_sig_sr_s = butter_lowpass_filter(normal_spectrum_SR_s, cut_off_freq, len(normal_spectrum_SR_s), filter_order)
    plt.plot(fiber_frequencies, filter_sig_sr_s, color='blue', label='filtered response (standard)')
    # plt.xlim(min(fiber_frequencies), 8e3)
    plt.xlim(4e3, max(fiber_frequencies))
    plt.vlines(5000, 0, 3000, colors='red', label='frequency limit')
    plt.hlines(1100/0.5/50, 4e3, 8100, colors='red', linestyles='dashed', label='Spontaneous rate')
    plt.ylim((0,3000/0.5/50))
    plt.ylabel('Spiking rate [sp/s]')
    plt.title('Spectral ripple: inverted versus standard (4.0 RPO)')
    plt.xlabel('Frequency [Hz]')
    # Spectral ripple inverted
    plt.bar(fiber_frequencies, normal_spectrum_SR_i, width=bar_width, alpha=alpha, color='magenta')
    filter_sig_sr_i = butter_lowpass_filter(normal_spectrum_SR_i, cut_off_freq, len(normal_spectrum_SR_i), filter_order)
    plt.plot(fiber_frequencies, filter_sig_sr_i, color='magenta', label='filtered response (inverted)')
    dB=3
    print(dB, 'dB point')
    _, id_5000 = find_nearest(fiber_frequencies, 5000)
    # 3 dB point:
    y_val_SR_s = normal_spectrum_SR_s[id_peak_sr_s]/(10**(dB/10))
    y_val_SR_i = normal_spectrum_SR_i[id_peak_sr_i]/(10**(dB/10))
    y_val_5000_s = normal_spectrum_SR_s[id_5000]/(10**(dB/10))
    y_val_5000_i = normal_spectrum_SR_i[id_5000]/(10**(dB/10))
    # needs to be larger than the y_val_SR!!!    
    SR_s_dB, idx_SR_s_dB = find_nearest(filter_sig_sr_s[id_peak_sr_s:], y_val_SR_s)
    SR_i_dB, idx_SR_i_dB = find_nearest(filter_sig_sr_i[id_peak_sr_i:], y_val_SR_i)

    SR_s_5000_dB, idx_5000_SR_s_dB = find_nearest(filter_sig_sr_s[id_peak_sr_s:], y_val_5000_s)
    SR_i_5000_dB, idx_5000_SR_i_dB = find_nearest(filter_sig_sr_i[id_peak_sr_i:], y_val_5000_i)
    idx_SR_s_dB += id_peak_sr_s
    idx_SR_i_dB += id_peak_sr_i

    idx_5000_SR_s_dB += id_peak_sr_s
    idx_5000_SR_i_dB += id_peak_sr_i

    f_SR_s_dB = fiber_frequencies[idx_SR_s_dB] # x dB point versus peak standard
    f_SR_i_dB = fiber_frequencies[idx_SR_i_dB] # x dB point versus peak inverted

    f_5000_SR_s_dB = fiber_frequencies[idx_5000_SR_s_dB] # x dB point versus 5000 Hz
    f_5000_SR_i_dB = fiber_frequencies[idx_5000_SR_i_dB] # x dB point versus 5000 Hz

    # SOMETHING IS OFF why is f_5000_SR_i_dB = 8028

    peak_marker = '*'
    peak_marker_size = 100
    ax = plt.subplot(1, 2, 1)
    plt.scatter(fiber_frequencies[id_peak_sr_s], filter_sig_sr_s[id_peak_sr_s], color='blue', s=peak_marker_size, marker=peak_marker, label= 'standard peak')
    plt.scatter(f_SR_s_dB, SR_s_dB, color='blue', label= str(dB) + ' dB (standard) from peak')
    # plt.scatter(f_5000_SR_s_dB, SR_s_5000_dB, color='r', label= str(dB) + ' dB (standard) from limit')
    plt.scatter(fiber_frequencies[id_peak_sr_i], filter_sig_sr_i[id_peak_sr_i], color='magenta', marker=peak_marker, s=peak_marker_size, label= 'inverted peak')
    plt.scatter(f_SR_i_dB, SR_i_dB, color='magenta', label= str(dB) + ' dB (inverted) from peak')
    # plt.scatter(f_5000_SR_i_dB, SR_i_5000_dB, color='r', label= str(dB) + ' dB (inverted) from limit')
    plt.legend()

    # SMRT
    _, id_6500 = find_nearest(fiber_frequencies, 6500)

    if num_CFs == 1903:
        time_str = ''
        if NH_dB == 50:
            dB_str = '2024-04-16'
        if NH_dB == 65:
            dB_str = '2023-11-24'
    if num_CFs == 2416:
        dB_str = ''
        time_str = '*2025'
        
    fname_NH_SMRT_RT = glob.glob(SMRT_data_dir + time_str +'*' + dB_str + '*100*width_*' + RPO[0] + '*.mat')[0]
    fname_NH_SMRT_R = glob.glob(SMRT_data_dir + time_str + '*100*width_20*'+ str(num_CFs) + 'CFs.mat')[0]
    # fname_NH_SMRT_RT1 = glob.glob(SMRT_data_dir + time_str + '*100*width_1_*' + str(num_CFs) +'CFs.mat')[0]
    print('SMRT RT:', fname_NH_SMRT_RT)
    print('SMRT R:', fname_NH_SMRT_R)
    normal_spectrum_SMRT_RT, _ =  get_spectrum(fname_NH_SMRT_RT)
    # normal_spectrum_SMRT_RT1, _ =  get_normalized_spectrum(fname_NH_SMRT_RT1)       
    normal_spectrum_SMRT_R, _ =  get_spectrum(fname_NH_SMRT_R)

    normal_spectrum_SMRT_RT = normal_spectrum_SMRT_RT/0.5/50
    normal_spectrum_SMRT_R = normal_spectrum_SMRT_R/0.5/50

    ax = plt.subplot(1, 2, 2)
    ax.text(-0.01, 0.1, 'B', transform=ax.transAxes + trans,
        fontsize=18, verticalalignment='top', color='white')
    plt.bar(fiber_frequencies, normal_spectrum_SMRT_RT, width=bar_width, alpha=alpha, color='blue')
    # plt.bar(fiber_frequencies, normal_spectrum_SMRT_RT1, width=bar_width, alpha=alpha, color='blue')
    plt.bar(fiber_frequencies, normal_spectrum_SMRT_R, width=bar_width, alpha=alpha, color='magenta')
    filter_sig_smrt_RT = butter_lowpass_filter(normal_spectrum_SMRT_RT, cut_off_freq, len(normal_spectrum_SMRT_RT), filter_order)
    # filter_sig_smrt_RT1 = butter_lowpass_filter(normal_spectrum_SMRT_RT1, cut_off_freq, len(normal_spectrum_SMRT_RT1), filter_order)
    filter_sig_smrt_R = butter_lowpass_filter(normal_spectrum_SMRT_R, cut_off_freq, len(normal_spectrum_SMRT_R), filter_order)
    plt.plot(fiber_frequencies, filter_sig_smrt_RT, color='blue', label='filtered response (4.0 RPO)')
    # plt.plot(fiber_frequencies, filter_sig_smrt_RT1, color='green', label='filtered response (1.0 RPO)')
    plt.plot(fiber_frequencies, filter_sig_smrt_R, color='magenta', label='filtered response (20.0 RPO)')
    plt.vlines(6500, 0, 3000, colors='red', label='frequency limit')
    plt.hlines(1100/0.5/50, 4e3, 8100, colors='red',linestyles='dashed', label='Spontaneous rate')
    plt.title('SMRT: 4.0 RPO versus 20.0 RPO')
    plt.xlabel('Frequency [Hz]')
    y_val_SMRT_RT = normal_spectrum_SMRT_RT[id_peak_smrt]/(10**(dB/10))
    y_val_6500_RT = normal_spectrum_SMRT_RT[id_6500]/(10**(dB/10)) 

    y_val_SMRT_R = normal_spectrum_SMRT_R[id_peak_smrt_R]/(10**(dB/10))
    y_val_6500_R = normal_spectrum_SMRT_R[id_6500]/(10**(dB/10)) 

    # needs to be larger than the y_val_SR!!!    
    SMRT_RT_dB, idx_SMRT_RT_dB = find_nearest(filter_sig_smrt_RT[id_peak_smrt:], y_val_SMRT_RT)
    y_SMRT_RT_6500, idx_SMRT_RT_dB = find_nearest(filter_sig_smrt_RT[id_peak_smrt:], y_val_SMRT_RT)    
    SMRT_6500_RT_dB, idx_6500_SMRT_RT_dB = find_nearest(filter_sig_smrt_RT[id_peak_smrt:], y_val_6500_RT)

    SMRT_R_dB, idx_SMRT_R_dB = find_nearest(filter_sig_smrt_R[id_peak_smrt_R:], y_val_SMRT_R)
    y_SMRT_R_6500, idx_SMRT_R_dB = find_nearest(filter_sig_smrt_R[id_peak_smrt_R:], y_val_SMRT_R)    
    SMRT_6500_R_dB, idx_6500_SMRT_R_dB = find_nearest(filter_sig_smrt_R[id_peak_smrt_R:], y_val_6500_R)

    idx_SMRT_RT_dB += id_peak_smrt
    idx_SMRT_R_dB += id_peak_smrt_R
    idx_6500_SMRT_RT_dB += id_peak_smrt
    idx_6500_SMRT_R_dB += id_peak_smrt_R
    f_SMRT_RT_dB = fiber_frequencies[idx_SMRT_RT_dB] # x dB point versus peak SMRT
    f_SMRT_R_dB = fiber_frequencies[idx_SMRT_R_dB] # x dB point versus peak SMRT
    f_6500_SMRT_RT_dB = fiber_frequencies[idx_6500_SMRT_RT_dB] # x dB point versus 6500 Hz
    f_6500_SMRT_R_dB = fiber_frequencies[idx_6500_SMRT_R_dB] # x dB point versus 6500 Hz

    plt.scatter(fiber_frequencies[id_peak_smrt], filter_sig_smrt_RT[id_peak_smrt], color='blue', s=peak_marker_size, marker=peak_marker, label= '4 RPO peak')
    plt.scatter(f_SMRT_RT_dB, SMRT_RT_dB, color='blue', label= str(dB) + ' dB (4.0 RPO) from peak')
    # plt.scatter(f_6500_SMRT_RT_dB, SMRT_6500_RT_dB, color='r', label= str(dB) + ' dB (4.0 RPO) from limit')
    plt.scatter(fiber_frequencies[id_peak_smrt_R], filter_sig_smrt_R[id_peak_smrt_R], color='magenta', s=peak_marker_size, marker=peak_marker, label= '20 RPO peak')
    plt.scatter(f_SMRT_R_dB, SMRT_R_dB, color='magenta', label= str(dB) + ' dB (20.0 RPO) from peak')
    # plt.scatter(f_6500_SMRT_R_dB, SMRT_6500_R_dB, color='r', label= str(dB) + ' dB (20.0 RPO) from limit')
    
    plt.legend(loc='center left')
    
    # Above frequency content
    # upper_4, upperfreq = get_spectrum(SMRT_data_dir + '/PSTH_filter2025-04-11_12-23-27_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_4_1000CFs.mat')
    # upper_20, _ = get_spectrum(SMRT_data_dir + '/PSTH_filter2025-04-11_12-24-26_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_20_1000CFs.mat')
    # upper_s, _ = get_spectrum(SMRT_data_dir + '/PSTH_filter2025-04-11_12-27-03_s_4.000_1_1000CFs.mat')
    # upper_i, _ = get_spectrum(SMRT_data_dir + '/PSTH_filter2025-04-11_12-32-35_i1_4.000_1_1000CFs.mat')
    # # all approximately 1080 spikes
    # np.mean(upper_4[678:])
    # np.mean(upper_20[678:])
    # np.mean(upper_s[678:])
    # np.mean(upper_i[678:])

    # print('SRs: C vs B', octave_from_frequency(f_SR_s_dB, 5000)) # frequency x dB from peak vs 5000 Hz
    # print('SRi: C vs B', octave_from_frequency(f_SR_i_dB, 5000)) # frequency x dB from peak vs 5000 Hz
    # print('SMRT: C vs B', octave_from_frequency(f_SMRT_dB, 6500)) # x frequency dB from peak vs 6500 Hz

    # x peak dB point vs peak
    print('SRs: C vs A', octave_from_frequency(f_SR_s_dB, fiber_frequencies[id_peak_sr_s])) # x frequency dB from peak vs peak
    print('SRi: C vs A', octave_from_frequency(f_SR_i_dB, fiber_frequencies[id_peak_sr_i])) # x frequency dB from peak vs peak
    print('SMRT: C vs A', octave_from_frequency(f_SMRT_RT_dB, fiber_frequencies[id_peak_smrt])) # frequency x dB from peak vs peak

    # x dB point of edge spectrum vs edge spectrum
    print('SRs: B vs D', octave_from_frequency(f_5000_SR_s_dB, 5000)) # frequency x dB from 5000 Hz vs 5000 Hz
    print('SRi: B vs D', octave_from_frequency(f_5000_SR_i_dB, 5000)) # frequency x dB from 5000 Hz vs 5000 Hz
    print('SMRT: B vs D', octave_from_frequency(f_6500_SMRT_RT_dB, 6500))# frequency x dB from 6500 Hz vs 6500 Hz   

    # 4 RPO vs 20 RPO
    print('SMRT 4: C vs A', octave_from_frequency(f_SMRT_RT_dB, fiber_frequencies[id_peak_smrt])) # frequency x dB from peak vs peak
    print('SMRT 20: C vs A', octave_from_frequency(f_SMRT_R_dB, fiber_frequencies[id_peak_smrt_R])) # frequency x dB from peak vs peak

    print('SMRT 4: B vs D', octave_from_frequency(f_6500_SMRT_RT_dB, 6500))# frequency x dB from 6500 Hz vs 6500 Hz  
    print('SMRT 20: B vs D', octave_from_frequency(f_6500_SMRT_R_dB, 6500))# frequency x dB from 6500 Hz vs 6500 Hz 

    return fig


def double_spectrum_one_fig(RPO, NH_dB, octave_spaced):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.079, bottom=0.11, right=0.967, top=0.929, wspace=0.105)
    bar_width = 15
    
    if NH_dB == 50:
        dB_str = '2024-04-16'
        # dB_str = '2024-04-02' # 2847
    if NH_dB == 65:
        dB_str = '2023-11-24'
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    alpha = 0.2
    # get EH
    fname_NH_SR = glob.glob(sr_data_dir + '*_s_*' + RPO + '*_1_*.mat')[0]
    print('SR:', fname_NH_SR)
    fname_NH_SMRT = glob.glob(SMRT_data_dir + '*' + dB_str + '*100*width_*' + RPO[0] + '*.mat')[0]
    print('SMRT:', fname_NH_SMRT)
    normal_spectrum_SR, fiber_frequencies = get_normalized_spectrum(fname_NH_SR) 
    normal_spectrum_SMRT, _ =  get_normalized_spectrum(fname_NH_SMRT)          
    
    _, id_5000 = find_nearest(fiber_frequencies, 5000)
    _, id_6500 = find_nearest(fiber_frequencies, 6500)

    #Spectral ripple
    ax = plt.subplot(1, 2, 1)
    ax.text(-0.01, 0.1, 'A', transform=ax.transAxes + trans,
        fontsize=18, verticalalignment='top', color='white')
    plt.bar(fiber_frequencies, normal_spectrum_SR, width=bar_width, alpha=alpha, color='blue')
    filter_sig_sr = butter_lowpass_filter(normal_spectrum_SR, cut_off_freq, len(normal_spectrum_SR), filter_order)
    plt.plot(fiber_frequencies, filter_sig_sr, color='blue')
    plt.xlim(min(fiber_frequencies), 8e3)
    plt.vlines(5000, 0, 3000, colors='red')
    plt.ylim((0,1))
    plt.ylabel(RPO + ' RPO \n normalized \n spiking')
    plt.title('Spectral ripple')
    plt.xlabel('Frequency [Hz]')

    if octave_spaced:
        plt.xscale('log', base=2)
        plt.xticks([500, 1000, 2000, 4000, 8000], labels=['500', '1000', '2000', '4000', '8000'])        

    # SMRT
    ax = plt.subplot(1, 2, 2)
    ax.text(-0.01, 0.1, 'B', transform=ax.transAxes + trans,
        fontsize=18, verticalalignment='top', color='white')
    plt.bar(fiber_frequencies, normal_spectrum_SMRT, width=bar_width, alpha=alpha, color='blue')
    filter_sig_smrt = butter_lowpass_filter(normal_spectrum_SMRT, cut_off_freq, len(normal_spectrum_SMRT), filter_order)
    plt.plot(fiber_frequencies, filter_sig_smrt, color='blue', label='filtered response')
    plt.vlines(6500, 0, 1, colors='red')
    plt.title('SMRT')
    plt.xlabel('Frequency [Hz]')
    if octave_spaced:
        plt.xscale('log', base=2)
        plt.xticks([500, 1000, 2000, 4000, 8000], labels=['500', '1000', '2000', '4000', '8000'])
    # plt.show()
    dB=3
    print(dB, 'dB point')
    # 3 dB point:
    y_val_SR = normal_spectrum_SR[id_peak_sr]/(10**(dB/10))
    y_val_SMRT = normal_spectrum_SMRT[id_peak_smrt]/(10**(dB/10))

    y_val_5000 = normal_spectrum_SR[id_5000]/(10**(dB/10))
    y_val_6500 = normal_spectrum_SMRT[id_6500]/(10**(dB/10)) 

    # needs to be larger than the y_val_SR!!!    
    SR_dB, idx_SR_dB = find_nearest(filter_sig_sr[id_peak_sr:], y_val_SR)
    SMRT_dB, idx_SMRT_dB = find_nearest(filter_sig_smrt[id_peak_smrt:], y_val_SMRT)

    y_SR_5000, idx_SR_dB = find_nearest(filter_sig_sr[id_peak_sr:], y_val_SR)
    y_SMRT_6500, idx_SMRT_dB = find_nearest(filter_sig_smrt[id_peak_smrt:], y_val_SMRT)    

    SR_5000_dB, idx_5000_SR_dB = find_nearest(filter_sig_sr[id_peak_sr:], y_val_5000)
    SMRT_6500_dB, idx_6500_SMRT_dB = find_nearest(filter_sig_smrt[id_peak_smrt:], y_val_6500)    

    idx_SR_dB += id_peak_sr
    idx_SMRT_dB += id_peak_smrt

    idx_5000_SR_dB += id_peak_sr
    idx_6500_SMRT_dB += id_peak_smrt

    f_SR_dB = fiber_frequencies[idx_SR_dB] # x dB point versus peak SR
    f_SMRT_dB = fiber_frequencies[idx_SMRT_dB] # x dB point versus peak SMRT

    f_5000_SR_dB = fiber_frequencies[idx_5000_SR_dB] # x dB point versus 5000 Hz
    f_6500_SMRT_dB = fiber_frequencies[idx_6500_SMRT_dB] # x dB point versus 6500 Hz
    
    ax = plt.subplot(1, 2, 1)
    plt.scatter(f_SR_dB, SR_dB, label= str(dB) + ' from peak')
    plt.scatter(f_5000_SR_dB, SR_5000_dB, color='r', label= str(dB) + ' from red line')
    plt.legend()
    # plt.hlines()

    ax = plt.subplot(1, 2, 2)
    plt.scatter(f_SMRT_dB, SMRT_dB, label= str(dB) + ' from peak')
    plt.scatter(f_6500_SMRT_dB, SMRT_6500_dB, color='r', label= str(dB) + ' from red line')
    plt.legend()
    
    # octave difference
    print('Octave differences')
    # print(dB, 'dB frequency SR from peak vs 5000', octave_from_frequency(f_SR_dB, 5000)) # frequency x dB from peak vs 5000 Hz
    # print(dB, 'dB frequency SMRT from peak vs 6500', octave_from_frequency(f_SMRT_dB, 6500)) # x frequency dB from peak vs 6500 Hz
    # print(dB, 'dB frequency SR vs peak frequency', octave_from_frequency(f_SR_dB, fiber_frequencies[id_peak_sr])) # x frequency dB from peak vs peak
    # print(dB, 'dB frequency SMRT vs peak frequency', octave_from_frequency(f_SMRT_dB, fiber_frequencies[id_peak_smrt])) # frequency x dB from peak vs peak

    # print('5000 Hz vs frequency at ', dB,  'dB', octave_from_frequency(f_5000_SR_dB, 5000)) # frequency x dB from 5000 Hz vs 5000 Hz
    # print('6500 Hz vs frequency at ', dB, 'dB', octave_from_frequency(f_6500_SMRT_dB, 6500))# frequency x dB from 6500 Hz vs 6500 Hz

    # x peak dB point vs end spectrum
    print('SR: C vs B', octave_from_frequency(f_SR_dB, 5000)) # frequency x dB from peak vs 5000 Hz
    print('SMRT: C vs B', octave_from_frequency(f_SMRT_dB, 6500)) # x frequency dB from peak vs 6500 Hz

    # x peak dB point vs peak
    print('SR: C vs A', octave_from_frequency(f_SR_dB, fiber_frequencies[id_peak_sr])) # x frequency dB from peak vs peak
    print('SMRT: C vs A', octave_from_frequency(f_SMRT_dB, fiber_frequencies[id_peak_smrt])) # frequency x dB from peak vs peak

    # x dB point of edge spectrum vs edge spectrum
    print('SR: B vs D', octave_from_frequency(f_5000_SR_dB, 5000)) # frequency x dB from 5000 Hz vs 5000 Hz
    print('SMRT: B vs D', octave_from_frequency(f_6500_SMRT_dB, 6500))# frequency x dB from 6500 Hz vs 6500 Hz    
    
    x=3

    return fig





if __name__ == "__main__":
    SR_vs_SMRT = False
    standard_vs_inverted = False
    SR_both_vs_SMRT = True
    SR_both_vs_SMRT_normalized = False
    SMRT_data_dir = './data/SMRT/'

    filter_order = 4
    cut_off_freq = 100
    RPO = '4.0'
    NH_dB = 50

    # sr_data_dir = './data/spectrum/'+str(NH_dB) +'dB_1903F/'
    sr_data_dir = './data/spectrum/'+str(NH_dB) + 'dB_2416CF/'
    octave_spaced = True
    if octave_spaced:
        octave_str = 'octave_spaced_'
    else:
        octave_str = ''
    num_CFs = 2416

    # sound spectrum
    # fig = double_sound_spectrum(RPO)
    # fig.savefig('./figures/SRvsSMRT_audio' + '_'.join(RPO) + 'RPO.png')
    
    if standard_vs_inverted:
        fig = spectrum_standard_inverted(RPO)

    if SR_both_vs_SMRT_normalized:
        fig = double_normalized_spectrum_standard_inverted_one_fig(RPO, NH_dB, octave_spaced)

    if SR_both_vs_SMRT:
        fig = double_spectrum_standard_inverted_one_fig(RPO, NH_dB, octave_spaced)
        fig.savefig('./figures/SRiSRsvsSMRT_filteredSpikeRate' + '_'.join(RPO) + 'RPO_' + octave_str + str(NH_dB) +'dB'+str(num_CFs)+'Fib.png')

    if SR_vs_SMRT:
        # neural activation
        fig = double_spectrum_one_fig(RPO, NH_dB, octave_spaced)
        fig.savefig('./figures/SRvsSMRT_filtered' + '_'.join(RPO) + 'RPO_' + octave_str + str(NH_dB) +'dB'+str(num_CFs)+'Fib.png')
    plt.show()

  