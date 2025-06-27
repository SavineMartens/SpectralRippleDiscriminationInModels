import numpy as np
from pymatreader import read_mat
from scipy.signal import butter, filtfilt, lfilter, spectrogram
import audio2numpy as a2n
import matplotlib.pyplot as plt
import load_gekke_matlab
import yaml
import os
import math
import glob

def load_mat_structs_Hamacher(fname, unfiltered_type = 'Hamacher'):
    matfile = read_mat(fname)['structPSTH']
    spike_rate_unfiltered = matfile['unfiltered_neurogram']
    sound_name = matfile['fname']
    if sound_name == None:
        sound_name = fname[fname.find('smrt'):fname.find('phase_0')+len('phase_0')]
    dBlevel = matfile['stimdb']
    fiber_frequencies = matfile['CF']
    t_unfiltered = matfile['t_unfiltered']
    try:
        Fs_down = matfile['down_sampling_rate']
    except:
        Fs_down = 'Newer version so not downsampled'
    if len(matfile) == 12:
        spike_rate_filtered_downsampled = spike_rate_unfiltered[:,::20] # Hamacher with Fs = 5000 --> 1e5/5e3 = 20
        t_filtered_downsampled = t_unfiltered[::20]
    else:
        try:
            spike_rate_filtered_downsampled = matfile['downsampled_Hamacher_neurogram']
            t_filtered_downsampled = matfile['t_downsampled_filtered_neurogram']
        except:
            spike_rate_filtered_downsampled = 'Newer version so not downsampled'
            t_filtered_downsampled = 'Newer version so not downsampled'
    if len(matfile) == 13:
        return spike_rate_unfiltered, spike_rate_filtered_downsampled, sound_name, dBlevel, fiber_frequencies, Fs_down, t_unfiltered, t_filtered_downsampled
    else:
        if unfiltered_type == 'Hamacher':
            Hamacher_neurogram = matfile['Hamacher_neurogram']
            print('Unfiltered version is Hamacher without downsampling')
            return Hamacher_neurogram, spike_rate_filtered_downsampled, sound_name, dBlevel, fiber_frequencies, Fs_down, t_unfiltered, t_filtered_downsampled
        elif unfiltered_type == 'OG' or unfiltered_type == 'Bruce':
            print('Unfiltered version is Bruce\'s version')
            return spike_rate_unfiltered, spike_rate_filtered_downsampled, sound_name, dBlevel, fiber_frequencies, Fs_down, t_unfiltered, t_filtered_downsampled

def load_matrices_from_vectors_Bruce_struct(fname):
    matfile = read_mat(fname)['structPSTH']
    sound_name = matfile['fname']
    dBlevel = matfile['stimdb']
    fiber_frequencies = matfile['CF']
    t_unfiltered = matfile['t_unfiltered']
    spike_rate_unfiltered = np.zeros((len(fiber_frequencies), len(t_unfiltered)))
    unfiltered_row_indices = matfile['unfiltered_row_indices'].astype(int)
    unfiltered_column_indices = matfile['unfiltered_column_indices'].astype(int)
    unfiltered_values = matfile['unfiltered_values']
    # fill in matrix
    for row, column, value in zip(unfiltered_row_indices, unfiltered_column_indices, unfiltered_values):
        spike_rate_unfiltered[row-1, column-1] = value
    # Hamacher 
    Hamacher_neurogram = np.zeros(spike_rate_unfiltered.shape)
    # new = np.zeros(spike_rate_unfiltered.shape)
    # for fiber in np.arange(len(fiber_frequencies)):
        # this one does not have a time delay
        # Hamacher_neurogram[fiber,:] = discrete_gaussian_filter(spike_rate_unfiltered[fiber,:], 1/1e5, sigma_c=1e-3)
    print('Don\'t use HAMACHER!!!!!!!!!!!!!!!!!!!!')
    Hamacher_Fs = 5000
    Fs_ratio = int(1e5/Hamacher_Fs)
    spike_rate_filtered_downsampled = Hamacher_neurogram[:,::Fs_ratio] # Hamacher with Fs = 5000 --> 1e5/5e3 = 20
    t_filtered_downsampled = t_unfiltered[::Fs_ratio]
    return spike_rate_unfiltered, spike_rate_filtered_downsampled, sound_name, dBlevel, fiber_frequencies, Hamacher_Fs, t_unfiltered, t_filtered_downsampled

def load_matrices_from_vectors_Bruce_multi_trial(fname):
    matfile = read_mat(fname)['structPSTH']
    sound_name = matfile['fname']
    dBlevel = matfile['stimdb']
    fiber_frequencies = matfile['CF']
    t_unfiltered = matfile['t_unfiltered']
    spike_rate_unfiltered = np.zeros((len(fiber_frequencies), len(t_unfiltered)))
    trial_row_indices = matfile['trial_row_indices'].astype(int)
    trial_column_indices = matfile['trial_column_indices'].astype(int)
    trial_values = matfile['trial_spikes']
    trial_count = matfile['trial_count']
    trial_idx = np.insert(np.cumsum(np.asarray(trial_count)), 0, 1 )# cumulative sum of trial counts to get the indices
    num_trials = int(matfile['nrep'])
    trials_neurogram = np.zeros((num_trials, len(fiber_frequencies), len(t_unfiltered)))
    # fill in matrix
    for trial in range(num_trials):
        row_indices = trial_row_indices[int(trial_idx[trial])-1:int(trial_idx[trial+1])]
        column_indices = trial_column_indices[int(trial_idx[trial])-1:int(trial_idx[trial+1])]
        values = trial_values[int(trial_idx[trial])-1:int(trial_idx[trial+1])]
        for row, column, value in zip(row_indices, column_indices, values):
            trials_neurogram[trial, row-1, column-1] = int(value)
    summed_neurogram = np.sum(spike_rate_unfiltered, axis=0) # sum over trials
    return summed_neurogram, trials_neurogram, sound_name, dBlevel, fiber_frequencies, t_unfiltered

from scipy.optimize import curve_fit
def sigmoid(x, L ,x0, k, b):
    # L is responsible for scaling the output range from [0,1] to [0,L]
    # b adds bias to the output and changes its range from [0,L] to [b,L+b]
    # k is responsible for scaling the input, which remains in (-inf,inf)
    # x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].
    # b = min(max(50, b), 33) # max(33, b)
    # L = max(100, L) # min(100-b, L)
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def sigmoid2(x, L ,x0, k, b):
    # L is responsible for scaling the output range from [0,1] to [0,L]
    # b adds bias to the output and changes its range from [0,L] to [b,L+b]
    # k is responsible for scaling the input, which remains in (-inf,inf)
    # x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].
    L = max(100, L)
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def sigmoid3(x, L ,x0, k, b):
    # L is responsible for scaling the output range from [0,1] to [0,L]
    # b adds bias to the output and changes its range from [0,L] to [b,L+b]
    # k is responsible for scaling the input, which remains in (-inf,inf)
    # x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].
    b = max(33, b)    
    L = min(100-b, L)
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def sigmoid4(x, L ,x0, k, b):
    # L is responsible for scaling the output range from [0,1] to [0,L]
    # b adds bias to the output and changes its range from [0,L] to [b,L+b]
    # k is responsible for scaling the input, which remains in (-inf,inf)
    # x0 is the point in the middle of the Sigmoid, i.e. the point where Sigmoid should originally output the value 1/2 [since if x=x0, we get 1/(1+exp(0)) = 1/2].
    # b = max(33, b)    
    L = min(100, L)
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def fit_sigmoid(xdata, ydata):
            # L              x0            k  b 
    p0 =    [max(ydata), np.median(xdata), 1, min(ydata)] #[max(ydata), np.median(xdata), 1, min(ydata)] # this is an mandatory initial guess)
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox', maxfev=2e6)
    y1 = sigmoid(xdata, *popt)
    sse1 = np.sum((y1 - ydata)**2)
    popt, pcov = curve_fit(sigmoid2, xdata, ydata, p0, method='dogbox', maxfev=2e6)
    y2 = sigmoid2(xdata, *popt)
    sse2 = np.sum((y2 - ydata)**2) # sum of squared errors
    popt, pcov = curve_fit(sigmoid3, xdata, ydata, p0, method='dogbox', maxfev=2e6)
    y3 = sigmoid3(xdata, *popt)
    sse3 = np.sum((y3 - ydata)**2)
    popt, pcov = curve_fit(sigmoid4, xdata, ydata, p0, method='dogbox', maxfev=2e6)
    y4 = sigmoid4(xdata, *popt)
    sse4 = np.sum((y4 - ydata)**2)
    y = [y1, y2, y3, y4][np.argmin([sse1, sse2, sse3, sse4])] # choose the best fit
    print(np.argmin([sse1, sse2, sse3, sse4]))
    print('summed squared error = ', np.sum((y - ydata)**2))
    return y

def find_closest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def symmetric_moving_average(data, window_size):
    """
    Calculate the symmetric moving average of a data array.
    
    Args:
        data (list or np.ndarray): Input data.
        window_size (int): Size of the moving window (should be odd for symmetry).
    
    Returns:
        np.ndarray: Smoothed data using symmetric moving average.
    """
    if window_size % 2 == 0:
        # raise ValueError("Window size must be odd for symmetry.")
        window_size += 1
    
    half_window = window_size // 2
    padded_data = np.pad(data, (half_window, half_window), mode='edge')
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.convolve(padded_data, kernel, mode='valid')
    
    return smoothed_data

def load_mat_virtual_all_thresholds(matfile, nerve_model_type=3, state=1, array_type =2):
    # for now only received nerve model 3(?) and array type MS
    c = nerve_model_type-1 # cochlear model, nerve_model_type, CM3 is most average according to Randy
    a = array_type-1 # 1 is HiFocus 1J in lateral position, 2 is HiFocus MS in mid-scalar position, Most of Jacob's files are with MS
    m = state-1 # 1: healthy fibres, 2: fibres with shortened periferal ending, 3: fibres without dendrites
    PW = float(matfile[matfile.find('Morphologies ')+len('Morphologies '):matfile.rfind('us ')])*1e-6 # [s]
    mat = read_mat(matfile)
    # [16x1] T-level of electrode e (monopolair stimulation) in [mA] ALREADY FROM APICAL TO BASAL!
    T_levels = mat['Df120']['T'][m]*1e-3 # [mA] --> [A]
    # [16x1] T-level of electrode e (monopolair stimulation)
    M_levels = mat['Df120']['M'][m]*1e-3 # [mA] --> [A]
    # [15x9x3200]=[ep,n,f] thresholds for fibre f stimulated with electrode pair ep and alpha(n)
    # same unit as output of AB's hilbert function (log2 units)
    TI_env_log2 = mat['Df120']['TI_env_log2'][m] # --> if I want to use output of the hilbert function
    TI_env_log2 = np.nan_to_num(TI_env_log2, nan=1000) # NaNs mean the threshold was higher than the current range
    # [15x9x3200]=[ep,n,f] thresholds for fibre f stimulated with electrode pair ep and alpha(n)
    # Current on apical electrode [mA]
    TIa = mat['Df120']['TIa'][m] *1e-3 # turn to [A]
    TIa = np.nan_to_num(TIa, nan=1000) # NaNs mean the threshold was higher than the current range
    TIb = mat['Df120']['TIb'][m] *1e-3 # turn to [A]
    TIb = np.nan_to_num(TIb, nan=1000)
    # Ln needs to be reversed 
    Ln = np.flipud(mat['Df120']['Ln'][m]) # 3200 fibres
    Le = mat['Df120']['Le'][m] # 16 electrodes
    Fn = np.flip(mat['Df120']['Fn'][m])*1e3 # [Hz] 3200 fibers
    Fe = np.flip(mat['Df120']['Fe'][m])*1e3 # [Hz] 3200 fibers
    # x=3

    return T_levels, M_levels, TI_env_log2, TIa, TIb, Ln, Le, PW, Fn, Fe

def load_cochlear_parms_and_pulse_train(data_dir, matfile_thresholds, f_elgram):
    pulse_train = read_mat(os.path.join(data_dir, f_elgram + '.mat'))['electrodogram']
    matrix_denom = f_elgram[f_elgram.find('TPD')+len('TPD'):f_elgram.find('TPD')+len('TPD')+3]
    signal_period = float(f_elgram[f_elgram.find('_SiPe')+len('_SiPe'):f_elgram.rfind('_elec')])*1e-6 # [s]
    c = int(matrix_denom[0])-1
    a = int(matrix_denom[1])-1
    m = int(matrix_denom[2])-1

    threshold_dir = os.path.join(os.path.dirname(__file__), "data")

    # load thresholds
    T_levels = np.squeeze(load_gekke_matlab.load_matlab_met_gekke_nested_shit(os.path.join(threshold_dir, matfile_thresholds + '.mat'), 'TPD', 'T')[c,a,m])*1e-3 # [A]
    M_levels = np.squeeze(load_gekke_matlab.load_matlab_met_gekke_nested_shit(os.path.join(threshold_dir, matfile_thresholds + '.mat'), 'TPD', 'M')[c,a,m])*1e-3 # [A]
    I_det_Cochlear = np.squeeze(load_gekke_matlab.load_matlab_met_gekke_nested_shit(os.path.join(threshold_dir, matfile_thresholds + '.mat'), 'TPD', 'TI')[c,a,m])*1e-3 # [A]
    Ln = np.squeeze(load_gekke_matlab.load_matlab_met_gekke_nested_shit(os.path.join(threshold_dir, matfile_thresholds + '.mat'), 'TPD', 'Ln')[c,a,m]) # 3200 fibres
    Fn = np.flip(np.squeeze(load_gekke_matlab.load_matlab_met_gekke_nested_shit(os.path.join(threshold_dir, matfile_thresholds + '.mat'), 'TPD', 'Fn')[c,a,m]))*1e3 # [Hz] 3200 fibres
    Le = np.squeeze(load_gekke_matlab.load_matlab_met_gekke_nested_shit(os.path.join(threshold_dir, matfile_thresholds + '.mat'), 'TPD', 'Le')[c,a,m]) # 22 electrodes
    PW_real = np.squeeze(load_gekke_matlab.load_matlab_met_gekke_nested_shit(os.path.join(threshold_dir,matfile_thresholds + '.mat'), 'TPD', 'Tphase')[c,a,m])*1e-6 # [s]
    IPG = np.squeeze(load_gekke_matlab.load_matlab_met_gekke_nested_shit(os.path.join(threshold_dir,matfile_thresholds + '.mat'), 'TPD', 'IPG')[c,a,m])*1e-6 # [s]
    # electrode 1 = basal
    if 'double' in matfile_thresholds:
        I_det = np.flipud(I_det_Cochlear)
    else:
        # I_det = np.flipud(I_det_Cochlear)
        I_det_LUMC = np.flipud(np.fliplr(I_det_Cochlear))
        I_det = I_det_LUMC
    # print('test')
    # I_det = I_det_Cochlear
    Ln = np.sort(Ln)[::-1] # should be in decending order
    Le = np.sort(Le)[::-1] # should be in decending order

    # for e in range(10):
        # plt.scatter(range(I_det.shape[0]), I_det[:,e*2], label='electrode ' + str(2*e+1))
    # plt.legend()
    # breakpoint()

    return pulse_train, signal_period, T_levels, M_levels, Ln, Le, I_det, PW_real, IPG, Fn

def transform_pulse_train_to_121_virtual(pulse_train, weights_matrix):
    (num_electrodes, num_samples) = weights_matrix.shape
    num_channels = num_electrodes -1
    pulse_times, pulse_electrodes = np.where(pulse_train.T < 0)
    pulse_train121 = np.zeros((121, num_samples))
    # # turn weights and I_given into 121 integers 
    weights_121_all_samples = np.zeros(num_samples)
    for el in np.arange(num_channels):
        el_pair = [el, el+1]
        pulse_times_electrode = pulse_times[pulse_electrodes == el]
        for pt in pulse_times_electrode:
            weights_pair = weights_matrix[el_pair, pt]
            # print(weights_pair)
            if (weights_pair == np.array([1.0, 0.0])).all():
                weights_121_all_samples[pt] = 1 + el*8
            elif (weights_pair == np.array([0.875, 0.125]) ).all():
                weights_121_all_samples[pt] = 2 + el*8
            elif(weights_pair == np.array([0.75, 0.25]) ).all():
                weights_121_all_samples[pt] = 3 + el*8
            elif (weights_pair == np.array([0.625, 0.375]) ).all():
                weights_121_all_samples[pt] = 4 + el*8
            elif (weights_pair == np.array([0.5, 0.5]) ).all():
                weights_121_all_samples[pt] = 5 + el*8
            elif (weights_pair == np.array([0.375, 0.625]) ).all():
                weights_121_all_samples[pt] = 6 + el*8
            elif (weights_pair == np.array([0.25, 0.75]) ).all():
                weights_121_all_samples[pt] = 7 + el*8
            elif (weights_pair == np.array([0.125, 0.875]) ).all():
                weights_121_all_samples[pt] = 8 + el*8
            elif (weights_pair == np.array([0.0, 1.0]) ).all():
                weights_121_all_samples[pt] = 9 + el*8
            else:
                continue
            pulse_pair = pulse_train[el_pair, pt]
            virtual_channel_id = int(weights_121_all_samples[pt] -1)
            pulse_train121[virtual_channel_id, pt] = np.sum(pulse_pair)# apical + basal
    kernel = np.array([1, -1]) # biphasic pulses, already negative first pulse
    pulse_train121 = lfilter(kernel, 1, pulse_train121)

    return pulse_train121

def plot_sound_spectrum(ax, frequency, outline):
    ax.plot(frequency, outline)
    ax.set_xscale('log', base=2)
    ax.set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
    return ax

def rebin_spikes(spike_list, old_binsize, new_binsize):
    from decimal import Decimal
    num_fibers = spike_list.shape[0]
    if Decimal(str(new_binsize)) % Decimal(str(old_binsize)) != 0:
        raise ValueError('New binsize must be a multiple of old binsize: ' + str(old_binsize) + ' s.')
    bin_ratio = int(Decimal(str(new_binsize)) / Decimal(str(old_binsize)))
    new_num_bins = int(spike_list.shape[1]/bin_ratio)
    new_spikes_list = np.zeros((num_fibers, new_num_bins))
    for fibre in range(num_fibers):
        for bin in range(new_num_bins):
            new_spikes_list[fibre, bin] = sum(spike_list[fibre, bin*bin_ratio:(bin+1)*bin_ratio])

    return new_spikes_list

def create_EH_freq_vector_electrode_allocation(type_scaling_fibres='log'): # 'log'/'lin'
    # type_scaling_fibres = 'lin' 

    # frequencies in FFT channels
    edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]

    # get fiber location and frequency from Randy's file
    matfile = './data/Fidelity120 HC3A MS All Morphologies 18us CF.mat'
    mat = read_mat(matfile)
    m=0
    # Greenwood frequency IDs
    Fn = np.flip(mat['Df120']['Fn'][m])*1e3 # [Hz] 3200 fibers
    Fe = np.flip(mat['Df120']['Fe'][m])*1e3 # [Hz] 16 electrodes

    # basilar membrane IDs
    Ln = np.flip(mat['Df120']['Ln'][m]) # [mm] 3200 fibers
    Le = np.flip(mat['Df120']['Le'][m]) # [mm] 16 electrodes

    # match fibers 
    fiber_id_electrode = np.zeros(16)
    for e, mm in enumerate(Le):
        fiber_id_electrode[e] = int(find_closest_index(Ln, mm) )

    num_fibers_between_electrode = abs(np.diff(fiber_id_electrode))
    half_electrode_range = int(np.mean(num_fibers_between_electrode)/2)

    # selected fiber IDs:
    fiber_id_list_FFT = np.arange(start=int(np.min(fiber_id_electrode)), stop=int(np.max(fiber_id_electrode)+ half_electrode_range)) # do I need to flip this?
    np.save('./data/fiber_ID_list_FFT.npy', fiber_id_list_FFT)

    # 272 is the frequency edge of an FFT bin one step before
    if type_scaling_fibres == 'log':
        freq_x_fft = list(np.logspace(np.log10(272), np.log10(edges[0]), half_electrode_range, base=10, endpoint=False)) 
    elif type_scaling_fibres == 'lin':
        freq_x_fft = list(np.linspace(272, edges[0], half_electrode_range, endpoint=False)) 

    for e in range(len(edges)-1):
        freq_range = (edges[e], edges[e+1])
        if type_scaling_fibres == 'log':
            freq_fft_band = list(np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), int(num_fibers_between_electrode[e]), base=10, endpoint=False))
        elif type_scaling_fibres == 'lin':
            freq_fft_band = list(np.linspace(freq_range[0], freq_range[1], int(num_fibers_between_electrode[e]), endpoint=False))
        freq_x_fft.extend(freq_fft_band)

    # np.save('./data/EH_freq_vector_electrode_allocation_'+ type_scaling_fibres + 'spaced.npy', freq_x_fft)
    return freq_x_fft, fiber_id_electrode, half_electrode_range

# # create_EH_freq_vector_electrode_allocation()
# edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]
# lin =  np.load('./data/EH_freq_vector_electrode_allocation_linspaced.npy')
# log =  np.load('./data/EH_freq_vector_electrode_allocation_logspaced.npy')
# fiber_id = np.load('./data/fiber_ID_list_FFT.npy')
# # get fiber location and frequency from Randy's file
# matfile = './data/Fidelity120 HC3A MS All Morphologies 18us CF.mat'
# mat = read_mat(matfile)
# m=0
# # basilar membrane IDs
# Ln = np.flip(mat['Df120']['Ln'][m]) # [mm] 3200 fibers
# Le = np.flip(mat['Df120']['Le'][m]) # [mm] 16 electrodes
# fiber_selection_mm = Ln[fiber_id]
# Le_location_flipped = Ln.max()-Le 
# import matplotlib.pyplot as plt
# plt.figure()
# # plt.scatter(range(len(lin)), lin, label='lin', s=1)
# from_zero_mm = Ln.max()-fiber_selection_mm - (Ln.max()-fiber_selection_mm.max())
# # plt.scatter(Ln.max()-fiber_selection_mm -(Ln.max()-fiber_selection_mm.max()), log, label='log', s=1)
# plt.scatter(fiber_selection_mm, log, label='log', s=1)
# # plt.vlines(Le_location_flipped, 300, 8000)
# plt.vlines(Le, 300, 8000)
# # plt.vlines(Ln.max()-Le-(Ln.max()-Le.max()), 300, 8000)
# plt.xlabel('Position fiber & electrode (mm)')
# plt.ylabel('Learnt frequency [Hz]')
# plt.title('Log-scale allocated frequency')
# # plt.legend()

# # plt.figure()
# # plt.scatter(Ln.max()-fiber_selection_mm -(Ln.max()-fiber_selection_mm.max()), log, label='log', s=1)
# # plt.hlines(edges, 0, 19)
# # plt.xlabel('Position fiber & electrode (mm)')
# # plt.ylabel('Learnt frequency [Hz]')
# # plt.title('Log-scale allocated frequency')

# plt.show()

def plot_spectrogram(fname):
        audio_signal, Fs =a2n.audio_from_file(fname)
        fig = plt.figure()
        f, t, Sxx = spectrogram(audio_signal, Fs)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.ylim((274,8000))
        return fig

def ax_spectrogram(fname, ax):
        audio_signal, Fs =a2n.audio_from_file(fname)
        print(len(audio_signal))
        f, t, Sxx = spectrogram(audio_signal, Fs)
        ax.pcolormesh(t, f*1e-3, Sxx, shading='gouraud')
        # ax.set_ylabel('Frequency [Hz]')
        # ax.set_xlabel('Time [sec]')
        ax.set_ylim((0.274, 8))
        return ax

def ax_neurogram(fname, bin_size, ax, clim=None, flim=None, norm=None, cmap_type = 'viridis'):
    spike_rate_matrix, fiber_frequencies = load_spike_matrix(fname, bin_size)
    sound_duration = bin_size*spike_rate_matrix.shape[1]
    x = np.arange(bin_size, sound_duration+bin_size, bin_size)
    mesh = ax.pcolormesh(x, fiber_frequencies*1e-3, spike_rate_matrix, cmap=cmap_type, norm=norm) #
    if clim:
        mesh.set_clim(clim)
    if flim:
        ax.set_ylim((0.274, flim))
    else:
        ax.set_ylim((0.274, 6.5))
    return ax

def Greenwood_function_mm_to_f(mm, max_Ln=35, A = 165.4, alpha = 2.1, k = 0.88):
    if hasattr(mm, "__len__"): # if vector
        f = []
        for m in mm:
            rel_mm = (max_Ln-m)/max_Ln
            f.append(A*(10**(alpha*rel_mm)-k))
    else: # if scalar
        rel_mm = (max_Ln-mm)/max_Ln
        f = A*(10**(alpha*rel_mm)-k)
    return f

def load_spike_matrix(fname, new_bin_size=0.005):
    if fname[-3:] == 'npy':
        spike_rates_list = np.load(fname)
        time = fname[:fname.index('spike_matrix')]
        config_file = glob.glob(time + 'config_output*.yaml')[0]
        with open(config_file, "r") as yamlfile: 
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        binsize_original = config['binsize']
        fiber_frequencies = np.load('./data/EH_freq_vector_electrode_allocation_logspaced.npy')
        fiber_id_list = np.load('./data/fiber_ID_list_FFT.npy')
        spikes_matrix = spike_rates_list[fiber_id_list,:]
    if fname[-3:] == 'mat':
        try:
            spikes_matrix, _, _, _, fiber_frequencies, _, t_unfiltered, _ = load_mat_structs_Hamacher(fname)
        except:
            spikes_matrix, _, _, _, fiber_frequencies, _, t_unfiltered, _ = load_matrices_from_vectors_Bruce_struct(fname)
        binsize_original = t_unfiltered[1]-t_unfiltered[0]
    if new_bin_size !=binsize_original:
        spike_rate_matrix = rebin_spikes(spikes_matrix, binsize_original, new_bin_size)/new_bin_size
    else: 
        spike_rate_matrix = spikes_matrix
    return spike_rate_matrix, fiber_frequencies

def spike_matrix_to_critical_band(fname, critical_band_type, new_bin_size=0.005):
    spike_rate_matrix, fiber_frequencies = load_spike_matrix(fname, new_bin_size)
    # def select_critical_bands(spike_rate_matrix, fiber_frequencies, type='single', num_critical_band = 42, number_of_fibers = 10):
    # taken from: https://www.sfu.ca/sonic-studio-webdav/handbook/Appendix_E.htmL changed edge 0 >20 according to https://en.wikipedia.org/wiki/Bark_scale
    if critical_band_type.lower() == 'bark': # Bark scale, N=24
        centre_frequency_critical_bands =np.array([50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 
                                                    1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500,10500, 13500])
        edge_frequency_critical_bands = np.array([20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 
                                        2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500])
    elif critical_band_type.lower() == 'hamacher': # Hamacher, N=42
        edges_mm = np.arange(0.25, 32.5, 0.75)[::-1]
        edge_frequency_critical_bands = Greenwood_function_mm_to_f(edges_mm) # or should the max be max(edges_mm)
        centre_frequency_critical_bands = edge_frequency_critical_bands[:-1] + np.diff(edge_frequency_critical_bands)/2
    elif critical_band_type.lower() == 'mel': # Mel scale https://isip.piconepress.com/courses/msstate/ece_8463/lectures/current/lecture_04/lecture_04_05.html
        centre_frequency_critical_bands = np.concatenate((np.arange(100, 1100, 100), np.array([1149, 1320, 1516, 1741, 2000, 2297, 2639, 3031, 3482, 4000, 4595, 5278, 6063, 6964])))
        BW = np.concatenate((100*np.ones(9), np.array([124, 160, 184, 211, 242, 278, 320, 367, 422, 484, 556, 639, 734, 843, 969])))
        edge_frequency_critical_bands = [] 
        for b in np.arange(len(BW)):
            edge_frequency_critical_bands.append(centre_frequency_critical_bands[b]-BW[b]/2)
        edge_frequency_critical_bands.append(centre_frequency_critical_bands[b]+BW[b]/2)
    elif critical_band_type.lower() == 'slim': # N=171
        edges_mm = np.arange(0.25, 32.5, 0.75/4)[::-1]
        edge_frequency_critical_bands = Greenwood_function_mm_to_f(edges_mm) # or should the max be max(edges_mm)
        centre_frequency_critical_bands = edge_frequency_critical_bands[:-1] + np.diff(edge_frequency_critical_bands)/2
    # elif critical_band_type.lower() == 'erb': # ERB scale
    #     pass
    # elif critical_band_type.lower() == 'semi' or critical_band_type.lower() == 'semitone': 
    #     pass
    else:
        raise ValueError('Don\'t have this option')
    num_critical_band = len(centre_frequency_critical_bands)
    start_critical_band = find_closest_index(centre_frequency_critical_bands, fiber_frequencies[0]) # Bruce's lowest possible frequency as input is 125    
    end_critical_band = find_closest_index(edge_frequency_critical_bands, fiber_frequencies[-1]) # 
    if num_critical_band == 42:
        end_critical_band -= 1 # centre frequency is otherwise exactly at edge Bruce matrix
        start_critical_band += 1
    list_frequency_idx = []
    new_fiber_frequencies_CF = []
    remaining_frequency_edges = edge_frequency_critical_bands[start_critical_band: end_critical_band+1]
    centre_remaining_frequency_bands = centre_frequency_critical_bands[start_critical_band: end_critical_band]
    selected_spikes = np.zeros((len(centre_remaining_frequency_bands), spike_rate_matrix.shape[1]))
    for i in range(start_critical_band, len(centre_frequency_critical_bands[start_critical_band:end_critical_band+1])):
        low_frequency = edge_frequency_critical_bands[i]
        high_frequency = edge_frequency_critical_bands[i+1]
        low_idx = (np.abs(fiber_frequencies - low_frequency)).argmin()
        idx = (np.abs(fiber_frequencies - centre_frequency_critical_bands[i])).argmin()
        high_idx = (np.abs(fiber_frequencies - high_frequency)).argmin()
        list_frequency_idx.append(idx)
        new_fiber_frequencies_CF.append(fiber_frequencies[idx])
        selected_spikes[i-start_critical_band,:] = np.sum(spike_rate_matrix[low_idx:high_idx,:], axis=0)
        # print(i, ':', high_idx-low_idx, 'fibers')
    # x=3
    return selected_spikes, new_fiber_frequencies_CF, centre_remaining_frequency_bands, remaining_frequency_edges

def closestDivisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    return a,n//a

def is_prime(n):
  for i in range(2,n):
    if (n%i) == 0:
      return False
  return True

def plot_fig_critical_bands(fname_list, critical_band_type, new_bin_size=0.005):
    for f, fname in enumerate(fname_list):
        selected_spikes, new_fiber_frequencies_CF, centre_remaining_frequency_bands, remaining_frequency_edges = spike_matrix_to_critical_band(fname, critical_band_type, new_bin_size)
        num_remaining_crit_bands, _ = selected_spikes.shape
        if 'SMRT' in fname and num_remaining_crit_bands == 115:
            selected_spikes = selected_spikes[:-20,:]
            num_remaining_crit_bands = 95
        if f == 0:
            number_bands_prime = num_remaining_crit_bands
            row_plot = 0
            while row_plot<3:
                number_bands_prime += 1
                if not is_prime(number_bands_prime):
                    row_plot, column_plot = closestDivisors(number_bands_prime)
            fig, ax = plt.subplots(row_plot, column_plot, sharey=True, sharex=True, figsize=(12, 9))
            # if row_plot == 4 and column_plot == 6:
            plt.subplots_adjust(left=0.067, right=0.979, hspace=0.338)
            # if row_plot == 4 and column_plot == 5:
                # plt.subplots_adjust(left=0.067, right=0.979, hspace=0.338)
            axes = ax.flatten()

        if '_u_' in fname:
            label = 'up'
            color = 'red'
        if '_d_' in fname:
            label = 'down'
            color = 'blue'
        if '_i1_' in fname:
            label = 'inverted'
            color = 'red'
        if '_s_' in fname:
            label = 'standard'
            color = 'blue'
        if 'width_' in fname:
            if 'width_20' in fname:
                label = '20 RPO'
                color = 'red'
            else:
                label = 'X RPO'
                color = 'blue'



        x = np.arange(new_bin_size, new_bin_size*selected_spikes.shape[1]+new_bin_size, new_bin_size)
        for n in range(num_remaining_crit_bands):
            axes[n].plot(x, selected_spikes[n,:], color=color, label=label)
            # if len(frequency_bands) == num_remaining_crit_bands:
            axes[n].set_title('Critical band ' + str(n+1) + '\n (CF=' + str(round(centre_remaining_frequency_bands[n])) + ' Hz)')
            # else:    
                # axes[n].set_title('Critical band ' + str(n+1) + '\n (' + str(round(centre_remaining_frequency_bands[n])) + '-' + str(round(frequency_bands[n+1])) + 'Hz)')
            axes[n].set_xlim((x[0], x[-1]))
            if n == num_remaining_crit_bands-1:
                axes[n].legend()

    # fig.text(0.08, 0.35, 'Internal Representations (IR)', ha='center', rotation='vertical', fontsize=font_size)
    fig.text(0.5, 0.04, 'Time [s]', ha='center', fontsize=20)
    return fig