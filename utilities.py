import numpy as np
from pymatreader import read_mat
from scipy.signal import butter, filtfilt, lfilter

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

# create_EH_freq_vector_electrode_allocation()

# lin =  np.load('./data/EH_freq_vector_electrode_allocation_linspaced.npy')
# log =  np.load('./data/EH_freq_vector_electrode_allocation_logspaced.npy')

# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(range(len(lin)), lin, label='lin', s=1)
# plt.scatter(range(len(lin)), log, label='log', s=1)
# plt.legend()

# plt.show()