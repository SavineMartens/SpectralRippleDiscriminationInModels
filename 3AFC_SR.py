import numpy as np
from utilities import *
from scipy.signal import find_peaks

freq_x_fft = np.load('./data/AB_MS_based_on_min_filtered_thresholdsfreq_x_fft.npy')
fiber_id_selection = np.load('./data/AB_MS_based_on_min_filtered_thresholdsfiber_ID_list_FFT.npy')
data_dir = './data/spectrum/65dB_2416CF/'

def get_normalized_spectrum(fname, 
                            filter_bool=True, 
                            filter_type= 'mavg', 
                            filter_order = 4, 
                            cut_off_freq = 100,
                            window_size = 33):
    if '.mat' in fname:
        try:
            spike_matrix, _, _, _, fiber_frequencies, _, _, _ = load_mat_structs_Hamacher(fname, unfiltered_type = 'OG')
        except:
            spike_matrix, _, _, _, fiber_frequencies, _, _, _ = load_matrices_from_vectors_Bruce_struct(fname)
        spectrum = (np.mean(spike_matrix, axis=1)-np.min(np.mean(spike_matrix, axis=1)))/(np.max(np.mean(spike_matrix, axis=1))-np.min(np.mean(spike_matrix, axis=1)))
    if '.npy' in fname:
        spike_matrix = np.load(fname, allow_pickle=True) 
        spike_matrix = spike_matrix[fiber_id_selection, :]  # Select fibers 
        spike_vector = np.mean(spike_matrix, axis=1)
        spectrum = (spike_vector-np.min(spike_vector))/(np.max(spike_vector)-np.min(spike_vector))  
        if filter_bool:   
            if filter_type == 'butter':  
                spectrum = butter_lowpass_filter(spectrum, cut_off_freq, len(spectrum), filter_order) # Can't LP f because the Fs is not consistent
            elif filter_type == 'mavg':
                spectrum = symmetric_moving_average(spectrum, window_size=window_size)
        return spectrum
    
def load_trials(fname_list, num_fibers):
    trial_matrix = np.zeros((len(fname_list), num_fibers))
    for i, fname in enumerate(fname_list):
        trial_matrix[i, :] = get_normalized_spectrum(fname)
    return trial_matrix

def load_trials_from_train(fname, 
                           filter_bool=False,
                           filter_type= 'mavg', 
                          filter_order = 4, 
                          cut_off_freq = 100,
                          window_size = 99): # 33
    """
    Load trials from a list of filenames and return a matrix of shape (n_trials, n_fibers).
    
    Parameters:
    - fname_list: list of filenames to load
    - num_fibers: number of fibers to select
    
    Returns:
    - trial_matrix: np.ndarray of shape (n_trials, n_fibers)
    """
    trains = np.load(fname, allow_pickle=True)
    num_fibers, num_trials = trains.shape
    trial_matrix = np.zeros((num_trials, len(fiber_id_selection)))
    for  t in range(num_trials):
        spectrum = np.zeros(len(fiber_id_selection))
        for f, fiber in enumerate(fiber_id_selection):
            trial_fiber_spike_times = trains[fiber, t]
            spectrum[f] = len(trial_fiber_spike_times)
        #normalize the spectrum
        spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))  # Normalize the spectrum
        if filter_bool:
            if filter_type == 'butter':  
                spectrum = butter_lowpass_filter(spectrum, cut_off_freq, len(spectrum), filter_order) # Can't LP f because the Fs is not consistent
            elif filter_type == 'mavg':
                spectrum = symmetric_moving_average(spectrum, window_size=window_size)
        trial_matrix[t, :] = spectrum
    return trial_matrix

def compute_d_prime_trials_freq(standard_matrix, inverted_matrix):
    """
    Compute d-prime across frequency channels.
    
    Parameters:
    - standard_matrix: np.ndarray of shape (n_trials, n_freqs)
    - neurograms_stim2: np.ndarray of shape (n_trials, n_freqs)
    
    Returns:
    - d_prime_vector: np.ndarray of shape (n_freqs,), one d′ per frequency channel
    """
    mu1 = np.mean(standard_matrix, axis=0)
    mu2 = np.mean(inverted_matrix, axis=0)
    
    var1 = np.var(standard_matrix, axis=0, ddof=1)
    var2 = np.var(inverted_matrix, axis=0, ddof=1)
    
    pooled_std = np.sqrt(0.5 * (var1 + var2))
    d_prime_vector = np.abs(mu1 - mu2) / (pooled_std + 1e-9)  # add small constant to avoid divide-by-zero
    
    return d_prime_vector

def compute_correlation_trials_freq(standard_matrix, inverted_matrix):
    """
    Compute correlation across frequency channels.
    
    Parameters:
    - standard_matrix: np.ndarray of shape (n_trials, n_freqs)
    - inverted_matrix: np.ndarray of shape (n_trials, n_freqs)
    
    Returns:
    - correlation_vector: np.ndarray of shape (n_freqs,), one correlation per frequency channel
    """
    correlation_vector = np.zeros(standard_matrix.shape[0])
    
    for i in range(standard_matrix.shape[0]):
        correlation_vector[i] = np.corrcoef(standard_matrix[i, :], inverted_matrix[i, :])[0, 1]
                                # scipy.stats.pearsonr(standard_matrix[:, i], inverted_matrix[:, i])[0]
    return correlation_vector


def compute_max_adjacent_peak_to_valley_db(vector):
    vector = np.asarray(vector)
    
    peaks, _ = find_peaks(vector)
    valleys, _ = find_peaks(-vector)
    
    if len(peaks) == 0 or len(valleys) == 0:
        return None  # No peaks or valleys
    
    max_ratio_db = -np.inf

    for peak_idx in peaks:
        peak_val = vector[peak_idx]

        # Find left and right valleys
        left_valleys = valleys[valleys < peak_idx]
        right_valleys = valleys[valleys > peak_idx]

        ratios_db = []

        if len(left_valleys) > 0:
            left_valley_val = vector[left_valleys[-1]]  # closest on the left
            if left_valley_val > 0:
                ratio = peak_val / left_valley_val
                ratios_db.append(20 * np.log10(ratio))
        
        if len(right_valleys) > 0:
            right_valley_val = vector[right_valleys[0]]  # closest on the right
            if right_valley_val > 0:
                ratio = peak_val / right_valley_val
                ratios_db.append(20 * np.log10(ratio))

        if ratios_db:
            local_max = max(ratios_db)
            if local_max > max_ratio_db:
                max_ratio_db = local_max

    return max_ratio_db if max_ratio_db > -np.inf else None


def get_3AFC_RPO_separate_phase(RPO, 
                                hearing_type, 
                                metric='d', 
                                phase_trials=5, 
                                plotting=False):
    num_trials = 5
    if type(RPO) == float:
        RPO = str(RPO)

    if metric == 'd':
        metric_matrix_i1 = np.zeros((phase_trials, len(fiber_id_selection)))
        metric_matrix_i2 = np.zeros((phase_trials, len(fiber_id_selection)))
        metric_matrix_i = np.zeros((phase_trials, len(fiber_id_selection)))
    elif metric == 'c' or metric == 'correlation':
        metric_matrix_i1 = np.zeros((phase_trials, num_trials))
        metric_matrix_i2 = np.zeros((phase_trials, num_trials))
        metric_matrix_i = np.zeros((phase_trials, num_trials))     

    for phase_trial in range(1,phase_trials+1):
        if hearing_type == 'NH':    
            # get NH
            fname_i1 = glob.glob(data_dir + '*i1*' + RPO + '*_'+str(phase_trial) + '*.mat')
            fname_i2 = glob.glob(data_dir + '*i2*' + RPO + '*_'+str(phase_trial) + '*.mat')
            fname_s = glob.glob(data_dir + '*_s_*' + RPO + '*_'+str(phase_trial) + '*.mat')
        elif hearing_type == 'EH':                
            # # get EH
            fname_i1 = glob.glob(data_dir + '2025*trains_[!alpha05]*i1*'  + RPO + '*_'+str(phase_trial)+'.npy')[0]
            fname_i2 = glob.glob(data_dir + '2025*trains_[!alpha05]*i2*'  + RPO + '*_'+str(phase_trial)+'.npy')[0]
            fname_s = glob.glob(data_dir + '2025*trains_[!alpha05]*_s_*' + RPO + '*_'+str(phase_trial)+'.npy')[0]


        trial_matrix_i1 = load_trials_from_train(fname_i1, len(fiber_id_selection))
        trial_matrix_i2 = load_trials_from_train(fname_i2, len(fiber_id_selection))
        trial_matrix_s = load_trials_from_train(fname_s, len(fiber_id_selection))
    
        compute_max_adjacent_peak_to_valley_db(trial_matrix_i1[0,:])

        if plotting:
            # plotting spectra
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3,1, figsize=(12, 8), sharex=True)
            plt.subplot(3, 1, 1)
            plt.plot(freq_x_fft, trial_matrix_i1.T, label='i1', color='blue', alpha=0.5)
            plt.plot(freq_x_fft, trial_matrix_s.T, label='s2', color='red', alpha=0.5)
            plt.title('i1 and s1')
            plt.ylabel('Normalized Spectrum')
            plt.xscale('log', base=2)
            plt.xlim((min(freq_x_fft), max(freq_x_fft)))
            plt.xticks([min(freq_x_fft), 500, 1000, 2000, 4000, 8000], labels=[str(int(min(freq_x_fft))), '500', '1000', '2000', '4000', '8000'])
            plt.xlabel('Frequency (Hz)')
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(freq_x_fft, trial_matrix_i2.T, label='i2', color='blue', alpha=0.5)
            plt.plot(freq_x_fft, trial_matrix_s.T, label='s2', color='red', alpha=0.5)
            plt.title('i2 and s2')
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(freq_x_fft, trial_matrix_i1.T, label='i1', color='blue', alpha=0.5)
            plt.plot(freq_x_fft, trial_matrix_i2.T, label='i2', color='orange', alpha=0.5)
            plt.title('i1 and i2')
            plt.legend()
            plt.suptitle(f'RPO {RPO} - {hearing_type} at phase {phase_trial}')

        if metric == 'd':
            metric_str = "d′" 
            # Compute d-prime for each frequency channel
            metric_i1 = compute_d_prime_trials_freq(trial_matrix_i1, trial_matrix_s)
            metric_i2 = compute_d_prime_trials_freq(trial_matrix_i2, trial_matrix_s)
            metric_i = compute_d_prime_trials_freq(trial_matrix_i1, trial_matrix_i2)

        elif metric == 'c' or metric=='correlation':
            # Compute correlation
            metric_str = 'correlation'
            metric_i1 = compute_correlation_trials_freq(trial_matrix_i1, trial_matrix_s)
            metric_i2 = compute_correlation_trials_freq(trial_matrix_i2, trial_matrix_s)
            metric_i = compute_correlation_trials_freq(trial_matrix_i1, trial_matrix_i2)



        # print('mean d′ i1:', np.mean(metric_i1))
        # print('mean d′ i2:', np.mean(metric_i2))   
        # print('mean d′ i:', np.mean(metric_i))
        metric_matrix_i1[phase_trial-1, :] = metric_i1
        metric_matrix_i2[phase_trial-1, :] = metric_i2
        metric_matrix_i[phase_trial-1, :] = metric_i

        if plotting:
            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(freq_x_fft, metric_i1, label=metric_str + ' i1', color='blue')
            plt.plot(freq_x_fft, metric_i2, label=metric_str + ' i2', color='orange')    
            plt.plot(freq_x_fft, metric_i, label=metric_str + ' i', color='red')
            plt.ylabel(metric_str)
            plt.xscale('log', base=2)
            plt.xlim((min(freq_x_fft), max(freq_x_fft)))
            plt.xticks([min(freq_x_fft), 500, 1000, 2000, 4000, 8000], labels=[str(int(min(freq_x_fft))), '500', '1000', '2000', '4000', '8000'])
            plt.legend()

    if plotting:    
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(freq_x_fft, np.mean(metric_matrix_i1,axis=0), label='d′ i1', color='blue')
        plt.plot(freq_x_fft, np.mean(metric_matrix_i2,axis=0), label='d′ i2', color='orange')    
        plt.plot(freq_x_fft, np.mean(metric_matrix_i, axis=0), label='d′ i', color='red')

        plt.xscale('log', base=2)
        plt.xlim((min(freq_x_fft), max(freq_x_fft)))
        plt.xticks([min(freq_x_fft), 500, 1000, 2000, 4000, 8000], labels=[str(int(min(freq_x_fft))), '500', '1000', '2000', '4000', '8000'])
        plt.legend()
        plt.ylabel(metric_str)
    return np.mean(metric_matrix_i1), np.mean(metric_matrix_i2), np.mean(metric_matrix_i)

plt.figure()

metric= 'c'  # Change this to 'c' for correlation

if metric == 'd':
    metric_str = "d′" 
elif metric == 'c' or metric == 'correlation':
    metric_str = 'correlation'

for RPO in [0.125, 0.176, 0.250, 0.354, 0.500, 0.707, 1.000, 1.414, 2.000, 2.828, 4.000]:
    d_i1, d_i2, d_i = get_3AFC_RPO_separate_phase(RPO, 'EH', metric=metric)
    plt.plot(RPO, d_i1, 'o', color='orange', label=metric_str + ' i1')
    plt.plot(RPO, d_i2, 'o',color='red', label=metric_str + ' i2')
    plt.plot(RPO, d_i, 'o',color='blue', label=metric_str + ' i')


plt.ylabel(metric_str)
plt.xlabel('RPO')
plt.show()