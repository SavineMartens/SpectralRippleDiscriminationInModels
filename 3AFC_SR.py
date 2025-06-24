import numpy as np
from utilities import *
from scipy.signal import find_peaks
from scipy.stats import norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import platform
from scipy.stats import norm
from scipy.integrate import quad

# TO DO:
# [X] NH with trials
# [X] EH w/o current steering
# [X] add noise?
# [X] EH with > 4.0 RPO
# [X] add sigmoid function to d′ to percent correct 
# [ ] how to fit sigmoid well when log scale

freq_x_fft = np.load('./data/AB_MS_based_on_min_filtered_thresholdsfreq_x_fft.npy')
fiber_id_selection = np.load('./data/AB_MS_based_on_min_filtered_thresholdsfiber_ID_list_FFT.npy')
# data_dir = './data/spectrum/65dB_2416CF/all_phases/'
spectrum_dir = '.\\data\\spectrum\\65dB_2416CF\\all_phases\\NH\\'

def dprime_to_pc_3afc_sim(dprime, n_trials):
    """
    Simulates a 3AFC task to estimate percent correct for a given d'.
    Assumes equal-variance Gaussian noise.

    Parameters:
        dprime (float): The d' value
        n_trials (int): Number of trials to simulate

    Returns:
        float: Percent correct (0–100)
    """
    # Signal is added to one of the three intervals
    signal = dprime
    correct = 0

    for _ in range(n_trials):
        responses = np.random.normal(0, 1, 3)  # noise-only trials
        target_interval = np.random.randint(0, 3)
        responses[target_interval] += signal
        if np.argmax(responses) == target_interval:
            correct += 1

    return 100 * correct / n_trials


def dprime_lookup_table(dprime):
    # Define table:  These are approximations, either simulated or derived from literature such as Macmillan & Creelman (2005).
    dprime_vals = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4,
                            1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0,
                            3.5, 4.0])
    pc_vals = np.array([33.3, 35, 38, 43, 48, 55, 62, 68,
                        74, 79, 83, 87, 90, 92.5, 94.5, 96,
                        98, 99])
    # Interpolation function
    dprime_to_pc_3afc_lut = interp1d(dprime_vals, pc_vals, kind='linear', bounds_error=False, fill_value=(33.3, 99))
    PC = dprime_to_pc_3afc_lut(dprime)    
    return PC

def interpolate_threshold(RPO_list, y, threshold=66.667):
    """
    Interpolates the RPO value corresponding to a given threshold using linear interpolation.
    
    Parameters:
    - RPO_list: List of RPO values
    - y: Corresponding values (e.g., percent correct)
    - threshold: The threshold value to interpolate for
    
    Returns:
    - interpolated_RPO: The RPO value corresponding to the threshold
    """
    if len(RPO_list) != len(y):
        raise ValueError("RPO_list and y must have the same length.")
    
    # Create an interpolation function
    interp_func = interp1d(RPO_list, y, bounds_error=False, fill_value=(0, 100))
    
    # Find the RPO value corresponding to the threshold
    interpolator = interp1d(y, RPO_list, kind='linear', bounds_error=False, fill_value=(33.3, 99))
    interpolated_RPO = interpolator(threshold)

    return interpolated_RPO

def pc_m_afc(d_prime, m=3):
    return 100 * quad(lambda x: norm.cdf(x)**(m - 1) * norm.pdf(x - d_prime), -np.inf, np.inf)[0]


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

def apply_filter(spectrum, filter_type='mavg', filter_order=4, cut_off_freq=100, window_size=33):
    """
    Apply a filter to the spectrum if filter_bool is True.
    
    Parameters:
    - spectrum: np.ndarray, the spectrum to filter
    - filter_bool: bool, whether to apply the filter
    - filter_type: str, type of filter ('butter' or 'mavg')
    - filter_order: int, order of the Butterworth filter
    - cut_off_freq: float, cutoff frequency for the Butterworth filter
    - window_size: int, size of the moving average window
    
    Returns:
    - filtered_spectrum: np.ndarray, the filtered spectrum
    """
    if filter_type == 'butter':
        return butter_lowpass_filter(spectrum, cut_off_freq, len(spectrum), filter_order)
    elif filter_type == 'mavg':
        return symmetric_moving_average(spectrum, window_size=window_size)
    return spectrum


def load_trials_from_train(fname, 
                           filter_bool=True,
                           filter_type= 'mavg', 
                          filter_order = 4, 
                          cut_off_freq = 100,
                          window_size = 33, # 33
                          add_noise=None): 
    """
    Load trials from a list of filenames and return a matrix of shape (n_trials, n_fibers).
    
    Parameters:
    - fname_list: list of filenames to load
    - num_fibers: number of fibers to select
    
    Returns:
    - trial_matrix: np.ndarray of shape (n_trials, n_fibers)
    """
    if '.mat' in fname:
        numpy_name = os.path.basename(fname.replace('.mat', '.npy'))
        numpy_name =  numpy_name.replace('PSTH','spectrum') # Change PSTH to spectrum
        if os.path.exists(spectrum_dir + numpy_name):
            print(f"File {numpy_name} already exists. Skipping loading and processing.")
            trial_matrix =  np.load(spectrum_dir + numpy_name, allow_pickle=True)
            num_trials, num_fibers = trial_matrix.shape
            for t in range(num_trials):
                spectrum = trial_matrix[t, :]
                if add_noise:
                    noise = np.random.normal(0, add_noise, spectrum.shape)
                    spectrum += noise  # Add noise to the spectrum
                # Apply filter if specified
                if filter_bool:
                    spectrum = apply_filter(spectrum, filter_type, filter_order, cut_off_freq, window_size)
                trial_matrix[t, :] = spectrum
            return trial_matrix
        if 'filter' in fname:
            neurogram, _ , _, _, _, _, _, _ = load_matrices_from_vectors_Bruce_struct(fname)        
            num_trials = 1
        if 'multi' in fname:
            _, neurogram, _, _, _, _ = load_matrices_from_vectors_Bruce_multi_trial(fname)
            [num_trials, _, _] = neurogram.shape
        trial_matrix = np.zeros((num_trials, len(fiber_id_selection)))
        for t in range(num_trials):
            if len(neurogram.shape) == 3:
                spectrum = np.sum(neurogram[t, :, :], axis=-1)  # Sum across time steps
            elif len(neurogram.shape) == 2:
                spectrum = np.sum(neurogram, axis=-1)  # Sum across time steps
            # normalize the spectrum
            spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
            if add_noise:
                noise = np.random.normal(0, add_noise, spectrum.shape)
                spectrum += noise  # Add noise to the spectrum
            # Apply filter if specified
            if filter_bool:
                spectrum = apply_filter(spectrum, filter_type, filter_order, cut_off_freq, window_size)
            trial_matrix[t, :] = spectrum
        if not filter_bool and not add_noise:
            np.save(spectrum_dir + numpy_name, trial_matrix, allow_pickle=True)
            if fname.startswith('S:\\'):
                fname = fname.replace(data_dir, 'C:\\python\\SpectralRippleDiscriminationInModels\\data\\spectrum\\65dB_2416CF\\')
                try:
                    os.remove(fname)  # Remove the .mat file if on Windows
                    print(f"Removed {fname} after saving as .npy")
                except:
                    print(f"Could not remove {fname}. Has already been deleted.")
    elif '.npy' in fname:
        trains = np.load(fname, allow_pickle=True)
        num_fibers, num_trials = trains.shape
        if num_fibers != 3200:
            return None
        trial_matrix = np.zeros((num_trials, len(fiber_id_selection)))
        for  t in range(num_trials):
            spectrum = np.zeros(len(fiber_id_selection))
            for f, fiber in enumerate(fiber_id_selection):
                trial_fiber_spike_times = trains[fiber, t]
                spectrum[f] = len(trial_fiber_spike_times)
            #normalize the spectrum
            spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))  # Normalize the spectrum
            # add noise
            if add_noise:
                noise = np.random.normal(0, add_noise, spectrum.shape)
                spectrum += noise  # Add noise to the spectrum
            # Apply filter if specified
            if filter_bool:
                spectrum = apply_filter(spectrum, filter_type, filter_order, cut_off_freq, window_size)
            trial_matrix[t, :] = spectrum
    return trial_matrix

def compute_d_prime_2matrices_trials_freq(standard_matrix, inverted_matrix):
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


def compute_d_prime_3matrices(standard_matrix, inverted1_matrix, inverted2_matrix, per_frequency=False):
    """
    Compute d-prime values for all 3 pairwise comparisons between three neurograms.

    Parameters:
        standard_matrix, inverted1_matrix, inverted2_matrix: np.ndarray of shape (trials, frequencies)
        per_frequency: bool, whether to compute d' per frequency or collapse into scalar

    Returns:
        dict with d-prime values for each pairwise comparison
    """
    def dprime(a, b):
        mean_diff = np.mean(a, axis=0) - np.mean(b, axis=0)
        var_a = np.var(a, axis=0, ddof=1)
        var_b = np.var(b, axis=0, ddof=1)
        pooled_std = np.sqrt(0.5 * (var_a + var_b))
        d = mean_diff / pooled_std
        d[np.isnan(d)] = 0  # handle divide-by-zero
        if per_frequency:
            return d
        else:
            return np.linalg.norm(d)

    d_s1 = dprime(standard_matrix, inverted1_matrix)
    d_s2 = dprime(standard_matrix, inverted2_matrix)
    d_i = dprime(inverted1_matrix, inverted2_matrix)

    return {
        "d_prime_s_vs_i1": abs(d_s1),
        "d_prime_s_vs_i2": abs(d_s2),
        "d_prime_i1_vs_i2": abs(d_i)
    }

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


def compute_max_adjacent_peak_to_valley_db(trial_matrix, num_trials=1, RPO=0.00):
    pvr = np.zeros(trial_matrix.shape[0])
    for i in range(num_trials):
        vector = trial_matrix[i, :]
        vector = np.asarray(vector)
        
        peak_width = 100
        peaks, _ = find_peaks(vector, width=peak_width)
        valleys, _ = find_peaks(-vector, width=peak_width)
        
        plotting = True
        if plotting:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(freq_x_fft, vector, label='Spectrum')
            plt.scatter(freq_x_fft[peaks], vector[peaks], color='red', label='Peaks')
            plt.scatter(freq_x_fft[valleys], vector[valleys], color='green', label='Valleys')
            plt.title('Peaks and Valleys in Spectrum')
            plt.xlabel('Frequency Channel')
            plt.ylabel('Normalized Amplitude')
            plt.legend()
            plt.title(f'RPO: {RPO}, Trial: {i+1}')
            plt.xscale('log', base=2)
            # plt.show()

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
            # if max_ratio_db > -np.inf:
            #     max_ratio_db = None   # Return the maximum ratio in dB, or None if no valid ratios were found
            
        pvr[i] = max_ratio_db

    return pvr


def get_3AFC_RPO_separate_phase(RPO, 
                                hearing_type, 
                                filter_bool=True,
                                filter_type='mavg',
                                metric='d', 
                                phase_trials=2, 
                                plotting=False,
                                window_size=33,
                                add_noise=None,
                                data_dir='./data/spectrum/65dB_2416CF/all_phases/'):
    # print('collecting data from ' + data_dir)
    
    num_trials = 5

    if type(RPO) == float:
        RPO = str(RPO)

    if metric == 'd':
        metric_str = "d′" 
        metric_matrix_i1 = np.zeros((phase_trials, len(fiber_id_selection)))
        metric_matrix_i2 = np.zeros((phase_trials, len(fiber_id_selection)))
        metric_matrix_i = np.zeros((phase_trials, len(fiber_id_selection)))
    elif metric == 'c' or metric == 'correlation' or metric == 'pvr':
        metric_matrix_i1 = np.zeros((phase_trials, num_trials))
        metric_matrix_i2 = np.zeros((phase_trials, num_trials))
        metric_matrix_i = np.zeros((phase_trials, num_trials))     

    successful_trials = 0
    phase_trial = 1 
    while successful_trials < phase_trials and phase_trial < 30:
    # for phase_trial in range(1,phase_trials+1):
        print(f'Processing RPO {RPO} - {hearing_type} at phase {phase_trial}')
        try:
            if hearing_type == 'NH':    
                # get NH
                fname_i1 = glob.glob(data_dir + '*multi*i1*' + RPO + '*_'+str(phase_trial) + '_2416CFs.mat')[0]
                print(fname_i1)
                fname_i2 = glob.glob(data_dir + '*multi*i2*' + RPO + '*_'+str(phase_trial) + '_2416CFs.mat')[0]
                print(fname_i2)
                fname_s = glob.glob(data_dir + '*multi*_s_*' + RPO + '*_'+str(phase_trial) + '_2416CFs.mat')[0]
                print(fname_s)
                
                if 'filter' in fname_i1:
                    if metric == 'd':
                        print('cannot use d′ for old Bruce struct')
                        metric = 'c'
                        num_trials = 1
            elif hearing_type == 'EH':                
                # # get EH
                fname_i1 = glob.glob(data_dir + '2025*trains_[!alpha05]*i1*'  + RPO + '*_'+str(phase_trial)+'.npy')[0]
                fname_i2 = glob.glob(data_dir + '2025*trains_[!alpha05]*i2*'  + RPO + '*_'+str(phase_trial)+'.npy')[0]
                fname_s = glob.glob(data_dir + '2025*trains_[!alpha05]*_s_*' + RPO + '*_'+str(phase_trial)+'.npy')[0]

        except:
            print(f"Error loading files for RPO {RPO} - {hearing_type} at phase {phase_trial}. Skipping...")
            phase_trial += 1
            continue
        # load trials
        trial_matrix_i1 = load_trials_from_train(fname_i1, filter_bool=filter_bool, filter_type=filter_type, window_size=window_size, add_noise=add_noise)
        trial_matrix_i2 = load_trials_from_train(fname_i2, filter_bool=filter_bool, filter_type=filter_type, window_size=window_size, add_noise=add_noise)
        trial_matrix_s = load_trials_from_train(fname_s, filter_bool=filter_bool, filter_type=filter_type, window_size=window_size, add_noise=add_noise)
    

        if trial_matrix_i1 is None or trial_matrix_i2 is None or trial_matrix_s is None:
            print(f"Error loading trials for RPO {RPO} - {hearing_type} at phase {phase_trial}. Skipping...")
            phase_trial += 1
            continue

        phase_trial += 1
        successful_trials += 1

        if plotting:
            # plotting spectra
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
            # Compute d-prime for each frequency channel
            # metric_i1 = compute_d_prime_trials_freq(trial_matrix_i1, trial_matrix_s)
            # metric_i2 = compute_d_prime_trials_freq(trial_matrix_i2, trial_matrix_s)
            # metric_i = compute_d_prime_trials_freq(trial_matrix_i1, trial_matrix_i2)

            collected_metrics = compute_d_prime_3matrices(trial_matrix_s, trial_matrix_i1, trial_matrix_i2, per_frequency=True)
            metric_i1 = collected_metrics["d_prime_s_vs_i1"]
            metric_i2 = collected_metrics["d_prime_s_vs_i2"]
            metric_i = collected_metrics["d_prime_i1_vs_i2"]

        elif metric == 'c' or metric=='correlation':
            # Compute correlation
            metric_str = 'correlation'
            metric_i1 = compute_correlation_trials_freq(trial_matrix_i1, trial_matrix_s)
            metric_i2 = compute_correlation_trials_freq(trial_matrix_i2, trial_matrix_s)
            metric_i = compute_correlation_trials_freq(trial_matrix_i1, trial_matrix_i2)

        elif metric == 'pvr':
            metric_str = 'peak to valley [dB]'
            metric_i1 = compute_max_adjacent_peak_to_valley_db(trial_matrix_i1, RPO=RPO)
            metric_i2 = compute_max_adjacent_peak_to_valley_db(trial_matrix_i2, RPO=RPO)
            metric_i = compute_max_adjacent_peak_to_valley_db(trial_matrix_s, RPO=RPO)  

        metric_matrix_i1[successful_trials-1, :] = metric_i1
        metric_matrix_i2[successful_trials-1, :] = metric_i2
        metric_matrix_i[successful_trials-1, :] = metric_i

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


def run_dprime_multi_noise(RPO, 
                            hearing_type, 
                            noise_list, 
                            filter_bool=True,
                            filter_type='mavg', 
                            phase_trials=2, 
                            window_size=33,
                            data_dir='./data/spectrum/65dB_2416CF/all_phases/'):
    print('collecting data from ' + data_dir)

    dprime_matrix = np.zeros((len(noise_list), len(RPO_list), 3))  # Columns for i1, i2, and i
    PC_matrix = np.zeros((len(noise_list), len(RPO_list), 3))  # Columns for i1, i2, and i
    
    for n, add_noise in enumerate(noise_list):
        for r, RPO in enumerate(RPO_list):
            print(f'Running for RPO: {RPO}, Noise: {add_noise}')
            d_i1, d_i2, d_i = get_3AFC_RPO_separate_phase(RPO, 
                                                        hearing_type, 
                                                        metric='d', 
                                                        filter_bool=filter_bool, 
                                                        phase_trials=phase_trials, 
                                                        plotting=False,
                                                        window_size=window_size,
                                                        add_noise=add_noise,
                                                        data_dir=data_dir)
            dprime_matrix[n, r, 0] = d_i1
            dprime_matrix[n, r, 1] = d_i2
            dprime_matrix[n, r, 2] = d_i
            # Convert d' to percent correct
            PC_matrix[n, r, 0] = pc_m_afc(d_i1) # dprime_lookup_table(d_i1)
            PC_matrix[n, r, 1] = pc_m_afc(d_i2) # dprime_lookup_table(d_i2)
            PC_matrix[n, r, 2] = pc_m_afc(d_i) # dprime_lookup_table(d_i)

    return dprime_matrix, PC_matrix


if __name__ == "__main__":
    run_single_noise = False
    run_multiple_noise = True
    load_created_output = False
    hearing_type = 'EH'
    true_axis = False  # Set to True if you want the x-axis to be the RPO, False if you want the x-axis to be the index of the RPO

    filter_bool = True  # Set to False if you don't want to filter the spectra
    filter_type = 'mavg'  # 'butter' or 'mavg'
    phase_trials = 30  # Number of phase trials to average over
    RPO_list = [0.125, 0.176, 0.250, 0.354, 0.500, 0.707, 1.000, 1.414, 2.000, 2.828, 4.000, 5.657, 8.000, 11.314] #
    window_size = 33
    add_noise = None # None if no noise is to be added, or a float value for the standard deviation of the noise
    save_bool = True  # Set to True if you want to save the results
    CS_off = True
    metric= 'd'  # Change this to 'c' for correlation  
    

    if platform.system() == 'Windows':
        if hearing_type == 'EH':
            data_dir = './data/spectrum/65dB_2416CF/all_phases/EH/'
            if CS_off:
                data_dir = 'S:\\python\\temporal-phast-plus\\output\\'# './data/spectrum/65dB_2416CF/all_phases/noCS/'
        elif hearing_type == 'NH':
            data_dir = 'S:\\Matlab\\BEZ2018model\\Output\\' #'./data/spectrum/65dB_2416CF/'
    elif platform.system() == 'Linux':
        if hearing_type == 'EH':
                if CS_off:
                    data_dir = '/exports/kno-shark/users/Savine/python/temporal-phast-plus/output/'
                else:
                    raise ValueError('CS not on cluster')
        elif hearing_type == 'NH':
            data_dir = '/exports/kno-shark/users/Savine/Matlab/BEZ2018model/Output/'


    char_str = ''
    if filter_bool:
        char_str = 'filter (window=' + str(window_size) + ')'
    else:
        char_str = 'no filter'

    if metric == 'd':
        metric_str = "d′" 
    elif metric == 'c' or metric == 'correlation':
        metric_str = 'correlation'
    elif metric == 'pvr':
        metric_str = 'peak to valley [dB]'

    if run_single_noise:
        metric_i1_list = []
        metric_i2_list = []     
        metric_i_list = []

        if add_noise:
            char_str += ' + noise ('r'$\sigma$=' + str(add_noise) + ')'
        else:
            char_str += ' no noise'

        for RPO in RPO_list:
            d_i1, d_i2, d_i = get_3AFC_RPO_separate_phase(RPO, 
                                                            hearing_type, 
                                                            metric=metric, 
                                                            filter_bool=filter_bool, 
                                                            filter_type=filter_type,
                                                            phase_trials=phase_trials, 
                                                            plotting=False,
                                                            window_size=window_size,
                                                            add_noise=add_noise,
                                                            data_dir=data_dir)
            metric_i1_list.append(d_i1)
            metric_i2_list.append(d_i2)
            metric_i_list.append(d_i)
        fig1 = plt.figure(figsize=(10,5))

        plt.ylabel(metric_str)
        plt.xlabel('RPO')
        if true_axis:
            plt.plot(RPO_list, metric_i1_list, 'o', color='orange', label=metric_str + ': i1 vs s')
            plt.plot(RPO_list, metric_i2_list, 'o',color='red', label=metric_str + ': i2 vs s')
            plt.plot(RPO_list, metric_i_list, 'o',color='blue', label=metric_str + ': i1 vs i2')
            plt.xscale('log', base=1.414)  # Logarithmic scale for RPO
            plt.xticks(RPO_list, labels=[str(int(rpo)) for rpo in RPO_list], rotation=45)
        else:
            plt.plot(metric_i1_list, 'o', color='orange', label=metric_str + ': i1 vs s')
            plt.plot(metric_i2_list, 'o',color='red', label=metric_str + ': i2 vs s')
            plt.plot(metric_i_list, 'o',color='blue', label=metric_str + ': i1 vs i2')
            plt.xticks(range(len(RPO_list[:-1])), labels=[str((rpo)) for rpo in RPO_list[:-1]], rotation=45)
        plt.legend()
        plt.title('Spectral ripple test with ' + hearing_type + ' ('+ str(phase_trials) +' trials) : ' + metric_str)
        if save_bool:
            fig1.savefig('./figures/3AFC_SR_' + hearing_type + '_' + metric + '_RPO_' + str(RPO_list) + '_' + char_str + '.png')


        if metric == 'd':
            pc_i_list = []
            pc_i1_list = []
            pc_i2_list = []

            for m in range(len(metric_i_list)):
                # pc_i_list.append(dprime_lookup_table(metric_i_list[m]))
                # pc_i1_list.append(dprime_lookup_table(metric_i1_list[m]))
                # pc_i2_list.append(dprime_lookup_table(metric_i2_list[m]))
                pc_i_list.append(pc_m_afc(metric_i_list[m]))
                pc_i1_list.append(pc_m_afc(metric_i1_list[m]))
                pc_i2_list.append(pc_m_afc(metric_i2_list[m]))
            fig2 = plt.figure(figsize=(10,5))
            if true_axis:
                plt.plot(RPO_list, pc_i1_list, 'o', color='orange', label=metric_str + ': i1 vs s')
                plt.plot(RPO_list, pc_i2_list, 'o',color='red', label=metric_str + ': i2 vs s')
                plt.plot(RPO_list, pc_i_list, 'o',color='blue', label=metric_str + ': i1 vs i2')
            else:
                plt.plot(pc_i1_list, 'o', color='orange', label=metric_str + ': i1 vs s')
                plt.plot(pc_i2_list, 'o',color='red', label=metric_str + ': i2 vs s')
                plt.plot(pc_i_list, 'o',color='blue', label=metric_str + ': i1 vs i2')
            plt.ylabel('Percent Correct')
            plt.title('Spectral ripple test with ' + hearing_type + ' ('+ str(phase_trials) +' trials) ' + char_str )
            plt.xlabel('RPO')
            plt.legend()
            plt.ylim((30, 100))
            plt.xlim((min(RPO_list), max(RPO_list)))
            avg = (np.asarray(pc_i1_list) + np.asarray(pc_i2_list))/2
            avg = np.append(avg, 33.33333)  # Add chance level for 3AFC
            RPO_list.append(20)
            if true_axis:
                y = fit_sigmoid(RPO_list, avg)
                plt.plot(RPO_list, y, color='red', label='Sigmoid fit')
                threshold = interpolate_threshold(RPO_list, y, threshold=66.667)
                plt.xscale('log', base=1.414)  # Logarithmic scale for RPO
                plt.xticks(RPO_list, labels=[str(int(rpo)) for rpo in RPO_list], rotation=45)
            else:
                y = fit_sigmoid(range(len(RPO_list)), avg)
                plt.plot( y, color='red', label='Sigmoid fit')
                threshold = interpolate_threshold(range(len(RPO_list)), y, threshold=66.667)
                threshold = 1.414**((threshold-6))  # Convert to RPO scale
                plt.xticks(range(len(RPO_list)), labels=[str((rpo)) for rpo in RPO_list], rotation=45)
            plt.hlines(33.33333, min(RPO_list), max(RPO_list), colors='black', linestyles='dashed', label='Chance level (33.333%)')
            plt.hlines((100+33.33333)/2, min(RPO_list), max(RPO_list), colors='black', linestyles='dotted', label='Threshold (66.667%) = ' + str(round(threshold,2)) + ' RPO')
            plt.legend()
            print('threshold = ', threshold, ' for noise = ', add_noise)
            if save_bool:
                fig2.savefig('./figures/3AFC_SR_' + hearing_type + '_RPO_' + str(RPO_list) + '_' + char_str + '_PC.png')

    if run_multiple_noise:
        noise_list = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]#[0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        char_str += ' and noise'
        dprime_matrix, PC_matrix = run_dprime_multi_noise(RPO_list,
                                                            hearing_type, 
                                                            noise_list, 
                                                            filter_bool=filter_bool, 
                                                            filter_type=filter_type,
                                                            phase_trials=phase_trials, 
                                                            window_size=window_size,
                                                            data_dir=data_dir)
        fig3 = plt.figure(figsize=(10, 5))
        
        true_axis = False
        if true_axis:
            plt.xlim((min(RPO_list), max(RPO_list)))
        else:
            plt.xlim((0, len(RPO_list)))

        if hearing_type == 'EH' and CS_off:
            hearing_type = 'EH (CS off)'

        RPO_list.append(20)
        threshold_list = np.zeros(len(noise_list))
        fit_list = np.zeros((len(noise_list), len(RPO_list)))
        for n, add_noise in enumerate(noise_list):
            if true_axis:
                plt.plot(RPO_list[:len(PC_matrix[n,:,0])], PC_matrix[n, :, 0], 'o', color='orange')#, label=f'i1 vs s (noise={add_noise})' if n == 0 else "")
                plt.plot(RPO_list[:len(PC_matrix[n,:,0])], PC_matrix[n, :, 1], 'o', color='red')#, label=f'i2 vs s (noise={add_noise})' if n == 0 else "")
                # plt.plot(RPO_list[:len(PC_matrix[n,:,0])], PC_matrix[n, :, 2], 'o', color='blue')#, label=f'i1 vs i2 (noise={add_noise})' if n == 0 else "")
            else:
                plt.plot(PC_matrix[n, :, 0], 'o', color='orange')#, label=f'i1 vs s (noise={add_noise})' if n == 0 else "")
                plt.plot(PC_matrix[n, :, 1], 'o', color='red')#, label=f'i2 vs s (noise={add_noise})' if n == 0 else "")
                # plt.plot(PC_matrix[n, :, 2], 'o', color='blue')#, label=f'i1 vs i2 (noise={add_noise})' if n == 0 else "")            
            avg = (PC_matrix[n, :, 0] + PC_matrix[n, :, 1])/2
            avg = np.append(avg, 33.33333)  # Add chance level for 3AFC
            y = fit_sigmoid(RPO_list, avg)
            threshold = float(interpolate_threshold(RPO_list, y, threshold=66.667))
            print('threshold = ', threshold, ' for noise = ', add_noise)
            if true_axis:
                plt.plot(RPO_list, y, label=f'fit with noise={add_noise}, threshold={round(threshold,2)}')
            else:
                plt.plot(y, label=f'fit with noise={add_noise}, threshold={round(threshold,2)}')
            threshold_list[n] = threshold
            fit_list[n, :] = y
        plt.hlines(33.33333, min(RPO_list), max(RPO_list), colors='black', linestyles='dashed', label='Chance level (33.333%)')
        plt.hlines((100+33.33333)/2, min(RPO_list), max(RPO_list), colors='black', linestyles='dotted', label='Threshold (66.667%)')
        plt.ylabel('Percent Correct')
        plt.xlabel('RPO')
        plt.legend()
        plt.ylim((30, 100))
        plt.title('Spectral ripple test  ('+ str(phase_trials) +' trials) with ' + hearing_type + ' ' + char_str )

        
        if true_axis:
            plt.xticks(RPO_list, labels=[str(int(rpo)) for rpo in RPO_list], rotation=45)
        else:
            plt.xticks(range(len(RPO_list[:-1])), labels=[str((rpo)) for rpo in RPO_list[:-1]], rotation=45)

        if save_bool:
                fig3.savefig('./figures/spectrum/3AFC/SR_' + hearing_type + '_' +  str(phase_trials) + '_trials_' + char_str.replace(' ', '_') + 'list_'+ str(noise_list).replace(', ', '_') +'.png')
                dict_save = {}
                dict_save['dprime_matrix'] = dprime_matrix
                dict_save['PC_matrix'] = PC_matrix
                dict_save['RPO_list'] = RPO_list
                dict_save['noise_list'] = noise_list
                dict_save['hearing_type'] = hearing_type
                dict_save['phase_trials'] = phase_trials
                dict_save['filter_bool'] = filter_bool
                dict_save['window_size'] = window_size
                dict_save['filter_type'] = filter_type
                dict_save['fit'] = fit_list
                dict_save['threshold'] = threshold_list
                np.save('./output/3AFC_SR_' + hearing_type + '_' + str(phase_trials) + '_trials_' + char_str.replace(' ', '_') + 'list_'+ str(noise_list).replace(', ', '_') +'.npy', dict_save, allow_pickle=True)


    if load_created_output:
        import matplotlib as mpl
        fig4, axs = plt.subplots(3, 2, figsize=(10, 5), sharex=True, sharey=True)  
        axs = axs.flatten()     
        true_axis = False

        hearing_type_list = ['NH', 'NH', 'EH', 'EH', 'EH (CS off)', 'EH (CS off)']
        char_str_list = ['no filter', 'filter (window=33)', 'no filter', 'filter (window=33)', 'no filter', 'filter (window=33)']

        # create colour map
        n_lines = len(noise_list)
        cmap = mpl.colormaps['plasma']
        colors = cmap(np.linspace(0, 1, n_lines))

        if true_axis:
            plt.xlim((min(RPO_list), max(RPO_list)))
        else:
            plt.xlim((0, len(RPO_list)))

        for ax, hearing_type, char_str in zip(axs, hearing_type_list, char_str_list):
            # Load the saved output
            data = np.load('./output/3AFC_SR_' + hearing_type + '_' + str(phase_trials) + '_trials_' + char_str.replace(' ', '_') + 'list_'+ str(noise_list).replace(', ', '_') +'.npy', allow_pickle=True).item()
            threshold_list = data['threshold']
            fit_list = data['fit']
            RPO_list = data['RPO_list']
            PC_matrix = data['PC_matrix']
            noise_list = data['noise_list']

            for n, add_noise in enumerate(noise_list):
                if true_axis:
                    ax.plot(RPO_list[:len(PC_matrix[n,:,0])], PC_matrix[n, :, 0], 'o', color=colors[n])
                    ax.plot(RPO_list[:len(PC_matrix[n,:,1])], PC_matrix[n, :, 1], 'o', color=colors[n])
                    ax.plot(RPO_list, y, label=f'fit with noise={add_noise}, threshold={round(threshold,2)}', color=colors[n])
                else:
                    ax.plot(RPO_list[:len(PC_matrix[n,:,0])], PC_matrix[n, :, 0], 'o', color=colors[n])
                    ax.plot(RPO_list[:len(PC_matrix[n,:,1])], PC_matrix[n, :, 1], 'o', color=colors[n])
                    ax.plot(y, label=f'fit with noise={add_noise}, threshold={round(threshold,2)}', color=colors[n])
            ax.hlines(33.33333, min(RPO_list), max(RPO_list), colors='black', linestyles='dashed', label='Chance level (33.333%)')
            ax.hlines((100+33.33333)/2, min(RPO_list), max(RPO_list), colors='black', linestyles='dotted', label='Threshold (66.667%)')
            if ax == axs[0] or ax == axs[2] or ax == axs[4]:
                ax.set_ylabel(hearing_type + '\n Percentage correct')
            if ax == axs[4] or ax == axs[5]:
                ax.set_xlabel('RPO')
            if ax == axs[0] or ax == axs[1]:
                ax.set_title(char_str)
            # ax.set_ylabel('Percent Correct')
            # ax.set_xlabel('RPO')
            ax.legend()

    plt.show()
