import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from pymatreader import read_mat
from scipy import signal

# TO DO
# [x] Lower RPO also for SMRT 
# [X] Look at upper edge
# [X] Look at the effect of slimmer critical bands
# [X] Check SMRT with interpretation with slimmer critical bands & check internal representations for fluctuations

# [] check SMRT correlation within critical bands and correlation with 5 Hz signal
# [] STRIPES in interpretation?
# [] Flip time signal SMRT? Compare SMRT vs iSMRT
# [] compare STRIPES w/ and w/o noise

# def correlate_critical_bands(fname):

def create_sine(bin_size, duration):
    f = 3
    t = np.arange(0, duration, bin_size)
    signal = np.sin(2*np.pi*f*t)
    return signal, t


test = 'SMRT' # 'SMRT'/'STRIPES'/ 'SR'
critical_band_type = 'Mel' # 'Bark'/'Mel'/'slim'/'Hamacher'
bin_size = 0.005


if test == 'SMRT':
    EH_data_dir = './data/SMRT/'
    NH_data_dir = './data/SMRT/'
elif test == 'STRIPES':
    EH_data_dir = './data/STRIPES/'
    NH_data_dir = './data/STRIPES/NH_normal_spont/'
elif test == 'SR':
    EH_data_dir = './data/SR/'
    NH_data_dir = './data/SR/'
else:
    raise ValueError('Invalid test')


# loop folder
fname_NH_list = glob.glob(NH_data_dir + '*1903CFs.mat')
fname_EH_list = glob.glob(EH_data_dir + '*.npy')
for fname in fname_NH_list:
    print(fname)
    selected_spikes, new_fiber_frequencies_CF, centre_remaining_frequency_bands, remaining_frequency_edges = spike_matrix_to_critical_band(fname, critical_band_type)
    sine, t = create_sine(bin_size, 2*len(selected_spikes[0,:])*bin_size)
    selected_spikes = selected_spikes[:len(new_fiber_frequencies_CF),:]

    # check subplot size
    number_bands_prime = selected_spikes.shape[0]
    row_plot = 0
    while row_plot<3:
        number_bands_prime += 1
        if not is_prime(number_bands_prime):
            row_plot, column_plot = closestDivisors(number_bands_prime)

    plt.subplots(row_plot, column_plot, sharex=True,figsize=(16, 6))
    for band in np.arange(selected_spikes.shape[0]):
        norm_spikes = 2*(selected_spikes[band,:]-np.min(selected_spikes[band,:])) / (np.max(selected_spikes[band,:])-np.min(selected_spikes[band,:])) - 1
        # correlation = np.corrcoef(signal, norm_spikes)[0, 1]
        correlation = signal.correlate(sine, norm_spikes, mode='full')
        lags = signal.correlation_lags(len(sine), len(norm_spikes), mode='full')
        lags = lags[np.argmax(correlation)]
        print(max(correlation))
        plt.subplot(row_plot, column_plot, band+1)
        plt.plot(range(len(sine))+lags, sine)
        plt.plot(sine)
        plt.plot(norm_spikes)
        plt.xlim((0, len(norm_spikes)))
        plt.title('corr: ' + str(round(max(correlation),3)))
    plt.suptitle(os.path.basename(fname) )


plt.show()

