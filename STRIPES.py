import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from spectrum_from_spikes import get_normalized_spectrum
from SMRTvsSR import get_FFT_spectrum

data_dir = './data/STRIPES/'
sound_dir = './sounds/STRIPES/'

ripple_list = np.arange(1, 5.5, 0.5)
fig, ax = plt.subplots(len(ripple_list), 2, figsize=(12, 9))

for rr, ripple_id in enumerate(ripple_list):
    if ripple_id == 1.0:
        ripple_id = 1.1
    # get NH
    fname_NH_u = glob.glob(data_dir + '*u*' + str(ripple_id) + '*.mat')[0]
    fname_NH_d = glob.glob(data_dir + '*d_*' + str(ripple_list) + '*.mat')[0]
    normal_spectrum_u, fiber_frequencies = get_normalized_spectrum(fname_NH_u) 
    normal_spectrum_d, _ =  get_normalized_spectrum(fname_NH_d) 

    bar_width = 15
    alpha = 0.2
    filter_order = 4
    cut_off_freq = 100

    # sound
    fname_u = glob.glob(sound_dir + '*u*' + str(ripple_id) + '*')[0]
    fname_d = glob.glob(sound_dir + '*d*' + str(ripple_id) + '*')[0]
    get_FFT_spectrum(fname_u)

    #NH
    plt.subplot(len(ripple_list), 2, rr+1)
    print(rr)
    plt.bar(fiber_frequencies, normal_spectrum_u, width=bar_width, alpha=alpha, color='orange')
    plt.bar(fiber_frequencies, normal_spectrum_d, width=bar_width, alpha=alpha, color='blue')
    filter_sig_u = butter_lowpass_filter(normal_spectrum_u, cut_off_freq, len(normal_spectrum_u), filter_order)
    filter_sig_d = butter_lowpass_filter(normal_spectrum_d, cut_off_freq, len(normal_spectrum_d), filter_order)
    plt.plot(fiber_frequencies, filter_sig_u, color='orange', label='inverted')
    plt.plot(fiber_frequencies, filter_sig_d, color='blue', label='standard')
    plt.ylim((0,1))
    plt.ylabel(str(ripple_id)) #  + ' RPO \n normalized \n spiking'
    plt.legend()
    plt.ylim((0,1))

plt.show()