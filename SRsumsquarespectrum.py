import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from pymatreader import read_mat
import matplotlib.transforms as mtransforms # labeling axes
from spectrum_from_spikes_new_freq_axis import get_normalized_spectrum
import scipy

RPO_list = [0.125, 0.176, 0.250, 0.354, 0.500, 0.707, 1.000, 1.414, 2.000, 2.828, 4.000, 5.657, 8.000, 11.314 ] 
dB = 65
data_dir = './data/spectrum/' + str(dB) + 'dB_2416CF/'
filter_bool = True
filter_type = 'mavg' # 'mavg' /'butter' 
window_size = 33 # moving average window size
cut_off_freq = 100 # for butterworth filter
filter_order = 4 # for butterworth filter
fiber_id_selection = np.load('./data/AB_MS_based_on_min_filtered_thresholdsfiber_ID_list_FFT.npy')

for RPO in RPO_list:
    print(RPO)
    RPO = str(RPO)
    # get NH
    try:
        fname_NH_i = glob.glob(data_dir + '*i1*' + RPO + '*.mat')[0]
        fname_NH_s = glob.glob(data_dir + '*_s_*' + RPO + '*.mat')[0]
        # print(fname_NH_i, fname_NH_s)
    except:
        print('No data for RPO = ' + RPO)
        continue
    normal_spectrum_i, fiber_frequencies = get_normalized_spectrum(fname_NH_i) 
    normal_spectrum_s, _ =  get_normalized_spectrum(fname_NH_s) 
    if filter_type == 'butter':
        filter_sig_i = butter_lowpass_filter(normal_spectrum_i, cut_off_freq, len(normal_spectrum_i), filter_order)
        filter_sig_s = butter_lowpass_filter(normal_spectrum_s, cut_off_freq, len(normal_spectrum_s), filter_order)
    elif filter_type == 'mavg':
        filter_sig_i = symmetric_moving_average(normal_spectrum_i, window_size=window_size)
        filter_sig_s = symmetric_moving_average(normal_spectrum_s, window_size=window_size)

    plt.subplot(2, 1, 1)
    ssd = sum((normal_spectrum_s-normal_spectrum_i)**2)
    plt.scatter(float(RPO), ssd, color ='b')
    filtered_ssd =  sum((filter_sig_s-filter_sig_i)**2)
    plt.scatter(float(RPO), filtered_ssd, color ='darkblue')
    plt.xlabel('RPO')

    plt.subplot(2, 1, 2)
    pearson = scipy.stats.pearsonr(normal_spectrum_i, normal_spectrum_s)[0]
    plt.scatter(float(RPO), pearson, color ='b')
    pearson = scipy.stats.pearsonr(filter_sig_s, filter_sig_i)[0]
    plt.scatter(float(RPO), pearson, color ='darkblue')
    print(scipy.stats.pearsonr(normal_spectrum_i, normal_spectrum_s)[1])

    # get EH
    fname_EH_i_ = glob.glob(data_dir + '2025*matrix*i1*'  + RPO + '*.npy')
    fname_EH_s_ = glob.glob(data_dir + '2025*matrix*_s_*' + RPO + '*.npy')
    fname_EH_i = [l for l in fname_EH_i_ if ('alpha' not in l)][0]
    fname_EH_s = [l for l in fname_EH_s_ if ('alpha' not in l)][0]
    # print(fname_EH_i, fname_EH_s)
    electric_spectrum_i, electric_spectrum2_i = get_normalized_spectrum(fname_EH_i, filter_bool)
    electric_spectrum_s, electric_spectrum2_s = get_normalized_spectrum(fname_EH_s, filter_bool)
    if filter_type == 'mavg':
        electric_spectrum2_i = symmetric_moving_average(electric_spectrum_i, window_size=window_size)
        electric_spectrum2_s = symmetric_moving_average(electric_spectrum_s, window_size=window_size)

    electric_spectrum_i = electric_spectrum_i[fiber_id_selection[0]:fiber_id_selection[-1]+1]
    electric_spectrum_s = electric_spectrum_s[fiber_id_selection[0]:fiber_id_selection[-1]+1]
    electric_spectrum2_i = electric_spectrum2_i[fiber_id_selection[0]:fiber_id_selection[-1]+1]
    electric_spectrum2_s = electric_spectrum2_s[fiber_id_selection[0]:fiber_id_selection[-1]+1]

    plt.subplot(2, 1, 1)
    ssd = sum((electric_spectrum_i-electric_spectrum_s)**2)
    plt.scatter(float(RPO), ssd, color ='red')
    filtered_ssd =  sum((electric_spectrum2_i-electric_spectrum2_s)**2)
    plt.scatter(float(RPO), filtered_ssd, color ='darkred')
    plt.xlabel('RPO')

    plt.subplot(2, 1, 2)
    pearson = scipy.stats.pearsonr(electric_spectrum_i, electric_spectrum_s)[0]
    plt.scatter(float(RPO), pearson, color ='red')
    pearson = scipy.stats.pearsonr(electric_spectrum2_i, electric_spectrum2_s)[0]
    plt.scatter(float(RPO), pearson, color ='darkred')
    print(scipy.stats.pearsonr(electric_spectrum_i, electric_spectrum_s)[1])

plt.show()