import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from pymatreader import read_mat
from spectrum_from_spikes import get_normalized_spectrum

# This doesn't work because I determine the X in the input, the x point of the centroid will always be the same

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]
# freq_x_fft = np.load('./data/EH_freq_vector_electrode_allocation_logspaced.npy')
freq_x_fft, fiber_id_electrode, half_electrode_range= create_EH_freq_vector_electrode_allocation('log')

data_dir = './data/spectrum/'
RPO = '2.000'
filter_bool = True # filter spike spectrum
filter_order = 4
cut_off_freq = 100

# get NH
fname_NH_i = glob.glob(data_dir + '*i1*' + RPO + '*.mat')[0]
fname_NH_s = glob.glob(data_dir + '*_s_*' + RPO + '*.mat')[0]
normal_spectrum_i, fiber_frequencies = get_normalized_spectrum(fname_NH_i) 
normal_spectrum_s, _ =  get_normalized_spectrum(fname_NH_s) 
filter_NH_i = butter_lowpass_filter(normal_spectrum_i, cut_off_freq, len(normal_spectrum_i), filter_order)
filter_NH_s = butter_lowpass_filter(normal_spectrum_s, cut_off_freq, len(normal_spectrum_s), filter_order)

# # get EH
# fname_EH_i = glob.glob(data_dir + '*matrix*i1*'  + RPO + '*.npy')[0]
# fname_EH_s = glob.glob(data_dir + '*matrix*_s_*' + RPO + '*.npy')[0]

# electric_spectrum_i, filter_EH_i = get_normalized_spectrum(fname_EH_i, filter_bool)
# electric_spectrum_s, filter_EH_s = get_normalized_spectrum(fname_EH_s, filter_bool)

# # remove outside of region of interest
# electric_spectrum_i = electric_spectrum_i[int(min(fiber_id_electrode))-half_electrode_range:int(max(fiber_id_electrode))]
# electric_spectrum_s = electric_spectrum_s[int(min(fiber_id_electrode))-half_electrode_range:int(max(fiber_id_electrode))]


# len(electric_spectrum_i)