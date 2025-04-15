import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from spectrum_from_spikes import get_normalized_spectrum, fiber_id_electrode, half_electrode_range, freq_x_fft
from SMRTvsSR import get_FFT_spectrum


NH_data_dir = './data/STRIPES/BookendsDifference/'

plot_edges = True

ripple_list = np.arange(2, 10, 1) # [3.0, 5.0, 7.0, 9.0]
font_size = 20

# plotting characteristics


color_u = 'blue'
color_d = 'red'
filter_bool = True
rr = 1
octave_spaced = True
bar_width = 15
alpha = 0.15
filter_order = 4
cut_off_freq = 100
if plot_edges:
    if len(ripple_list)>6:
        num_rows = 3
        num_columns = int(np.ceil(len(ripple_list)/3))
    if len(ripple_list)>3:
        num_rows = 2
        num_columns = int(np.ceil(len(ripple_list)/2))
    else:
        num_rows = 1
        num_columns = len(ripple_list)
    fig = plt.figure()


# Looping over glide densities
for ripple_ud in ripple_list:
    if ripple_ud == 1.0:
        ripple_ud = 1.1 


    # get NH
    fname_NH_u = glob.glob(NH_data_dir + '*BS_d*' + str(ripple_ud) + '*0.01_d*.mat')[0]
    fname_NH_d = glob.glob(NH_data_dir + '*BS_d*' + str(ripple_ud) + '*0.01_d*noise*.mat')[0]
    # fname_NH_d = glob.glob(NH_data_dir + '*BS_d*' + str(ripple_ud) + '*0.01_d*.mat')[0]

    print(fname_NH_u)
    print(fname_NH_d)


    plt.subplot(num_rows, num_columns, rr)
    normal_spectrum_u, fiber_frequencies = get_normalized_spectrum(fname_NH_u) 
    normal_spectrum_d, _ =  get_normalized_spectrum(fname_NH_d) 
    plt.bar(fiber_frequencies, normal_spectrum_u, width=bar_width, alpha=alpha, color=color_u)
    plt.bar(fiber_frequencies, normal_spectrum_d, width=bar_width, alpha=alpha, color=color_d)
    filter_sig_u = butter_lowpass_filter(normal_spectrum_u, cut_off_freq, len(normal_spectrum_u), filter_order)
    filter_sig_d = butter_lowpass_filter(normal_spectrum_d, cut_off_freq, len(normal_spectrum_d), filter_order)
    plt.plot(fiber_frequencies, filter_sig_u, color=color_u, label='down')
    plt.plot(fiber_frequencies, filter_sig_d, color=color_d, label='noise')
    plt.vlines(8000, 0, 1, color='black', linestyle='dashed')
    plt.vlines(8700, 0, 1, color='black', linestyle='dashed')
    plt.ylabel(str(ripple_ud))
    plt.ylim(0,1)
    if octave_spaced:
        plt.xscale('log', base=2)
        plt.xticks([500, 1000, 2000, 4000, 8000], labels=['500', '1000', '2000', '4000', '8000'])   
        octave_str = 'octave_spaced'   
    else:
        octave_str = 'linear_spaced'
    rr += 1            
    plt.xlim(min(fiber_frequencies), max(fiber_frequencies))
    plt.legend()
    if ripple_ud == ripple_list[-1]:
        plt.suptitle('STRIPES frequency edge (NH)', fontsize=font_size)
        fig.text(0.02, 0.47, 'Glide density', ha='center', rotation='vertical', fontsize=font_size)
        fig.text(0.5, 0.03, 'Frequency [Hz]', ha='center', fontsize=font_size)
        # fig.savefig('./figures/STRIPES/spectrum_edges_NH_'+ str(ripple_list) + octave_str + '.png')



        




plt.show()