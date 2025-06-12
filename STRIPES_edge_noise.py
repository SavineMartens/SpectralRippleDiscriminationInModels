import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from spectrum_from_spikes import get_normalized_spectrum, fiber_id_electrode, half_electrode_range, freq_x_fft
from SMRTvsSR import get_FFT_spectrum, get_spectrum


NH_data_dir = './data/STRIPES/BookendsDifference/'

plot_edges = False
plot_critical_bands = True

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
use_normalized_spectrum = True
num_fibers = 1903

bin_size = 0.005
critical_band_type = 'slim' # 'Bark'/'Mel'/'slim'/'Hamacher'
wo_noise = True


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
    if wo_noise:
        fname_NH_1 = glob.glob(NH_data_dir + '*BS_d*' + str(ripple_ud) + '_*0.01_u*noise*.mat')[0]
        label_1 = 'up without noise'
        fname_NH_2 = glob.glob(NH_data_dir + '*BS_d*' + str(ripple_ud) + '_*0.01_d*noise*.mat')[0]
        label_2 = 'down without noise'
        str_save = 'without_noise'
    else:
        fname_NH_1 = glob.glob(NH_data_dir + '*BS_d*' + str(ripple_ud) + '_*0.01_u_'+ str(num_fibers) +'*.mat')[0]
        label_1 = 'up with noise'
        fname_NH_2 = glob.glob(NH_data_dir + '*BS_d*' + str(ripple_ud) + '_*0.01_d_'+ str(num_fibers) +'*.mat')[0]
        label_2 = 'down with noise'
        str_save = 'with_noise'


    print(fname_NH_1)
    print(fname_NH_2)

    if plot_edges:
        plt.subplot(num_rows, num_columns, rr)
        if use_normalized_spectrum:
            spectrum_1, fiber_frequencies = get_normalized_spectrum(fname_NH_1) 
            spectrum_2, _ =  get_normalized_spectrum(fname_NH_2) 
        else:
            spectrum_1, fiber_frequencies = get_spectrum(fname_NH_1) #get_normalized_spectrum(fname_NH_1) 
            spectrum_2, _ =  get_spectrum(fname_NH_2)#get_normalized_spectrum(fname_NH_2) 
        plt.bar(fiber_frequencies, spectrum_1, width=bar_width, alpha=alpha, color=color_u)
        plt.bar(fiber_frequencies, spectrum_2, width=bar_width, alpha=alpha, color=color_d)
        filter_sig_1 = butter_lowpass_filter(spectrum_1, cut_off_freq, len(spectrum_1), filter_order)
        filter_sig_2 = butter_lowpass_filter(spectrum_2, cut_off_freq, len(spectrum_2), filter_order)
        plt.plot(fiber_frequencies, filter_sig_1, color=color_u, label=label_1)
        plt.plot(fiber_frequencies, filter_sig_2, color=color_d, label=label_2)
        plt.vlines(8000, 0, 1, color='black', linestyle='dashed')
        plt.vlines(8700, 0, 1, color='black', linestyle='dashed')
        plt.ylabel(str(ripple_ud))
        # plt.ylim(0,1)
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

    if plot_critical_bands:
        fig = plot_fig_critical_bands([fname_NH_2, fname_NH_1], critical_band_type, bin_size)
        plt.suptitle('NH with '+  str(ripple_ud) + ' density on '+ critical_band_type + ' scale', fontsize=20)

        fig.savefig('./figures/STRIPES/double_critical_bands/noise_bookends/NH_critical_'+ critical_band_type + 'bands_'+ str(ripple_ud)  + str_save + '.png')

        




plt.show()