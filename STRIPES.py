import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from spectrum_from_spikes import get_normalized_spectrum, fiber_id_electrode, half_electrode_range, freq_x_fft
from SMRTvsSR import get_FFT_spectrum


EH_data_dir = './data/STRIPES/'
NH_data_dir = './data/STRIPES/1e-4Spont/'
sound_dir = './sounds/STRIPES/'

# can't do at the same time
plot_spectrum = False
plot_spectroneurogram = True
plot_critical_bands = False

ripple_list = np.arange(5.5, 10.5, 0.5)
font_size = 20

if plot_critical_bands:
    critical_band_type = 'Bark'
    bin_size = 0.005

if plot_spectrum:
    fig, ax = plt.subplots(len(ripple_list), 3, figsize=(12, 9), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.085, bottom=0.10, right=0.957, top=0.924, wspace=0.11)
    color_u = 'blue'
    color_d = 'red'
    filter_bool = True
    rr = 1

if plot_spectroneurogram:
    bin_size = 0.005
    clim = None #(0, 1000)
    norm = None
    flim=8 # None

for ripple_ud in ripple_list:
    if ripple_ud == 1.0:
        ripple_ud = 1.1 

    # sound
    fname_u = glob.glob(sound_dir + '*u*' + str(ripple_ud) + '*')[0]
    fname_d = glob.glob(sound_dir + '*d*' + str(ripple_ud) + '*')[0]
    # get NH
    fname_NH_u = glob.glob(NH_data_dir + '*u*' + str(ripple_ud) + '*.mat')[0]
    fname_NH_d = glob.glob(NH_data_dir + '*d_*' + str(ripple_ud) + '*.mat')[0]
    # get EH
    fname_EH_u = glob.glob(EH_data_dir + '*matrix*u*' + str(ripple_ud) + '*.npy')[0]
    fname_EH_d = glob.glob(EH_data_dir + '*matrix*d_*' + str(ripple_ud) + '*.npy')[0]

    if plot_critical_bands:
        # NH
        fig = plot_fig_critical_bands([fname_NH_d, fname_NH_u], critical_band_type, bin_size)
        plt.suptitle('NH with '+  str(ripple_ud) + ' density on '+ critical_band_type + ' scale', fontsize=20)
        fig.savefig('./figures/STRIPES/criticalbands_NH_' + critical_band_type +'scale_binsize_' +str(bin_size) + 's_'+ str(ripple_ud) +'density.png')
        # EH
        fig = plot_fig_critical_bands([fname_EH_d, fname_EH_u], critical_band_type, bin_size)
        plt.suptitle('EH with ' +  str(ripple_ud) +' density on '+ critical_band_type + ' scale', fontsize=20)
        fig.savefig('./figures/STRIPES/criticalbands_EH_' + critical_band_type +'scale_binsize_' +str(bin_size) + 's_'+ str(ripple_ud) +'density.png')

    
    if plot_spectroneurogram:
        fig, ax = plt.subplots(3,2, figsize=(12, 9))
        plt.subplots_adjust(left=0.067, bottom=0.08, right=0.957, top=0.94, wspace=0.11)
        axes = ax.flatten()
        
        # sound
        ax_spectrogram(fname_d, axes[0])
        axes[0].set_ylabel('Frequency [kHz]')
        # axes[0].set_ylabel('Stimulus \n Frequency [kHz]')
        fig.text(0.02, 0.765, 'Stimulus', ha='center', rotation='vertical', fontsize=font_size-2)
        ax_spectrogram(fname_u, axes[1])
        axes[1].set_ylabel('Frequency [kHz]')

        # NH
        ax_neurogram(fname_NH_u, bin_size, axes[2], flim=flim)
        axes[2].set_ylabel('Frequency [kHz]')
        fig.text(0.02, 0.405, 'Normal hearing', ha='center', rotation='vertical', fontsize=font_size-2)
        ax_neurogram(fname_NH_d, bin_size, axes[3], flim=flim)
        axes[3].set_ylabel('Frequency [kHz]')

        # EH
        ax_neurogram(fname_EH_u, bin_size, axes[4], flim=flim)
        axes[4].set_ylabel('Frequency [kHz]')
        fig.text(0.02, 0.105, 'Electric hearing', ha='center', rotation='vertical', fontsize=font_size-2)
        ax_neurogram(fname_EH_d, bin_size, axes[5], flim=flim)
        axes[5].set_ylabel('Frequency [kHz]')
        plt.suptitle(str(ripple_ud) + ' density', fontsize=font_size)
        fig.text(0.515, 0.02, 'Time [s]', ha='center', fontsize=font_size)
        fig.savefig('./figures/STRIPES/spectroneurograms_clim_' + str(clim) + '_flim_' + str(flim) + '_binsize_' +str(bin_size) + '_norm_' + str(norm) + '_'+ str(ripple_ud) +'density.png')

    if plot_spectrum:
        bar_width = 15
        alpha = 0.15
        filter_order = 4
        cut_off_freq = 100

        # sound
        outline_u, frequency = get_FFT_spectrum(fname_u)
        outline_d, frequency = get_FFT_spectrum(fname_d)
        
        ax = plt.subplot(len(ripple_list), 3, rr)
        plt.plot(frequency, outline_u, color=color_u, linestyle='dashed', alpha=0.5)
        plt.plot(frequency, outline_u, color=color_d, linestyle='dashed', alpha=0.5)
        plt.xscale('log', base=2)
        plt.xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
        plt.xlim(274, 8e3)
        plt.ylim(0,1)
        plt.ylabel(str(ripple_ud)) 
        rr += 1

        # NH
        normal_spectrum_u, fiber_frequencies = get_normalized_spectrum(fname_NH_u) 
        normal_spectrum_d, _ =  get_normalized_spectrum(fname_NH_d) 
        plt.subplot(len(ripple_list), 3, rr)
        plt.bar(fiber_frequencies, normal_spectrum_u, width=bar_width, alpha=alpha, color=color_u)
        plt.bar(fiber_frequencies, normal_spectrum_d, width=bar_width, alpha=alpha, color=color_d)
        filter_sig_u = butter_lowpass_filter(normal_spectrum_u, cut_off_freq, len(normal_spectrum_u), filter_order)
        filter_sig_d = butter_lowpass_filter(normal_spectrum_d, cut_off_freq, len(normal_spectrum_d), filter_order)
        plt.plot(fiber_frequencies, filter_sig_u, color=color_u, label='up')
        plt.plot(fiber_frequencies, filter_sig_d, color=color_d, label='down')
        plt.ylabel(str(ripple_ud))
        rr += 1

        # get EH
        electric_spectrum_u, electric_spectrum2_u = get_normalized_spectrum(fname_EH_u, filter_bool)
        electric_spectrum_d, electric_spectrum2_d = get_normalized_spectrum(fname_EH_d, filter_bool)
        
        electric_spectrum_u = electric_spectrum_u[int(min(fiber_id_electrode))-half_electrode_range:int(max(fiber_id_electrode))]
        electric_spectrum_d = electric_spectrum_d[int(min(fiber_id_electrode))-half_electrode_range:int(max(fiber_id_electrode))]
        electric_spectrum2_u = electric_spectrum2_u[int(min(fiber_id_electrode))-half_electrode_range:int(max(fiber_id_electrode))]
        electric_spectrum2_d = electric_spectrum2_d[int(min(fiber_id_electrode))-half_electrode_range:int(max(fiber_id_electrode))]

        plt.subplot(len(ripple_list), 3, rr)
        plt.bar(freq_x_fft, electric_spectrum_u, width=bar_width, alpha=alpha, color=color_u)
        plt.bar(freq_x_fft, electric_spectrum_d, width=bar_width, alpha=alpha, color=color_d)
        plt.plot(freq_x_fft, electric_spectrum2_u, color=color_u, label='up')
        plt.plot(freq_x_fft, electric_spectrum2_d, color=color_d, label='down')
        plt.legend()
        rr += 1

        if ripple_ud == ripple_list[-1]:
            plt.subplot(len(ripple_list), 3, 1)
            plt.title('Stimulus normalized PSD')
            plt.subplot(len(ripple_list), 3, 2)
            plt.title('Normal hearing')
            plt.subplot(len(ripple_list), 3, 3)
            plt.title('Electric hearing')
            fig.text(0.02, 0.47, 'Glide density', ha='center', rotation='vertical', fontsize=font_size)
            fig.text(0.5, 0.03, 'Time [s]', ha='center', fontsize=font_size)
            fig.savefig('./figures/STRIPES/spectrum_'+ str(ripple_list) +'.png')



plt.show()