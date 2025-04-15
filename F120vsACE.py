import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from spectrum_from_spikes import get_normalized_spectrum, fiber_id_electrode, half_electrode_range, freq_x_fft
from SMRTvsSR import get_FFT_spectrum
from SMRT_neurograms import ax_colour_map_SMRT_per_kHz


def create_EH_freq_vector_electrode_allocation(Ln, Le, type_scaling_fibres='log', SCS='F120'): # 'log'/'lin'
    # type_scaling_fibres = 'lin' 

    if SCS == 'F120':
        # frequencies in FFT channels edges of bands
        edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]
    if SCS == 'ACE':
        edges = [187.5, 312.5, 437.50, 562.50, 687.50, 812.50, 937.50, 1062.5, 1187.5, 1312.5, 1562.5, 1812.5, 2062.5, 2312.5, 2687.5, 3062.5, 3562.5, 4062.5, 4687.5, 5312.5, 6062.5, 6937.5, 7937.5]

    # match fibers 
    fiber_id_electrode = np.zeros(len(Le))
    for e, mm in enumerate(Le):
        fiber_id_electrode[e] = int(find_closest_index(Ln, mm) )
    num_fibers_between_electrode = abs(np.diff(fiber_id_electrode))
    half_electrode_range = int(np.mean(num_fibers_between_electrode)/2)
  
    if SCS == 'F120':
        # selected fiber IDs:
        fiber_id_list_FFT = np.arange(start=int(np.min(fiber_id_electrode)), stop=int(np.max(fiber_id_electrode)+ half_electrode_range)) # do I need to flip this?

        if type_scaling_fibres == 'log':
        # 272 is the frequency edge of an FFT bin one step before
            freq_x_fft = list(np.logspace(np.log10(272), np.log10(edges[0]), half_electrode_range, base=10, endpoint=False)) 
        elif type_scaling_fibres == 'lin':
            freq_x_fft = list(np.linspace(272, edges[0], half_electrode_range, endpoint=False)) 
    if SCS == 'ACE':
        fiber_id_inbetween_electrode = np.zeros(len(Le)-1)
        for e in range(len(fiber_id_inbetween_electrode)):
            fiber_id_inbetween_electrode[e] = int((fiber_id_electrode[e] +fiber_id_electrode[e+1])/2)
        # breakpoint()
        if all(np.diff(fiber_id_inbetween_electrode)<0):
            half_electrode_range *= -1
        fiber_id_inbetween_electrode = np.insert(fiber_id_inbetween_electrode, 0, int(fiber_id_electrode[0]-half_electrode_range))
        fiber_id_inbetween_electrode = np.append(fiber_id_inbetween_electrode, int(fiber_id_electrode[-1]+half_electrode_range))
        # selected fiber IDs:
        fiber_id_list_FFT = np.arange(start=int(np.min(fiber_id_inbetween_electrode)), stop=int(np.max(fiber_id_inbetween_electrode)))
        # now numbers around electrode
        num_fibers_between_electrode = abs(np.diff(fiber_id_inbetween_electrode))
        # print('inbetween', fiber_id_inbetween_electrode)
        # print('num_fibers_between_electrode', num_fibers_between_electrode)
        freq_x_fft = []

    for e in range(len(edges)-1):
        freq_range = (edges[e], edges[e+1])
        if type_scaling_fibres == 'log':
            freq_fft_band = list(np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), int(num_fibers_between_electrode[e]), base=10, endpoint=False))
        elif type_scaling_fibres == 'lin':
            freq_fft_band = list(np.linspace(freq_range[0], freq_range[1], int(num_fibers_between_electrode[e]), endpoint=False))
        freq_x_fft.extend(freq_fft_band)

    plotting_bool = False
    if plotting_bool:
        # plotting
        plt.figure()
        plt.plot(Ln[fiber_id_list_FFT], freq_x_fft)
        plt.vlines(Le, min(freq_x_fft)*np.ones(len(Le)), max(freq_x_fft)*np.ones(len(Le)))
        if SCS == 'ACE':
            plt.hlines([250, 375, 500, 625, 750, 875, 1000, 1125, 1250, 1437, 1687, 1937, 2187, 2500, 2875, 3312, 3812, 4375, 5000, 5687, 6500, 7437], min(Ln[fiber_id_list_FFT])*np.ones(len(Le)), max(Ln[fiber_id_list_FFT])*np.ones(len(Le)))
            plt.hlines(1250, min(Ln[fiber_id_list_FFT]), max(Ln[fiber_id_list_FFT]), color='red')
            plt.vlines(Le[8], min(freq_x_fft), max(freq_x_fft), color='red')
        plt.xlabel('Position along basilar membrane [mm]')
        plt.ylabel('Freq [Hz]')
        plt.title(SCS + ' ' + type_scaling_fibres + ' spaced')

    return np.asarray(freq_x_fft), np.asarray(fiber_id_list_FFT)


def plot_dual_neurograms(fname_list, config_list, electric_scale = 'log_fft'):
    num_rows = int(len(fname_list)/2)
    fig, ax = plt.subplots(num_rows, 2)
    axes = ax.flatten()
    norm = None
        
    clim = (0, 1000)
    flim = 8
    for f_i, fname in enumerate(fname_list):
        spike_rates_list = np.load(fname)
        [num_fibers, num_bins] = spike_rates_list.shape 
        with open(config_list[f_i]) as yamlfile: 
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        start_bin = config['binsize']
        binsize = config['binsize']

        if config['SCS'] == 'F120':
            _, _, _, _, _, Ln, Le, _, _, _ = load_mat_virtual_all_thresholds(os.path.join('./data', config['virtual_thresholds_file']), 
                                                                        nerve_model_type=config['nerve_model_type_AB'], 
                                                                        array_type=config['array_type_AB'], 
                                                                        state=config['nerve_state_AB']) # [A]
        else:
            _, _, _, _, Ln, Le, _, _, _, _ = load_cochlear_parms_and_pulse_train('./data/STRIPES/ACE/', config['cochlear_thresholds_file'], config['cochlear_f_elgram'])
        
        frequency_list, fiber_id_list = create_EH_freq_vector_electrode_allocation(Ln, Le, 'log', SCS=config['SCS'])
        spike_matrix = spike_rates_list[fiber_id_list,:]
        y_axis_str = 'Frequency [kHz]'

        [mesh, axe] = ax_colour_map_SMRT_per_kHz(axes[f_i], spike_matrix, frequency_list*1e-3, config['sound_duration'], y_axis_str, binsize=binsize, clim=clim, norm=norm, flim=flim)


if __name__ == '__main__':  

    Glide_list = ['1.1', '3.0', '5.0', '7.0', '9.0']
    direction = 'd'
    fname_list = []
    config_list = []

    for glide in Glide_list:
        F120_name = glob.glob('./data/STRIPES/*matrix*' + direction + '_' + glide + '*.npy')[0]
        F120_config = glob.glob('./data/STRIPES/*config*' + direction + '_' + glide + '*.yaml')[0]
        ACE_name = glob.glob('./data/STRIPES/ACE/*matrix*' + direction + '_' + glide + '*.npy')[0]
        ACE_config = glob.glob('./data/STRIPES/ACE/*config*' + direction + '_' + glide + '*.yaml')[0]
        fname_list.append(F120_name)
        fname_list.append(ACE_name)
        config_list.append(F120_config)
        config_list.append(ACE_config)
    
    plot_dual_neurograms(fname_list, config_list)

    plt.show()