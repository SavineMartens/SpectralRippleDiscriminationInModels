from utilities import load_mat_virtual_all_thresholds, rebin_spikes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os 
import yaml
from pymatreader import read_mat
import glob
import matplotlib.transforms as mtransforms # labeling axes

# to do
# [X] y-scale allocated according to FFT -- Frijns says do Greenwood
# [ ] add color bar to all
# [X] find out unit of color bar --> spikes/s



font_size = 14
save_dir = './figures/SMRT/'

cmap_type = 'viridis' # 'inferno'/'nipy_spectral'
    
def ax_colour_map_SMRT_per_kHz(ax, spike_rates_list, Fn_sel, sound_duration, y_axis_str='Frequency [kHz]', binsize=0.01, clim=[], norm=None, flim=None):

    x = np.arange(binsize, sound_duration+binsize, binsize)
    mesh = ax.pcolormesh(x, Fn_sel, spike_rates_list, cmap=cmap_type, norm=norm) #
    if clim:
        mesh.set_clim(clim)
    # cbar = plt.colorbar()
    # cbar.set_label('Spike rate [spikes/s]')
    
    first_after_500 = next((x for x in Fn_sel if x > 0.500), None)
    if flim:
        ax.set_ylim((first_after_500, flim))
    else:
        ax.set_ylim((first_after_500, 6.5))
    ax.set_xlim(binsize, sound_duration)
    ax.set_ylabel(y_axis_str, fontsize=font_size-2)

    return mesh, ax




def load_mat_structs_Hamacher(fname, unfiltered_type = 'Bruce'):
    matfile = read_mat(fname)['structPSTH']
    fiber_frequencies = matfile['CF']
    t_unfiltered = matfile['t_unfiltered']
    
    try:
        spike_rate_unfiltered = matfile['unfiltered_neurogram']
    except:
        spike_rate_unfiltered = np.zeros((len(fiber_frequencies), len(t_unfiltered)))
        unfiltered_row_indices = matfile['unfiltered_row_indices'].astype(int)
        unfiltered_column_indices = matfile['unfiltered_column_indices'].astype(int)
        unfiltered_values = matfile['unfiltered_values']
        # fill in matrix
        for row, column, value in zip(unfiltered_row_indices, unfiltered_column_indices, unfiltered_values):
            spike_rate_unfiltered[row-1, column-1] = value
    sound_name = matfile['fname']
    if sound_name == None:
        sound_name = fname[fname.find('smrt'):fname.find('phase_0')+len('phase_0')]
    dBlevel = matfile['stimdb']
    try:
        Fs_down = matfile['down_sampling_rate']
    except:
        Fs_down = 'Newer version so not downsampled'
    if len(matfile) == 12:
        spike_rate_filtered_downsampled = spike_rate_unfiltered[:,::20] # Hamacher with Fs = 5000 --> 1e5/5e3 = 20
        t_filtered_downsampled = t_unfiltered[::20]
    else:
        try:
            spike_rate_filtered_downsampled = matfile['downsampled_Hamacher_neurogram']
            t_filtered_downsampled = matfile['t_downsampled_filtered_neurogram']
        except:
            spike_rate_filtered_downsampled = 'Newer version so not downsampled'
            t_filtered_downsampled = 'Newer version so not downsampled'
    if len(matfile) == 13:
        return spike_rate_unfiltered, spike_rate_filtered_downsampled, sound_name, dBlevel, fiber_frequencies, Fs_down, t_unfiltered, t_filtered_downsampled
    else:
        if unfiltered_type == 'Hamacher':
            Hamacher_neurogram = matfile['Hamacher_neurogram']
            print('Unfiltered version is Hamacher without downsampling')
            return Hamacher_neurogram, spike_rate_filtered_downsampled, sound_name, dBlevel, fiber_frequencies, Fs_down, t_unfiltered, t_filtered_downsampled
        elif unfiltered_type == 'OG' or unfiltered_type == 'Bruce':
            print('Unfiltered version is Bruce\'s version')
            return spike_rate_unfiltered, spike_rate_filtered_downsampled, sound_name, dBlevel, fiber_frequencies, Fs_down, t_unfiltered, t_filtered_downsampled



def select_RPO_from_string(fname):
    RPO = fname[fname.index('width_')+len('width_'):]
    if '.' not in RPO:
        RPO = RPO + '.0'
    carrier = fname[fname.index('dens_')+len('dens_'):fname.index('_rate')]
    return RPO, carrier


if __name__ == '__main__':  
    bin_size = 0.005
    sound_duration = 0.5
    flim= 8 # None

    electric = True
    electric_scale = 'log_min_fft' # 'log_fft', 'lin_fft', 'greenwood' 
    normal = False
    data_dir = os.path.join(os.path.dirname(__file__), "data/SMRT/")
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cbar_bool = False

    ###########################################################################################################
    # Electric Hearing 
    if electric:
        print('Electric hearing:') 
        # output_data_dir =  './output/EH_SMRT/' 

        norm = None
        
        clim = []# (0, 1500)

        # fname_list = ['2023-11-24_16h33m43.98s spike_matrix_F120_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_1', # or 1.5?
        #               '2023-11-24_16h40m12.69s spike_matrix_F120_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_3']
        # fname_list = ['2025-03-25_09h53m37.29s spike_matrix_F120_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_1', # or 1.5?
        #             #   '2025-03-25_12h19m13.66s spike_matrix_F120_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_2',
        #               '2025-03-25_09h53m37.13s spike_matrix_F120_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_3']
        # fname_list = ['2025-04-01_16h28m47.23s spike_matrix_F120_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_1',
        #             #   '2025-04-01_16h28m40.58s spike_matrix_F120_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_2',
        #               '2025-04-01_12h47m53.12s spike_matrix_F120_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_3']
        fname_list = ['2025-04-14_15h46m8.97s spike_matrix_F120_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_1',
                        '2025-04-14_15h46m38.13s spike_matrix_F120_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_2',
                      '2025-04-14_15h46m38.13s spike_matrix_F120_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_3']
        fig, ax = plt.subplots(1,len(fname_list), figsize= (5*int(len(fname_list)),4))
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        plt.subplots_adjust(left=0.07, right=0.98, bottom=0.144 , top=0.92) # 
        axes = ax.flatten()

        for f_i, fname in enumerate(fname_list):
            print(fname)
            spike_rates_list = np.load(data_dir + fname + '.npy')
            time = fname[:fname.index('spike_matrix')-1]
            config_file = time + ' config_output.yaml'
            config_file = glob.glob(data_dir + time + ' config_output*.yaml')[0]
            [num_fibers, num_bins] = spike_rates_list.shape 
            with open(config_file, "r") as yamlfile: #'./config_test.yaml'
                config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            start_bin = config['binsize']
            binsize = config['binsize']

            # print('binsize:', binsize)
            T_levels, M_levels, _, TIa, TIb, Ln, Le, PW, Fn, Fe = load_mat_virtual_all_thresholds(os.path.join('./data', config['virtual_thresholds_file']), 
                                                                            nerve_model_type=config['nerve_model_type_AB'], 
                                                                            array_type=config['array_type_AB'], 
                                                                            state=config['nerve_state_AB']) # [A]

            
            if electric_scale == 'log_fft':
                frequency_list = np.load('./data/EH_freq_vector_electrode_allocation_logspaced.npy')*1e-3
                fiber_id_list = np.load('./data/fiber_ID_list_FFT.npy')
                spike_matrix = spike_rates_list[fiber_id_list,:]
                y_axis_str = 'Frequency [kHz]'# \n logspaced based on FFT '
            elif electric_scale == 'lin_fft':
                frequency_list = np.load('./data/EH_freq_vector_electrode_allocation_linspaced.npy')*1e-3
                fiber_id_list = np.load('./data/fiber_ID_list_FFT.npy')
                spike_matrix = spike_rates_list[fiber_id_list,:]
                y_axis_str = 'Frequency [kHz] \n linspaced based on FFT '
            elif electric_scale == 'greenwood':
                fiber_id_list = range(config['fiber_start'], config['fiber_end'], config['fiber_stepsize'])
                frequency_list = Fn[fiber_id_list]*1e-3
                y_axis_str = 'Frequency [kHz]'
                spike_matrix = spike_rates_list
            elif electric_scale == 'log_min_fft':
                frequency_list = np.load('./data/AB_MS_based_on_min_filtered_thresholdsfreq_x_fft.npy')*1e-3
                fiber_id_list = np.load('./data/AB_MS_based_on_min_filtered_thresholdsfiber_ID_list_FFT.npy')
                y_axis_str = 'Frequency [kHz]'
                spike_matrix = spike_rates_list[fiber_id_list,:]
            RPO = fname[fname.index('width_')+len('width_'):]
            [mesh, axe] = ax_colour_map_SMRT_per_kHz(axes[f_i], spike_matrix, frequency_list, sound_duration, y_axis_str, binsize=bin_size, clim=clim, norm=norm, flim=flim)
            axe.set_title(RPO + '.0 RPO', fontsize=font_size)
            axe.set_xlabel('Time [s]', fontsize=font_size)
            axes[f_i].text(-0.015, 0.1, labels[f_i], transform=axes[f_i].transAxes + trans,
                fontsize=18, verticalalignment='top', fontfamily='Open Sans', color='white')

            if 'p' in RPO:
                RPO.replace('p', '.')
            else:
                RPO + '.0'
            # plt.title(RPO + ' RPO')
        time_stamp = fname[:fname.find(' spike')]
        if cbar_bool:
            # standard
            # fig.colorbar(mesh, ax=axes.ravel().tolist())
            # adjust location to liking
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.14, 0.03, 0.78])
            fig.colorbar(mesh, cax=cbar_ax, label='Spike rate [spikes/s]')
            # cbar_ax.set_label('Spike rate [spikes/s]')
            cbar_str = 'Cbar'
        else:
            cbar_str = 'NoCbar'
        fig.savefig(save_dir + 'EH_cmap_' + cmap_type + '_clim_' + str(clim) + '_flim_' + str(flim) + '_binsize_' +str(bin_size) + '_norm_' + str(norm) + cbar_str +'_'+ electric_scale +'2416Fib111.6dB.png')

    ###########################################################################################################
    # Normal Hearing 
    if normal:
        print('Normal hearing:') 
        num_fibers = 2416
        dB = 50
        # data_dir =  './data/SMRT/' 
        if dB == 65:
            fname_list = ['PSTH_filter2023-11-24_17-42-22_SMRT_stimuli_C_dens_33_rate_5_depth_20_width_4_2847CFs',
                        'PSTH_filter2023-11-24_16-14-12_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_4_2847CFs',
                        'PSTH_filter2023-11-24_16-19-59_SMRT_stimuli_C_dens_33_rate_5_depth_20_width_16_2847CFs',
                        'PSTH_filter2023-11-24_16-24-11_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_16_2847CFs']
        if dB == 50: 
            if num_fibers == 2847:
                fname_list = ['PSTH_filter2024-04-02_16-57-24_SMRT_stimuli_C_dens_33_rate_5_depth_20_width_4_2847CFs',
                            'PSTH_filter2024-04-02_18-26-57_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_4_2847CFs',
                            'PSTH_filter2024-04-02_16-34-06_SMRT_stimuli_C_dens_33_rate_5_depth_20_width_16_2847CFs',
                            'PSTH_filter2024-04-02_16-56-58_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_16_2847CFs']
            elif num_fibers == 1903:
                fname_list = ['PSTH_filter2024-04-16_11-50-31_SMRT_stimuli_C_dens_33_rate_5_depth_20_width_4_1903CFs',
                            'PSTH_filter2024-04-16_11-52-24_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_4_1903CFs',
                            'PSTH_filter2024-04-16_11-52-40_SMRT_stimuli_C_dens_33_rate_5_depth_20_width_16_1903CFs',
                            'PSTH_filter2024-04-16_11-52-45_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_16_1903CFs']
            elif num_fibers == 2416:
                fname_list = ['PSTH_filter2025-03-24_15-53-11_SMRT_stimuli_C_dens_33_rate_5_depth_20_width_4_2416CFs',
                              'PSTH_filter2025-03-24_15-53-59_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_4_2416CFs',
                              'PSTH_filter2025-03-24_16-00-40_SMRT_stimuli_C_dens_33_rate_5_depth_20_width_16_2416CFs',
                              'PSTH_filter2025-03-24_16-03-26_SMRT_stimuli_C_dens_100_rate_5_depth_20_width_16_2416CFs']

        # clim=(2000, None)
        clim = []
        norm=None#matplotlib.colors.Normalize() #matplotlib.colors.PowerNorm(.75) #None
        sound_duration = 0.5
        # flim=None
        fig, ax = plt.subplots(2,2, figsize= (10,8))
        plt.subplots_adjust(left=0.059, bottom=0.063, right=0.98, top=0.92)
        trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
        # fig.set_size_inches((10, 8))
        axes = ax.flatten()
        # plt.tight_layout()
        for f_i, fname in enumerate(fname_list):
            print(fname)
           
            spikes_unfiltered, _, sound_name, _, fiber_frequencies, _, t_unfiltered, _ = load_mat_structs_Hamacher(data_dir + fname + '.mat')
            [num_fibers, num_bins_unfiltered] = spikes_unfiltered.shape 
            binsize_unfiltered = t_unfiltered[1]-t_unfiltered[0]
            spike_rates_list = rebin_spikes(spikes_unfiltered, binsize_unfiltered, bin_size)/bin_size
            [mesh, axe] = ax_colour_map_SMRT_per_kHz(axes[f_i], spike_rates_list, fiber_frequencies*1e-3, sound_duration, binsize=bin_size, clim=clim, norm=norm, flim=flim)
            RPO = fname[fname.index('width_')+len('width_'):fname.index('_' + str(num_fibers))]
            axes[f_i].text(-0.015, 0.1, labels[f_i], transform=axes[f_i].transAxes + trans,
                fontsize=16, verticalalignment='top', fontfamily='Open Sans', color='white')

            if f_i < 2:
                carrier = fname[fname.index('dens_')+len('dens_'):fname.index('_rate')]
                axe.set_title(carrier + ' carrier density \n' + RPO + '.0 RPO', fontsize=font_size)
                #    axe.set_ylabel(carrier + ' carrier density \n Greenwood frequency [kHz]', fontsize=font_size)
            else:
                axe.set_title(RPO + '.0 RPO', fontsize=font_size)
            if f_i > 1:
                axe.set_xlabel('Time [s]', fontsize=font_size)
            
            if norm:
                norm_str = str(norm)[str(norm).index('colors.')+len('colors.'):str(norm).index(' object')]
            else:
                norm_str = None

        if cbar_bool:
            # standard
            # fig.colorbar(mesh, ax=axes.ravel().tolist())
            # adjust location to liking
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.06, 0.03, 0.86])
            fig.colorbar(mesh, cax=cbar_ax, label='Spike rate [spikes/s]')
            # cbar_ax.set_label('Spike rate [spikes/s]')
            cbar_str = 'Cbar'
        else:
            cbar_str = 'NoCbar'
        fig.savefig(save_dir + 'NH_RPO4vs16_both_density_cmap_' + cmap_type + '_clim_' + str(clim) + '_flim_' + str(flim) + '_binsize_' + str(bin_size) + '_norm_' + str(norm_str) + 'NoTitle'+ cbar_str+ str(dB)+'dB' + str(num_fibers) + 'NumFib.png')

    plt.show()

