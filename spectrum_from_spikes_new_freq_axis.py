import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from pymatreader import read_mat
import matplotlib.transforms as mtransforms # labeling axes

# TO DO
# [X] Check new files if they are similar to online ones --> not exactly, but close

x_start = 250

edges = [306, 442, 578, 646, 782, 918, 1054, 1250, 1529, 1801, 2141, 2549, 3025, 3568, 4248, 8054] # [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]

freq_x_fft = np.load('./data/AB_MS_based_on_min_filtered_thresholdsfreq_x_fft.npy')
fiber_id_selection = np.load('./data/AB_MS_based_on_min_filtered_thresholdsfiber_ID_list_FFT.npy')

def get_normalized_spectrum(fname, filter_bool=True, filter_order = 4, cut_off_freq = 100):
    if '.mat' in fname:
        try:
            unfiltered, _, _, _, fiber_frequencies, _, _, _ = load_mat_structs_Hamacher(fname, unfiltered_type = 'OG')
        except:
            unfiltered, _, _, _, fiber_frequencies, _, _, _ = load_matrices_from_vectors_Bruce_struct(fname)
        normal_spectrum = (np.mean(unfiltered, axis=1)-np.min(np.mean(unfiltered, axis=1)))/(np.max(np.mean(unfiltered, axis=1))-np.min(np.mean(unfiltered, axis=1)))
        return normal_spectrum, fiber_frequencies
    if '.npy' in fname:
        spike_matrix = np.load(fname, allow_pickle=True)  
        spike_vector = np.mean(spike_matrix, axis=1)
        electric_spectrum = (spike_vector-np.min(spike_vector))/(np.max(spike_vector)-np.min(spike_vector))   
        if filter_bool:     
            electric_spectrum2 = butter_lowpass_filter(electric_spectrum, cut_off_freq, len(spike_vector), filter_order) # Can't LP f because the Fs is not consistent
            # electric_spectrum2 = (spike_vector2-np.min(spike_vector2))/(np.max(spike_vector2)-np.min(spike_vector2))
        else:
            electric_spectrum2 = None
        return electric_spectrum, electric_spectrum2

def create_single_spectrum(normal_spectrum, electric_spectrum, fiber_frequencies, filter_bool, electric_spectrum2=None, fname_NH='', fname_EH=''):
    fig = plt.figure()
    bar_width = 15
    alpha = 0.2
    #NH
    plt.subplot(2,1,1)
    if filter_bool: # show spiking and filtered spiking
        plt.bar(fiber_frequencies, normal_spectrum, width=bar_width, alpha=alpha)
        filter_sig = butter_lowpass_filter(normal_spectrum, cut_off_freq, len(normal_spectrum), filter_order)
        plt.plot(fiber_frequencies, filter_sig)
    else:
        plt.plot(fiber_frequencies, normal_spectrum)
    plt.ylabel('normalized spiking NH')
    plt.ylim((0,1))
    # match NH x-axis
    plt.xlim((x_start, np.max(edges)))
    plt.title(fname_NH)
    
    #EH
    plt.subplot(2,1,2)
    if filter_bool:
        plt.bar(freq_x_fft, electric_spectrum[fiber_id_selection[0]:fiber_id_selection[-1]+1], width=bar_width, alpha=alpha)
        plt.plot(freq_x_fft, electric_spectrum2[fiber_id_selection[0]:fiber_id_selection[-1]+1])
    else:
        plt.plot(freq_x_fft, electric_spectrum[fiber_id_selection[0]:fiber_id_selection[-1]+1])
    plt.xlim((x_start, np.max(edges)))
    plt.vlines(edges, 0, 1.1, color='k')
    plt.ylabel('normalized spiking EH')
    plt.ylim((0,1))
    plt.xlabel('Frequency [Hz]')
    plt.title(fname_EH)
    return fig

def create_double_spectrum(normal_spectrum_i, 
                           normal_spectrum_s, 
                           electric_spectrum_i, 
                           electric_spectrum_s, 
                           fiber_frequencies, 
                           filter_bool, 
                           electric_spectrum2_i=None,
                           electric_spectrum2_s=None, 
                           vlines_nh=False,
                           vlines_eh=False):
    fig = plt.figure()
    bar_width = 15
    # vlines_nh = True
    alpha = 0.2
    #NH
    plt.subplot(2,1,1)
    if filter_bool:
        plt.bar(fiber_frequencies, normal_spectrum_i, width=bar_width, alpha=alpha, color='orange')
        plt.bar(fiber_frequencies, normal_spectrum_s, width=bar_width, alpha=alpha, color='blue')
        filter_sig_i = butter_lowpass_filter(normal_spectrum_i, cut_off_freq, len(normal_spectrum_i), filter_order)
        filter_sig_s = butter_lowpass_filter(normal_spectrum_s, cut_off_freq, len(normal_spectrum_s), filter_order)
        plt.plot(fiber_frequencies, filter_sig_i, color='orange', label='i')
        plt.plot(fiber_frequencies, filter_sig_s, color='blue', label='s')
        plt.ylim((0,1))
    else:
        plt.plot(fiber_frequencies, normal_spectrum_i, label='i')
        plt.plot(fiber_frequencies, normal_spectrum_s, label='s')
    plt.ylabel('normalized spiking NH')
    plt.legend()
    plt.ylim((0,1))
    if vlines_nh:
        plt.vlines(edges, 0, 1.1, color='k')
        
    # match NH x-axis
    plt.xlim((x_start, np.max(edges)))
    
    #EH
    plt.subplot(2,1,2)
    plt.legend()
    if filter_bool:
        plt.bar(freq_x_fft, electric_spectrum_i[fiber_id_selection[0]:fiber_id_selection[-1]+1], width=bar_width, alpha=alpha, color='orange')
        plt.bar(freq_x_fft, electric_spectrum_s[fiber_id_selection[0]:fiber_id_selection[-1]+1], width=bar_width, alpha=alpha, color='blue')
        plt.plot(freq_x_fft, electric_spectrum2_i[fiber_id_selection[0]:fiber_id_selection[-1]+1], color='orange', label='i')
        plt.plot(freq_x_fft, electric_spectrum2_s[fiber_id_selection[0]:fiber_id_selection[-1]+1], color='blue', label='s')
    else:
        plt.plot(freq_x_fft, electric_spectrum_i[fiber_id_selection[0]:fiber_id_selection[-1]+1], label='i', color='orange')
        plt.plot(freq_x_fft, electric_spectrum_s[fiber_id_selection[0]:fiber_id_selection[-1]+1], label='s', color='blue')
    plt.xlim((x_start, np.max(edges)))
    plt.legend()
    if vlines_eh:
        plt.vlines(edges, 0, 1.1, color='k')
    plt.ylim((0,1))
    plt.ylabel('normalized spiking EH')
    plt.xlabel('Frequency [Hz]')
    return fig


def double_spectrum_one_fig(RPO_list,
                           vlines_nh=False,
                           vlines_eh=False,
                           octave_spaced=False,
                           filter_type='butter'):
    fig, axes = plt.subplots(len(RPO_list), 2, figsize=(12, 9))
    plt.subplots_adjust(left=0.079, bottom=0.062, right=0.98, top=0.96, hspace=0.3)
    bar_width = 15
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    alpha = 0.2
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    rr = 1
    axs=axes.flatten()
    for RPO in RPO_list:
        # get NH
        fname_NH_i = glob.glob(data_dir + '*i1*' + RPO + '*.mat')[0]
        fname_NH_s = glob.glob(data_dir + '*_s_*' + RPO + '*.mat')[0]
        normal_spectrum_i, fiber_frequencies = get_normalized_spectrum(fname_NH_i) 
        normal_spectrum_s, _ =  get_normalized_spectrum(fname_NH_s) 
                
        # get EH
        fname_EH_i_ = glob.glob(data_dir + '2025-04*matrix*i1*'  + RPO + '*.npy')
        fname_EH_s_ = glob.glob(data_dir + '2025-04*matrix*_s_*' + RPO + '*.npy')
        # if alpha_05_bool:
            # fname_EH_i = [l for l in fname_EH_i_ if ('alpha' in l)][0]
            # fname_EH_s = [l for l in fname_EH_s_ if ('alpha' in l)][0]
        # else:
        fname_EH_i = [l for l in fname_EH_i_ if ('alpha' not in l)][0]
        fname_EH_s = [l for l in fname_EH_s_ if ('alpha' not in l)][0]
        electric_spectrum_i, electric_spectrum2_i = get_normalized_spectrum(fname_EH_i, filter_bool)
        electric_spectrum_s, electric_spectrum2_s = get_normalized_spectrum(fname_EH_s, filter_bool)
        if filter_type == 'mavg':
            electric_spectrum2_i = symmetric_moving_average(electric_spectrum_i, window_size=window_size)
            electric_spectrum2_s = symmetric_moving_average(electric_spectrum_s, window_size=window_size)

        electric_spectrum_i = electric_spectrum_i[fiber_id_selection[0]:fiber_id_selection[-1]+1]
        electric_spectrum_s = electric_spectrum_s[fiber_id_selection[0]:fiber_id_selection[-1]+1]
        electric_spectrum2_i = electric_spectrum2_i[fiber_id_selection[0]:fiber_id_selection[-1]+1]
        electric_spectrum2_s = electric_spectrum2_s[fiber_id_selection[0]:fiber_id_selection[-1]+1]

        #NH
        plt.subplot(len(RPO_list), 2, rr)
        print(rr)
        if octave_spaced:
            plt.xscale('log', base=2)
            plt.xticks([x_start, 500, 1000, 2000, 4000, 8000], labels=[str(x_start), '500', '1000', '2000', '4000', '8000'])
            bar_width = 8
        plt.bar(fiber_frequencies, normal_spectrum_i, width=bar_width, alpha=alpha, color=color_i)
        plt.bar(fiber_frequencies, normal_spectrum_s, width=bar_width, alpha=alpha, color=color_s)
        if filter_type == 'butter':
            filter_sig_i = butter_lowpass_filter(normal_spectrum_i, cut_off_freq, len(normal_spectrum_i), filter_order)
            filter_sig_s = butter_lowpass_filter(normal_spectrum_s, cut_off_freq, len(normal_spectrum_s), filter_order)
        elif filter_type == 'mavg':
            filter_sig_i = symmetric_moving_average(normal_spectrum_i, window_size=window_size)
            filter_sig_s = symmetric_moving_average(normal_spectrum_s, window_size=window_size)
        plt.plot(fiber_frequencies, filter_sig_i, color=color_i, label='inverted')
        plt.plot(fiber_frequencies, filter_sig_s, color=color_s, label='standard')
        plt.ylim((0,1))
        while RPO[-2:] == '00':
            RPO = RPO[:-2]
        plt.ylabel(RPO + ' RPO \n normalized \n spiking')
        plt.legend()
        plt.ylim((0,1))
        if rr == 2*len(RPO_list)-1:
            plt.xlabel('Frequency [Hz]')
        if vlines_nh:
            plt.vlines(edges, 0, 1.1, color='k')
        # match NH x-axis
        plt.xlim((x_start, np.max(edges)))
        # plt.title('NH:' + RPO)
        if rr == 1:
            plt.title('Normal hearing')
        axs[rr-1].text(-0.015, 1, labels[rr-1], transform=axs[rr-1].transAxes + trans,
                fontsize=16, verticalalignment='top', fontfamily='Open Sans', color='black')
                # bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
        # np.save('./data/spectrum/NH_fiber_freqs.npy', fiber_frequencies)
        # np.save('./data/spectrum/NH_i1_'+ RPO +'_1_spectrum.npy', normal_spectrum_i)
        # np.save('./data/spectrum/NH_s_'+ RPO +'_1_spectrum.npy', normal_spectrum_s)
        # np.save('./data/spectrum/NH__filtered_i1_'+ RPO +'_1_spectrum.npy', filter_sig_i)
        # np.save('./data/spectrum/NH__filtered_s_'+ RPO +'_1_spectrum.npy', filter_sig_s)
        rr += 1
        #EH
        plt.subplot(len(RPO_list), 2, rr)
        print(rr)
        plt.legend()
        if octave_spaced:
            plt.xscale('log', base=2)
            plt.xticks([x_start, 500, 1000, 2000, 4000, 8000], labels=[str(int(x_start)), '500', '1000', '2000', '4000', '8000'])
        plt.bar(freq_x_fft, electric_spectrum_i, width=bar_width, alpha=alpha, color=color_i)
        plt.bar(freq_x_fft, electric_spectrum_s, width=bar_width, alpha=alpha, color=color_s)
        plt.plot(freq_x_fft, electric_spectrum2_i, color=color_i, label='inverted')
        plt.plot(freq_x_fft, electric_spectrum2_s, color=color_s, label='standard')
        plt.ylabel(RPO + ' RPO \n normalized \n spiking')
        plt.xlim((x_start, np.max(edges)))
        plt.legend()
        if vlines_eh:
            plt.vlines(edges, 0, 1.1, color='k')
        plt.ylim((0,1))
        # plt.ylabel('normalized spiking EH')
        # plt.title('EH:' + RPO)
        if rr == 2*len(RPO_list):
            plt.xlabel('Frequency [Hz]')
        if rr == 2:
            plt.title('Electric hearing')
        axs[rr-1].text(-0.015, 1, labels[rr-1], transform=axs[rr-1].transAxes + trans,
                fontsize=16, verticalalignment='top', color='black')
        rr += 1 
        # np.save('./data/spectrum/EH_fiber_freqs.npy', freq_x_fft)
        # np.save('./data/spectrum/EH_i1_'+ RPO +'_1_spectrum.npy', electric_spectrum_i)
        # np.save('./data/spectrum/EH_s_'+ RPO +'_1_spectrum.npy', electric_spectrum_s)
        # np.save('./data/spectrum/EH__filtered_i1_'+ RPO +'_1_spectrum.npy', electric_spectrum2_i)
        # np.save('./data/spectrum/EH__filtered_s_'+ RPO +'_1_spectrum.npy', electric_spectrum2_s)
    x=4
    return fig


def CS_off_vs_on(alpha_i, alpha_s, alpha2_i, alpha2_s, # alpha = 0.5
                 CS_i, CS_s, CS2_i, CS2_s, # current steering on
                 filter_bool=True,
                 vlines = False, 
                 v_ellips = True,
                 octave_spaced=False):
    fig = plt.figure()
    bar_width = 15
    electrode_width = bar_width*3
    alpha_bar_width = bar_width/2
    alpha = 0.10
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    #CS
    ax=plt.subplot(2,1,1)
    ax.text(-0.015, 1, 'A', transform=ax.transAxes + trans,
                fontsize=16, verticalalignment='top', color='black')
    if filter_bool:
        # individual bars
        plt.bar(freq_x_fft, CS_i[fiber_id_selection[0]:fiber_id_selection[-1]+1], width=bar_width, alpha=alpha, color=color_i)
        plt.bar(freq_x_fft, CS_s[fiber_id_selection[0]:fiber_id_selection[-1]+1], width=bar_width, alpha=alpha, color=color_s)
        # filtered lines
        plt.plot(freq_x_fft, CS2_i[fiber_id_selection[0]:fiber_id_selection[-1]+1], color=color_i, label='inverted')
        plt.plot(freq_x_fft, CS2_s[fiber_id_selection[0]:fiber_id_selection[-1]+1], color=color_s, label='standard')
    else:
        plt.plot(freq_x_fft, CS_i[fiber_id_selection[0]:fiber_id_selection[-1]+1], label='i', color=color_i)
        plt.plot(freq_x_fft, CS_s[fiber_id_selection[0]:fiber_id_selection[-1]+1], label='s', color=color_s)
    plt.ylabel('CS (F120) \n normalized spiking')
    plt.legend()
    plt.ylim((0,1))
    if octave_spaced:
        plt.xscale('log', base=2)
        plt.xticks([x_start, 500, 1000, 2000, 4000, 8000], labels=[str(int(x_start)), '500', '1000', '2000', '4000', '8000'])
    if vlines:
        plt.vlines(edges, 0, 1.1, color='k')
    if v_ellips:
        from matplotlib.patches import Ellipse
        xx=1
        for x_i, x in enumerate(edges):
            ax.add_patch(Ellipse((x,0), bar_width*xx, 0.15, color=color_e))
            if x_i>=1:
                section = (edges[x_i] - edges[x_i-1])/8
                for b in range(1,8):
                    x_between = edges[x_i-1]+b*section
                    # if x_i==15 and b==6:
                        # breakpoint()
                    # ax.add_patch(Ellipse((x_between,0), alpha_bar_width, 0.1, color=color_e))
            xx+=1
    # match NH x-axis
    plt.xlim((x_start, np.max(edges)))
    
    #CS off
    ax = plt.subplot(2,1,2)
    ax.text(-0.015, 1, 'B', transform=ax.transAxes + trans,
                fontsize=16, verticalalignment='top', color='black')
    plt.legend()
    if filter_bool:
        plt.bar(freq_x_fft, alpha_i[fiber_id_selection[0]:fiber_id_selection[-1]+1], width=bar_width, alpha=alpha, color=color_i)
        plt.bar(freq_x_fft, alpha_s[fiber_id_selection[0]:fiber_id_selection[-1]+1], width=bar_width, alpha=alpha, color=color_s)
        plt.plot(freq_x_fft, alpha2_i[fiber_id_selection[0]:fiber_id_selection[-1]+1], color=color_i, label='inverted')
        plt.plot(freq_x_fft, alpha2_s[fiber_id_selection[0]:fiber_id_selection[-1]+1], color=color_s, label='standard')
    else:
        plt.plot(freq_x_fft, alpha_i[fiber_id_selection[0]:fiber_id_selection[-1]+1], label='i', color=color_i)
        plt.plot(freq_x_fft, alpha_s[fiber_id_selection[0]:fiber_id_selection[-1]+1], label='s', color=color_s)
    plt.xlim((x_start, np.max(edges)))
    plt.legend()
    if vlines:
        plt.vlines(edges, 0, 1.1, color='k')
    plt.ylim((0,1))
    plt.ylabel('CS off \n normalized spiking')
    # plt.title('Current steering off')
    plt.xlabel('Frequency [Hz]')
    if octave_spaced:
        plt.xscale('log', base=2)
        plt.xticks([x_start, 500, 1000, 2000, 4000, 8000], labels=[str(int(x_start)), '500', '1000', '2000', '4000', '8000'])
    if v_ellips:
        xx = 1
        for x in range(len(edges)-1):
            ax.add_patch(Ellipse(((edges[x]+edges[x+1])/2,0), bar_width*xx, 0.15, color=color_e))
            xx+=1
    x=3
    return fig



if __name__ == "__main__":

    dB = 65
    # data_dir = './data/spectrum/' + str(dB) + 'dB/'
    data_dir = './data/spectrum/' + str(dB) + 'dB_2416CF/'
    # pick figure
    double_spectrum_bool = False # show i1 AND s in one fig per RPO
    single_spectrum_bool = False
    double_spectrum_one_fig_bool = True # show i1 AND s in all RPO in one fig
    versus_alpha = True # 2.828 CS vs CS off
    critical_bands_fig = False
    octave_spaced = True 
    if octave_spaced:
        octave_str = 'octave_spaced'
    else:
        octave_str = ''

    # fig characteristics
    filter_bool = True # filter spike spectrum
    alpha_05_bool = False # use EH with sort of CS off, always the peak in the middle of the electrodes
    filter_type = 'mavg' # 'mavg' / 'butter'
    vlines_nh = False
    vlines_eh = False
    v_ellips = True
    filter_order = 4
    cut_off_freq = 100
    window_size = 33 # for moving average filter
    # for single spectrum
    type_phase = 'i1' #'i1' / 's'
    color_s = 'blue'
    color_i = 'red'
    color_e = 'orange'
    # save strings
    if vlines_nh:
        vline_nh_str = 'wVlinesNH'
    else:
        vline_nh_str = ''

    # save strings
    if vlines_eh:
        vline_eh_str = 'wVlinesEH'
    else:
        vline_eh_str = ''

    if v_ellips:
        v_ellips_str = 'wEllipsEH'
    else:
        v_ellips_str = ''

    if filter_bool:
        filter_str = 'filtered'
    else:
        filter_str = 'notfiltered'

    # RPO's to plot
    # if alpha_05_bool:
    RPO_list = ['0.500','1.414', '2.000', '4.000'] # , '2.828'
    # else:
    #     RPO_list = ['0.500', '1.000', '1.414', '2.000', '2.828', '4.000'] # , '2.828'

    if critical_bands_fig:
        critical_band_type = 'slim'
        bin_size = 0.005
        # for r_i, RPO in enumerate(RPO_list):
        #     # NH
        #     # Spectral ripple 
        #     fname_NH_i = glob.glob(data_dir + '*_i1_*' + RPO + '*.mat')[0]
        #     fname_NH_s = glob.glob(data_dir + '*_s_*' + RPO + '*.mat')[0]
        #     print(fname_NH_i, fname_NH_s)
        #     fig = plot_fig_critical_bands([fname_NH_i, fname_NH_s], critical_band_type, bin_size)
        #     plt.suptitle('NH with spectral ripples ('+  str(RPO) + ' RPO) on '+ critical_band_type + ' scale', fontsize=20)
        #     fig.savefig('./figures/SR/criticalbands_NH_' + critical_band_type +'scale_binsize_' +str(bin_size) + 's_'+ str(RPO) +'density.png')
    
        #     # EH
        #     fname_EH_i_ = glob.glob(data_dir + '*matrix*i1*'  + RPO + '*.npy')[0]
        #     fname_EH_s_ = glob.glob(data_dir + '*matrix*_s_*' + RPO + '*.npy')[0]
        #     print(fname_EH_i_, fname_EH_s_)
        #     fig = plot_fig_critical_bands([fname_EH_i_, fname_EH_s_], critical_band_type, bin_size)
        #     plt.suptitle('EH with spectral ripples ('+  str(RPO) + ' RPO) on '+ critical_band_type + ' scale', fontsize=20)
        #     fig.savefig('./figures/SR/criticalbands_EH_' + critical_band_type +'scale_binsize_' +str(bin_size) + 's_'+ str(RPO) +'density.png')
        
        # SMRT
        carrier = '100'
        for RPO in ['1', '4', '16']:
            fname_RT = glob.glob('./data/SMRT/*dens_' + carrier + '*width_' + RPO + '_1903CFs.mat')[0]
            fname_R = glob.glob('./data/SMRT/*dens_' + carrier + '*width_20*1903CFs.mat')[0]
            print(fname_RT, fname_R)
            fig = plot_fig_critical_bands([fname_RT, fname_R], critical_band_type, bin_size)
            plt.suptitle('NH with SMRT ('+  str(RPO) + ' RPO) on '+ critical_band_type + ' scale', fontsize=20)
            # fig.savefig('./figures/SMRT/criticalbands_NH_' + critical_band_type +'scale_binsize_' +str(bin_size) + 's_'+ str(RPO) +'density.png')

        for RPO in ['1', '3']:
            fname_RT = glob.glob('./data/SMRT/*dens_' + carrier + '*width_' + RPO + '.npy')[0]
            fname_R = glob.glob('./data/SMRT/*dens_' + carrier + '*width_20.npy')[0]
            print(fname_RT, fname_R)
            fig = plot_fig_critical_bands([fname_RT, fname_R], critical_band_type, bin_size)
            plt.suptitle('EH with SMRT ('+  str(RPO) + ' RPO) on '+ critical_band_type + ' scale', fontsize=20)
            # fig.savefig('./figures/SMRT/criticalbands_EH_' + critical_band_type +'scale_binsize_' +str(bin_size) + 's_'+ str(RPO) +'density.png')


    if double_spectrum_one_fig_bool:
        fig = double_spectrum_one_fig(RPO_list, vlines_nh=vlines_nh, vlines_eh=vlines_eh, octave_spaced=octave_spaced, filter_type=filter_type)
        fig.savefig('./figures/spectrum/EH_NH_onefig_'+ filter_type +'filteredNewFreqAxis_'+ vline_nh_str + vline_eh_str + octave_str + '_'.join(RPO_list) + 'RPO_'+ str(dB)+'dB_111.6dB'+ color_s + color_i +'.png')

    if single_spectrum_bool or double_spectrum_bool:
        for r_i, RPO in enumerate(RPO_list):
            print(RPO) 
            if double_spectrum_bool:
                # get NH
                fname_NH_i = glob.glob(data_dir + '*i1*' + RPO + '*.mat')[0]
                fname_NH_s = glob.glob(data_dir + '*_s_*' + RPO + '*.mat')[0]
                normal_spectrum_i, fiber_frequencies = get_normalized_spectrum(fname_NH_i) 
                normal_spectrum_s, _ =  get_normalized_spectrum(fname_NH_s) 
                
                # get EH
                fname_EH_i_ = glob.glob(data_dir + '*matrix*i1*'  + RPO + '*.npy')
                fname_EH_s_ = glob.glob(data_dir + '*matrix*_s_*' + RPO + '*.npy')
                if alpha_05_bool:
                    alpha_save_str = '_alpha_0.5'
                    fname_EH_i = [l for l in fname_EH_i_ if ('alpha' in l)][0]
                    fname_EH_s = [l for l in fname_EH_s_ if ('alpha' in l)][0]
                else:
                    alpha_save_str = ''
                    fname_EH_i = [l for l in fname_EH_i_ if ('alpha' not in l)][0]
                    fname_EH_s = [l for l in fname_EH_s_ if ('alpha' not in l)][0]
                electric_spectrum_i, electric_spectrum2_i = get_normalized_spectrum(fname_EH_i, filter_bool)
                electric_spectrum_s, electric_spectrum2_s = get_normalized_spectrum(fname_EH_s, filter_bool)

                fig = create_double_spectrum(normal_spectrum_i, 
                                            normal_spectrum_s, 
                                            electric_spectrum_i, 
                                            electric_spectrum_s, 
                                            fiber_frequencies, 
                                            filter_bool, 
                                            electric_spectrum2_i, 
                                            electric_spectrum2_s,
                                            vlines_nh=vlines_nh)
                plt.suptitle('Ripple density: '+ RPO + ' RPO')
                # fig.savefig('./figures/spectrum/' + filter_str + '_' + type_scaling_fibres + '_both_' + RPO + alpha_save_str + vline_nh_str + vline_eh_str + '.png')
            if single_spectrum_bool:
                # get NH
                fname_NH = glob.glob(data_dir + '*_' +type_phase +'_*' + RPO + '*.mat')[0]
                print(fname_NH)
                normal_spectrum, fiber_frequencies = get_normalized_spectrum(fname_NH)#(np.mean(unfiltered, axis=1)-np.min(np.mean(unfiltered, axis=1)))/(np.max(np.mean(unfiltered, axis=1))-np.min(np.mean(unfiltered, axis=1)))
                
                # get EH
                fname_EH_ = glob.glob(data_dir + '*matrix*_' + type_phase + '_*'  + RPO + '*.npy')
                if alpha_05_bool:
                    alpha_save_str = '_alpha_0.5'
                    fname_EH = [l for l in fname_EH_ if ('alpha' in l)][0]
                else:
                    alpha_save_str = ''
                    fname_EH = [l for l in fname_EH_ if ('alpha' not in l)][0]
                print(fname_EH)
                electric_spectrum, electric_spectrum2 = get_normalized_spectrum(fname_EH, filter_bool)
                
                fig = create_single_spectrum(normal_spectrum, electric_spectrum, fiber_frequencies, filter_bool, electric_spectrum2=electric_spectrum2)        
                plt.suptitle('Ripple density: '+ type_phase + '_'  + RPO + ' RPO')
                # fig.savefig('./figures/spectrum/' + filter_str + '_' + type_scaling_fibres + '_' + type_phase + '_' + RPO + alpha_save_str +'.png')

    if versus_alpha:
        RPO = '2.828'
        # get all EH files
        fname_EH_i_ = glob.glob(data_dir + '2025-04*matrix*i1*'  + RPO + '*.npy')
        fname_EH_s_ = glob.glob(data_dir + '2025-04*matrix*_s_*' + RPO + '*.npy')
        # no CS (alpha=0.5)
        fname_EH_i_alpha05 = [l for l in fname_EH_i_ if ('alpha' in l)][0]
        fname_EH_s_alpha05 = [l for l in fname_EH_s_ if ('alpha' in l)][0]
        # CS
        fname_EH_i_CS = [l for l in fname_EH_i_ if ('alpha' not in l)][0]
        fname_EH_s_CS = [l for l in fname_EH_s_ if ('alpha' not in l)][0]
        
        electric_spectrum_i_alpha05, electric_spectrum2_i_alpha05 = get_normalized_spectrum(fname_EH_i_alpha05, filter_bool)
        electric_spectrum_s_alpha05, electric_spectrum2_s_alpha05 = get_normalized_spectrum(fname_EH_s_alpha05, filter_bool)
        electric_spectrum_i_CS, electric_spectrum2_i_CS = get_normalized_spectrum(fname_EH_i_CS, filter_bool)
        electric_spectrum_s_CS, electric_spectrum2_s_CS = get_normalized_spectrum(fname_EH_s_CS, filter_bool)
        if filter_type == 'mavg':
            electric_spectrum2_i_alpha05 = symmetric_moving_average(electric_spectrum_i_alpha05, window_size=window_size)
            electric_spectrum2_s_alpha05 = symmetric_moving_average(electric_spectrum_s_alpha05, window_size=window_size)
            electric_spectrum2_i_CS = symmetric_moving_average(electric_spectrum_i_CS, window_size=window_size) 
            electric_spectrum2_s_CS = symmetric_moving_average(electric_spectrum_s_CS, window_size=window_size)

        squared_difference_CS = sum((electric_spectrum_s_CS-electric_spectrum_i_CS)**2)
        squared_difference_alpha = sum((electric_spectrum_s_alpha05-electric_spectrum_i_alpha05)**2)
        print('ALL error CS:', squared_difference_CS)
        print('ALL error CS off:', squared_difference_alpha)

        squared_difference_CS2 = sum((electric_spectrum2_s_CS-electric_spectrum2_i_CS)**2)
        squared_difference_alpha2 = sum((electric_spectrum2_s_alpha05-electric_spectrum2_i_alpha05)**2)
        print('ALL filtered error CS:', squared_difference_CS2)
        print('ALL filtered error CS off:', squared_difference_alpha2)

        squared_difference_CS_in_fig = sum((electric_spectrum_s_CS[fiber_id_selection[0]:fiber_id_selection[-1]+1]-electric_spectrum_i_CS[fiber_id_selection[0]:fiber_id_selection[-1]+1])**2)
        squared_difference_alpha_in_fig = sum((electric_spectrum_s_alpha05[fiber_id_selection[0]:fiber_id_selection[-1]+1]-electric_spectrum_i_alpha05[fiber_id_selection[0]:fiber_id_selection[-1]+1])**2)
        print('in fig error CS:', squared_difference_CS_in_fig)
        print('in fig error CS off:', squared_difference_alpha_in_fig)

        squared_difference_CS_in_fig2 = sum((electric_spectrum2_s_CS[fiber_id_selection[0]:fiber_id_selection[-1]+1]-electric_spectrum2_i_CS[fiber_id_selection[0]:fiber_id_selection[-1]+1])**2)
        squared_difference_alpha_in_fig2 = sum((electric_spectrum2_s_alpha05[fiber_id_selection[0]:fiber_id_selection[-1]+1]-electric_spectrum2_i_alpha05[fiber_id_selection[0]:fiber_id_selection[-1]+1])**2)
        print('in fig filtered error CS:', squared_difference_CS_in_fig2)
        print('in fig filtered error CS off:', squared_difference_alpha_in_fig2)

        fig = CS_off_vs_on(alpha_i=electric_spectrum_i_alpha05, alpha_s=electric_spectrum_s_alpha05, alpha2_i=electric_spectrum2_i_alpha05, alpha2_s=electric_spectrum2_s_alpha05,
                 CS_i=electric_spectrum_i_CS, CS_s=electric_spectrum_s_CS, CS2_i=electric_spectrum2_i_CS, CS2_s=electric_spectrum2_s_CS,
                 filter_bool=filter_bool, vlines=vlines_eh, v_ellips=v_ellips, octave_spaced=octave_spaced)
        plt.suptitle('Ripple density: ' + RPO + ' RPO')
        fig.savefig('./figures/spectrum/CSvsCSoff_' + filter_str + '_full_labels_' + filter_type + 'filteredNewFreqAxis_' + v_ellips_str + RPO + 'RPO'+ str(dB)+'dB_111.6dB'+ color_s + color_i + octave_str + '.png')
    plt.show()