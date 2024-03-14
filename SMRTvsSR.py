import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt


def get_normalized_spectrum(fname, filter_bool=True, filter_order = 4, cut_off_freq = 100):
    if '.mat' in fname:
        unfiltered, _, _, _, fiber_frequencies, _, _, _ = load_mat_structs_Hamacher(fname, unfiltered_type = 'OG')
        normal_spectrum = (np.mean(unfiltered, axis=1)-np.min(np.mean(unfiltered, axis=1)))/(np.max(np.mean(unfiltered, axis=1))-np.min(np.mean(unfiltered, axis=1)))
        return normal_spectrum, fiber_frequencies
    if '.npy' in fname:
        spike_matrix = np.load(fname, allow_pickle=True)  
        spike_vector = np.mean(spike_matrix, axis=1)
        electric_spectrum = (spike_vector-np.min(spike_vector))/(np.max(spike_vector)-np.min(spike_vector))   
        if filter_bool:
            spike_vector2 = butter_lowpass_filter(spike_vector, cut_off_freq, len(spike_vector), filter_order) # Can't LP f because the Fs is not consistent
            electric_spectrum2 = (spike_vector2-np.min(spike_vector2))/(np.max(spike_vector2)-np.min(spike_vector2))
        else:
            electric_spectrum2 = None
        return electric_spectrum, electric_spectrum2

def double_spectrum_one_fig(RPO_list):
    fig, ax = plt.subplots(len(RPO_list), 2, figsize=(12, 4), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.079, bottom=0.11, right=0.967, top=0.929, wspace=0.105)
    bar_width = 15
    
    alpha = 0.2
    for RPO in RPO_list:
        # get EH
        fname_NH_SR = glob.glob(sr_data_dir + '*i1*' + RPO + '_1_*.mat')[0]
        print('SR:', fname_NH_SR)
        fname_NH_SMRT = glob.glob(SMRT_data_dir + '*width_*' + RPO + '*.mat')[0]
        print('SMRT:', fname_NH_SR)
        normal_spectrum_SR, fiber_frequencies = get_normalized_spectrum(fname_NH_SR) 
        normal_spectrum_SMRT, _ =  get_normalized_spectrum(fname_NH_SMRT)          
       
        #Spectral ripple
        plt.subplot(len(RPO_list), 2, 1)
        plt.bar(fiber_frequencies, normal_spectrum_SR, width=bar_width, alpha=alpha, color='blue')
        filter_sig_i = butter_lowpass_filter(normal_spectrum_SR, cut_off_freq, len(normal_spectrum_SR), filter_order)
        plt.plot(fiber_frequencies, filter_sig_i, color='blue', label='inverted')
        plt.xlim(min(fiber_frequencies), 1e4)
        plt.vlines(5000, 0, 1, colors='red')
        plt.ylim((0,1))
        plt.ylabel(RPO + ' RPO \n normalized \n spiking')
        plt.title('Spectral ripple')
        plt.xlabel('Frequency [Hz]')
        
        # SMRT
        plt.subplot(len(RPO_list), 2, 2)
        plt.bar(fiber_frequencies, normal_spectrum_SMRT, width=bar_width, alpha=alpha, color='blue')
        filter_sig_s = butter_lowpass_filter(normal_spectrum_SMRT, cut_off_freq, len(normal_spectrum_SMRT), filter_order)
        plt.plot(fiber_frequencies, filter_sig_s, color='blue', label='standard')
        plt.vlines(6500, 0, 1, colors='red')
        plt.title('SMRT')
        plt.xlabel('Frequency [Hz]')
        # plt.show()
        # x=3
       
    return fig





if __name__ == "__main__":

    sr_data_dir = './data/spectrum/'
    SMRT_data_dir = './data/SMRT/'

    filter_order = 4
    cut_off_freq = 100
    RPO = '4'

    fig = double_spectrum_one_fig(RPO)
    fig.savefig('./figures/SRvsSMRT_filtered' + '_'.join(RPO) + 'RPO.png')
    plt.show()

  