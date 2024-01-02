import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from pymatreader import read_mat

type_scaling_fibres = 'log'

# freq_x_fft = np.load('EH_freq_vector_electrode_allocation_'+ type_scaling_fibres + 'spaced.npy')
# freq_x_fft, fiber_id_electrode, half_electrode_range = create_EH_freq_vector_electrode_allocation(type_scaling_fibres)
# frequencies in FFT channels
edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]

# get fiber location and frequency from Randy's file
matfile = './data/Fidelity120 HC3A MS All Morphologies 18us CF.mat'
mat = read_mat(matfile)
m=0
# Greenwood frequency IDs
Fn = np.flip(mat['Df120']['Fn'][m])*1e3 # [Hz] 3200 fibers
Fe = np.flip(mat['Df120']['Fe'][m])*1e3 # [Hz] 16 electrodes

# basilar membrane IDs
Ln = np.flip(mat['Df120']['Ln'][m]) # [mm] 3200 fibers
Le = np.flip(mat['Df120']['Le'][m]) # [mm] 16 electrodes

# match fibers 
fiber_id_electrode = np.zeros(16)
for e, mm in enumerate(Le):
    fiber_id_electrode[e] = int(find_closest_index(Ln, mm) )

num_fibers_between_electrode = abs(np.diff(fiber_id_electrode))
half_electrode_range = int(np.mean(num_fibers_between_electrode)/2)
# 272 is the frequency edge of an FFT bin one step before
if type_scaling_fibres == 'log':
    freq_x_fft = list(np.logspace(np.log10(272), np.log10(edges[0]), half_electrode_range, base=10, endpoint=False)) 
elif type_scaling_fibres == 'lin':
    freq_x_fft = list(np.linspace(272, edges[0], half_electrode_range, endpoint=False)) 

for e in range(len(edges)-1):
    freq_range = (edges[e], edges[e+1])
    if type_scaling_fibres == 'log':
        freq_fft_band = list(np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), int(num_fibers_between_electrode[e]), base=10, endpoint=False))
    elif type_scaling_fibres == 'lin':
        freq_fft_band = list(np.linspace(freq_range[0], freq_range[1], int(num_fibers_between_electrode[e]), endpoint=False))
    freq_x_fft.extend(freq_fft_band)

band_to_pick = 0
alpha_05_bool = False
data_dir = './data/spectrum/'

type_phase = 'i1' #'i1' / 's'
if alpha_05_bool:
    RPO_list = ['0.500','1.414', '2.000', '2.828', '4.000'] # , '2.828'
else:
    RPO_list = ['0.500', '1.000', '1.414', '2.000', '2.828', '4.000'] # , '2.828'

for r_i, RPO in enumerate(RPO_list):
    print(RPO) 
    # get EH
    fname_EH_ = glob.glob(data_dir + '*matrix*_' + type_phase + '_*'  + RPO + '*.npy')
    if alpha_05_bool:
        alpha_save_str = '_alpha_0.5'
        fname_EH = [l for l in fname_EH_ if ('alpha' in l)][0]
    else:
        alpha_save_str = ''
        fname_EH = [l for l in fname_EH_ if ('alpha' not in l)][0]
    print(fname_EH)
    spike_matrix = np.load(fname_EH, allow_pickle=True)  
    fiber_band = spike_matrix[int(fiber_id_electrode[band_to_pick])]
