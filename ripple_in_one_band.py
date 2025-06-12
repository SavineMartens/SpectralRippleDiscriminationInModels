import numpy as np
from utilities import *
import glob
import matplotlib.pyplot as plt
from pymatreader import read_mat

# SMRT!!!!
# [ ] add smrt data to folder?
# [ ] get STRIPES files



# frequencies in FFT channels
edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]

# get fiber location and frequency from Randy's file
matfile = './data/Fidelity120 HC3A MS All Morphologies 18us CF.mat'
mat = read_mat(matfile)
m=0

# basilar membrane IDs
Ln = np.flip(mat['Df120']['Ln'][m]) # [mm] 3200 fibers
Le = np.flip(mat['Df120']['Le'][m]) # [mm] 16 electrodes

# match fibers 
fiber_id_electrode = np.zeros(16)
for e, mm in enumerate(Le):
    fiber_id_electrode[e] = int(find_closest_index(Ln, mm) )

data_dir = './data/SMRT/'

RPO_list = np.arange(1, 3)

for r_i, RPO in enumerate(RPO_list):
    fig, ax = plt.subplots(3, 5, sharex=True, figsize=(16, 15))   
    axes = ax.flatten()
    plt.suptitle('SMRT RPO: ' + str(RPO))
    for band in range(len(edges)-1):
        print(RPO) 
        # get EH
        fname_EH_ = glob.glob(data_dir + '*matrix*_100*width_'  + str(RPO) + '*.npy')[0]
        print(fname_EH_)
        spike_matrix = np.load(fname_EH_, allow_pickle=True)  
        fiber_band = spike_matrix[int(fiber_id_electrode[band+1]):int(fiber_id_electrode[band]), :]
        band_vector = np.sum(fiber_band, axis=0)

        # plt.figure()
        axes[band].plot(band_vector)
        axes[band].set_title('FFT band: ' + str(band) + ' (' + str(edges[band]) + '-'  + str(edges[band+1]) + 'Hz)' )
    # plt.savefig('./figures/SMRT/Temporal_depiction_per_FFT_band_RPO' + str(RPO) + '.png')
plt.show()
