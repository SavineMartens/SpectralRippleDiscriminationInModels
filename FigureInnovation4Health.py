import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from scipy.io import wavfile
from matplotlib.patches import Rectangle

edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]
frequency = np.load('./data/spectrum/frequency_vector_FFT.npy')
fig, ax = plt.subplots(ncols=3, nrows=4, sharey=True, figsize=(18,15))
plt.subplots_adjust(hspace=0.152, wspace=0.076)

name_list = ['s_0.500_1', 'i1_0.500_1', 's_2.000_1', 'i1_2.000_1', 's_4.000_1', 'i1_4.000_1']
iterator = 0
bar_width = 15
xlim0 = 340 # 272
fontsize = 16
fontsize_label = 14

alpha = 0.2
for i_n, name in enumerate(name_list):
    previous1 = 0
    previous2 = 0   
    ################# ACOUSTIC
    outline = np.load('./data/spectrum/Poster/normalized_FFT_bins_'+ name + '.npy')
    # winn plot
    if  name[0] == 's':
        color = 'deepskyblue'
        label = 'standard'
    elif name[0] == 'i':
        color = 'black'
        label = 'inverted'
    if i_n !=0 and i_n%2 == 0:
        iterator += 1
    ax[0,iterator].plot(frequency, outline, color = color, label=label)
    ax[0,iterator].set_xscale('log', base=2)
    ax[0,iterator].set_xlim((200, 8700))
    ax[0,iterator].set_ylim((0, 1))
    ax[0,iterator].set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
    edges = [340, 476, 612, 680, 816, 952, 1088, 1292, 1564, 1836, 2176, 2584, 3060, 3604, 4284, 8024]
    ax[0,iterator].legend(fontsize=fontsize_label)
    # for edge in edges:
    #     ax[0,iterator].vlines(edge, 0, 1, color='lightgray')
    RPO = re.search('_(.*)_', name)
    ax[0,0].set_ylabel('Acoustic \n normalized \n spectrum',fontsize=fontsize)
    ax[0,iterator].set_xlim((xlim0, np.max(edges)))

    ################# ELECTRIC
    normalized_bins = np.load('./data/spectrum/Poster/normalized_SCS_bins'+ name + '.npy')
    for i, bin in enumerate(normalized_bins): 
        # winn plot
        if  name[0] == 's':
            ax[1,iterator].hlines(bin, edges[i], edges[i+1], colors='deepskyblue', linewidth= 3) # 
            ax[1,iterator].hlines(bin, edges[i], edges[i+1], colors='white', linewidth= 1)
            ax[1,iterator].vlines(edges[i], previous1, bin, colors='deepskyblue', linewidth= 3) # 
            previous1 = bin 
        elif  name[0] == 'i':
            ax[1,iterator].hlines(bin, edges[i], edges[i+1], colors='black', linewidth= 3) 
            ax[1,iterator].vlines(edges[i], previous2, bin, colors='black', linewidth= 3) # 
            previous2 = bin
            # ax[1, iterator].hlines(bin, edges[i], edges[i+1], colors='white', linewidth= 1) 
        ax[1,iterator].set_xscale('log', base=2)
        ax[1,iterator].set_xlim((200, 8700))
        ax[1,iterator].set_ylim((0, 1))
        ax[1,iterator].set_xticks([250, 500, 1000, 2000, 4000, 8000], labels=['250', '500', '1000', '2000', '4000', '8000'])
        # ax[1,iterator].set_xlabel('Frequency [Hz]')
        # foedge in edges:
        ax[1,iterator].vlines(edges, 0, 1, color='lightgray')
        ax[1,0].set_ylabel('Electric \n normalized \n spectrum',fontsize=fontsize)
        ax[1,iterator].set_xlim((xlim0, np.max(edges)))
    # #add rectangle to plot
    # ax[0,iterator].add_patch(Rectangle((0,0), edges[0],1,
    #                 edgecolor = 'grey',
    #                 facecolor = 'grey',
    #                 fill=True,
    #                 alpha=0.25))
    # ax[0,iterator].add_patch(Rectangle((edges[-1],0), edges[-1]+2000,1,
    #                 edgecolor = 'grey',
    #                 facecolor = 'grey',
    #                 fill=True,
    #                 alpha=0.25))
    # ax[1,iterator].add_patch(Rectangle((0,0), edges[0],1,
    #             edgecolor = 'grey',
    #             facecolor = 'grey',
    #             fill=True,
    #             alpha=0.25))
    # ax[1,iterator].add_patch(Rectangle((edges[-1],0), edges[-1]+2000,1,
    #             edgecolor = 'grey',
    #             facecolor = 'grey',
    #             fill=True,
    #             alpha=0.25))  
    
    ############ NH
    fiber_frequencies = np.load('./data/spectrum/Poster/NH_fiber_freqs.npy')
    normal_spectrum = np.load('./data/spectrum/Poster/NH_'+ name +'_spectrum.npy')
    filtered_normal_spectrum = np.load('./data/spectrum/Poster/NH__filtered_'+ name +'_spectrum.npy')
    ax[2,iterator].bar(fiber_frequencies, normal_spectrum, width=bar_width, alpha=alpha, color=color)
    ax[2,iterator].plot(fiber_frequencies, filtered_normal_spectrum, color=color, label=label)
    ax[2,iterator].set_ylim((0,1))
    ax[2,iterator].set_xlim((xlim0, np.max(edges)))
    ax[2,0].set_ylabel('Normal hearing \n normalized \n spiking',fontsize=fontsize)
    ax[2,iterator].legend(fontsize=fontsize_label)

    ############ EH
    freq_x_fft = np.load('./data/spectrum/Poster/EH_fiber_freqs.npy')
    electric_spectrum = np.load('./data/spectrum/Poster/EH_'+ name +'_spectrum.npy')
    filtered_electric_spectrum = np.load('./data/spectrum/Poster/EH__filtered_'+ name +'_spectrum.npy')
    ax[3,iterator].bar(freq_x_fft, electric_spectrum, width=bar_width, alpha=alpha, color=color)
    ax[3,iterator].plot(freq_x_fft, filtered_electric_spectrum, color=color, label=label)
    ax[3,0].set_ylabel('Electric hearing \n normalized \n spiking',fontsize=fontsize)
    ax[3,iterator].set_xlim((xlim0, np.max(edges)))
    ax[3,iterator].legend(fontsize=fontsize_label)

# ax[0,0].set_title('Acoustic spectrum')
# ax[1,0].set_title('Electric spectrum')
# ax[2,0].set_title('Normal hearing spectrum')
# ax[3,0].set_title('Electric hearing spectrum')

ax[0,0].set_title('0.5 RPO',fontsize=fontsize)
ax[0,1].set_title('2.0 RPO',fontsize=fontsize)
ax[0,2].set_title('4.0 RPO',fontsize=fontsize)

for i in range(3):
    ax[3,i].set_xlabel('Frequency [Hz]',fontsize=fontsize)

fig.savefig('./figures/spectrum/Innovation4HealthSpectrum.png')    
plt.show()