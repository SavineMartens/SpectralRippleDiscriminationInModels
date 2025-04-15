import numpy as np
import matplotlib.pyplot as plt
from spectrum_from_spikes import get_normalized_spectrum, butter_lowpass_filter
from SMRTvsSR import get_spectrum
import glob

SMRT_data_dir = 'C:\\Matlab\\BEZ2018model\\Output\\SMRT_100_sel\\'
# SMRT
bar_width = 15
cut_off_freq = 100
filter_order = 4

cmap = plt.get_cmap('Greens')

plt.figure()

for RPO in range(2,22,2):
    fname = glob.glob(SMRT_data_dir + '*100*width_' + str(RPO) + '_*.mat')[0]
    print(fname)
    normal_spectrum, fiber_frequencies = get_spectrum(fname) #get_normalized_spectrum(fname)    
    # plt.bar(fiber_frequencies, normal_spectrum, width=bar_width)
    filtered = butter_lowpass_filter(normal_spectrum, cut_off_freq, len(normal_spectrum), filter_order)
    plt.plot(fiber_frequencies, filtered, label='filtered response (' + str(RPO) +' RPO)', color=cmap(RPO*10))
    plt.title('SMRT')
    plt.xlim((4000, 8000))
    plt.xlabel('Frequency [Hz]')

plt.legend()
plt.show()