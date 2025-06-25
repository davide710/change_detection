import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy


def low_pass_filter(x, y):
    sampling_period = x[1] - x[0]
    sampling_rate = 1.0 / sampling_period
    max_period = (np.max(x) - np.min(x)) / 25
    cutoff_frequency = 1 / max_period
    order = 4

    nyquist_rate = sampling_rate * 0.5
    normalized_cutoff = cutoff_frequency / nyquist_rate
    b, a = scipy.signal.butter(order, normalized_cutoff, btype='low', analog=False)
    return scipy.signal.filtfilt(b, a, y)

files = os.listdir('data')
j = 1

for f in files:
    df = pd.read_csv(f'data/{f}', sep='\t')
    df = df[['Secondi', 'Canale_1', 'Canale_2', 'Canale_4', 'Canale_5', 'Canale_6', 'Canale_7', 'Canale_8', 'RH%']].copy()
    x = df['Secondi']
    for i, c in enumerate(['Canale_1', 'Canale_2', 'Canale_4', 'Canale_5', 'Canale_6', 'Canale_7', 'Canale_8', 'RH%']):
        y = df[c]
        #y = low_pass_filter(x, y)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f'images/{j}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        j += 1
