import numpy as np
import pandas as pd 
import mne
import matplotlib.pyplot as plt
import os
from pyprep.find_noisy_channels import NoisyChannels

# Channels used
ch_names = ['EEG_FP1', 'EEG_F7', 'EEG_F8', 'EEG_T4', 'EEG_T6', 'EEG_T5', 'EEG_T3',
       'EEG_FP2', 'EEG_O1', 'EEG_P3', 'EEG_Pz', 'EEG_F3', 'EEG_Fz', 'EEG_F4',
       'EEG_C4', 'EEG_P4', 'EEG_POz', 'EEG_C3', 'EEG_Cz', 'EEG_O2']
sfreq=256 # Sampling freq from metadata
info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types='eeg')


# All the pilot patients, each has num_CA.csv, num_DA.csv, num_LOFT.csv, num_SS.csv
pilots = [5,6,9,10,11,12,13,14,17,18,19,21,22,23,24,25,26]
runs = ["CA", "DA", "SS","LOFT"]

# Blank dictionary for the bad channels to add to with pyprep
auto_bad_dict = {f"{p}_{run}": [] for p in pilots for run in runs}

# Preprocess all 68 files
for pilot in pilots: 
    # Create the directory for this pilot if it doesn't exist
    output_dir = f'vandana/raw_filtered_eeg_plots/{pilot}'
    os.makedirs(output_dir, exist_ok=True)

    for run in runs:
        # Data read
        data_path = f"data/raw/{pilot}/{pilot}_{run}.csv"
        data = pd.read_csv(data_path, skiprows=0, usecols=[*range(3, 23)]) 

        # Load raw EEG data onto MNE
        raw = mne.io.RawArray(data.transpose().values.copy()/ 1e6, info, verbose=False)
        
        mapping = {name: name.replace('EEG_', '').replace('FP', 'Fp') for name in raw.ch_names}
        raw.rename_channels(mapping)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        # Filtering
        raw.notch_filter(60, verbose=False)
        raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)

        # # Generate raw squiggle plot and save
        # fig = raw.plot(duration=10, scalings='auto', show=False)
        # fig.savefig(f'{output_dir}/{run}_raw.png')
        # plt.close(fig)

        # # Generate and save Save PSD (Frequency check)
        # psd_fig = raw.compute_psd().plot(show=False)
        # psd_fig.savefig(f'{output_dir}/{run}_psd.png')
        # plt.close(psd_fig)

        # Assuming 'raw' is an MNE Raw object
        nd = NoisyChannels(raw)
        # Run all detection methods
        nd.find_all_bads() 

        # Get the suggested bad channels and add it to bad channel dict
        current_key = f"{pilot}_{run}"
        auto_bad_dict[current_key] = nd.get_bads()

        print(f"Completed {current_key}. Bads found: {nd.get_bads()}")

print(auto_bad_dict)

# Json save the auto_bad_dict
output_path = 'vandana/pyprep_results.json'

os.makedirs('vandana', exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(auto_bad_dict, f, indent=4)