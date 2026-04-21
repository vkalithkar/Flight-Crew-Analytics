import numpy as np
import pandas as pd 
import mne
import os
import json

# Channels used
ch_names = ['EEG_FP1', 'EEG_F7', 'EEG_F8', 'EEG_T4', 'EEG_T6', 'EEG_T5', 'EEG_T3',
       'EEG_FP2', 'EEG_O1', 'EEG_P3', 'EEG_Pz', 'EEG_F3', 'EEG_Fz', 'EEG_F4',
       'EEG_C4', 'EEG_P4', 'EEG_POz', 'EEG_C3', 'EEG_Cz', 'EEG_O2']
sfreq=256 # Sampling freq from metadata
info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types='eeg')

# All the pilot patients, each has num_CA.csv, num_DA.csv, num_LOFT.csv, num_SS.csv
pilots = [5,6,9,10,11,12,13,14,17,18,19,21,22,23,24,25,26]
runs = ["CA", "DA", "SS","LOFT"]

# Load the Bad Channel Dictionary
with open('data/eeg_preprocess/pyprep_results.json', 'r') as f:
    bad_dict = json.load(f)

# Preprocess all 68 files
for pilot in pilots: 
    output_dir = f'data/eeg_preprocess/interpolated_data/{pilot}'
    os.makedirs(output_dir, exist_ok=True)

    for run in runs:
        # Data read
        current_key = f"{pilot}_{run}"
        data_path = f"data/raw/{pilot}/{pilot}_{run}.csv"
        data = pd.read_csv(data_path, skiprows=0, usecols=[*range(3, 23)]) 
        print("data read")

        # Load raw EEG data onto MNE
        raw = mne.io.RawArray(data.transpose().values.copy()/ 1e6, info, verbose=False)
        
        mapping = {name: name.replace('EEG_', '').replace('FP', 'Fp') for name in raw.ch_names}
        raw.rename_channels(mapping)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        print("data loaded")

        # Interpolation
        raw.info['bads'] = bad_dict.get(current_key, [])

        if len(raw.info['bads']) > 0:
            # Reconstruct the signal from neighbors so we have 20 clean channels
            raw.interpolate_bads(reset_bads=True)
            print(f"Interpolated bads for {current_key}")
        else:
            print(f"No bad channels to interpolate for {current_key}")

        # Filtering
        raw.notch_filter(60, verbose=False)
        raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)
        print("frequency filtered")

        # Re-reference
        raw.set_eeg_reference('average', projection=False, verbose=False)

        # Save as fif file for the next step
        save_file = f"{output_dir}/{pilot}_{run}_cleaned_raw.fif"
        raw.save(save_file, overwrite=True, verbose=False)
        
        print(f"Finished processing {current_key}. Saved to {save_file}")

print("\nAll files have been cleaned and saved as .fif files.")
