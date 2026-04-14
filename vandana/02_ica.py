import numpy as np
import pandas as pd 
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
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

# Preprocess all 68 files
for pilot in pilots: 
    input_dir = f'vandana/cleaned_data/{pilot}'
    output_dir = f'vandana/ica_data/{pilot}'
    os.makedirs(output_dir, exist_ok=True)

    for run in runs:
        current_key = f"{pilot}_{run}"
        file_name = f"{current_key}_cleaned_raw.fif"
        input_path = os.path.join(input_dir, file_name)

        # Load pre-cleaned filtered EEG data onto MNE
        raw = mne.io.read_raw_fif(input_path, preload=True, verbose=False)        
        print("data loaded")

        # Initialize and run ICA
        ica = ICA(n_components=15, random_state=42, method='fastica')
        ica.fit(raw, verbose=False)

        # Automated labeling with a pre-trained EEG ML model 
        ic_labels = label_components(raw, ica, method='iclabel')
        labels = ic_labels["labels"]

        # Which components to remove
        exclude_idx = [
            idx for idx, label in enumerate(labels) 
            if label in ["eye", "muscle", "heart", "line_noise"]
        ]

        # Apply and save
        ica.exclude = exclude_idx
        raw_ica = raw.copy()
        ica.apply(raw_ica, verbose=False)

        save_path = os.path.join(output_dir, f"{current_key}_ica_raw.fif")
        raw_ica.save(save_path, overwrite=True, verbose=False)

        print(f"Processed {current_key}: Excluded {len(exclude_idx)} artifact components. Saved to {save_path}")

print("ICA Cleaning Complete.")