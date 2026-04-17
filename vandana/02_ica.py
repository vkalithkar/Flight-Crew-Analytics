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
        mapping = {name: name.replace('EEG_', '').replace('FP', 'Fp') for name in raw.ch_names}
        raw.rename_channels(mapping)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
      
        print("data loaded")

        # Reassert this
        raw.filter(l_freq=1.0, h_freq=None, verbose=False)

        # Initialize and run ICA
        ica = ICA(n_components=0.99, random_state=42, method='picard', fit_params=dict(ortho=False, extended=True))
        ica.fit(raw, verbose=False)

        # Automated labeling with a pre-trained EEG ML model 
        ic_labels = label_components(raw, ica, method='iclabel')
        labels = ic_labels["labels"]
        probs = ic_labels["y_pred_proba"] # This is the secret sauce

        artifact_categories = ["eye", "muscle", "heart"]
        exclude_idx = []
        for i in range(len(labels)):
            # Force the brain_prob extraction based on actual array shape
            # If 2D: [component_index, brain_column]
            # If 1D: we need to investigate why, but usually it's [brain_column]
            current_brain_prob = probs[i][0] if isinstance(probs[i], (np.ndarray, list)) else probs[0]
            
            if labels[i] in ["eye", "muscle", "heart"] or current_brain_prob < 0.30:
                exclude_idx.append(i)  

        # Apply and save
        ica.exclude = exclude_idx
        raw_ica = raw.copy()
        ica.apply(raw_ica, verbose=False)

        save_path = os.path.join(output_dir, f"{current_key}_ica_raw.fif")
        raw_ica.save(save_path, overwrite=True, verbose=False)

        print(f"Processed {current_key}: Excluded {len(exclude_idx)} artifact components. Saved to {save_path}")

print("ICA Cleaning Complete.")