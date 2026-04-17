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

# Configuration
duration = 2.0  # seconds per epoch
overlap = 1.0   # seconds of overlap (50%)
pilots = [5,6,9,10,11,12,13,14,17,18,19,21,22,23,24,25,26]
runs = ["CA", "DA", "SS", "LOFT"]
event_id = {'SS': 1, 'CA': 2, 'DA': 5}

# Process all 68 files
for pilot in pilots: 
    input_dir = f'vandana/ica_data/{pilot}'
    output_dir = f'vandana/epoched_data/{pilot}'
    os.makedirs(output_dir, exist_ok=True)

    for run in runs:
        file_path = os.path.join(input_dir, f"{pilot}_{run}_ica_raw.fif")

        # Load pre-cleaned filtered EEG data onto MNE
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)  
        mapping = {name: name.replace('EEG_', '').replace('FP', 'Fp') for name in raw.ch_names}
        raw.rename_channels(mapping)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        print("data loaded")

        # Create fixed-length events (every 1 second we start a 2-second window)
        events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)

        event_channel_idx = raw.ch_names.index('Event')
        for i in range(len(events)):
            # Get the time sample index for this specific epoch
            start_sample = events[i, 0]
            
            # Look at the 'Event' channel at this specific moment
            # .get_data returns [channel, sample], so we take [0, 0]
            actual_label = raw.get_data(picks=[event_channel_idx], 
                                        start=start_sample, 
                                        stop=start_sample+1)[0, 0]
            
            # Overwrite the generic '1' with the actual event (1, 2, or 5)
            events[i, 2] = int(actual_label)

        # Slice data into epochs
        # epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, baseline=None, preload=True, verbose=False)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, baseline=None, preload=True, verbose=False)

        # Save
        save_path = os.path.join(output_dir, f"{pilot}_{run}_epoch.fif")
        epochs.save(save_path, overwrite=True, verbose=False)

        print(f"[{pilot}_{run}] Created {len(epochs)} epochs.")


print("Epoching Complete.")