import numpy as np
import pandas as pd 
import mne
import os

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

summary_data = []

# Process all 68 files
for pilot in pilots: 
    input_dir = f'data/eeg_preprocess/ica_data/{pilot}'
    raw_csv_dir = f'./data/raw/{pilot}' 
    output_dir = f'data/eeg_preprocess/epoched_data/{pilot}'
    os.makedirs(output_dir, exist_ok=True)

    for run in runs:
        # Construct paths
        current_key = f"{pilot}_{run}"
        ica_path = os.path.join(input_dir, f"{current_key}_ica_raw.fif")
        csv_path = os.path.join(raw_csv_dir, f"{current_key}.csv")
       
        # Load pre-cleaned filtered EEG data onto MNE
        raw = mne.io.read_raw_fif(ica_path, preload=True, verbose=False)  

        mapping = {name: name.replace('EEG_', '').replace('FP', 'Fp') for name in raw.ch_names}
        raw.rename_channels(mapping)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        print("data loaded")

        # Event loading
        df_events = pd.read_csv(csv_path, usecols=['Event'])
        event_labels = df_events['Event'].values

        # Create fixed-length events (every 1 second we start a 2-second window)
        events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)

        # Map the labels from CSV to MNE events
        # We look at the 'Event' value at the exact start sample of each epoch
        valid_indices = []
        for i in range(len(events)):
            sample_idx = events[i, 0]
            
            if sample_idx < len(event_labels):
                raw_label = event_labels[sample_idx]
                
                # Instruction: 3 and 4 are treated as 0 (No Event)
                if raw_label in [3, 4]:
                    events[i, 2] = 0
                else:
                    events[i, 2] = int(raw_label)
                valid_indices.append(i)

        # Only keep events that were within the CSV range
        events = events[valid_indices]

        # Slice and Save
        # 'baseline=None' because you've already high-passed at 1Hz
        epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, 
                            baseline=None, preload=True, verbose=False)
        
        save_path = os.path.join(output_dir, f"{current_key}_epo.fif")
        epochs.save(save_path, overwrite=True, verbose=False)
        
        unique, counts = np.unique(epochs.events[:, 2], return_counts=True)
        counts_dict = dict(zip(unique, counts))
        
        summary_data.append(f"{current_key} | Total: {len(epochs)} | 0: {counts_dict.get(0,0)}, 1: {counts_dict.get(1,0)}, 2: {counts_dict.get(2,0)}, 5: {counts_dict.get(5,0)}")
       
        print(f"[{current_key}] Created {len(epochs)} labeled epochs.")
        del raw, epochs, df_events

print("\n" + "="*50)
print("FINAL EPOCHING SUMMARY")
print("="*50)
for line in summary_data:
    print(line)
print("="*50)