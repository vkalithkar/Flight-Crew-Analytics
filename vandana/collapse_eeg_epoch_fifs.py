import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load one of the epoch generated files
epochs = mne.read_epochs('data/eeg_preprocess/epoched_data/5/5_LOFT_epo.fif')

# figs = epochs.plot_image(picks='eeg', combine='mean', show=False)
# figs[0].savefig('vandana/snapshot_epochs_image.png')
# print("Successfully saved snapshot_epochs.png")

# Get data with MNE
data = epochs.get_data()

n_epochs, n_channels, n_times = data.shape
data_2d = data.reshape(n_epochs, n_channels * n_times)

# Create column names (e.g., "Fp1_0ms", "Fp1_4ms", etc.)
ch_names = epochs.ch_names
col_names = [f"{ch}_{t}" for ch in ch_names for t in range(n_times)]

# Build df
df = pd.DataFrame(data_2d, columns=col_names)

# MNE target add to the df too
df['target'] = epochs.events[:, 2]

print(f"DataFrame created with shape: {df.shape}")
print(df.head())

column_log_path = os.path.join("vandana/column_names.txt")

# Write the column names one per line
with open(column_log_path, 'w') as f:
    for col in df.columns:
        f.write(f"{col}\n")