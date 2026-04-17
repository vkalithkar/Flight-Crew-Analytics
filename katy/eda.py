#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# getting the shape of the CA df for pilot 5
df5 = pd.read_csv('katy/clean/25_LOFT_clean.csv')
print('user 5 CA df is of shape:')
print(df5.shape)
print('---xxx---')

# understanding how long the data is, temporally
first_time = df5['TimeSecs'].iloc[0]
last_time = df5['TimeSecs'].iloc[-1]
print(f'Time is {(last_time-first_time):.3f} seconds, which is {((last_time-first_time)/60):.3f} minutes.')
print('---xxx---')

# understanding what the other columns are
# 0 = no event, 2 = CA state
print('The two events experienced in this data and their instances are:')
print(df5['Event'].value_counts())
print('---xxx---')

state_colors = {
    1: 'red',
    2: 'yellow',
    5: 'green',
    # add as many states as you need
}

# define CA state blocks once, reuse across all plots
state_col = df5['Event']
changes = state_col != state_col.shift()
blocks = df5[changes].copy()
blocks['end_time'] = blocks['TimeSecs'].shift(-1).fillna(df5['TimeSecs'].iloc[-1])

# Plot 1: Respirations
plt.figure(figsize=(20,8))
plt.plot(df5['TimeSecs'], df5['R'], linewidth = 0.1)
for state, color in state_colors.items():
    for _, row in blocks[blocks['Event'] == state].iterrows():
        plt.axvspan(row['TimeSecs'], row['end_time'], color=color, alpha=0.3)
plt.xlabel('Time (seconds)')
plt.ylabel('Respirations')
plt.title('Time vs. Respirations, 25 LOFT Cleaned')
plt.savefig('katy/loft_resp_cleaned_25.png')
plt.clf()

# Plot 2: ECG
plt.figure(figsize=(20,8))
plt.plot(df5['TimeSecs'], df5['ECG'], linewidth=0.1)
for state, color in state_colors.items():
    for _, row in blocks[blocks['Event'] == state].iterrows():
        plt.axvspan(row['TimeSecs'], row['end_time'], color=color, alpha=0.3)
plt.xlabel('Time (seconds)')
plt.ylabel('ECG')
plt.title('Time vs. ECG, 25 LOFT Cleaned')
plt.savefig('katy/loft_ecg_25_cleaned.png')
plt.clf()

# Plot 3: GSR
plt.figure(figsize=(20,8))
plt.plot(df5['TimeSecs'], df5['GSR'], linewidth=0.3)
for state, color in state_colors.items():
    for _, row in blocks[blocks['Event'] == state].iterrows():
        plt.axvspan(row['TimeSecs'], row['end_time'], color=color, alpha=0.3)
plt.xlabel('Time (seconds)')
plt.ylabel('GSR')
plt.title('Time vs. GSR, 25 LOFT Cleaned')
plt.savefig('katy/loft_gsr_25_cleaned.png')
plt.clf()