#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt

pilots = [5, 6, 9, 10, 11, 12, 13, 14, 17, 18, 19, 21, 22, 23, 24, 25, 26]
file_types = ['CA', 'DA', 'SS', 'LOFT']

state_colors = {
    1: 'red',
    2: 'yellow',
    5: 'green',
}

for pilot_id in pilots:
    for file_type in file_types:
        FILE_NAME = f'{pilot_id}_{file_type}_clean'  # was int + str, now f-string
        out_dir = f'./eda_plots/{pilot_id}'
        os.makedirs(out_dir, exist_ok=True)  # create per-pilot dir if needed

        df = pd.read_csv(f'./data/ecg_gsr_resp_preprocess/{FILE_NAME}.csv')
        print(f'{FILE_NAME} shape:')
        print(df.shape)
        print('---xxx---')

        print('Events in this data:')
        print(df['Event'].value_counts())
        print('---xxx---')

        state_col = df['Event']
        changes = state_col != state_col.shift()
        blocks = df[changes].copy()
        blocks['end_time'] = blocks['TimeSecs'].shift(-1).fillna(df['TimeSecs'].iloc[-1])

        def add_state_shading(blocks, state_colors):
            for event_id, color in state_colors.items():  # renamed to avoid collision
                for _, row in blocks[blocks['Event'] == event_id].iterrows():
                    plt.axvspan(row['TimeSecs'], row['end_time'], color=color, alpha=0.3)

        # Plot 1: Respirations
        plt.figure(figsize=(20, 8))
        plt.plot(df['TimeSecs'], df['R'], linewidth=1.0)
        add_state_shading(blocks, state_colors)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Respirations')
        plt.title(f'Time vs. Respirations, {FILE_NAME}')
        plt.savefig(f'{out_dir}/{FILE_NAME}_resp.png')
        plt.clf()

        # Plot 2: ECG
        plt.figure(figsize=(20, 8))
        plt.plot(df['TimeSecs'], df['ECG'], linewidth=1.0)
        add_state_shading(blocks, state_colors)
        plt.xlabel('Time (seconds)')
        plt.ylabel('ECG')
        plt.title(f'Time vs. ECG, {FILE_NAME}')
        plt.savefig(f'{out_dir}/{FILE_NAME}_ecg.png')
        plt.clf()

        # Plot 3: GSR
        plt.figure(figsize=(20, 8))
        plt.plot(df['TimeSecs'], df['GSR'], linewidth=1.0)
        add_state_shading(blocks, state_colors)
        plt.xlabel('Time (seconds)')
        plt.ylabel('GSR')
        plt.title(f'Time vs. GSR, {FILE_NAME}')
        plt.savefig(f'{out_dir}/{FILE_NAME}_gsr.png')
        plt.clf()