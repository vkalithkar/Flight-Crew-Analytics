#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# getting the shape of the CA df for pilot 5
df5 = pd.read_csv('data/raw/5/5_CA.csv')
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

# Simple line plot
x = df5['TimeSecs']
y = df5['R']
y2=df5['ECG']

plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('time vs. respirations')
plt.savefig('katy/test.png')

y2=df5['ECG']
plt.plot(x, y2)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('time vs. ECG')
plt.savefig('katy/test2.png')