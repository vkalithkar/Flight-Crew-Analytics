#!/usr/bin/env python3

import pandas as pd

# getting the shape of the CA df for pilot 5
df5 = pd.read_csv('data/raw/5/5_CA.csv')
print(df5.shape)

# understanding how long the data is, temporally
first_time = df5.head(1)['TimeSecs']
last_time = df5.tail(1)['TimeSecs']
print(f'Time is {last_time-first_time} seconds, which is {(last_time-first_time)/60} minutes.')