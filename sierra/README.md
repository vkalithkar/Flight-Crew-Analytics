# Big Picture

-reads sensor data (EEG, ECG, GSR, respiration) from files

-chops it into time windows

-extracts “features” (useful summaries like averages, noise, spikes)

-saves everything into one clean CSV for machine learning

## Toolbox

**numpy (np)** → math (averages, arrays, etc.)
**pandas (pd)** → tables
**Path** → helps find files
**neurokit2 (nk)** → library for brain/heart/skin signal processing 
**skew, kurtosis** → measures shape of data (weirdness / peaks)

## Important info

Sampling rate = 256 measurements per second

We look at 30-second chunks
We move forward by 20% of that window each time

Convert seconds → number of rows
So now everything is in “samples”

## Feature functions

### EEG features (brain waves)

We break EEG into 3-second chunks

“power” = how strong the brain activity is

“variance” = how jumpy/noisy it is

Instead of raw waves, we keep:

average power, variability of power, average noise, noise variability

### ECG features (heart)

This library:

finds heartbeats and computes heart rate variability (HRV)

We return:

average heart rate, heart rate variation, RMSSD (short-term HR variation), SDNN (overall HR variation)

### GSR


We extract:

tonic level (baseline stress), variability of stress, number of spikes (SCR peaks = “stress blips”)

### Resp

We get:

breathing rate average and breathing rate variation

## Basic Stats

It calculates:

mean (average)
std (spread)
skew (is it lopsided?)
kurtosis (how spike-y it is)

## Main Feature Maker (def create_feature_table(df):):

slide through data -> take a slice (one chunk of time) -> build feature dictionary -> check what signals exist -> label (what was happening, most common event in window) -> add meta data -> save to results

X = features (inputs for ML)
y = labels (answers)

## MAIN program to run everything

finds all cleaned sensor files -> Turn raw file to features -> attach labels so features + answers together -> stack all subjects/files into one big dataset and save as csv
