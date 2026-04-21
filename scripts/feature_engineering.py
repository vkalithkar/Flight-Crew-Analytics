#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
import neurokit2 as nk
from scipy.stats import skew, kurtosis
import mne

FS = 256  # sampling rate

WINDOW_SEC = 30
STEP_RATIO = 0.2

WINDOW_SIZE = FS * WINDOW_SEC
STEP_SIZE = int(WINDOW_SIZE * STEP_RATIO)


INPUT_DIR = Path("./data/ecg_gsr_resp_preprocess")
EEG_DIR = Path("./data/eeg_preprocess/epoched_data")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def eeg_features(epoch):
    """
    epoch: (n_channels, n_times)
    """

    feats = {}

    # collapse spatially (robust baseline)
    signal = epoch.mean(axis=0)

    sub_win = FS * 3
    powers, variances = [], []

    for i in range(0, len(signal) - sub_win, sub_win):
        chunk = signal[i:i + sub_win]
        powers.append(np.sum(chunk ** 2))
        variances.append(np.var(chunk))

    if len(powers) > 0:
        feats.update({
            "eeg_power_mean": np.mean(powers),
            "eeg_power_std": np.std(powers),
            "eeg_var_mean": np.mean(variances),
            "eeg_var_std": np.std(variances),
        })

    # spatial structure
    channel_power = np.sum(epoch ** 2, axis=1)
    feats.update({
        "eeg_channel_power_mean": np.mean(channel_power),
        "eeg_channel_power_std": np.std(channel_power),
        "eeg_spatial_var": np.mean(np.var(epoch, axis=1)),
    })

    feats.update({
        "eeg_mean": np.mean(signal),
        "eeg_std": np.std(signal),
        "eeg_skew": skew(signal),
        "eeg_kurtosis": kurtosis(signal),
    })

    return feats

def ecg_features(signal):
    try:
        signals, info = nk.ecg_process(signal, sampling_rate=FS)
        hrv = nk.hrv(info, sampling_rate=FS)

        return {
            "hr_mean": np.mean(signals["ECG_Rate"]),
            "hr_std": np.std(signals["ECG_Rate"]),
            "rmssd": hrv["HRV_RMSSD"].values[0],
            "sdnn": hrv["HRV_SDNN"].values[0],
        }
    except Exception:
        return {}

def gsr_features(signal):
    try:
        eda_signals, info = nk.eda_process(signal, sampling_rate=FS)

        return {
            "gsr_tonic_mean": np.mean(eda_signals["EDA_Tonic"]),
            "gsr_tonic_std": np.std(eda_signals["EDA_Tonic"]),
            "gsr_scr_count": len(info["SCR_Peaks"]),
        }
    except Exception:
        return {}


def rsp_features(signal):
    try:
        rsp_signals, info = nk.rsp_process(signal, sampling_rate=FS)

        return {
            "rsp_rate_mean": np.mean(rsp_signals["RSP_Rate"]),
            "rsp_rate_std": np.std(rsp_signals["RSP_Rate"]),
        }
    except Exception:
        return {}


def basic_stats(signal, prefix):
    return {
        f"{prefix}_mean": np.mean(signal),
        f"{prefix}_std": np.std(signal),
        f"{prefix}_skew": skew(signal),
        f"{prefix}_kurtosis": kurtosis(signal),
    }


def create_feature_table(df, eeg_epochs):
    """
    df = CSV with ECG/GSR/RSP + labels
    eeg_epochs = MNE Epochs object aligned to df windows
    """

    X, y = [], []

    eeg_data = eeg_epochs.get_data()  # (n_epochs, ch, time)

    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end]

        feats = {}

        epoch_idx = start // STEP_SIZE

        if epoch_idx < len(eeg_data):
            feats.update(eeg_features(eeg_data[epoch_idx]))

        if "ECG" in df.columns:
            feats.update(ecg_features(window["ECG"].values))
            feats.update(basic_stats(window["ECG"].values, "ecg"))

        if "GSR" in df.columns:
            feats.update(gsr_features(window["GSR"].values))
            feats.update(basic_stats(window["GSR"].values, "gsr"))

        if "R" in df.columns:
            feats.update(rsp_features(window["R"].values))
            feats.update(basic_stats(window["R"].values, "rsp"))

        if "Event" in df.columns:
            y.append(window["Event"].mode()[0])

        feats["subject_id"] = window["subject_id"].iloc[0]
        feats["file_type"] = window["file_type"].iloc[0]

        X.append(feats)

    return pd.DataFrame(X), np.array(y)


def main():
    all_features = []

    for file in INPUT_DIR.glob("*_clean.csv"):
        print(f"Processing {file.name}")
        df = pd.read_csv(file)

        subject_id = file.stem.split("_")[0]
        eeg_file = EEG_DIR / subject_id / file.name.replace("_clean.csv", "_epo.fif")

        eeg_epochs = mne.read_epochs(eeg_file, preload=True, verbose=False)

        X, y = create_feature_table(df, eeg_epochs)

        X["label"] = y
        all_features.append(X)

    final_df = pd.concat(all_features, ignore_index=True)

    out_path = OUTPUT_DIR / "features.csv"
    final_df.to_csv(out_path, index=False)

    print(f"\nSaved features → {out_path}")
    print(f"Shape: {final_df.shape}")

if __name__ == "__main__":
    main()
