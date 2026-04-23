#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
import neurokit2 as nk
from scipy.stats import skew, kurtosis
import mne

FS = 256  

WINDOW_SEC = 30
STEP_RATIO = 0.2

WINDOW_SIZE = FS * WINDOW_SEC
STEP_SIZE = int(WINDOW_SIZE * STEP_RATIO)

INPUT_DIR = Path("./data/ecg_gsr_resp_preprocess")
EEG_DIR = Path("./data/eeg_preprocess/epoched_data")
# CHANGED THIS!!!
OUTPUT_DIR = Path("./data/feature_extracted")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def eeg_features(epoch, ch_names):
    feats = {}

    n_channels = epoch.shape[0]

    # CHANGED THIS!!!
    for i in range(n_channels):
        ch = epoch[i]
        ch_name = ch_names[i].replace(" ", "").lower()

        feats[f"eeg_{ch_name}_mean"] = np.mean(ch)
        feats[f"eeg_{ch_name}_std"] = np.std(ch)
        feats[f"eeg_{ch_name}_power"] = np.sum(ch ** 2)

    signal_global = epoch.mean(axis=0)

    sub_win = FS * 3
    powers, variances = [], []

    for i in range(0, len(signal_global) - sub_win, sub_win):
        chunk = signal_global[i:i + sub_win]
        powers.append(np.sum(chunk ** 2))
        variances.append(np.var(chunk))

    if len(powers) > 0:
        feats.update({
            "eeg_power_mean": np.mean(powers),
            "eeg_power_std": np.std(powers),
            "eeg_var_mean": np.mean(variances),
            "eeg_var_std": np.std(variances),
        })

    feats.update({
        "eeg_mean": np.mean(signal_global),
        "eeg_std": np.std(signal_global),
        "eeg_skew": skew(signal_global),
        "eeg_kurtosis": kurtosis(signal_global),
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
            "gsr_scr_count": len(info.get("SCR_Peaks", [])),
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


def label_window(window_df, state_col, baseline_label=0):
    states = window_df[state_col]
    events = states[states != baseline_label]

    if events.empty:
        return baseline_label

    return events.value_counts().idxmax()


def create_feature_table(df, eeg_epochs):
    X, y = [], []

    eeg_data = eeg_epochs.get_data()
    #CHANGED THIS!!!
    ch_names = eeg_epochs.ch_names  

    n_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE

    for w in range(n_windows):
        start = w * STEP_SIZE
        end = start + WINDOW_SIZE

        window = df.iloc[start:end]

        feats = {}

        if w < len(eeg_data):
            feats.update(eeg_features(eeg_data[w], ch_names))

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
            y.append(label_window(window, "Event", baseline_label=0))

        feats["subject_id"] = window["subject_id"].iloc[0]
        feats["file_type"] = window["file_type"].iloc[0]
        feats["window_id"] = w

        X.append(feats)

    return pd.DataFrame(X), np.array(y)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    for file in INPUT_DIR.glob("*_clean.csv"):
        print(f"Processing {file.name}")

        df = pd.read_csv(file)

        subject_id = file.stem.split("_")[0]
        file_type = df["file_type"].iloc[0] 

        eeg_file = EEG_DIR / subject_id / file.name.replace("_clean.csv", "_epo.fif")

        if not eeg_file.exists():
            print(f"Missing EEG file for {subject_id}, skipping.")
            continue

        eeg_epochs = mne.read_epochs(eeg_file, preload=True, verbose=False)

        X, y = create_feature_table(df, eeg_epochs)

        X["label"] = y

        #CHANGED THIS!!!
        subject_dir = OUTPUT_DIR / subject_id
        subject_dir.mkdir(exist_ok=True, parents=True)

        out_path = subject_dir / f"{subject_id}_{file_type}_features.csv"
        X.to_csv(out_path, index=False)

        print(f"Saved → {out_path} | shape={X.shape}")


if __name__ == "__main__":
    main()