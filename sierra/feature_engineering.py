import numpy as np
import pandas as pd
from pathlib import Path
import neurokit2 as nk
from scipy.stats import skew, kurtosis


FS = 256  # sampling rate

WINDOW_SEC = 30
STEP_RATIO = 0.2

WINDOW_SIZE = FS * WINDOW_SEC
STEP_SIZE = int(WINDOW_SIZE * STEP_RATIO)

INPUT_DIR = Path("./katy/")
OUTPUT_DIR = Path("./output")

def eeg_features(signal):
    sub_win = FS * 3  # 3 second chunks
    powers = []
    variances = []

    for i in range(0, len(signal) - sub_win, sub_win):
        chunk = signal[i:i + sub_win]

        powers.append(np.sum(chunk ** 2))
        variances.append(np.var(chunk))

    if len(powers) == 0:
        return {}

    return {
        "eeg_power_mean": np.mean(powers),
        "eeg_power_std": np.std(powers),
        "eeg_var_mean": np.mean(variances),
        "eeg_var_std": np.std(variances),
    }


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


def create_feature_table(df):
    X = []
    y = []

    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end]

        feats = {}


        if "EEG" in df.columns:
            feats.update(eeg_features(window["EEG"].values))
            feats.update(basic_stats(window["EEG"].values, "eeg"))


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
            label = window["Event"].mode()[0]
            y.append(label)


        feats["subject_id"] = window["subject_id"].iloc[0]
        feats["file_type"] = window["file_type"].iloc[0]

        X.append(feats)

    return pd.DataFrame(X), np.array(y)


def main():
    all_features = []

    for file in INPUT_DIR.glob("*_clean.csv"):
        print(f"Processing {file.name}")
        df = pd.read_csv(file)

        X, y = create_feature_table(df)

        X["label"] = y
        all_features.append(X)

    final_df = pd.concat(all_features, ignore_index=True)

    out_path = OUTPUT_DIR / "features.csv"
    final_df.to_csv(out_path, index=False)

    print(f"\nSaved features → {out_path}")
    print(f"Shape: {final_df.shape}")

if __name__ == "__main__":
    main()
