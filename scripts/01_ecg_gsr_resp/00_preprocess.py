"""
00_preprocess.py  –   physiological preprocessing script for ECG, GSR, and Resp (NOT EEG!)
Flight-Crew-Analytics
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import neurokit2 as nk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR   = REPO_ROOT / "data" / "raw"
OUT_DIR   = REPO_ROOT / "data" / "ecg_gsr_resp_preprocess"

OUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 256

FILE_TYPES = ["CA", "DA", "LOFT", "SS"]

NEXUS_ECG_SCALE = 1.22e-4

BAD_EVENT_CODES = {3: 0, 4: 0}


# ---------------------------------------------------------------------------
# Signal-cleaning helpers
# ---------------------------------------------------------------------------

def clean_event_column(series: pd.Series) -> pd.Series:
    return series.replace(BAD_EVENT_CODES).astype(int)


def clean_ecg(signal: pd.Series) -> pd.Series:
    arr = signal.to_numpy(dtype=float) * NEXUS_ECG_SCALE
    try:
        cleaned = nk.ecg_clean(arr, sampling_rate=FS, method="neurokit")
    except Exception:
        cleaned = nk.signal_filter(
            arr, sampling_rate=FS,
            lowcut=0.5, highcut=40.0,
            method="butterworth", order=4,
        )
    return pd.Series(cleaned, index=signal.index, name=signal.name)


def clean_respiration(signal: pd.Series) -> pd.Series:
    arr = signal.to_numpy(dtype=float)
    try:
        cleaned = nk.rsp_clean(arr, sampling_rate=FS, method="biosppy")
        cleaned = nk.signal_filter(
            cleaned, sampling_rate=FS,
            highcut=1.0, method="butterworth", order=2,
        )
    except Exception:
        cleaned = nk.signal_filter(
            arr, sampling_rate=FS,
            lowcut=0.1, highcut=1.0,
            method="butterworth", order=2,
        )
    return pd.Series(cleaned, index=signal.index, name=signal.name)


def clean_gsr(signal: pd.Series) -> pd.Series:
    arr = signal.to_numpy(dtype=float)
    try:
        cleaned = nk.eda_clean(arr, sampling_rate=FS, method="neurokit")
    except Exception:
        cleaned = nk.signal_filter(
            arr, sampling_rate=FS,
            highcut=3.0,
            method="butterworth", order=4,
        )
    return pd.Series(cleaned, index=signal.index, name=signal.name)


# ---------------------------------------------------------------------------
# Per-file preprocessing
# ---------------------------------------------------------------------------

def preprocess_file(csv_path: Path, subject_id: str, file_type: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Event" in df.columns:
        df["Event"] = clean_event_column(df["Event"])

    if "ECG" in df.columns:
        df["ECG"] = clean_ecg(df["ECG"])

    if "R" in df.columns:
        df["R"] = clean_respiration(df["R"])

    if "GSR" in df.columns:
        df["GSR"] = clean_gsr(df["GSR"])

    df.insert(0, "subject_id", subject_id)
    df.insert(1, "file_type",  file_type)

    return df


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    subject_dirs = sorted(
        [d for d in RAW_DIR.iterdir() if d.is_dir() and re.fullmatch(r"\d+", d.name)],
        key=lambda d: int(d.name),
    )

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        print(f"── Subject {subj_id} ──────────────────────────────────", flush=True)

        for ftype in FILE_TYPES:
            csv_path = subj_dir / f"{subj_id}_{ftype}.csv"

            if not csv_path.exists():
                print(f"  Missing: {csv_path.name} – skipping.", flush=True)
                continue

            try:
                cleaned_df = preprocess_file(csv_path, subj_id, ftype)
                out_path = OUT_DIR / f"{subj_id}_{ftype}_clean.csv"
                cleaned_df.to_csv(out_path, index=False)
                print(f"  Written → {out_path.name}", flush=True)

            except Exception as exc:
                print(f"  FAILED – Subject {subj_id} / {ftype}: {exc}", flush=True)


if __name__ == "__main__":
    main()