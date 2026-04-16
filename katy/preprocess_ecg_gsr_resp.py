"""
preprocess.py  –  Katy's physiological preprocessing script
Flight-Crew-Analytics / katy/

Usage (from repo root):
    python katy/preprocess.py

For each subject folder in data/raw/, reads the four CSVs (CA, DA, LOFT, SS),
cleans EEG, ECG, Respiration, and GSR signals with NeuroKit2, and writes one
cleaned CSV per subject per file type into katy/clean/.

Sampling rate : 256 Hz (all sensors, B-Alert X24 + NeXus-10)
"""

import os
import re
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import neurokit2 as nk

# ---------------------------------------------------------------------------
# Paths  (relative to repo root – adjust if running from a different cwd)
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
RAW_DIR    = REPO_ROOT / "data" / "raw"
OUT_DIR    = REPO_ROOT / "katy" / "clean"
LOG_FILE   = REPO_ROOT / "katy" / "preprocess.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)   # nk can be noisy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 256   # Hz – all sensors

FILE_TYPES = ["CA", "DA", "LOFT", "SS"]

# EEG_CHANNELS = [
#     "EEG_FP1", "EEG_F7",  "EEG_F8",  "EEG_T4",  "EEG_T6",
#     "EEG_T5",  "EEG_T3",  "EEG_FP2", "EEG_O1",  "EEG_P3",
#     "EEG_Pz",  "EEG_F3",  "EEG_Fz",  "EEG_F4",  "EEG_C4",
#     "EEG_P4",  "EEG_POz", "EEG_C3",  "EEG_Cz",  "EEG_O2",
# ]

# Event codes 3 and 4 are documented as erroneous – remap to 0
BAD_EVENT_CODES = {3: 0, 4: 0}


# ---------------------------------------------------------------------------
# Signal-cleaning helpers
# ---------------------------------------------------------------------------

def clean_event_column(series: pd.Series) -> pd.Series:
    """Remap erroneous event codes (3, 4) → 0."""
    return series.replace(BAD_EVENT_CODES).astype(int)


def clean_ecg(signal: pd.Series) -> pd.Series:
    """
    Clean the ECG signal with NeuroKit2.
    Returns a cleaned 1-D Series of the same length.
    Falls back to bandpass-only on processing errors.
    """
    arr = signal.to_numpy(dtype=float)
    try:
        cleaned = nk.ecg_clean(arr, sampling_rate=FS, method="neurokit")
    except Exception as exc:
        log.warning("ECG clean failed (%s) – using basic bandpass fallback.", exc)
        cleaned = nk.signal_filter(
            arr, sampling_rate=FS,
            lowcut=0.5, highcut=40.0,
            method="butterworth", order=4,
        )
    return pd.Series(cleaned, index=signal.index, name=signal.name)


def clean_respiration(signal: pd.Series) -> pd.Series:
    """
    Clean the Respiration signal.
    NeuroKit2's rsp_clean applies a bandpass (0.1–0.35 Hz by default)
    suited to chest-belt respiration data.
    """
    arr = signal.to_numpy(dtype=float)
    try:
        cleaned = nk.rsp_clean(arr, sampling_rate=FS, method="khodadad2018")
    except Exception as exc:
        log.warning("RSP clean failed (%s) – using basic bandpass fallback.", exc)
        cleaned = nk.signal_filter(
            arr, sampling_rate=FS,
            lowcut=0.1, highcut=0.35,
            method="butterworth", order=2,
        )
    return pd.Series(cleaned, index=signal.index, name=signal.name)


def clean_gsr(signal: pd.Series) -> pd.Series:
    """
    Clean the GSR / EDA signal.
    nk.eda_clean applies a low-pass filter (default 3 Hz) to remove
    high-frequency noise while preserving the slow skin-conductance response.
    """
    arr = signal.to_numpy(dtype=float)
    try:
        cleaned = nk.eda_clean(arr, sampling_rate=FS, method="neurokit")
    except Exception as exc:
        log.warning("GSR/EDA clean failed (%s) – using basic lowpass fallback.", exc)
        cleaned = nk.signal_filter(
            arr, sampling_rate=FS,
            highcut=3.0,
            method="butterworth", order=4,
        )
    return pd.Series(cleaned, index=signal.index, name=signal.name)


# def clean_eeg_channel(signal: pd.Series, ch_name: str) -> pd.Series:
#     """
#     Clean a single EEG channel:
#       1. Bandpass  1–40 Hz  (removes DC drift + high-freq noise)
#       2. Notch     60 Hz    (US power-line interference)
#     NeuroKit2 does not have a dedicated single-channel EEG cleaner, so we
#     chain signal_filter calls directly.
#     """
#     arr = signal.to_numpy(dtype=float)
#     # Replace any NaN with channel mean before filtering
#     nan_mask = np.isnan(arr)
#     if nan_mask.any():
#         arr[nan_mask] = np.nanmean(arr)

#     try:
#         # Bandpass
#         bp = nk.signal_filter(
#             arr, sampling_rate=FS,
#             lowcut=1.0, highcut=40.0,
#             method="butterworth", order=4,
#         )
#         # Notch at 60 Hz
#         cleaned = nk.signal_filter(
#             bp, sampling_rate=FS,
#             method="powerline", powerline=60,
#         )
#     except Exception as exc:
#         log.warning("EEG channel %s clean failed (%s) – skipping filter.", ch_name, exc)
#         cleaned = arr

#     return pd.Series(cleaned, index=signal.index, name=signal.name)


# ---------------------------------------------------------------------------
# Per-file preprocessing
# ---------------------------------------------------------------------------

def preprocess_file(csv_path: Path, subject_id: str, file_type: str) -> pd.DataFrame:
    """
    Load one raw CSV, preprocess all physiological signals, return cleaned DataFrame.
    """
    log.info("  Loading  %s", csv_path.name)
    df = pd.read_csv(csv_path)

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # ---- Event column -------------------------------------------------------
    if "Event" in df.columns:
        df["Event"] = clean_event_column(df["Event"])
    else:
        log.warning("  No 'Event' column found in %s", csv_path.name)

    # # ---- EEG channels -------------------------------------------------------
    # present_eeg = [ch for ch in EEG_CHANNELS if ch in df.columns]
    # missing_eeg = set(EEG_CHANNELS) - set(present_eeg)
    # if missing_eeg:
    #     log.warning("  Missing EEG channels: %s", sorted(missing_eeg))

    # for ch in present_eeg:
    #     df[ch] = clean_eeg_channel(df[ch], ch)
    # log.info("  EEG      cleaned (%d channels)", len(present_eeg))

    # ---- ECG ----------------------------------------------------------------
    if "ECG" in df.columns:
        df["ECG"] = clean_ecg(df["ECG"])
        log.info("  ECG      cleaned")
    else:
        log.warning("  No 'ECG' column found in %s", csv_path.name)

    # ---- Respiration --------------------------------------------------------
    if "R" in df.columns:
        df["R"] = clean_respiration(df["R"])
        log.info("  RSP      cleaned")
    else:
        log.warning("  No 'R' (Respiration) column found in %s", csv_path.name)

    # ---- GSR / EDA ----------------------------------------------------------
    if "GSR" in df.columns:
        df["GSR"] = clean_gsr(df["GSR"])
        log.info("  GSR      cleaned")
    else:
        log.warning("  No 'GSR' column found in %s", csv_path.name)

    # ---- Metadata columns ---------------------------------------------------
    df.insert(0, "subject_id", subject_id)
    df.insert(1, "file_type",  file_type)

    return df


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Flight Crew Analytics – physiological preprocessing")
    log.info("RAW_DIR : %s", RAW_DIR)
    log.info("OUT_DIR : %s", OUT_DIR)
    log.info("=" * 60)

    # Discover subject folders (numeric directory names only)
    subject_dirs = sorted(
        [d for d in RAW_DIR.iterdir() if d.is_dir() and re.fullmatch(r"\d+", d.name)],
        key=lambda d: int(d.name),
    )

    if not subject_dirs:
        log.error("No numeric subject folders found under %s", RAW_DIR)
        return

    log.info("Found %d subject folders: %s",
             len(subject_dirs), [d.name for d in subject_dirs])

    total_written = 0
    errors = []

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        log.info("")
        log.info("── Subject %s ──────────────────────────────────", subj_id)

        for ftype in FILE_TYPES:
            csv_path = subj_dir / f"{subj_id}_{ftype}.csv"

            if not csv_path.exists():
                log.warning("  Missing file: %s – skipping.", csv_path.name)
                continue

            try:
                cleaned_df = preprocess_file(csv_path, subj_id, ftype)

                out_path = OUT_DIR / f"{subj_id}_{ftype}_clean.csv"
                cleaned_df.to_csv(out_path, index=False)
                log.info("  Written  → %s", out_path.relative_to(REPO_ROOT))
                total_written += 1

            except Exception as exc:
                msg = f"Subject {subj_id} / {ftype}: {exc}"
                log.error("  FAILED – %s", msg)
                errors.append(msg)

    log.info("")
    log.info("=" * 60)
    log.info("Done.  %d files written.", total_written)
    if errors:
        log.warning("%d errors encountered:", len(errors))
        for e in errors:
            log.warning("  %s", e)
    log.info("=" * 60)


if __name__ == "__main__":
    main()