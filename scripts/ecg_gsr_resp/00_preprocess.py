"""
preprocess.py  –   physiological preprocessing script for ECG, GSR, and Resp (NOT EEG!)
Flight-Crew-Analytics

Usage (from repo root):
    python scripts/ecg_gsr_resp/00_preprocess.py

For each subject folder in data/raw/, reads the four CSVs (CA, DA, LOFT, SS),
cleans ECG, Respiration, and GSR signals with NeuroKit2, and writes one
cleaned CSV per subject per file type into data/clean2/.

Sampling rate : 256 Hz (all sensors, B-Alert X24 + NeXus-10)

Changes from v1:
  - RSP: switched to biosppy method + second-pass 1 Hz low-pass
  - ECG: ADC-to-physical-unit scaling before cleaning + R-peak validation
  - All signals: post-cleaning sanity check (flags >1% samples beyond ±5 SD)
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
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
RAW_DIR    = REPO_ROOT / "data" / "raw"
OUT_DIR    = REPO_ROOT / "data" / "ecg_gsr_resp_preprocess"
LOG_FILE = REPO_ROOT / "log" / "preprocessing_ecg_gsr_resp" / "preprocessing.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

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

# NeXus-10 ECG scaling factor: raw ADC counts → physical units (µV).
# VERIFY this against your NeXus-10 hardware manual before running –
# the exact factor depends on gain settings used during acquisition.
NEXUS_ECG_SCALE = 1.22e-4   # µV per LSB  (placeholder – confirm with manual)

# Sanity-check threshold: flag a signal if more than this fraction of samples
# exceed ±SANITY_SD_THRESHOLD standard deviations from the signal mean.
SANITY_SD_THRESHOLD  = 5.0   # SD
SANITY_OUTLIER_PCT   = 1.0   # percent

# Event codes 3 and 4 are documented as erroneous – remap to 0
BAD_EVENT_CODES = {3: 0, 4: 0}


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(signal: pd.Series, signal_name: str,
                 subject_id: str, file_type: str) -> None:
    """
    Warn if more than SANITY_OUTLIER_PCT % of samples lie beyond
    ±SANITY_SD_THRESHOLD SDs from the signal mean.
    Call this after every cleaning step.
    """
    arr = signal.to_numpy(dtype=float)
    if np.nanstd(arr) == 0:
        log.warning(
            "SANITY – subject %s / %s / %s: zero-variance signal.",
            subject_id, file_type, signal_name,
        )
        return
    z = np.abs((arr - np.nanmean(arr)) / np.nanstd(arr))
    pct_outlier = (z > SANITY_SD_THRESHOLD).mean() * 100
    if pct_outlier > SANITY_OUTLIER_PCT:
        log.warning(
            "SANITY – subject %s / %s / %s: %.1f%% of samples exceed %g SD",
            subject_id, file_type, signal_name, pct_outlier, SANITY_SD_THRESHOLD,
        )


# ---------------------------------------------------------------------------
# Signal-cleaning helpers
# ---------------------------------------------------------------------------

def clean_event_column(series: pd.Series) -> pd.Series:
    """Remap erroneous event codes (3, 4) → 0."""
    return series.replace(BAD_EVENT_CODES).astype(int)


def clean_ecg(signal: pd.Series) -> pd.Series:
    """
    Clean the ECG signal with NeuroKit2.

    Step 1 – scale raw ADC counts to physical units using NEXUS_ECG_SCALE.
             NeuroKit2's internal operations assume millivolt-scale input;
             passing raw counts (order of magnitude ~10^4) can degrade
             cleaning quality.
    Step 2 – nk.ecg_clean (neurokit method): 0.5 Hz high-pass + powerline notch.

    Falls back to bandpass-only on processing errors.
    """
    arr = signal.to_numpy(dtype=float) * NEXUS_ECG_SCALE   # ADC → physical units
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


def validate_ecg_peaks(signal: pd.Series, subject_id: str) -> None:
    """
    Detect R-peaks in a cleaned ECG and log the estimated heart rate.
    Emits a warning if the derived BPM is outside the plausible range 40–180.

    Call this on at least one subject before running the full pipeline to
    confirm that the ADC scaling and cleaning are producing detectable peaks.
    """
    arr = signal.to_numpy(dtype=float)
    try:
        _, info = nk.ecg_peaks(arr, sampling_rate=FS)
        n_peaks = len(info["ECG_R_Peaks"])
        duration_min = len(arr) / FS / 60
        bpm = n_peaks / duration_min if duration_min > 0 else 0
        log.info(
            "ECG validation – subject %s: %d R-peaks detected (%.1f bpm)",
            subject_id, n_peaks, bpm,
        )
        if not (40 < bpm < 180):
            log.warning(
                "ECG validation – implausible heart rate for subject %s: %.1f bpm "
                "(check NEXUS_ECG_SCALE constant)",
                subject_id, bpm,
            )
    except Exception as exc:
        log.warning("ECG peak validation failed for subject %s: %s", subject_id, exc)


def clean_respiration(signal: pd.Series) -> pd.Series:
    """
    Clean the Respiration signal.

    Uses biosppy method (more appropriate for chest-belt strain gauges than
    khodadad2018, which targets PPG-derived respiration).  A second-pass
    1 Hz low-pass further suppresses residual high-frequency noise while
    keeping the full breath-cycle waveform intact (normal RR ~0.2–0.5 Hz).
    """
    arr = signal.to_numpy(dtype=float)
    try:
        cleaned = nk.rsp_clean(arr, sampling_rate=FS, method="biosppy")
        # Second-pass low-pass to remove residual HF buzz
        cleaned = nk.signal_filter(
            cleaned, sampling_rate=FS,
            highcut=1.0, method="butterworth", order=2,
        )
    except Exception as exc:
        log.warning("RSP clean failed (%s) – using basic bandpass fallback.", exc)
        cleaned = nk.signal_filter(
            arr, sampling_rate=FS,
            lowcut=0.1, highcut=1.0,
            method="butterworth", order=2,
        )
    return pd.Series(cleaned, index=signal.index, name=signal.name)


def clean_gsr(signal: pd.Series) -> pd.Series:
    """
    Clean the GSR / EDA signal.
    nk.eda_clean applies a low-pass filter (default 3 Hz) to remove
    high-frequency noise while preserving the slow skin-conductance response.

    Note: tonic/phasic decomposition (nk.eda_process) should be performed
    downstream in the feature-extraction stage, not here.
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


# ---------------------------------------------------------------------------
# Per-file preprocessing
# ---------------------------------------------------------------------------

def preprocess_file(csv_path: Path, subject_id: str, file_type: str,
                    validate_ecg: bool = False) -> pd.DataFrame:
    """
    Load one raw CSV, preprocess all physiological signals, return cleaned DataFrame.

    Parameters
    ----------
    csv_path     : path to the raw CSV file
    subject_id   : numeric subject identifier string
    file_type    : one of CA / DA / LOFT / SS
    validate_ecg : if True, run R-peak validation on the cleaned ECG and log BPM.
                   Set True for the first subject processed as a pipeline sanity check.
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

    # ---- ECG ----------------------------------------------------------------
    if "ECG" in df.columns:
        df["ECG"] = clean_ecg(df["ECG"])
        log.info("  ECG      cleaned")
        sanity_check(df["ECG"], "ECG", subject_id, file_type)
        if validate_ecg:
            validate_ecg_peaks(df["ECG"], subject_id)
    else:
        log.warning("  No 'ECG' column found in %s", csv_path.name)

    # ---- Respiration --------------------------------------------------------
    if "R" in df.columns:
        df["R"] = clean_respiration(df["R"])
        log.info("  RSP      cleaned")
        sanity_check(df["R"], "R", subject_id, file_type)
    else:
        log.warning("  No 'R' (Respiration) column found in %s", csv_path.name)

    # ---- GSR / EDA ----------------------------------------------------------
    if "GSR" in df.columns:
        df["GSR"] = clean_gsr(df["GSR"])
        log.info("  GSR      cleaned")
        sanity_check(df["GSR"], "GSR", subject_id, file_type)
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
    first_subject = True   # used to trigger ECG peak validation once

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
                cleaned_df = preprocess_file(
                    csv_path, subj_id, ftype,
                    validate_ecg=first_subject,   # validate on first file only
                )
                first_subject = False

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