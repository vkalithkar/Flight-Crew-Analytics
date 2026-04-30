#!/bin/bash
#SBATCH --job-name=final_pipeline
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate tasks (stick to 1)
#SBATCH --cpus-per-task=20 # how many cores (bora max is 20)
#SBATCH --time=12:00:00 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=32G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH -o log/pipeline_%j.out #STDOUT to file (%j is jobID)
#SBATCH -e log/pipeline_%j.err #STDERR to file (%j is jobID)

set -euo pipefail

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flight2

# Downloads the data
./scripts/00_download_data.sh

# Preprocessing all of the data besides EEG
python ./scripts/01_ecg_gsr_resp/00_preprocess.py

# These scripts are for preprocessing EEG
python ./scripts/02_eeg/00_get_bad_channels.py
python ./scripts/02_eeg/01_interpolate_filter_reref.py
python ./scripts/02_eeg/02_ica.py
python ./scripts/02_eeg/03_epoching.py

# EDA Plot Generation Script
python ./scripts/03_eda_plots.py

# Feature engineering script to prepare data for xgboost
python ./scripts/04_feature_engineering.py

# XGBoost model
python ./scripts/05_model_creation_HPO.py

# cleaning up data directory
rm -rf data/ecg_gsr_resp_preprocess/
rm -rf data/eeg_preprocess/
rm -rf data/feature_extracted/