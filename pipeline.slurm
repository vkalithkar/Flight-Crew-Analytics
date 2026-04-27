#!/bin/bash
#SBATCH --job-name=final_pipeline
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate tasks (stick to 1)
#SBATCH --cpus-per-task=20 # how many cores (bora max is 20)
#SBATCH --time=12:00:00 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=32G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH -o log/pipeline/pipeline_%j.out #STDOUT to file (%j is jobID)
#SBATCH -e log/pipeline/pipeline_%j.err #STDERR to file (%j is jobID)

set -euo pipefail

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flight2

# Downloads the data
./scripts/01_download_data.sh

# Makes a combined version of all the data if needed for further analysis
# python ./02_combined_data.py

# Preprocessing all of the data besides EEG
python ./scripts/ecg_gsr_resp/00_preprocess.py

# These scripts are for preprocessing EEG
python ./scripts/eeg/00_get_bad_channels.py
python ./scripts/eeg/01_interpolate_filter_reref.py
python ./scripts/eeg/02_ica.py
python ./scripts/eeg/03_epoching.py

# Feature engineering script to prepare data for xgboost

python ./scripts/03_feature_engineering.py

# XGBoost model
python ./scripts/04_xgboost_modeling.py
