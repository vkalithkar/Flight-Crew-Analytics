#!/bin/bash
#SBATCH --job-name=eeg_preprocess
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate 'tasks' (stick to 1)
#SBATCH --cpus-per-task=8 # how many cores (bora max is 20)
#SBATCH --time=1:00:00 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=48G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH -o log/preprocessing_eeg/eeg_preprocess.out #STDOUT to file (%j is jobID)
#SBATCH -e log/preprocessing_eeg/eeg_preprocess.err #STDERR to file (%j is jobID)

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flight2

#python ./scripts/eeg/00_get_bad_channels.py
python ./scripts/eeg/01_interpolate_filter_reref.py
python ./scripts/eeg/02_ica.py
python ./scripts/eeg/03_epoching.py