#!/bin/bash
#SBATCH --job-name=mne_ica
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate 'tasks' (stick to 1)
#SBATCH --cpus-per-task=8 # how many cores (bora max is 20)
#SBATCH --time=2:00:00 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=48G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH --mail-user=vkalithkar@wm.edu # who to email
#SBATCH -o vandana/output_dump/mne_ica.out #STDOUT to file (%j is jobID)
#SBATCH -e vandana/output_dump/mne_ica.err #STDERR to file (%j is jobID)

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flight-env

#python ./vandana/00_get_bad_channels.py
#python ./vandana/01_interpolate_filter_reref.py
python ./vandana/02_ica.py