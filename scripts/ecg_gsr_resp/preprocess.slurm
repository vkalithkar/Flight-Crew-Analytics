#!/bin/bash
#SBATCH --job-name=ecg_gsr_resp_preprocess
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate 'tasks' (stick to 1)
#SBATCH --cpus-per-task=4 # how many cores (bora max is 20)
#SBATCH --time=45 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=16G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH -o log/preprocessing_ecg_gsr_resp/ecg_gsr_resp_preprocess.out #STDOUT to file (%j is jobID)
#SBATCH -e log/preprocessing_ecg_gsr_resp/ecg_gsr_resp_preprocess.err #STDERR to file (%j is jobID)

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flight2

mkdir -p log/preprocessing_ecg_gsr_resp
python ./scripts/ecg_gsr_resp/00_preprocess.py