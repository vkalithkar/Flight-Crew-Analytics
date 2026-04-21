#!/bin/bash
#SBATCH --job-name=feature_engineering
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate 'tasks' (stick to 1)
#SBATCH --cpus-per-task=4 # how many cores (bora max is 20)
#SBATCH --time=02:00:00 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=32G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH -o log/feature_engineering/feature_engineering.out #STDOUT to file (%j is jobID)
#SBATCH -e log/feature_engineering/feature_engineering.err #STDERR to file (%j is jobID)

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flight2

mkdir -p log/feature_engineering

./scripts/feature_engineering.py
