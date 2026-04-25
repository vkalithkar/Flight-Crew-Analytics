#!/bin/bash
#SBATCH --job-name=xgboost_modeling_test
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate 'tasks' (stick to 1)
#SBATCH --cpus-per-task=15 # how many cores (bora max is 20)
#SBATCH --time=3:00:00 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=48G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH -o log/xgboost_modeling_test/xgboost_modeling_test.out #STDOUT to file (%j is jobID)
#SBATCH -e log/xgboost_modeling_test/xgboost_modeling_test.err #STDERR to file (%j is jobID)

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flight2

python ./scripts/xgboost_modeling.py