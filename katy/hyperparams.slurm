#!/bin/bash
#SBATCH --job-name=hyperparams_flight
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate tasks (stick to 1)
#SBATCH --cpus-per-task=20 # how many cores (bora max is 20)
#SBATCH --time=90 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=32G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH -o log/hyperparams/hyperparams_%j.out #STDOUT to file (%j is jobID)
#SBATCH -e log/hyperparams/hyperparams_%j.err #STDERR to file (%j is jobID)

set -euo pipefail

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flight2

# Downloads the data
python ./katy/hyperparam_optimization.py