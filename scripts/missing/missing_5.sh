#!/usr/bin/env bash
#------------------------------------------------------------------------------
#
#           Ocean subgrid parameterization using machine learning
#
#                             Graduation work
#
#------------------------------------------------------------------------------
# @ Victor Mangeleer
#
# ---------------------
#    Slurm arguments
# ---------------------
#SBATCH --job-name=missing_5              # Name of the job
#SBATCH --export=ALL                      # Export all environment variables
#SBATCH --output=missing_5.log            # Log-file (important!)
#SBATCH --cpus-per-task=2                 # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G                 # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                      # Number of GPU's
#SBATCH --time=1:00:00                    # Max execution time
#
# Activate your Anaconda environment
source ~/.bashrc
conda activate TFE

# Moving to correct folder
cd ../../src/pyqg_parameterization_benchmarks/

# ---------------------
#   Script parameters
# ---------------------
python -u DATASET_5.py
    