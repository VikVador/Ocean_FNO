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
#SBATCH --job-name=__P3__FNO_MIXED_EDDIES                           # Name of the job
#SBATCH --export=ALL                             # Export all environment variables
#SBATCH --output=__P3__FNO_MIXED_EDDIES.log                         # Log-file (important!)
#SBATCH --cpus-per-task=2                        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                             # Number of GPU's
#SBATCH --time=1:00:00                           # Max execution time
#SBATCH --partition=1080ti,2080ti,quadro,tesla
#
# Activate your Anaconda environment
source ~/.bashrc
conda activate TFE

# Moving to correct folder
cd ../../src/pyqg_parameterization_benchmarks/

# ---------------------
#   Script parameters
# ---------------------
#
python -u offline.py --folder_offline FULL_OFFLINE --folder_models __P3__FNO_MIXED_EDDIES --memory 32 --type_sim FULL
