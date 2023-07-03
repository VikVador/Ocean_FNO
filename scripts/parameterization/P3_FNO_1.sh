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
#SBATCH --job-name=P3_FNO_1                    # Name of the job
#SBATCH --export=ALL                             # Export all environment variables
#SBATCH --output=P3_FNO_1.log                  # Log-file (important!)
#SBATCH --cpus-per-task=4                        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G                     # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                             # Number of GPU's
#SBATCH --time=2:00:00                  # Max execution time
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
python -u train_parameterization.py --inputs q --targets q_fluxes --param_type FNO --param_name FNO --sim_type P3 --folder_training EDDIES_TRAINING_MIXED_10000 --num_epochs 50 --memory 32 --zero_mean False --padding circular --folder_validation EDDIES_VALIDATION
    