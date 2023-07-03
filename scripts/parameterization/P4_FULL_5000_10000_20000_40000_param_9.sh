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
#SBATCH --job-name=P4_FULL_5000_10000_20000_40000_param_9                    # Name of the job
#SBATCH --export=ALL                             # Export all environment variables
#SBATCH --output=P4_FULL_5000_10000_20000_40000_param_9.log                  # Log-file (important!)
#SBATCH --cpus-per-task=4                        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=48G                     # Memory to allocate per allocated CPU core
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
python -u train_parameterization.py --inputs q --targets q_fluxes --param_type UNET --param_name UNET --sim_type P4_FULL_5000_10000_20000_40000 --folder_training FULL_TRAINING_MIXED_20000 --num_epochs 50 --memory 48 --zero_mean False --padding circular --folder_validation FULL_VALIDATION
    