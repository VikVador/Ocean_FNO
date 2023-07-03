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
#SBATCH --job-name=UNET_EDDIES_UNIQUE_param_3                    # Name of the job
#SBATCH --export=ALL                             # Export all environment variables
#SBATCH --output=UNET_EDDIES_UNIQUE_param_3.log                  # Log-file (important!)
#SBATCH --cpus-per-task=4                        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G                     # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                             # Number of GPU's
#SBATCH --time=1:00:00                  # Max execution time
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
python -u train_parameterization.py --inputs q --targets q_fluxes --param_type UNET --param_name UNET --sim_type UNET_EDDIES_UNIQUE --folder_training EDDIES_TRAINING_UNIQUE_5000 --num_epochs 50 --memory 32 --zero_mean False --padding circular --folder_validation EDDIES_VALIDATION
    