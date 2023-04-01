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
#SBATCH --job-name=baseline_FCNN_20000_8        # Name of the job
#SBATCH --export=ALL                           # Export all environment variables
#SBATCH --output=baseline_FCNN_20000_8.log      # Log-file (important!)
#SBATCH --cpus-per-task=4                      # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G                      # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                           # Number of GPU's
#SBATCH --time=2:00:00                         # Max execution time
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
numepoch=50
memcpu=32

# Note: 29 [Gb] needed to store training 1 -> 20 + validation 1 -> 10
# ---------------------
#         Helper
# ---------------------
# 5 training sets
# --folder_training eddies_training_1 eddies_training_2 eddies_training_3 eddies_training_4 eddies_training_5
#
# 10 training sets
# --folder_training eddies_training_1 eddies_training_2 eddies_training_3 eddies_training_4 eddies_training_5 eddies_training_6 eddies_training_7 eddies_training_8 eddies_training_9 eddies_training_10
#
# 20 training sets
# --folder_training eddies_training_1 eddies_training_2 eddies_training_3 eddies_training_4 eddies_training_5 eddies_training_6 eddies_training_7 eddies_training_8 eddies_training_9 eddies_training_10 eddies_training_11 eddies_training_12 eddies_training_13 eddies_training_14 eddies_training_15 eddies_training_16 eddies_training_17 eddies_training_18 eddies_training_19 eddies_training_20
#
# Choices of target
# --targets q_subgrid_forcing
# --targets q_forcing_total
# --targets u_subgrid_forcing
# --targets v_subgrid_forcing
# ---------------------
#   Script parameters
# ---------------------
#
python -u train_parameterization.py --inputs q u v --targets q_forcing_total --param_type FCNN --param_name FCNN_Baseline --sim_type fcnn/baseline --folder_training eddies_training_1 eddies_training_2 eddies_training_3 eddies_training_4 eddies_training_5 eddies_training_6 eddies_training_7 eddies_training_8 eddies_training_9 eddies_training_10 eddies_training_11 eddies_training_12 eddies_training_13 eddies_training_14 eddies_training_15 eddies_training_16 eddies_training_17 eddies_training_18 eddies_training_19 eddies_training_20 --num_epochs $numepoch --memory $memcpu --zero_mean True --padding circular --folder_validation eddies_validation_1 eddies_validation_2 eddies_validation_3 eddies_validation_4 eddies_validation_5 eddies_validation_6 eddies_validation_7 eddies_validation_8 eddies_validation_9 eddies_validation_10