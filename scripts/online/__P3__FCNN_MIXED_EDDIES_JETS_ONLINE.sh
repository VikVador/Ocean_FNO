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
#SBATCH --job-name=__P3__FCNN_MIXED_EDDIES                           # Name of the job
#SBATCH --export=ALL                             # Export all environment variables
#SBATCH --output=__P3__FCNN_MIXED_EDDIES_JETS_ONLINE.log                     # Log-file (important!)
#SBATCH --cpus-per-task=4                        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=40G                     # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                             # Number of GPU's
#SBATCH --time=1:00:00                      # Max execution time
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
python -u online.py --folder_online JETS_ONLINE --folder_models __P3__FCNN_MIXED_EDDIES --memory 40 --type_sim JETS
