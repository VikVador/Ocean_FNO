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
#SBATCH --job-name=eddies_training_4    # Name of the job
#SBATCH --export=ALL                    # Export all environment variables
#SBATCH --output=eddies_training_4.log  # Log-file (important!)
#SBATCH --cpus-per-task=4               # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G               # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                    # Number of GPU's
#SBATCH --time=1:00:00                  # Max execution time
#
# Activate your Anaconda environment
source ~/.bashrc
conda activate TFE

# Moving to correct folder
cd ../../src/pyqg_parameterization_benchmarks/

# ---------------------
#   Script parameters
# ---------------------
nbthreads=4
memcpu=32
savehr=False

# ---------------------
#   Script parameters
# ---------------------
python -u generate_dataset.py --save_folder eddies_training_4 --simulation_type 0 --target_sample_size 3000 --operator_cf 1 --skipped_time 0.5 --nb_threads $nbthreads --memory $memcpu --save_high_res $savehr