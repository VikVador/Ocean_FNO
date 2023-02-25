#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=eddies_HR        # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=eddies_HR.log      # Log-file (important!)
#SBATCH --cpus-per-task=1           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                # Number of GPU's
#SBATCH --time=0:05:00              # Max execution time

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ../notebooks/src/pyqg_parameterization_benchmarks/ 
python generate_data.py