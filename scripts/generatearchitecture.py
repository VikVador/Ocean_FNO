#------------------------------------------------------------------------------
#
#           Ocean subgrid parameterization using machine learning
#
#                             Graduation work
#
#------------------------------------------------------------------------------
# @ Victor Mangeleer
#
# -----------------
#    Parameters
# -----------------
# Total number of configurations
nb_conf = 126

"""

    7 * 3 * 6

modes_low_pass  = [(0, 8),  (0, 16), (0, 24), (0, 32)]
modes_pass_band = [(8, 16), (16, 24), (24, 32)]
modes           = modes_low_pass + modes_pass_band
width           = [32, 64, 128]
layers          = [4, 8, 12, 16, 20, 24]
weights         = [True]
"""

# -----------------
#       Fixed
# -----------------
dataset_training   = "FULL_TRAINING_MIXED_5000"
dataset_validation = "FULL_VALIDATION"
architecture       = "FFNO"
inputs             = "q u v"
targets            = "q_fluxes"
num_epochs         = "50"
learning_rate      = "0.001"
batch_size         = "32"
optimizer          = "adam"
scheduler          = "constant"

# -----------------
#    Generation
# -----------------
# Contains all the job names created
job_list = ""

for conf in range(nb_conf):

    # Creation of the Job name
    job_name = "ARCH_" + str(conf)

    # If width == 128, we need bigger equipement !
    if conf >= 84:
        time = "8"
        gpus = "4"
    else:
        time = "3"
        gpus = "2"

    job = f"""#!/usr/bin/env bash
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
#SBATCH --job-name={job_name}                    # Name of the job
#SBATCH --export=ALL                             # Export all environment variables
#SBATCH --output={job_name}.log                  # Log-file (important!)
#SBATCH --cpus-per-task=2                        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:{gpus}                        # Number of GPU's
#SBATCH --time={time}:00:00                      # Max execution time
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
python -u train_parameterization.py --folder_training {dataset_training} --folder_validation {dataset_validation} --param_name {architecture} --inputs {inputs} --targets {targets} --learning_rate {learning_rate} --batch_size {batch_size} --optimizer {optimizer} --scheduler {scheduler} --configuration {conf} --num_epochs {num_epochs} --zero_mean False --padding circular --memory 32 --param_type ARCH_{architecture} --sim_type ARCH_{architecture}
"""

    # Opening file
    job_file = open(f"architecture/{job_name}.sh", "w")
    job_file.write(job)
    job_file.close()

    # Complete list of jobs for sending sbatch
    job_list += f"sbatch {job_name}.sh\n"

# Creating sbatch file for cluster
job_file = open(f"architecture/CLUSTER_ARCHITECTURE.sh", "w")
job_file.write(job_list)
job_file.close()
