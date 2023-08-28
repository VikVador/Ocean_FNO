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
learning_rate = ["0.1", "0.01", "0.001"]
batch_size    = ["64"]
optimizer     = ["adam", "sgd"]
scheduler     = ["constant",
                 "multi_step",
                 "exponential",
                 "cosine",
                 "cosine_warmup_restart",
                 "cyclic"]

# -----------------
#       Fixed
# -----------------
dataset_training   = ["FULL_TRAINING_MIXED_5000"]
dataset_validation = "FULL_VALIDATION"
architecture       = "FFNO"
inputs             = "q u v"
targets            = "q_fluxes"
num_epochs         = "50"


# -----------------
#    Generation
# -----------------
# Keeps curent job index
job_index = 1

# Contains all the job names created
job_list = ""

for train in dataset_training:
    for bs in batch_size:
        for lr in learning_rate:
            for opt in optimizer:
                for sch in scheduler:

                    # Creation of the Job name
                    job_name = "SENS_" + str(job_index)

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
#SBATCH --gres=gpu:4                             # Number of GPU's
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
python -u train_parameterization.py --folder_training {train} --folder_validation {dataset_validation} --param_name {architecture} --inputs {inputs} --targets {targets} --learning_rate {lr} --batch_size {bs} --optimizer {opt} --scheduler {sch} --configuration default --num_epochs {num_epochs} --zero_mean False --padding circular --memory 32 --param_type SENSITIVITY_{architecture} --sim_type SENSITIVITY_{architecture}
"""

                    # Opening file
                    job_file = open(f"sensitivity/{job_name}.sh", "w")
                    job_file.write(job)
                    job_file.close()

                    # Complete list of jobs for sending sbatch
                    job_list += f"sbatch {job_name}.sh\n"
                    job_index = job_index + 1

# Creating sbatch file for cluster
job_file = open(f"sensitivity/CLUSTER_SENSITIVITY.sh", "w")
job_file.write(job_list)
job_file.close()
