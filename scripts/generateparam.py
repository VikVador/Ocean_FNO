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
training_folder   = ["EDDIES_TRAINING_UNIQUE_5000"]
param_name        = ["FCNN"]
inputs            = ["q", "q u", "q v", "q u v"]
targets           = ["q_forcing_total", "q_subgrid_forcing", "q_fluxes"]

validation_folder = "EDDIES_VALIDATION"
sim_type          = "FINAL_EDDIES_UNIQUE"
number_epoch      = 50

# -----------------
#    Generation
# -----------------
# Keeps curent job index
job_index = 1

# Contains all the job names created
job_list = ""

for d in training_folder:
    for p in param_name:
        for i in inputs:
            for t in targets:

                # Computing memory
                mem = "48" if "20000" in d else "32"

                # Simulation time
                sim_time = "3" if "20000" in d else "2"

                 # Complete name of the job
                job_name = f"{sim_type}_param_{job_index}"

                # Zero mean
                zm = "False" if t == "q_fluxes" else "True"

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
#SBATCH --cpus-per-task=4                        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu={mem}G                     # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                             # Number of GPU's
#SBATCH --time={sim_time}:00:00                  # Max execution time
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
python -u train_parameterization.py --inputs {i} --targets {t} --param_type {p} --param_name {p} --sim_type {sim_type} --folder_training {d} --num_epochs {str(number_epoch)} --memory {mem} --zero_mean {zm} --padding circular --folder_validation {validation_folder}
    """

                # Opening file
                job_file = open(f"{job_name}.sh", "w")
                job_file.write(job)
                job_file.close()
                job_index = job_index + 1

                # Complete list of jobs for sending sbatch
                job_list += f"sbatch {job_name}.sh\n"

# Creating sbatch file for cluster
job_file = open(f"CLUSTER_{sim_type}.sh", "w")
job_file.write(job_list)
job_file.close()
