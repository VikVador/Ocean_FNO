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
param_name        = ["UNET"]
inputs            = ["q", "q u", "q v", "q u v"]
targets           = ["q_forcing_total", "q_subgrid_forcing", "q_fluxes"]
sim_type          = "UNET_EDDIES_UNIQUE"
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

                # Initalization
                zm                = None
                mem               = None
                sim_time          = None
                validation_folder = None

                # ---------------------------------
                #   Automatic Parameters Tuning
                # ---------------------------------
                # RAM
                if "20000"   in d:
                    mem = "48"
                elif "40000" in d:
                    mem = "90"
                else:
                    mem = "32"

                # Job time
                if "20000"   in d:
                    sim_time = "2"
                elif "40000" in d:
                    sim_time = "4"
                else:
                    sim_time = "1"

                # Zero mean
                zm = "False" if t == "q_fluxes" else "True"

                # Validation folder
                if "EDDIES" in d:
                    validation_folder = "EDDIES_VALIDATION"
                if "JETS"   in d:
                     validation_folder = "JETS_VALIDATION"
                if "FULL"   in d:
                     validation_folder = "FULL_VALIDATION"

                 # Complete name of the job
                job_name = f"{sim_type}_param_{job_index}"

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
