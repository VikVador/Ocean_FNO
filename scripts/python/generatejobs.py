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
#     Librairies
# -----------------
#
# --------- Standard ---------


# -----------------
#    Parameters
# -----------------
# Number of jobs to generate
nb_jobs = 1

# Name of the folder containing dataset
dataset_name = "JETS_TRAINING_UNIQUE_5000"

# Simulation type
sim_type = 1

# Number of samples targeted
target_sample = 5000

# Skipped time before sampling
skipped_time = 6

# Number of threads for parallel computation
nb_threads = 4

# Ram memory needed
ram = 32

# Saving high resolution samples
save_HR = False

# -----------------
#    Generation
# -----------------
# Keeps curent job index
job_index = 1

# Contains all the job names created
job_list = ""

# Generation of the jobs
for i in range(nb_jobs):

    # Complete name of the job
    job_name = f"{dataset_name}_{job_index}"

    job =f"""#!/usr/bin/env bash
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
#SBATCH --job-name={job_name}             # Name of the job
#SBATCH --export=ALL                      # Export all environment variables
#SBATCH --output={job_name}.log           # Log-file (important!)
#SBATCH --cpus-per-task={str(nb_threads)} # Number of CPU cores to allocate
#SBATCH --mem-per-cpu={str(ram)}G         # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                      # Number of GPU's
#SBATCH --time=1:30:00                    # Max execution time
#
# Activate your Anaconda environment
source ~/.bashrc
conda activate TFE

# Moving to correct folder
cd ../../src/pyqg_parameterization_benchmarks/

# ---------------------
#   Script parameters
# ---------------------
python -u generate_dataset.py --save_folder {job_name} --simulation_type {str(sim_type)} --target_sample_size {str(target_sample)}  --operator_cf 1 --skipped_time {str(skipped_time)} --nb_threads {str(nb_threads)} --memory {str(ram)} --save_high_res {str(save_HR)}
    """

    # Opening file
    job_file = open(f"{job_name}.sh", "w")
    job_file.write(job)
    job_file.close()
    job_index = job_index + 1

    # Complete list of jobs for sending sbatch
    job_list += f"sbatch {job_name}.sh\n"

# Creating sbatch file for cluster
job_file = open(f"CLUSTER_{dataset_name}.sh", "w")
job_file.write(job_list)
job_file.close()
