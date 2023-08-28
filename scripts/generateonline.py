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
online_datasets = ["EDDIES_ONLINE", "JETS_ONLINE"]

"""
model_folders   = ["__P1__FCNN_UNIQUE_JETS",
                   "__P1__UNET_UNIQUE_EDDIES",
                   "__P1__UNET_UNIQUE_JETS",
                   "__P2__FCNN_MIXED_EDDIES",
                   "__P2__FCNN_MIXED_JETS",
                   "__P2__UNET_MIXED_EDDIES",
                   "__P2__UNET_MIXED_JETS",
                   "__P3__FCNN_MIXED_EDDIES",
                   "__P3__FCNN_MIXED_JETS",
                   "__P4__FCNN_FULL"]

model_folders     = ["__P1__UNET_UNIQUE_EDDIES",
                     "__P1__UNET_UNIQUE_JETS",
                     "__P2__UNET_MIXED_EDDIES",
                     "__P2__UNET_MIXED_JETS",
                     "__P3__UNET_MIXED_EDDIES",
                     "__P3__UNET_MIXED_JETS",
                     "__P4__UNET_FULL"]
"""

"""
model_folders     = ["__P1__FNO_UNIQUE_EDDIES",
                     "__P1__FNO_UNIQUE_JETS",
                     "__P1__FFNO_UNIQUE_EDDIES",
                     "__P1__FFNO_UNIQUE_JETS",
                     "__P2__FNO_MIXED_EDDIES",
                     "__P2__FNO_MIXED_JETS",
                     "__P2__FFNO_MIXED_EDDIES",
                     "__P2__FFNO_MIXED_JETS",
                     "__P3__FNO_MIXED_EDDIES",
                     "__P3__FNO_MIXED_JETS",
                     "__P3__FFNO_MIXED_EDDIES",
                     "__P3__FFNO_MIXED_JETS",
                     "__P4__FNO_FULL",
                     "__P4__FFNO_FULL"]

model_folders= ["__P7__ARCH_32_4",
                "__P7__ARCH_32_8",
                "__P7__ARCH_32_12",
                "__P7__ARCH_32_16",
                "__P7__ARCH_32_20",
                "__P7__ARCH_32_24",
                "__P7__ARCH_64_4",
                "__P7__ARCH_64_8",
                "__P7__ARCH_64_12",
                "__P7__ARCH_64_16",
                "__P7__ARCH_64_20",
                "__P7__ARCH_64_24",
                "__P7__ARCH_128_4",
                "__P7__ARCH_128_8",
                "__P7__ARCH_128_12",
                "__P7__ARCH_128_16",
                "__P7__ARCH_128_20",
                "__P7__ARCH_128_24",
                "__P7__ARCH_GOAT__"]
"""

model_folders= ["__P10__FINAL"]

#model_folders = [original_list[len(original_list) - i] for i in range(1, len(original_list)+1)]

# -----------------
#    Generation
# -----------------
# Contains all the job names created
job_list = ""

# Used to keep count of everything happening
index = 1

for i, m in enumerate(model_folders):
    for j, d in enumerate(online_datasets):

        # Initalization
        type_sim = None
        time     = None
        mem      = None

        # ---------------------------------
        #   Automatic Parameters Tuning
        # ---------------------------------
        if "EDDIES" in d:
            type_sim = "EDDIES"
        elif "JETS" in d:
            type_sim = "JETS"
        else:
            type_sim = "NONE"

        if "P4" in m:
            time = "24" if "FFNO" in m else "12"
            mem  = "50"
        else:
            time = "24" if "FFNO" in m else "12"
            mem  = "40"

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
#SBATCH --job-name={m}                           # Name of the job
#SBATCH --export=ALL                             # Export all environment variables
#SBATCH --output={m}_{d}.log             # Log-file (important!)
#SBATCH --cpus-per-task=4                        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu={mem}G                     # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                             # Number of GPU's
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
python -u online.py --folder_online {d} --folder_models {m} --memory 40 --type_sim {type_sim}
"""

        # Opening file
        job_file = open(f"online/{m}_{d}.sh", "w")
        job_file.write(job)
        job_file.close()

        # Complete list of jobs for sending sbatch
        job_list += f"sbatch {m}_{d}.sh\n"

        # Updating index
        index = index + 1

# Creating sbatch file for cluster
job_file = open(f"online/CLUSTER_ONLINE.sh", "w")
job_file.write(job_list)
job_file.close()
