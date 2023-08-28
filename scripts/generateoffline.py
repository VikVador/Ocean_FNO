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
offline_datasets  = ["EDDIES_OFFLINE", "JETS_OFFLINE", "FULL_OFFLINE"]

"""
model_folders     = ["__P1__FCNN_UNIQUE_JETS",
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
"""

"""
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

model_folders= ["__P8__ARCH_modes_0_8",
                "__P8__ARCH_modes_0_16",
                "__P8__ARCH_modes_0_24",
                "__P8__ARCH_modes_0_32",
                "__P8__ARCH_modes_8_16",
                "__P8__ARCH_modes_16_24",
                "__P8__ARCH_modes_24_32",
                "__P9__ARCH_layers_4",
                "__P9__ARCH_layers_8",
                "__P9__ARCH_layers_12",
                "__P9__ARCH_layers_16",
                "__P9__ARCH_layers_20",
                "__P9__ARCH_layers_24"]
"""

model_folders= ["__P10__FINAL"]

# -----------------
#    Generation
# -----------------
# Keeps curent job index
job_index = 1

# Contains all the job names created
job_list = ""

for m in model_folders:
    for d in offline_datasets:

        # Initalization
        type_sim = None

        # ---------------------------------
        #   Automatic Parameters Tuning
        # ---------------------------------
        if "EDDIES" in d:
            type_sim = "EDDIES"
        elif "JETS" in d:
            type_sim = "JETS"
        else:
            type_sim = "FULL"

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
#SBATCH --job-name={m}_{type_sim}                 # Name of the job
#SBATCH --export=ALL                             # Export all environment variables
#SBATCH --output={m}_{type_sim}.log               # Log-file (important!)
#SBATCH --cpus-per-task=2                        # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=32G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                             # Number of GPU's
#SBATCH --time=3:00:00                           # Max execution time
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
python -u offline.py --folder_offline {d} --folder_models {m} --memory 32 --type_sim {type_sim}
"""

        # Opening file
        job_file = open(f"offline/{m}_{d}.sh", "w")
        job_file.write(job)
        job_file.close()

        # Complete list of jobs for sending sbatch
        job_list += f"sbatch {m}_{d}.sh\n"

# Creating sbatch file for cluster
job_file = open(f"offline/CLUSTER_OFFLINE.sh", "w")
job_file.write(job_list)
job_file.close()
