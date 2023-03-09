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
# Move to correct folder
cd datasets/

# --------------------
#         EDDIES
# --------------------
#
# ----  Training  ----
sbatch eddies_training_1.sh
sbatch eddies_training_2.sh
sbatch eddies_training_3.sh
sbatch eddies_training_4.sh
sbatch eddies_training_5.sh
sbatch eddies_training_6.sh
sbatch eddies_training_7.sh
sbatch eddies_training_8.sh
sbatch eddies_training_9.sh
sbatch eddies_training_10.sh
sbatch eddies_training_11.sh
sbatch eddies_training_12.sh
sbatch eddies_training_13.sh
sbatch eddies_training_14.sh
sbatch eddies_training_15.sh
sbatch eddies_training_16.sh
sbatch eddies_training_17.sh
sbatch eddies_training_18.sh
sbatch eddies_training_19.sh
sbatch eddies_training_20.sh

# ---- Validation ----
sbatch eddies_validation_1.sh
sbatch eddies_validation_2.sh
sbatch eddies_validation_3.sh
sbatch eddies_validation_4.sh
sbatch eddies_validation_5.sh
sbatch eddies_validation_6.sh
sbatch eddies_validation_7.sh
sbatch eddies_validation_8.sh
sbatch eddies_validation_9.sh
sbatch eddies_validation_10.sh

# ----   Offline  ----
sbatch eddies_offline_1.sh
sbatch eddies_offline_2.sh
sbatch eddies_offline_3.sh
sbatch eddies_offline_4.sh
sbatch eddies_offline_5.sh
sbatch eddies_offline_6.sh
sbatch eddies_offline_7.sh
sbatch eddies_offline_8.sh
sbatch eddies_offline_9.sh
sbatch eddies_offline_10.sh

# ----   Online   ----
sbatch eddies_online_1.sh
sbatch eddies_online_2.sh
sbatch eddies_online_3.sh
sbatch eddies_online_4.sh
sbatch eddies_online_5.sh