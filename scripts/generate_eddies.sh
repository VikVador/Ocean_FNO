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
