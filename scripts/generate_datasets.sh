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
cd missing/

# --------------------
#         EDDIES
# --------------------
#
# ----  Training  ----
sbatch missing_1.sh
sbatch missing_2.sh
sbatch missing_3.sh
sbatch missing_4.sh
sbatch missing_5.sh
sbatch missing_6.sh