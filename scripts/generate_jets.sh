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
#         JETS
# --------------------
#
# ----  Training  ----
sbatch jets_training_1.sh
sbatch jets_training_2.sh
sbatch jets_training_3.sh
sbatch jets_training_4.sh
sbatch jets_training_5.sh
sbatch jets_training_6.sh
