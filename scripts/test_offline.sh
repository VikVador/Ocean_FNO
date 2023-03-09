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
cd parameterization/baseline/

# --------------------
#         5000
# --------------------
cd 5000/

sbatch baseline_FCNN_5000_1.sh
sbatch baseline_FCNN_5000_2.sh
sbatch baseline_FCNN_5000_3.sh
sbatch baseline_FCNN_5000_4.sh
sbatch baseline_FCNN_5000_5.sh
sbatch baseline_FCNN_5000_6.sh
sbatch baseline_FCNN_5000_7.sh
sbatch baseline_FCNN_5000_8.sh

# --------------------
#         10000
# --------------------
cd ../10000

sbatch baseline_FCNN_10000_1.sh
sbatch baseline_FCNN_10000_2.sh
sbatch baseline_FCNN_10000_3.sh
sbatch baseline_FCNN_10000_4.sh
sbatch baseline_FCNN_10000_5.sh
sbatch baseline_FCNN_10000_6.sh
sbatch baseline_FCNN_10000_7.sh
sbatch baseline_FCNN_10000_8.sh

# --------------------
#         20000
# --------------------
cd ../20000

sbatch baseline_FCNN_20000_1.sh
sbatch baseline_FCNN_20000_2.sh
sbatch baseline_FCNN_20000_3.sh
sbatch baseline_FCNN_20000_4.sh
sbatch baseline_FCNN_20000_5.sh
sbatch baseline_FCNN_20000_6.sh
sbatch baseline_FCNN_20000_7.sh
sbatch baseline_FCNN_20000_8.sh