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
# This script is used to retreive the main ressources of my TFE on Alan !
#
mkdir update
cd update
scp -r alan:/home/vmangeleer/TFE/pyqg_parameterization_benchmarks/notebooks notebooks
scp -r alan:/home/vmangeleer/TFE/pyqg_parameterization_benchmarks/models/base base
scp -r alan:/home/vmangeleer/TFE/pyqg_parameterization_benchmarks/models/kaskade kaskade
scp -r alan:/home/vmangeleer/TFE/pyqg_parameterization_benchmarks/src src
scp -r alan:/home/vmangeleer/TFE/pyqg_parameterization_benchmarks/scripts scripts
scp alan:/home/vmangeleer/TFE/pyqg_parameterization_benchmarks/environment.yml environment.yml
