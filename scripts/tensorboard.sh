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
# ------------
# Installation
# ------------
# 1 - Variables
#
# alan_username = vmangeleer
# remote_port   = 1280
# local_port    = 1280
# remote_path   = /home/vmangeleer/TFE/pyqg_parameterization_benchmarks/runs/
#
#
# 2 - Setting up local to listen on ALAN's port
#
# Symbolic : ssh -N -f -L localhost:{local_port}:localhost:{remote_port} {alan_username}@master.alan.priv
#
# Complete : ssh -N -f -L localhost:1280:localhost:1280 vmangeleer@master.alan.priv
#
#
# 3 - Running tensorboard on ALAN
#
# conda activate TFE
#
# Symbolic : tensorboard --logdir {remote_path} --port {remote_port}
#
# Complete : tensorboard --logdir /home/vmangeleer/TFE/pyqg_parameterization_benchmarks/runs/ --port 1280
#
# -------------------
# Running tensorboard
# -------------------
conda activate TFE
tensorboard --logdir /home/vmangeleer/TFE/pyqg_parameterization_benchmarks/runs/ --port 1280