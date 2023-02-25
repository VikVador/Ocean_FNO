#----------------------------------------------------------------------
#
# Ocean parameterizations in an idealized model using machine learning
#
#----------------------------------------------------------------------
# @ Victor Mangeleer
#
#---------------
# Documentation
#---------------
# This script has for purpose to generate datasets
#
# ---------------------------------- PARAMETERS ----------------------------------

# Name of the folder used to save the datasets (../datasets/save_folder)
save_folder = "eddies_1"

# Simulation type to use to generate the high resolution data
simulation_type = 4

# Frequency to save the results (Ex: if sampling = 1, each hour the data is saved)
sampling_frequency = 50

# Operator to apply on the high resolution data
operator_index = 1

# Size of the dataset that will be used as training data
train_size = 0.7

# Number of threads to use by PYQG
number_threads = 2

# Determine if the high resolution training set should be saved entirely or just the last sample 
save_HR_train = False
# ---------------------------------- PARAMETERS ----------------------------------
#
#
#
#
#
#-------------------------------------------------------------
#                           Librairies
#-------------------------------------------------------------
import os
import sys
import glob
import numpy as np

import json
import fsspec
import xarray as xr
import pyqg
import pyqg.diagnostic_tools
import coarsening_ops as coarsening

from utils import *
from scipy.stats import gaussian_kde
from tqdm.notebook import tqdm, trange
from pyqg.diagnostic_tools import calc_ispec as _calc_ispec

print("TEST")

