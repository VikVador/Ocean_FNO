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
# This script has for purpose to train a FCNN
#
# ---------------------------------- PARAMETERS ----------------------------------
#
# Define the name of the folder to store the resulting network
result_folder = "fcnn_test"

# Define the folder from which to load the datasets
loading_folder_data = ["eddies_small_test"]

# Inputs used by the FCNN ['q', 'u', 'v']
inputs     = ['q']

# Output of the FCNN ['q_subgrid_forcing']
targets    = ['q_subgrid_forcing']

# Number of epochs for training per layer
num_epochs = 1, 

# ---------------------------------- PARAMETERS ----------------------------------
#
#
#
#
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
import torch
import random
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import json
import fsspec
import xarray as xr
import pyqg
import pyqg.diagnostic_tools
import coarsening_ops as coarsening

from scipy.stats import gaussian_kde
from tqdm.notebook import tqdm, trange
from utils import *
from pyqg.diagnostic_tools import calc_ispec as _calc_ispec
from neural_networks import FullyCNN
from neural_networks import FCNNParameterization
from online_metrics import diagnostic_differences

#-------------------------------------------------------------
#                           Functions
#-------------------------------------------------------------
# Used to compute the power spectrum
calc_ispec = lambda *args, **kwargs: _calc_ispec(*args, averaging = False, truncate =False, **kwargs)

def getPath(folder_name, loading = False):
    """
    Used to determine the path to the results' folder for the FCNN
    """
    # Count number of already existing folders in the model folder
    nb_files = len(glob.glob("../../models/*"))

    # Creation of a non-existing folder name
    return f"../../models/{folder_name}/" if loading else f"../../models/{folder_name}_{nb_files}/"

def config_for(model):
    """
    Return the parameters needed to initialize a new pyqg.QGModel, except for nx and ny.
    """
    config = dict(H1 = model.Hi[0])
    for prop in ['L', 'W', 'dt', 'rek', 'g', 'beta', 'delta','U1', 'U2', 'rd']:
        config[prop] = getattr(model, prop)
    return config

#----------------------------------------------------------------------------------------------------------------------
#                                                      MAIN
#----------------------------------------------------------------------------------------------------------------------
#
#-------------------------------------------------------------
#                        Loading the data      
#-------------------------------------------------------------
#
print("Loading the data")
#
# Creation of the different paths
path_train_HR, path_test_HR, path_train_LR,\
path_test_LR, path_train_ALR, path_test_ALR = list(), list(), list(), list(), list(), list()

for p in loading_folder_data:
    path_train_HR.append( "../../datasets/" + p + "/train_HR.nc")
    path_train_LR.append( "../../datasets/" + p + "/train_LR.nc")
    path_train_ALR.append("../../datasets/" + p + "/train_ALR.nc")
    path_test_HR.append(  "../../datasets/" + p + "/test_HR.nc")
    path_test_LR.append(  "../../datasets/" + p + "/test_LR.nc")
    path_test_ALR.append( "../../datasets/" + p + "/test_ALR.nc")

# Loading initial dataset
train_HR  = xr.load_dataset(path_train_HR[0] )
train_LR  = xr.load_dataset(path_train_LR[0] )
train_ALR = xr.load_dataset(path_train_ALR[0])
# test_HR   = xr.load_dataset(path_test_HR[0]  )
test_LR   = xr.load_dataset(path_test_LR[0]  )
test_ALR  = xr.load_dataset(path_test_ALR[0] )

print("Finish loading initial data")

# Concatenation of the other datasets
for p in range(1, len(loading_folder_data)):
    train_HR  = xr.concat([train_HR , xr.load_dataset(path_train_HR[p])], dim = "time")
    train_LR  = xr.concat([train_LR , xr.load_dataset(path_train_LR[p])], dim = "time")
    train_ALR = xr.concat([train_ALR, xr.load_dataset(path_train_ALR[p])], dim = "time")
    # test_HR   = xr.concat([test_HR  , xr.load_dataset(path_test_HR[p])], dim = "time")
    test_LR   = xr.concat([test_LR  , xr.load_dataset(path_test_LR[p])], dim = "time")
    test_ALR  = xr.concat([test_ALR , xr.load_dataset(path_test_ALR[p])], dim = "time")

# Displaying information regarding number of samples
print("-----------------")
print("     Datasets    ")
print("-----------------")
for d in loading_folder_data:
    print(f"- {d}")
    
print("\n-----------------")
print("Number of samples")
print("-----------------")
print("Train HR  = " + str(train_HR.dims["time"] ))
print("Train LR  = " + str(train_LR.dims["time"] ))
print("Train ALR = " + str(train_ALR.dims["time"]))
# print("Test  HR  = " + str(test_HR.dims["time"]  ))
print("Test  LR  = " + str(test_LR.dims["time"]  ))
print("Test  ALR = " + str(test_ALR.dims["time"] ))
print("\nTotal : "   + str(train_ALR.dims["time"] + test_ALR.dims["time"]))

#-------------------------------------------------------------
#                 FCNN : Training & Saving     
#-------------------------------------------------------------
# Determine the complete path of the result folder
result_folder_path = getPath(result_folder)

# Check if cuda is available
print("-----------")
print("CUDA = " + str(torch.cuda.is_available()))
print("-----------")

# Training the parameterization
FCNN_trained = FCNNParameterization.train_on(dataset    = train_ALR, 
                                             directory  = result_folder_path,
                                             inputs     = inputs,
                                             targets    = targets,
                                             num_epochs = num_epochs, 
                                             zero_mean  = True, 
                                             padding    = 'circular')