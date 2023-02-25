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
save_folder = "eddies_6"

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
#
#-------------------------------------------------------------
#                           Librairies
#-------------------------------------------------------------
import os
import sys
import glob
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
    nb_files = len(glob.glob("../models/*"))

    # Creation of a non-existing folder name
    return f"../models/{folder_name}/" if loading else f"../models/{folder_name}_{nb_files}/"

def config_for(model):
    """
    Return the parameters needed to initialize a new pyqg.QGModel, except for nx and ny.
    """
    config = dict(H1 = model.Hi[0])
    for prop in ['L', 'W', 'dt', 'rek', 'g', 'beta', 'delta','U1', 'U2', 'rd']:
        config[prop] = getattr(model, prop)
    return config

def train_test_split_xarray(dataset, train_size = 0.7):
    """
    Split an xarray dataset into a training and test datasets
    """
    # Retreiving the size of the time dimension of the xarray
    size = dataset.dims["time"]

    # Computing indexes for training and testing sets
    index_train_start = 0
    index_train_end   = int(size * train_size)
    index_test_start  = index_train_end + 1
    index_test_end    = size - 1

    # Creating slicing vectors
    train_indexes = list(np.linspace(index_train_start, index_train_end, num = index_train_end + 1,               dtype = int))
    test_indexes  = list(np.linspace(index_test_start,  index_test_end , num = index_test_end - index_test_start, dtype = int))

    # Creation of the datasets
    dataset_train = dataset.isel(time = train_indexes)
    dataset_test  = dataset.isel(time = test_indexes)

    return dataset_train, dataset_test

def add_Forcing_Fluxes(coarsened_model):
    """
    From a coarsened model (result of coarsening operator), retreives the dataset 
    containing PYQG variables as well as the subgrid forcing and fluxes terms.
    """
    # Initial dataset (everything will be added to it)
    dataset_init = coarsened_model.m2.to_dataset().copy(deep = True)
    
    # Computing Subgrid Variables
    dqdt_bar, dqbar_dt, q_tot = coarsened_model.q_forcing_total
    q_sfor                    = coarsened_model.subgrid_forcing("q")
    u_sfor                    = coarsened_model.subgrid_forcing("u")
    v_sfor                    = coarsened_model.subgrid_forcing("v")
    uq_sf, vq_sf              = coarsened_model.subgrid_fluxes("q")
    uu_sf, vu_sf              = coarsened_model.subgrid_fluxes("u")
    uv_sf, vv_sf              = coarsened_model.subgrid_fluxes("v")

    # Creation of dataset with missing values
    missing_values = {
                        "q_forcing_total"   : (("lev", "x", "y"), q_tot),
                        "dqdt_bar"          : (("lev", "x", "y"), dqdt_bar),
                        "dqbar_dt"          : (("lev", "x", "y"), dqbar_dt),
                        "q_subgrid_forcing" : (("lev", "x", "y"), q_sfor),
                        "u_subgrid_forcing" : (("lev", "x", "y"), u_sfor),
                        "v_subgrid_forcing" : (("lev", "x", "y"), v_sfor),
                        "uq_subgrid_flux"   : (("lev", "x", "y"), uq_sf),
                        "vq_subgrid_flux"   : (("lev", "x", "y"), vq_sf),
                        "uu_subgrid_flux"   : (("lev", "x", "y"), uu_sf),
                        "vv_subgrid_flux"   : (("lev", "x", "y"), vv_sf),
                        "uv_subgrid_flux"   : (("lev", "x", "y"), uv_sf)}

    dataset_forcing_fluxes = xr.Dataset(missing_values)

    # Merging all the results    
    return xr.merge([dataset_init, dataset_forcing_fluxes])

def generateData(model_HR, coarsening_index = 1, sampling_freq = 1000, low_res_nx = 64, number_threads = 1):
    """
    Used to generate datasets corresponding to a: 
    - High resolution          (= HR,  nx = 256)
    - Augmented low resolution (= ALR, nx = 64 ) with q coming from HR after coarsening and filtering
    - Low resolution           (= LR,  nx = 64 )
    """
    snapshots_HIGH_RESOLUTION, snapshots_AUG_LOW_RESOLUTION, snapshots_LOW_RESOLUTION = list(), list(), list()

    # Stores the different models
    model_ALR = 1
    model_LR  = pyqg.QGModel(nx = low_res_nx, ntd = number_threads, **config_for(model_HR))
    
    #---------------------------------------
    #         Running the simulation 
    #---------------------------------------
    while model_HR.t < model_HR.tmax:
    
        # Computing next step of simulation
        model_HR._step_forward()
        model_LR._step_forward()
        
        # Sampling of data
        if model_HR.tc % sampling_freq == 0:

            # Creation of the coarsened data
            if coarsening_index   == 1:
                model_ALR = coarsening.Operator1(model_HR, low_res_nx = low_res_nx) # Spectral truncation + sharp filter
            elif coarsening_index == 2:
                model_ALR = coarsening.Operator2(model_HR, low_res_nx = low_res_nx) # Spectral truncation + sharp filter
            else:
                model_ALR = coarsening.Operator3(model_HR, low_res_nx = low_res_nx) # GCM-Filters + real-space coarsening

            # Saving everything
            snapshots_HIGH_RESOLUTION.append(   model_HR.to_dataset().copy(deep = True))
            snapshots_LOW_RESOLUTION.append(    model_LR.to_dataset().copy(deep = True))
            snapshots_AUG_LOW_RESOLUTION.append(add_Forcing_Fluxes(model_ALR))

    #---------------------------------------
    #            Xarray creations
    #---------------------------------------
    # Concatenation of all the results into a big dataset
    ds_HR  = xr.concat(snapshots_HIGH_RESOLUTION,     dim = 'time')
    ds_ALR = xr.concat(snapshots_AUG_LOW_RESOLUTION,  dim = 'time')
    ds_LR  = xr.concat(snapshots_LOW_RESOLUTION,      dim = 'time')

    # Diagnostics get dropped by this procedure since they're only present for part of the timeseries; 
    # resolve this by saving the most recent diagnostics (they're already time-averaged so this is ok)
    for k, v in snapshots_HIGH_RESOLUTION[-1].variables.items():
        if k not in ds_HR:
            ds_HR[k] = v.isel(time = -1)

    for k,v in snapshots_AUG_LOW_RESOLUTION[-1].variables.items():
        if k not in ds_ALR:
            ds_ALR[k] = v.isel(time = -1)
            
    for k,v in snapshots_LOW_RESOLUTION[-1].variables.items():
        if k not in ds_LR:
            ds_LR[k] = v.isel(time = -1)

    # Drop complex variables since they're redundant and can't be saved
    complex_vars = [k for k,v in ds_HR.variables.items() if np.iscomplexobj(v)]
    ds_HR = ds_HR.drop_vars(complex_vars)
    
    complex_vars = [k for k,v in ds_ALR.variables.items() if np.iscomplexobj(v)]
    ds_ALR = ds_ALR.drop_vars(complex_vars)

    complex_vars = [k for k,v in ds_LR.variables.items() if np.iscomplexobj(v)]
    ds_LR = ds_LR.drop_vars(complex_vars)

    return ds_HR, ds_ALR, ds_LR, model_LR


#-------------------------------------------------------------
#                         Generation
#-------------------------------------------------------------
# Definition of the parameters for each possible simulation type
nx        = [256        , 256   , 256        , 256  ,  256, 256]
dt        = [1          , 1     , 1          , 1    ,    1,   1]
tmax      = [10         , 10    , 0.5        , 0.5  ,   10,  10] 
tavestart = [5          , 5     , 1          , 1    ,    5,   5]
rek       = [5.789e-7   , 7e-08 , 5.789e-7   , 7e-08,   -1,  -1]
delta     = [0.25       , 0.1   , 0.25       , 0.1  , 0.25, 0.1]
beta      = [1.5 * 1e-11, 1e-11 , 1.5 * 1e-11, 1e-11,   -1,  -1]

# Radom parameter values
rek_random  = random.uniform(5.7, 5.9)   * 1e-7  if simulation_type < 5 else \
              random.uniform(6.9, 7.1)   * 1e-8

beta_random = random.uniform(1.45, 1.55) * 1e-11 if simulation_type < 5 else \
              random.uniform(0.95, 1.05) * 1e-11

# Dictionnary of simulation parameters (Conversion to seconds embedded)
simulation_parameters              = {}
simulation_parameters['nx']        = nx       [simulation_type]
simulation_parameters['dt']        = dt       [simulation_type] * 60 * 60
simulation_parameters['tmax']      = tmax     [simulation_type] * 24 * 60 * 60 * 360
simulation_parameters['tavestart'] = tavestart[simulation_type] * 24 * 60 * 60 * 360
simulation_parameters['rek']       = rek      [simulation_type] if simulation_type < 4 else rek_random
simulation_parameters['delta']     = delta    [simulation_type]
simulation_parameters['beta']      = beta     [simulation_type] if simulation_type < 4 else beta_random

# Creation of a PYQG model with corresponding parameters
model_HIGH_RESOLUTION = pyqg.QGModel(ntd = number_threads, **simulation_parameters)

# Generation of all the datasets
dataset_HR, dataset_ALR, dataset_LR, model_LOW_RESOLUTION = generateData(model_HIGH_RESOLUTION, 
                                                                         coarsening_index = operator_index, 
                                                                         sampling_freq    = sampling_frequency,
                                                                         number_threads   = number_threads)

#-------------------------------------------------------------
#                         Saving
#-------------------------------------------------------------
# Complete path to saving folder
saving_path = "../../datasets/" + save_folder

# Checks if the saving folder exists, if not, creates it
if not os.path.exists(saving_path):
   os.makedirs(saving_path)

# Creation of training and test sets
train_HR,  test_HR  = train_test_split_xarray(dataset_HR,  train_size = train_size)
train_ALR, test_ALR = train_test_split_xarray(dataset_ALR, train_size = train_size)
train_LR,  test_LR  = train_test_split_xarray(dataset_LR,  train_size = train_size)

# Saving everything to special format (recommended by xarray doccumentation)
train_LR.to_netcdf( saving_path + "/train_LR.nc")
train_ALR.to_netcdf(saving_path + "/train_ALR.nc" )
test_HR.to_netcdf(  saving_path + "/test_HR.nc" )
test_LR.to_netcdf(  saving_path + "/test_LR.nc" )
test_ALR.to_netcdf( saving_path + "/test_ALR.nc" )

# In order to simply observe what is inside the training set, the last sample 
# is uniquely saved (in order to save space). However, one has still the possibility 
# to save the entire dataset if needed.
train_HR.to_netcdf(saving_path + "/train_HR.nc") if save_HR_train == True else \
   train_HR.isel(time = [-1]).copy(deep = True).to_netcdf(saving_path + "/train_HR.nc")