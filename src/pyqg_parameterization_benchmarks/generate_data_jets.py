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

# Define which simulation type to use to generate the high resolution data
simulation_type = 1

# Define at which frequency to save the results (Ex: if sampling = 1, each hour the data is saved)
sampling_frequency = 8

# Define which operator to apply on the high resolution data
operator_index = 1

# Define the size of the dataset that will be used as training data
train_size = 0.7

# Define if the high resolution training set should be saved entirely or just the last sample (/!\ High memory cost if True)
save_HR_train = False

# Define the name of the folder used to save the datasets (../datasets/save_folder)
save_folder = "jets_big"

# ---------------------------------- PARAMETERS ----------------------------------
#
#
#
#
#
#
#----------
# Libraries
#----------
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

#----------
# Functions
#----------
#
# Used to compute the power spectrum
calc_ispec = lambda *args, **kwargs: _calc_ispec(*args, averaging = False, truncate =False, **kwargs)

def get_dataset(index, base_url = "https://g-402b74.00888.8540.data.globus.org"):
    """
    Documentation
    -------------
    Load the datasets used in L. Zanna & Al.'s paper  are hosted on globus as zarr files
    """
    paths = ['eddy/high_res', 'eddy/low_res', 'jet/high_res', 'jet/low_res', 'eddy/forcing1', 'eddy/forcing2', 'eddy/forcing3', 'eddy/low_res']
    mapper = fsspec.get_mapper(f"{base_url}/{paths[index]}.zarr")
    return xr.open_zarr(mapper, consolidated=True)

def run_save(model, sampling_freq = 1000):
    """
    Documentation
    -------------
    Saves simulation results every iterations % sampling_freq == 0 and 
    concatenates everything into a huge xarray dataset.
    """
    # Run the model and save snapshots
    snapshots = []
    while model.t < model.tmax:
        if model.tc % sampling_freq == 0:
            snapshots.append(model.to_dataset().copy(deep = True))
        model._step_forward()

    # Concatenation of all the results into a big dataset
    ds = xr.concat(snapshots, dim = 'time')

    # Diagnostics get dropped by this procedure since they're only present for
    # part of the timeseries; resolve this by saving the most recent
    # diagnostics (they're already time-averaged so this is ok)
    for k,v in snapshots[-1].variables.items():
        if k not in ds:
            ds[k] = v.isel(time=-1)

    # Drop complex variables since they're redundant and can't be saved
    complex_vars = [k for k,v in ds.variables.items() if np.iscomplexobj(v)]
    ds = ds.drop_vars(complex_vars)

    return ds

def train_test_split_xarray(dataset, train_size = 0.7):

    # Retreiving the size of the time dimension of the xarray
    size        = dataset.dims["time"]

    # Computing indexes for training and testing sets
    index_train_start = 0
    index_train_end   = int(size * train_size)
    index_test_start  = index_train_end + 1
    index_test_end    = size - 1

    # Creating slicing vectors
    train_indexes = list(np.linspace(index_train_start, index_train_end, num = index_train_end + 1, dtype = int))
    test_indexes  = list(np.linspace(index_test_start,  index_test_end , num = index_test_end - index_test_start, dtype = int))

    # Creation of the datasets
    dataset_train = dataset.isel(time = train_indexes)
    dataset_test  = dataset.isel(time = test_indexes)

    return dataset_train, dataset_test

def getDataset(model_LOW_RESOLUTION):

    # Initial dataset (everything will be added to it)
    dataset_LOW_RESOLUTION_base = model_LOW_RESOLUTION.m2.to_dataset().copy(deep = True)

    # Retreiving each corresponding variables
    dqdt_bar, dqbar_dt, q_tot = model_LOW_RESOLUTION.q_forcing_total
    q_sfor                    = model_LOW_RESOLUTION.subgrid_forcing("q")
    u_sfor                    = model_LOW_RESOLUTION.subgrid_forcing("u")
    v_sfor                    = model_LOW_RESOLUTION.subgrid_forcing("v")
    uq_sf, vq_sf              = model_LOW_RESOLUTION.subgrid_fluxes("q")
    uu_sf, vu_sf              = model_LOW_RESOLUTION.subgrid_fluxes("u")
    uv_sf, vv_sf              = model_LOW_RESOLUTION.subgrid_fluxes("v")

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

    dataset_LOW_RESOLUTION_missing = xr.Dataset(missing_values)

    # Merging all the results    
    return xr.merge([dataset_LOW_RESOLUTION_base, dataset_LOW_RESOLUTION_missing])

def run_simulations(model_HIGH_RESOLUTION, coarsening_index, sampling_freq = 1000):
    """
    Documentation
    -------------
    Saves simulation results at a frequency defined by the sampling frequency.
    """
    # Used to store high and low resolution data
    snapshots_HIGH_RESOLUTION, snapshots_LOW_RESOLUTION = list(), list()

    # Stores the low resolution model
    model_LOW_RESOLUTION = 1

    # Running the simulation
    while model_HIGH_RESOLUTION.t < model_HIGH_RESOLUTION.tmax:

        if model_HIGH_RESOLUTION.tc % sampling_freq == 0:

            # Saving high resolution data (only last step to save memory)
            #if model_HIGH_RESOLUTION.t == model_HIGH_RESOLUTION.tmax - 1:
            snapshots_HIGH_RESOLUTION.append(model_HIGH_RESOLUTION.to_dataset().copy(deep = True))

            # Creation of the coarsened data
            if coarsening_index   == 1:
                model_LOW_RESOLUTION = coarsening.Operator1(model_HIGH_RESOLUTION, low_res_nx = 64) # Spectral truncation + sharp filter
            elif coarsening_index == 2:
                model_LOW_RESOLUTION = coarsening.Operator2(model_HIGH_RESOLUTION, low_res_nx = 64) # Spectral truncation + sharp filter
            elif coarsening_index == 3:
                model_LOW_RESOLUTION = coarsening.Operator3(model_HIGH_RESOLUTION, low_res_nx = 64) # GCM-Filters + real-space coarsening
            else:
                print("ERROR - Operator index is invalid !")

            # Saving low resolution data
            snapshots_LOW_RESOLUTION.append(getDataset(model_LOW_RESOLUTION))

        # Computing next step of simulation
        model_HIGH_RESOLUTION._step_forward()

    # Concatenation of all the results into a big dataset
    ds_HR = xr.concat(snapshots_HIGH_RESOLUTION, dim = 'time')
    ds_LR = xr.concat(snapshots_LOW_RESOLUTION,  dim = 'time')

    print(ds_HR)
    
    # Diagnostics get dropped by this procedure since they're only present for part of the timeseries; 
    # resolve this by saving the most recent diagnostics (they're already time-averaged so this is ok)
    for k, v in snapshots_HIGH_RESOLUTION[-1].variables.items():
        if k not in ds_HR:
            ds_HR[k] = v.isel(time = -1)

    for k,v in snapshots_LOW_RESOLUTION[-1].variables.items():
        if k not in ds_LR:
            ds_LR[k] = v.isel(time = -1)

    # Drop complex variables since they're redundant and can't be saved
    complex_vars = [k for k,v in ds_HR.variables.items() if np.iscomplexobj(v)]
    ds_HR = ds_HR.drop_vars(complex_vars)
    
    complex_vars = [k for k,v in ds_LR.variables.items() if np.iscomplexobj(v)]
    ds_LR = ds_LR.drop_vars(complex_vars)

    return ds_HR, ds_LR, model_LOW_RESOLUTION


# -- Generating --
#
# Definition of the parameters for each simulation type
nx        = [256        , 256   , 256        , 256  ,  256, 256]
dt        = [1          , 1     , 1          , 1    ,    1,   1]
tmax      = [10         , 10    , 0.25       , 2    ,   10,  10]
tavestart = [5          , 5     , 1          , 1    ,    5,   5]
rek       = [5.789e-7   , 7e-08 , 5.789e-7   , 7e-08,   -1,  -1]
delta     = [0.25       , 0.1   , 0.25       , 0.1  , 0.25, 0.1]
beta      = [1.5 * 1e-11, 1e-11 , 1.5 * 1e-11, 1e-11,   -1,  -1]

# Radom parameter values
rek_random  = random.uniform(5.7, 5.9)   * 1e-7  if simulation_type < 5 else \
              random.uniform(6.9, 7.1)   * 1e-8

beta_random = random.uniform(1.45, 1.55) * 1e-11 if simulation_type < 5 else \
              random.uniform(0.95, 1.05) * 1e-11

# Creation of dictionnary (Conversion to seconds embedded)
simulation_parameters_training              = {}
simulation_parameters_training['nx']        = nx       [simulation_type]
simulation_parameters_training['dt']        = dt       [simulation_type] * 60 * 60
simulation_parameters_training['tmax']      = tmax     [simulation_type] * 24 * 60 * 60 * 360
simulation_parameters_training['tavestart'] = tavestart[simulation_type] * 24 * 60 * 60 * 360
simulation_parameters_training['rek']       = rek      [simulation_type] if simulation_type < 4 else rek_random
simulation_parameters_training['delta']     = delta    [simulation_type]
simulation_parameters_training['beta']      = beta     [simulation_type] if simulation_type < 4 else beta_random

# Creation of a PYQG model with corresponding parameters
model_HIGH_RESOLUTION = pyqg.QGModel(**simulation_parameters_training)

# Creation of the associated dataset
dataset_HIGH_RESOLUTION, dataset_LOW_RESOLUTION, model_LOW_RESOLUTION = run_simulations(model_HIGH_RESOLUTION, 
                                                                                        operator_index, 
                                                                                        sampling_freq = sampling_frequency)

# -- Saving --
#
# Complete path to saving folder
saving_path = "../../datasets/" + save_folder

# Checks if the saving folder exists, if not, creates it
if not os.path.exists(saving_path):
   os.makedirs(saving_path)

# Generating training and test sets
train_HR, test_HR = train_test_split_xarray(dataset_HIGH_RESOLUTION, train_size = train_size)
train_LR, test_LR = train_test_split_xarray(dataset_LOW_RESOLUTION,  train_size = train_size)

# Saving the datasets
train_LR.to_netcdf(saving_path + "/train_LR.nc")
test_HR.to_netcdf( saving_path + "/test_HR.nc" )
test_LR.to_netcdf( saving_path + "/test_LR.nc" )

# In order to simply observe what is inside the training set, the last sample is uniquely saved (in order to save space).
# However, one has still the possibility to save the entire dataset if needed.
train_HR.to_netcdf(saving_path + "/train_HR.nc") if save_HR_train == True else \
   train_HR.isel(time = [-1]).copy(deep = True).to_netcdf(saving_path + "/train_HR.nc")