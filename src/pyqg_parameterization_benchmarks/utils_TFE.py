#------------------------------------------------------------------------------
#
#           Ocean subgrid parameterization using machine learning
#
#                             Graduation work
#
#------------------------------------------------------------------------------
# @ Victor Mangeleer
#
# -----------------
#   Documentation
# -----------------
# This file contains all the functions used throughout all the .py
# (for asserts, loading data, computing subgrid terms, ...)
#
# -----------------
#     Librairies
# -----------------
#
# --------- Standard ---------
import os
import glob
import math
import torch
import random
import numpy             as np
import xarray            as xr
import matplotlib.pyplot as plt

# --------- PYQG ---------
import pyqg

# --------- PYQG Benchmark ---------
import pyqg_parameterization_benchmarks.coarsening_ops as coarsening
from   pyqg_parameterization_benchmarks.utils          import *
from   pyqg_parameterization_benchmarks.plots_TFE      import *


# ----------------------------------------------------------------------------------------------------------
#
#                                                    Asserts
#
# ----------------------------------------------------------------------------------------------------------
def check_choices(values, possibilities):
    """
    Documentation
    -------------
    Determine if the elements inside values are accepted, i.e. are in the list of possibilities.
    """
    for v in values:
        if v not in possibilities:
            return False
    return True

def check_datasets_availability(names):
    """
    Documentation
    -------------
    Determine whether or not, given a list of datasets name, they exists or not.
    """
    # Main directory path prefix
    main_directory = "../../datasets/"

    # Checking each path availability
    for n in names:
        if os.path.exists(main_directory + n) == False:
            return False
    return True

def needed_memory(nb_samples, save_high_res = False):
    """
    Documentation
    -------------
    Determine the total number of memory space needed to create and store the datasets.
    """
    # Space needed for each type per sample [Mb]
    memory_HR  = 17291/2355
    memory_LR  = 2779/2355
    memory_ALR = memory_LR * 2.5
    
    # Actual space needed for one sample
    space_sample = memory_HR + memory_LR + memory_ALR if save_high_res else memory_LR + memory_ALR
    
    # Memory Space Needed
    space_needed = space_sample * nb_samples

    return math.ceil(space_needed/1024) # Conversion to [Gb]

def get_datasets_memory(names, datasets_type = ["ALR"]):
    """
    Documentation
    -------------
    Determine the amount of memory needed to load and store all the datasets, given a list of their names.
    """
    # Contains the total size taken by all the folder contained in path
    total_size = 0

    # Security
    assert check_choices(datasets_type, ["HR", "LR", "ALR"]), \
        f"Assert (get_datasets_memory): Datasets type must be a list containing 'HR', 'LR' and/or 'ALR'."
    
    # Main path prefix
    main_directory = "../../datasets/"

    # Checking each path availability
    for n in names:

        # Current path to inside the dataset
        current_path = main_directory + n

        # Updating total size needed
        total_size  += os.path.getsize(current_path + "/dataset_HR.nc")  if "HR"  in datasets_type else 0
        total_size  += os.path.getsize(current_path + "/dataset_LR.nc")  if "LR"  in datasets_type else 0
        total_size  += os.path.getsize(current_path + "/dataset_ALR.nc") if "ALR" in datasets_type else 0

    return total_size/(1024 ** 3)

# ----------------------------------------------------------------------------------------------------------
#
#                                                    Models
#
# ----------------------------------------------------------------------------------------------------------
def get_model_path(model_name):
    """
    Documentation
    -------------
    Used to determine the path to the results' folder for the model
    """
    # Count number of already existing folders in the model folder
    nb_files = str(len(glob.glob("../../models/*")))
    
    # Main directory path prefix
    current_directory = "../../models/" + model_name + "/"
    
    # Final path to the model (make sure there are no overwritte of existing model)
    model_path = f"{current_directory}"

    # Final model name
    model_name = model_name + f"_{str(nb_files)}"
    
    # Creation of a non-existing folder name
    return model_name, model_path

def check_model_folder_availability(model_folder):
    """
    Documentation
    -------------
    Determine whether or not if a model folder exists and is not empty
    """
    # Main directory path prefix
    main_directory = "../../models/"

    # Checking each path availability
    if os.path.exists(main_directory + model_folder) == False:
        return False
    
    # Checking if model folder contains at least one folder
    if len(os.listdir(main_directory + model_folder + "/")) == 0:
        return False
    
    return True

def config_for(model):
    """
    Documentation
    -------------
    Return the parameters needed to initialize a new pyqg.QGModel, except for nx and ny.
    """
    config = dict(H1 = model.Hi[0])
    for prop in ['L', 'W', 'dt', 'rek', 'g', 'beta', 'delta','U1', 'U2', 'rd']:
        config[prop] = getattr(model, prop)
    return config

# ----------------------------------------------------------------------------------------------------------
#
#                                                Parameterizations
#
# ----------------------------------------------------------------------------------------------------------
def minibatch(*arrays, batch_size = 64, as_tensor = True, shuffle = True):

    # Since set removes duplicate, this assert make sure that inputs and outputs have same dimensions !
    assert len(set([len(a) for a in arrays])) == 1

    # Index vector
    order = np.arange(len(arrays[0]))
    if shuffle:
        np.random.shuffle(order)

    # Step size (arrondis vers le bas)
    steps = int(np.ceil(len(arrays[0]) / batch_size))

    # Choose data type to store the batch
    xform = torch.as_tensor if as_tensor else lambda x: x

    # Creation of all the mini batches !
    for step in range(steps):
        idx = order[step * batch_size : (step + 1) * batch_size]

        # Yield is the same as return except that it return a generator ! In other words,
        # it is an iterator that you can only go through once since values are discarded !
        # This really really really smart !
        yield tuple(xform(array[idx]) for array in arrays)
        
# ----------------------------------------------------------------------------------------------------------
#
#                                                Offline testing
#
# ----------------------------------------------------------------------------------------------------------
def get_param_type(param_names):
    """
    Documentation
    -------------
    Return a list containing the type of target predicted by the model where:
    """
    # Stores the different target types
    types = list()

    # Looping over all the parameterizations loaded:
    for p in param_names:

        if "q_subgrid_forcing" in p:
            types.append(0)
        if "q_forcing_total"   in p:
            types.append(1)
        if "uq_subgrid_flux"   in p:
            types.append(2)
        if "vq_subgrid_flux"   in p:
            types.append(3)
            
    return types

def imshow_offline(arr):
    """
    Documentation
    -------------
    Used to display the R^2/corr plot done for offline testing and return their mean value
    """
    # Computes mean value of measured variable
    mean = arr.mean().data
    
    # Creation of the plot
    plt.imshow(arr, vmin = 0, vmax = 1, cmap = 'inferno')
    plt.text(32, 32, f"{mean:.2f}", color = ('white' if mean < 0.75 else 'black'), \
             fontweight = 'bold', ha = 'center', va = 'center', fontsize = 16)
    plt.xticks([]); plt.yticks([])
    
    return mean
                
# ----------------------------------------------------------------------------------------------------------
#
#                                                    Statistics
#
# ----------------------------------------------------------------------------------------------------------
class BasicScaler(object):
    """
    Simple class to perform normalization and denormalization
    """
    def __init__(self, mu = 0, sd = 1):
        self.mu = mu
        self.sd = sd

    def transform(self, x):
        return (x - self.mu) / self.sd

    def inverse_transform(self, z):
        return z * self.sd + self.mu

class ChannelwiseScaler(BasicScaler):
    """
    Simple class to compute mean and standard deviation of each channel
    """
    def __init__(self, x, zero_mean = False):
        assert len(x.shape) == 4

        # Computation of the mean
        if zero_mean:
            mu = 0
        else:
            mu = np.array([x[:,i].mean() for i in range(x.shape[1])])[np.newaxis , : , np.newaxis , np.newaxis]

        # Computation of the standard deviation
        sd = np.array([x[:,i].std() for i in range(x.shape[1])])[np.newaxis , : , np.newaxis , np.newaxis]

        # Initialization of a scaler with correct mean and standard deviation
        super().__init__(mu, sd)
        
# ----------------------------------------------------------------------------------------------------------
#
#                                                    Datasets
#
# ----------------------------------------------------------------------------------------------------------
def train_test_split_xarray(dataset, train_size = 0.7):
    """
    Documentation
    -------------
    Split an xarray dataset into a training and test datasets.
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

def load_data(datasets_name, datasets_type = ["ALR"]):
    """
    Documentation
    -------------
    Load the datasets (HR, LR and/or ALR) with the use of a list of names and the type needed to be loaded
    """
    # Main path prefix
    main_directory = "../../datasets/"

    # Security
    assert check_choices(datasets_type, ["HR", "LR", "ALR"]), \
        f"Assert (get_datasets_memory): Datasets type must be a list containing 'HR', 'LR' and/or 'ALR'."
    
    # Initialization
    data_HR, data_LR, data_ALR = None, None, None

    # Loading the data
    for i, n in enumerate(datasets_name):

        # Current dataset
        curr_data = main_directory + n + "/"

        # Loading the different datasets
        if "HR" in datasets_type:
            loaded_HR  = xr.load_dataset(curr_data + "dataset_HR.nc")
            data_HR    = loaded_HR if i == 0 else \
                         xr.concat([data_HR, loaded_HR], dim = "time")
            loaded_HR.close()

        if "LR" in datasets_type:
            loaded_LR  = xr.load_dataset(curr_data + "dataset_LR.nc")
            data_LR    = loaded_LR if i == 0 else \
                         xr.concat([data_LR, loaded_LR], dim = "time")
            loaded_LR.close()

        if "ALR" in datasets_type:
            loaded_ALR = xr.load_dataset(curr_data + "dataset_ALR.nc")
            data_ALR   = loaded_ALR if i == 0 else \
                         xr.concat([data_ALR, loaded_ALR], dim = "time")

    return data_HR, data_LR, data_ALR
    
def add_Forcing_Fluxes(coarsened_model):
    """
    Documentation
    -------------
    From a coarsened model (result of coarsening operator), creates the dataset
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

def generateData(model_HR, save_HR, coarsening_index, skipped_time,
                sampling_freq = 1000, low_res_nx = 64, number_threads = 1):
    """
    Documentation
    -------------
    Used to generate datasets corresponding to a:
    - High resolution          (= HR,  nx = 256)
    - Augmented low resolution (= ALR, nx = 64 ) with q coming from HR after coarsening and filtering
    - Low resolution           (= LR,  nx = 64 )
    """
    # Determine the number of time steps to skip
    skipped_steps = skipped_time * 365 * 24

    # Stores the data associated to the different steps
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
        if model_HR.tc % sampling_freq == 0 and skipped_steps < model_HR.t:

            # Creation of the coarsened data
            if coarsening_index   == 1:
                model_ALR = coarsening.Operator1(model_HR, low_res_nx = low_res_nx) # Spectral truncation + sharp filter
            elif coarsening_index == 2:
                model_ALR = coarsening.Operator2(model_HR, low_res_nx = low_res_nx) # Spectral truncation + sharp filter
            else:
                model_ALR = coarsening.Operator3(model_HR, low_res_nx = low_res_nx) # GCM-Filters + real-space coarsening

            # The whole dataset is saved
            if save_HR == True:
                snapshots_HIGH_RESOLUTION.append(model_HR.to_dataset().copy(deep = True))
                
            snapshots_LOW_RESOLUTION.append(model_LR.to_dataset().copy(deep = True))
            snapshots_AUG_LOW_RESOLUTION.append(add_Forcing_Fluxes(model_ALR))

    # Only the last step is saved
    if save_HR == False:
        snapshots_HIGH_RESOLUTION.append(model_HR.to_dataset().copy(deep = True))

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

    return ds_HR, ds_ALR, ds_LR, model_ALR