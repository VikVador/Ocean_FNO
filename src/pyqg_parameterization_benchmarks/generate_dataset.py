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
#     Librairies
# -----------------
#
# --------- Standard ---------
import os
import sys
import json
import glob
import math
import torch
import random
import fsspec
import matplotlib
import numpy             as np
import xarray            as xr
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.stats import gaussian_kde

# --------- PYQG ---------
import pyqg
import pyqg.diagnostic_tools
from   pyqg.diagnostic_tools import calc_ispec         as _calc_ispec
import pyqg_parameterization_benchmarks.coarsening_ops as coarsening

calc_ispec = lambda *args, **kwargs: _calc_ispec(*args, averaging = False, truncate =False, **kwargs)

# --------- PYQG Benchmark ---------
from pyqg_parameterization_benchmarks.utils           import *
from pyqg_parameterization_benchmarks.utils_TFE       import *
from pyqg_parameterization_benchmarks.plots_TFE       import *
from pyqg_parameterization_benchmarks.online_metrics  import diagnostic_differences
from pyqg_parameterization_benchmarks.neural_networks import FullyCNN, FCNNParameterization

# -----------------------------------------------------
#                         Main
# -----------------------------------------------------
if __name__ == '__main__':

    # ----------------------------------
    # Parsing the command-line arguments
    # ----------------------------------
    # Definition of the help message that will be shown on the terminal
    usage = """
    USAGE: python generate_dataset.py --save_folder test     <X>                                                                         
                                      --simulation_type      <X>                                                                               
                                      --target_sample_size   <X>                                                                              
                                      --operator_cf          <X>                                                                              
                                      --skipped_time         <X>                                                                              
                                      --nb_threads           <X>                                                                              
                                      --memory               <X>                                                                              
                                      --save_high_res        <X>
    """
    # Initialization of the parser
    parser = ArgumentParser(usage)

    # Definition of the possible stuff to be parsed
    parser.add_argument(
        '--simulation_type',
        help = 'Choose the type of simulation used to generate the dataset',
        type = int,
        choices = [0, 1, 2, 3, 4, 5])

    parser.add_argument(
        '--target_sample_size',
        help = 'Choose the number of samples expected to be in the datasets (nb_sample >= target_sample_size)',
        type = int)

    parser.add_argument(
        '--operator_cf',
        help = 'Choose the coarsening and filtering operator applied on the high resolution simulation',
        type = int,
        choices = [1, 2, 3])

    parser.add_argument(
        '--nb_threads',
        help = 'Choose the number of threads used to run the simulation',
        type = int,
        choices = [1, 2, 3, 4])

    parser.add_argument(
        '--memory',
        help = 'Total number of memory allocated [GB] (used for security purpose)',
        type = int)

    parser.add_argument(
        '--save_folder',
        help = 'Choose the name of the folder used to save the datasets',
        type = str)

    parser.add_argument(
        '--save_high_res',
        help = 'Choose if the whole high resolution is saved or just the last sample (memory saving)',
        type = str,
        choices = ["True", "False"],
        default = "False")

    parser.add_argument(
        '--skipped_time',
        help = 'Choose the time [year] at which the sampling of the simulation starts',
        type = float,
        default = 0.5)

    # Retrieving the values given by the user
    args = parser.parse_args()

    # Display information over terminal (0)
    tfe_title()

    # ----------------------------------
    #              Asserts
    # ----------------------------------
    # Contains simulation duration
    sim_duration = [10, 10, 2, 2, 10, 10]

    # Checking if there are enough samples to reach the target sample size (1)
    assert sim_duration[args.simulation_type] > args.skipped_time, \
        f"Assert: Skipped time ({args.skipped_time} [Years]) must be lower than simulation duration ({sim_duration[args.simulation_type]} [Years])"

    # Determine the sampling frequency needed to reach the target sample size (HYP : 10 years - skipped time, dt = 1h)
    sampling_frequency = math.ceil((365 * (10 - args.skipped_time) * 24)/args.target_sample_size)

    # Real number of samples created
    nb_samples = (365 * 10 * 24)/sampling_frequency

    # Compute the needed memory
    nd_memory = needed_memory(nb_samples, args.save_high_res)

    # Checking if enough memory has been allocated
    assert args.memory > nd_memory, \
        f"Assert: Not enough memory to generate dataset, {nd_memory} [Gb] are needed"

    # Checking if there are enough samples to reach the target sample size (2)
    assert nb_samples >= args.target_sample_size, \
        f"Assert: Impossible to reach target sample size ({args.target_sample_size}), reduce the skipped time ({args.skipped_time} [Years])"

    # Update of save_high_res argument to bool type
    args.save_high_res = True if args.save_high_res == "True" else False

    # ----------------------------------
    #      Simulation initialization
    # ----------------------------------
    # Retreives the simulation type from the parser
    simulation_type = args.simulation_type

    # Definition of the parameters for each simulation type
    nx        = [256        , 256   , 256        , 256  ,  256, 256]
    dt        = [1          , 1     , 1          , 1    ,    1,   1]
    tmax      = [10         , 10    , 2          , 2    ,   10,  10]
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
    model_HIGH_RESOLUTION = pyqg.QGModel(ntd = args.nb_threads, **simulation_parameters_training)

    # Display information over terminal (1)
    section("Properties of the datasets")
    show_sim_parameters(args)
    section("Generating datasets")

    # ----------------------------------
    #        Running simulation
    # ----------------------------------
    dataset_HR, dataset_ALR, dataset_LR, model_ALR = \
        generateData(model_HR         = model_HIGH_RESOLUTION,
                     coarsening_index = args.operator_cf,
                     skipped_time     = args.skipped_time,
                     save_HR          = args.save_high_res,
                     sampling_freq    = sampling_frequency,
                     number_threads   = args.nb_threads)

    # Display information over terminal (2)
    section("Saving datasets")

    # ----------------------------------
    #          Saving datasets
    # ----------------------------------
    # Complete path to saving folder
    saving_path = "../../datasets/" + args.save_folder

    # Checks if the saving folder exists, if not, creates it
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    # Saving everything to special format (recommended by xarray doccumentation)
    dataset_HR.to_netcdf( saving_path + "/dataset_HR.nc")
    dataset_ALR.to_netcdf(saving_path + "/dataset_ALR.nc" )
    dataset_LR.to_netcdf( saving_path + "/dataset_LR.nc" )
    
    # Display information over terminal (3)
    print("\nDone\n")
    section("Plotting state variables")
    
    # -----------------------------------------------
    #           Plotting state variables
    # -----------------------------------------------
    # -- Potential vorticity q -- 
    plotStateVariable(dataset_HR, dataset_LR, state_variable = "q",     save_path = saving_path)
    
    # -- Horizontal velocity u -- 
    plotStateVariable(dataset_HR, dataset_LR, state_variable = "u",     save_path = saving_path)
    
    # -- Vertical velocity v -- 
    plotStateVariable(dataset_HR, dataset_LR, state_variable = "v",     save_path = saving_path)
    
    # -- Horizontal velocity with background flow ufull -- 
    plotStateVariable(dataset_HR, dataset_LR, state_variable = "ufull", save_path = saving_path)
    
    # -- Vertical velocity with background flow vfull -- 
    plotStateVariable(dataset_HR, dataset_LR, state_variable = "vfull", save_path = saving_path)
    
    # Display information over terminal (4)
    print("\nDone\n")
    section("Plotting subgrid variables")
    
    # -----------------------------------------------
    #           Plotting subgrid variables
    # -----------------------------------------------
    # -- Potential vorticity --
    plotPowerSpectrum(  model_ALR, forcing_variable = "q",    save_path = saving_path)
    plotForcingVariable(model_ALR, forcing_variable = "sq",   save_path = saving_path)
    
    # -- Velocity --
    plotPowerSpectrum(  model_ALR, forcing_variable = "u",    save_path = saving_path)
    plotForcingVariable(model_ALR, forcing_variable = "su",   save_path = saving_path)
    
    # -- Vocticity --
    plotForcingVariable(model_ALR, forcing_variable = "flux", save_path = saving_path)
    
    # Display information over terminal (5)
    print("\nDone\n")