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
from torch.utils.tensorboard import SummaryWriter

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
    USAGE:      python train_parameterization.py --folder_training    <X>      
                                                 --folder_validation  <X> 
                                                 --save_directory     <X>                   
                                                 --inputs             <X>
                                                 --targets            <X>         
                                                 --num_epochs         <X>         
                                                 --zero_mean          <X>             
                                                 --padding            <X>        
                                                 --memory             <X>
                                                 --sim_type           <X>
    """
    # Initialization of the parser
    parser = ArgumentParser(usage)

    # Definition of the possible stuff to be parsed
    parser.add_argument(
        '--folder_training',
        help  = 'Choose the folder used to load data as training data',
        nargs = '+')

    parser.add_argument(
        '--folder_validation',
        help  = 'Choose the folder used to load data as training data',
        nargs = '+')

    parser.add_argument(
        '--save_directory',
        help = 'Choose the folder used to load data as training data',
        type = str)

    parser.add_argument(
        '--inputs',
        help  = 'Choose the type of inputs given to the parameterization for training',
        nargs = '+')
    
    parser.add_argument(
        '--targets',
        help = 'Choose the parameterization ouptut',
        type = str,
        choices = ['q_forcing_total', 'q_subgrid_forcing', 'u_subgrid_forcing', 'v_subgrid_forcing'])

    parser.add_argument(
        '--num_epochs',
        help = 'Choose the number of epochs made by the paremeterization while training',
        type = int)
    
    parser.add_argument(
        '--zero_mean',
        help = 'Choose the type of pre-processing made on the datasets',
        type = str,
        default = True)

    parser.add_argument(
        '--padding',
        help = 'Choose the type of padding used by the parameterization',
        type = str,
        choices = ['circular'],
        default = 'circular')

    parser.add_argument(
        '--memory',
        help = 'Total number of memory allocated [GB] (used for security purpose)',
        type = int)
    
    parser.add_argument(
        '--sim_type',
        help = 'Type of fluid simulation studied (used to order tensorboard folders)',
        type = str,
        default = "eddies")

    # Retrieving the values given by the user
    args = parser.parse_args()

    # Display information over terminal (0)
    tfe_title()
        
    # ----------------------------------
    #              Asserts
    # ----------------------------------
    # Check if the path of each dataset exist
    assert check_datasets_availability(args.folder_training), \
        f"Assert: One (or more) of the traininig dataset does not exist,  check again the name of the folders"
    
    assert check_datasets_availability(args.folder_validation), \
        f"Assert: One (or more) of the validation dataset does not exist, check again the name of the folders"
        
    # Check if there is enough memory allocated to load the datasets
    needed_memory = get_datasets_memory(args.folder_training  , datasets_type = ["ALR"]) + \
                    get_datasets_memory(args.folder_validation, datasets_type = ["ALR"])
    
    assert math.ceil(needed_memory) < args.memory , \
        f"Assert: Not enough memory allocated to store the train and validation datasets ({math.ceil(needed_memory)} [Gb])"

    # Check if the inputs are a combination of q, u, v
    assert check_choices(args.inputs, ["q", "u", "v"]), \
        f"Assert: Input variable(s) for the parameterization must be 'q', 'u', 'v' or a combination"
    
    # Check that num epoch is positive
    assert 0 < args.num_epochs, \
        f"Assert: Number of epochs must be greater than 0"
    
    # Update of zero mean argument to bool type
    args.zero_mean = True if args.zero_mean == "True" else False

    # Display information over terminal (1)
    section("Loading datasets")
    show_param_parameters(args)
    
    # ----------------------------------
    #          Loading datasets
    # ----------------------------------
    # Training
    _, _, data_ALR_train = load_data(args.folder_training,   datasets_type = ["ALR"])
    
    # Validation
    _, _, data_ALR_valid = load_data(args.folder_validation, datasets_type = ["ALR"])
    
    # ----------------------------------
    #     Training parameterization
    # ----------------------------------
    # Write down inputs as string for model name
    input_str = ""
    for i in args.inputs:
        input_str += "_" + i
    
    # Creation of a more complete model name
    curr_model_name = f"{args.save_directory}{input_str}_to_{args.targets}_{str(1000 * len(args.folder_training))}_{args.num_epochs}"
    
    # Determine the complete path of the result folder
    model_name, model_path = get_model_path(curr_model_name)
    
    # Initialization of tensorboard
    tsb = SummaryWriter(f"../../runs/{args.sim_type}/{model_name}")

    # Training the parameterization
    FCNN_trained = FCNNParameterization.train_on(dataset     = data_ALR_train, 
                                                 dataset_val = data_ALR_valid,
                                                 directory   = model_path,
                                                 inputs      = args.inputs,
                                                 targets     = [args.targets],
                                                 num_epochs  = args.num_epochs, 
                                                 zero_mean   = args.zero_mean, 
                                                 padding     = args.padding,
                                                 tensorboard = tsb)
    
    # Display information over terminal (2)
    print("\nDone\n")