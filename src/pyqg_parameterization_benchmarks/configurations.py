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
# This file contains all the configurations of networks that will be tested !
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
import numpy as np

# ----------------------------------------------------------------------------------------------------------
#
#                                                   Functions
#
# ----------------------------------------------------------------------------------------------------------
def get_neural_network_configuration(model_name, model_config):
    
    # Conversion for simplicity
    config_value = int(model_config)
    
    # Contains the model configuration
    model_config_parameters = dict()
    
    # ------------------------------------------------------------------------------------------------------
    #                                 Fully Convolutional Neural Network
    # ------------------------------------------------------------------------------------------------------
    if model_name == "FCNN":    
        raise Exception(f"ERROR (get_neural_network_configuration), parameterization {model_name} in configuration {model_config} does not exist")
    
    # ------------------------------------------------------------------------------------------------------
    #                                                 U-Net
    # ------------------------------------------------------------------------------------------------------
    elif model_name == "UNET":
        
        if config_value == 1:
            model_config_parameters = {
                "init_features" : 4
            }
            
        else:
            raise Exception(f"ERROR (get_neural_network_configuration), parameterization {model_name} in configuration {model_config} does not exist")

    # ------------------------------------------------------------------------------------------------------
    #                                        Fourier Neural Operator
    # ------------------------------------------------------------------------------------------------------
    elif model_name == "FNO":

        if config_value == 1:
            model_config_parameters = {
                "Modes - X" : 16,
                "Modes - Y" : 16,
                "Width    " : 32,
                "Layers   " : 6,
            }
            
        else:
            raise Exception(f"ERROR (get_neural_network_configuration), parameterization {model_name} in configuration {model_config} does not exist")
    
    # ------------------------------------------------------------------------------------------------------
    #                                    Factorized Fourier Neural Operator
    # ------------------------------------------------------------------------------------------------------
    elif model_name == "FFNO":

        # Parameters to explore        
        modes_low_pass  = [(0, 8),  (0, 16), (0, 24), (0, 32)]
        modes_pass_band = [(8, 16), (16, 24), (24, 32)]
        modes           = modes_low_pass + modes_pass_band
        width           = [32, 64, 128]
        layers          = [4, 8, 12, 16, 20, 24]
        weights         = [True]   
        
        # Total number of combinations possible (here 48 in total)
        total_comb = len(modes) * len(width) * len(layers) * len(weights)
        
        # Keep count of the configuration index
        config_index = 0
        
        if config_value < total_comb:
            
            for we in weights:
                for w in width:
                    for l in layers:
                        for m in modes:
                            
                            # Configuration is found !
                            if config_index == config_value:
                                model_config_parameters = {
                                    "Modes - X" : m,
                                    "Modes - Y" : m,
                                    "Width    " : w,
                                    "Layers   " : l,
                                    "Weights sharing" : we 
                                }
                                config_index = config_index + 1
                            else:
                                config_index = config_index + 1
            
        else:
            raise Exception(f"ERROR (get_neural_network_configuration), parameterization {model_name} in configuration {model_config} does not exist")
    
    # ------------------------------------------------------------------------------------------------------
    #                                                Kaskade
    # ------------------------------------------------------------------------------------------------------
    elif model_name == "KASKADE":
        raise Exception(f"ERROR (get_neural_network_configuration), parameterization {model_name} in configuration {model_config} does not exist")
    
    # ------------------------------------------------------------------------------------------------------
    #                                                 Error
    # ------------------------------------------------------------------------------------------------------
    else:
        raise Exception(f"ERROR (get_neural_network_configuration), parameterization name {model_name} does not exist")
        
    return model_config_parameters