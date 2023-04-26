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
import torch
import numpy    as np
import torch.nn as nn

# --------- PYQG Benchmark ---------
from pyqg_parameterization_benchmarks.nn_kaskade_blocs import *

class Kaskade(nn.Module):

    def __init__(self, inputs, targets, padding = 'circular', zero_mean = True):
        super().__init__()

        # Storing more information about the data and parameters
        self.padding      = padding
        self.inputs       = inputs
        self.targets      = targets
        self.is_zero_mean = zero_mean
        self.n_in         = len(inputs)
        
        # Dimension of input and output data
        n_in  = len(inputs)
        n_out = len(targets)
        
        # --------------------------------------------------------------------------------------------------
        #                                     Architecture (Parameters)
        # --------------------------------------------------------------------------------------------------
        # Number of blocks to use
        nb_blocks = 2
        
        # Depthwise convolution
        dept      = False
        
        # Activation function
        act       = nn.GELU()
        
        # Kernels
        kernel_large  = 11
        kernel_medium = 7
        kernel_small  = 3
        
        # Paddings
        padding_large  = 5
        padding_medium = 3 
        padding_small  = 1
        
        # Add block output to next block input
        self.kaskade_in  = True
        
        # Concatenation or addition of outputs
        self.kaskade_out = True
        
        # --------------------------------------------------------------------------------------------------
        #                                            Architecture
        # --------------------------------------------------------------------------------------------------
        self.large_1  = Block(input_size = n_in, kernel_size = kernel_large,  padding_size = padding_large,  nb_resblock = nb_blocks, 
                              activation = act, depthwise = dept, projector = True, se = True)
        self.large_2  = Block(input_size = n_in, kernel_size = kernel_large,  padding_size = padding_large,  nb_resblock = nb_blocks, 
                              activation = act, depthwise = dept, projector = True, se = True)
        self.large_3  = Block(input_size = n_in, kernel_size = kernel_large,  padding_size = padding_large,  nb_resblock = nb_blocks, 
                              activation = act, depthwise = dept, projector = False, se = False)
        
        self.medium_1 = Block(input_size = n_in, kernel_size = kernel_medium, padding_size = padding_medium, nb_resblock = nb_blocks, 
                              activation = act, depthwise = dept, projector = True, se = True)
        self.medium_2 = Block(input_size = n_in, kernel_size = kernel_medium, padding_size = padding_medium, nb_resblock = nb_blocks, 
                              activation = act, depthwise = dept, projector = True, se = True)
        self.medium_3 = Block(input_size = n_in, kernel_size = kernel_medium, padding_size = padding_medium, nb_resblock = nb_blocks, 
                              activation = act, depthwise = dept, projector = False, se = False)
        
        self.small_1  = Block(input_size = n_in, kernel_size = kernel_small,  padding_size = padding_small,  nb_resblock = nb_blocks, 
                              activation = act, depthwise = dept, projector = True, se = True)
        self.small_2  = Block(input_size = n_in, kernel_size = kernel_small,  padding_size = padding_small,  nb_resblock = nb_blocks, 
                              activation = act, depthwise = dept, projector = True, se = True)
        self.small_3  = Block(input_size = n_in, kernel_size = kernel_small,  padding_size = padding_small,  nb_resblock = nb_blocks, 
                              activation = act, depthwise = dept, projector = False, se = False)
            
        # Computing decoder in/out
        input_decoder = [3 * n_in, 2 * n_in] if self.kaskade_out else [1 * n_in, 1 * n_in]
        
        self.reductor = nn.Sequential(
            nn.Conv2d(in_channels = input_decoder[0], out_channels = input_decoder[1], kernel_size = 7, padding = 3, groups = n_in),
            nn.Conv2d(in_channels = input_decoder[1], out_channels = n_in,             kernel_size = 7, padding = 3, groups = n_in),
        )  
        
        self.decoder = nn.Sequential(
            Block(input_size = n_in, kernel_size = (3, 3), padding_size = 1, nb_resblock = 2, activation = nn.GELU(), depthwise = False, projector = False, se = True),
            nn.Conv2d(in_channels = n_in, out_channels = n_out, kernel_size = 5, padding = 2),
        ) 
                        
    # ------------------------------------------------------------------------------------------------------
    #                                                Forward
    # ------------------------------------------------------------------------------------------------------
    def forward(self, x):
        
        # ---- LARGE SCALE ----
        l1 = self.large_1(x)
        l2 = self.large_2(l1)
        l3 = self.large_3(l2)
        
        # ---- MEDIUM SCALE ----
        m1 = self.medium_1(torch.add(x, l3) if self.kaskade_in else x)
        m2 = self.medium_2(m1)
        m3 = self.medium_3(m2)
        
        # ---- SMALL SCALE ----
        s1 = self.small_1(torch.add(x, m3) if self.kaskade_in else x)
        s2 = self.small_2(s1)
        s3 = self.small_3(s2)
        
        # ---- DECODING ----
        S = None
        
        if self.kaskade_out:
            S = torch.cat((l3, m3, s3), axis = 1)
        else:
            S = torch.add(l3, m3)
            S = torch.add(S, s3)
            
        S = self.reductor(S)
        S = self.decoder(S)
    
        # Normalization of prediction
        if self.is_zero_mean:
            return S - S.mean(dim = (1, 2, 3), keepdim = True)
        else:
            return S