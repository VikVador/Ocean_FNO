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
        # Modes to keep per extractor
        modes_small  = (0,   8)
        modes_medium = (8,  16)
        modes_large  = (16, 24)
        
        # Width of Fourier Blocs
        width = 32
        
        # Number of Fourier Blocs per extractor
        number_layers = 4
            
        # --------------------------------------------------------------------------------------------------
        #                                            Architecture
        # --------------------------------------------------------------------------------------------------
        # Extractor for the different scales
        self.extractor_small  = Extractor(input_size = n_in, output_size = 1, modes = modes_small,  width = width, number_layers = number_layers)
        self.extractor_medium = Extractor(input_size = n_in, output_size = 1, modes = modes_medium, width = width, number_layers = number_layers)
        self.extractor_large  = Extractor(input_size = n_in, output_size = 1, modes = modes_large,  width = width, number_layers = number_layers)

        # Decoding resulting output
        self.decoder = Decoder(input_size = 6, output_size = n_out, channels = 6)

    # ------------------------------------------------------------------------------------------------------
    #                                                Forward
    # ------------------------------------------------------------------------------------------------------
    def forward(self, x):
        
        # ---- Extraction ----
        e_small  = self.extractor_small(x)
        e_medium = self.extractor_medium(x)
        e_large  = self.extractor_large(x)

        # ---- Decoding ----
        S = self.decoder(torch.cat((e_small, e_medium, e_large), dim = 1))
                
        # Normalization of prediction
        if self.is_zero_mean:
            return S - S.mean(dim = (1, 2, 3), keepdim = True)
        else:
            return S
        
    def count_parameters(self,):
        print("Model parameters  =", sum(p.numel() for p in self.parameters() if p.requires_grad))