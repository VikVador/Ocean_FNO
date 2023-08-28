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
import torch.nn                                              as nn

# Library used until part 5
#import fourierflow.FactorizedFourierNeuralOperator._2D_.Base as FFNO_2D

# Revisited library since part 5 (code re-work for //)
import fourierflow.FactorizedFourierNeuralOperator._2D_.Base_V4 as FFNO_2D

class FactorizedFNO(nn.Module):

    def __init__(self, in_channels, out_channels, modes_x, modes_y, width, n_lay = 4, weight_sharing = False, zero_mean = True):
        super(FactorizedFNO, self).__init__()
        
        # Storing more information about the data and parameters
        self.is_zero_mean = True
        
        # ----------------------------------------------------
        #                    Architecture
        # ----------------------------------------------------
        self.model = FFNO_2D.FFNO(input_dim      = in_channels,
                                  output_dim     = out_channels,
                                  modes_x        = modes_x,
                                  modes_y        = modes_y,
                                  width          = width,
                                  n_layers       = n_lay,
                                  share_weight   = weight_sharing,
                                  factor         = 4,
                                  n_ff_layers    = 2,
                                  ff_weight_norm = False,
                                  layer_norm     = False)

    def forward(self, x):
    
        # ---- Shaping to [BS, X, Y, C] -----
        x = x.permute(0, 2, 3, 1)
        
        # ----------- Forwarding ------------
        x = self.model.forward(x)
        
        # ---- Shaping to [BS, C, X, Y] -----
        pred = x.permute(0, 3, 1, 2)
        
        # Normalization of prediction
        if self.is_zero_mean:
            return pred - pred.mean(dim = (1,2,3), keepdim = True)
        else:
            return pred
    
    def count_parameters(self,):
        print("Model parameters  =", sum(p.numel() for p in self.parameters() if p.requires_grad))