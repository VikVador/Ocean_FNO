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
import torch.nn                                    as nn
import fourierflow.FourierNeuralOperator._2D_.Base as FNO_2D

class FourierNO(nn.Module):

    def __init__(self, in_channels, out_channels, modes_x, modes_y, width, n_lay = 4, zero_mean = True):
        super(FourierNO, self).__init__()
        
        # Storing more information about the data and parameters
        self.is_zero_mean = True
        
        # ----------------------------------------------------
        #                    Architecture
        # ----------------------------------------------------
        self.model = FNO_2D.FNO( input_channels = in_channels,
                                output_channels = out_channels,
                                         modes1 = modes_x,
                                         modes2 = modes_y,
                                          width = width,
                                       n_layers = n_lay)

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