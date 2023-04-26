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

#----------------------------------------------------------------------------------------------------------------
#
#                                                  Squeeze & Excitation
#
#----------------------------------------------------------------------------------------------------------------
# -- Functions --
def GlobalAveragePooling2D(x):
    
    # Global averging
    x = torch.mean(x, dim = (2, 3))

    # Adding missing dimensions
    x = torch.unsqueeze(x, axis = 0)
    x = torch.unsqueeze(x, axis = 1)
    
    return x

class SqueezeExcitationBlock(nn.Module):
    """
    -------------
    Documentation
    -------------
    The Squeeze-and-Excitation Block is an architectural unit designed to improve the representational 
    power of a network by enabling it to perform dynamic channel-wise feature recalibration:

        (H, W, C) --> (H, W, C) with H = H_0 * SE & C = C_0 * SE
    
    ---------
    Variables
    ---------
    - c : number of channels in the network.
    - r : filter reduction ratio
    """
    def __init__(self, c, r):
        super().__init__()

        # Security
        assert c > 0 and r > 0, \
            f"Assert (SE_Block): c and r must be positive integers"
        
        assert c >= r, \
            f"Assert (SE_Block): c must be greater than r"
        
        # Initialization of the layers
        self.squeeze        = nn.Linear(in_features = c, out_features = np.ceil(c/r).astype("int"))
        self.squeeze_act    = nn.ReLU()

        self.excitation     = nn.Linear(in_features = np.ceil(c/r).astype("int"), out_features = c)
        self.excitation_act = nn.Sigmoid()

    def forward(self, x):
        y = GlobalAveragePooling2D(x)
        y = self.squeeze_act(self.squeeze(y))
        y = self.excitation_act(self.excitation(y))
        return x * torch.transpose(torch.transpose(y, 0, 2), 1, 3)

#----------------------------------------------------------------------------------------------------------------
#
#                                             Projection & Reduction (Bottleneck)
#
#----------------------------------------------------------------------------------------------------------------
class ProjectorBlock(nn.Module):
    def __init__(self, input_size, projector_kernel, projector_padding, projection_size = 4):
        super().__init__()    

        self.projection     = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = projector_kernel,
                                        padding      = projector_padding,
                                        groups       = input_size)
        
        self.intermediate   = nn.Conv2d(in_channels  = input_size,
                                        out_channels = input_size * projection_size,
                                        kernel_size  = (1, 1),
                                        padding      = 0,
                                        groups       = input_size)

        self.reduction      = nn.Conv2d(in_channels  = input_size * projection_size, 
                                        out_channels = input_size, 
                                        kernel_size  = (1, 1),
                                        padding      = 0,
                                        groups       = input_size)
        
        self.normalization  = nn.BatchNorm2d(input_size)
        self.activation     = nn.GELU()
        
    def forward(self, x):
        y = self.projection(x)
        y = self.normalization(y)
        y = self.intermediate(y)
        y = self.activation(y)
        x = x + self.reduction(y)
        return x
    
#----------------------------------------------------------------------------------------------------------------
#
#                                                    Residual Block
#
#----------------------------------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, input_size, kernel_size, padding_size, dilation = None, activation = None, normalization = None, depthwise = False):
        super().__init__()

        # Determine whether or not to perform depthwise convolution
        self.group_size    = input_size                 if depthwise     == True else 1
        self.activation    = nn.GELU()                  if activation    == None else activation
        self.normalization = nn.BatchNorm2d(input_size) if normalization == None else normalization

        self.conv_1 = nn.Conv2d(in_channels  = input_size, 
                                out_channels = input_size, 
                                kernel_size  = kernel_size,
                                padding      = padding_size,
                                groups       = self.group_size)
        
        self.conv_2 = nn.Conv2d(in_channels  = input_size, 
                                out_channels = input_size, 
                                kernel_size  = kernel_size,
                                padding      = padding_size,
                                groups       = self.group_size)
        
    def forward(self, x):
        y = self.conv_1(x)
        y = self.conv_2(y)
        x = y + x
        x = self.activation(x)
        x = self.normalization(x)
        return x
    
#----------------------------------------------------------------------------------------------------------------
#
#                                                    Block module
#
#----------------------------------------------------------------------------------------------------------------
class Block(nn.Module):

    def __init__(self, input_size, kernel_size, padding_size, nb_resblock, 
                activation = None, depthwise = False, projector = False, se = False):
        super().__init__()
        
        # Adding residual blocks
        subblocks = [ResBlock(input_size = input_size, kernel_size = kernel_size, padding_size = padding_size, activation = activation, depthwise = depthwise)
                     for i in range(nb_resblock)]
        
        # Adding projection block
        if projector:
            subblocks.append(ProjectorBlock(input_size, projector_kernel = (7, 7), projector_padding = 3, projection_size = 8))
        
        # Adding squeeze & excitation block
        if se:
            subblocks.append(SqueezeExcitationBlock(input_size, 2))
        
        # Compiling everything togheter
        self.blocks = nn.Sequential(*subblocks)

    def forward(self, x):
        x = self.blocks(x)
        return x
    


    

