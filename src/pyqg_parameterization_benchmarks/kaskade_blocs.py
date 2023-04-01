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
import numpy as np
import torch.nn as nn

class KKB_EXTRACTOR(nn.Module):
    def __init__(self, input_size, extractor_kernel, extractor_padding, feature_amplification = 2):
        super().__init__()
        
        self.depthwise_1    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)
        
        self.depthwise_2    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)
        
        self.depthwise_3    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)
        
        self.depthwise_4    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)

        self.depthwise_5    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)
        
        self.depthwise_6    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)

        self.reduce         = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size * feature_amplification, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size,
                                        stride       = 2)
        
        self.activation_1   = nn.GELU()
        self.activation_2   = nn.GELU()
        self.activation_3   = nn.GELU()
        self.norm_1         = nn.BatchNorm2d(input_size)
        self.norm_2         = nn.BatchNorm2d(input_size)
        self.norm_3         = nn.BatchNorm2d(input_size)
        self.norm_4         = nn.BatchNorm2d(input_size * feature_amplification)

    def forward(self, x):
        y = self.depthwise_1(x)
        y = self.depthwise_2(y)
        y = torch.add(x, y)
        y = self.activation_1(y)
        y = self.norm_1(y)

        w = self.depthwise_3(y)
        w = self.depthwise_4(w)
        w = torch.add(y, w)
        w = self.activation_2(w)
        w = self.norm_2(w)

        z = self.depthwise_5(w)
        z = self.depthwise_6(z)
        z = torch.add(w, z)
        z = self.activation_3(z)
        z = self.norm_3(z)

        s = self.reduce(z)
        s = self.norm_4(s)
        return s
    
    
class KKB_PROJECTOR(nn.Module):
    def __init__(self, input_size, projector_kernel, projector_padding, projection_size = 2):
        super().__init__()

        self.depthwise_1    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = projector_kernel,
                                        padding      = projector_padding,
                                        groups       = input_size)
        
        self.depthwise_2    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = projector_kernel,
                                        padding      = projector_padding,
                                        groups       = input_size)
    

        self.projection     = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size * projection_size, 
                                        kernel_size  = projector_kernel,
                                        padding      = projector_padding,
                                        groups       = input_size)
        
        self.intermediate   = nn.Conv2d(in_channels  = input_size * projection_size,
                                        out_channels = input_size * projection_size,
                                        kernel_size  = (1, 1),
                                        padding      = 0,
                                        groups       = input_size * projection_size)

        self.reduction      = nn.Conv2d(in_channels  = input_size * projection_size, 
                                        out_channels = input_size, 
                                        kernel_size  = (1, 1),
                                        padding      = 0,
                                        groups       = input_size)
        
        self.activation_1 = nn.GELU()
        self.activation_2 = nn.GELU()
        self.activation_3 = nn.GELU()
        self.norm_1       = nn.BatchNorm2d(input_size)
        self.norm_2       = nn.BatchNorm2d(input_size * projection_size)
        self.norm_3       = nn.BatchNorm2d(input_size * projection_size)
        self.norm_4       = nn.BatchNorm2d(input_size)
    

    def forward(self, x):
        y = self.depthwise_1(x)
        y = self.depthwise_2(y)
        y = torch.add(x, y)
        y = self.activation_1(y)
        y = self.norm_1(y)

        y = self.projection(y)
        y = self.norm_2(y)
        
        y = self.intermediate(y)
        y = self.norm_3(y)

        y = self.reduction(y)
        s = self.norm_4(y)

        return s

# ----------------------------------------------------------------------------------------------------------
#
#                                       Kaskade Bloc - Functions
#
# ----------------------------------------------------------------------------------------------------------
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

class KKB_AMPLIFICATOR(nn.Module):
    def __init__(self, input_size, extractor_kernel, extractor_padding, ratio = 2):
        super().__init__()
        
        self.depthwise_1    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)
        
        self.depthwise_2    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)
        
        self.depthwise_3    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)
        
        self.depthwise_4    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)

        self.depthwise_5    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)
        
        self.depthwise_6    = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = extractor_kernel,
                                        padding      = extractor_padding,
                                        groups       = input_size)
        
        self.se_1            = SqueezeExcitationBlock(input_size,  ratio) 
        self.se_2            = SqueezeExcitationBlock(input_size,  ratio) 
        self.se_3            = SqueezeExcitationBlock(input_size,  ratio) 
        self.activation_1   = nn.GELU()
        self.activation_2   = nn.GELU()
        self.activation_3   = nn.GELU()
        self.norm_1         = nn.BatchNorm2d(input_size)
        self.norm_2         = nn.BatchNorm2d(input_size)
        self.norm_3         = nn.BatchNorm2d(input_size)

    def forward(self, x):
        y = self.depthwise_1(x)
        y = self.depthwise_2(y)
        y = torch.add(x, y)
        y = self.activation_1(y)
        y = self.norm_1(y)
        y = self.se_1(y)

        w = self.depthwise_3(y)
        w = self.depthwise_4(w)
        w = torch.add(y, w)
        w = self.activation_2(w)
        w = self.norm_2(w)
        w = self.se_2(w)

        z = self.depthwise_5(w)
        z = self.depthwise_6(z)
        z = torch.add(w, z)
        z = self.activation_3(z)
        z = self.norm_3(z)
        s = self.se_3(z)

        return s

# ----------------------------------------------------------------------------------------------------------
#
#                                         Kaskade Bloc - Reduction 
#
# ----------------------------------------------------------------------------------------------------------
class KKB_RED(nn.Module):
    """
    -------------
    Documentation
    -------------
    This block is an architectural unit designed to perform a feature reduction:

        (H, W, C) --> (H/2, W/2, C * feature_amplification)

    """
    def __init__(self, input_size, feature_amplification = 2):
        super().__init__()

        self.residual_1     = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = (7, 7),
                                        padding      = 3)
        
        self.residual_2     = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = (7, 7),
                                        padding      = 3)
        
        self.residual_3     = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = (7, 7),
                                        padding      = 3)
        
        self.residual_4     = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size, 
                                        kernel_size  = (7, 7),
                                        padding      = 3)
        
        self.reduce         = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size * feature_amplification, 
                                        kernel_size  = (3, 3),
                                        padding      = 1,
                                        stride       = 2)

        self.se_1         = SqueezeExcitationBlock(input_size,  1)
        self.se_2         = SqueezeExcitationBlock(input_size,  1)
        self.activation_1 = nn.GELU()
        self.activation_2 = nn.GELU()
        self.norm_1       = nn.BatchNorm2d(input_size)    
        self.norm_2       = nn.BatchNorm2d(input_size)
    
    def forward(self, x):
        y = self.residual_1(x)
        y = self.residual_2(y)
        y = torch.add(x, y)
        y = self.se_1(y)
        y = self.activation_1(y)
        y = self.norm_1(y)
        
        w = self.residual_3(y)
        w = self.residual_4(w)
        w = torch.add(y, w)
        w = self.se_1(w)
        w = self.activation_2(w)
        w = self.norm_2(w)
        
        w = self.reduce(w)
        
        return w

# ----------------------------------------------------------------------------------------------------------
#
#                                       Kaskade Bloc - Projection & Reduction 
#
# ----------------------------------------------------------------------------------------------------------
class KKB_LPR(nn.Module):
    """
    -------------
    Documentation
    -------------
    This block is an architectural unit designed to perform a combination of linear projection and reduction (bottleneck)

        (H, W, C) --> (H, W, C)

    """
    def __init__(self, input_size, projection_size = 2):
        super().__init__()


        self.projection     = nn.Conv2d(in_channels  = input_size, 
                                        out_channels = input_size * projection_size, 
                                        kernel_size  = (7, 7),
                                        padding      = 3)
        
        self.intermediate   = nn.Conv2d(in_channels  = input_size * projection_size,
                                        out_channels = input_size * projection_size,
                                        kernel_size  = (1, 1),
                                        padding      = 0)

        self.reduction      = nn.Conv2d(in_channels  = input_size * projection_size, 
                                        out_channels = input_size, 
                                        kernel_size  = (1, 1),
                                        padding      = 0)
        
        self.projection_act   = nn.LayerNorm(input_size * projection_size)
        self.intermediate_act = nn.GELU()
        
    def forward(self, x):
        y = self.projection(x)
        y = torch.transpose(self.projection_act(torch.transpose(y, 1, 3)), 3, 1)
        y = self.intermediate(y)        
        y = self.intermediate_act(y)
        y = self.reduction(y)
        x = torch.add(x, y)
        return y

# ----------------------------------------------------------------------------------------------------------
#
#                                          Kaskade Bloc - Squeeze & Excitation
#
# ----------------------------------------------------------------------------------------------------------    
class KKB_SER(nn.Module):
    """
    -------------
    Documentation
    -------------
    This block is an architectural unit designed to perform a combination of squeeze-excitation and feature reconstitution

        (H, W, C) --> (H, W, C)

    """
    def __init__(self, input_size, output_size = 2, kernel_size = (4, 4), reconstitution_factor = 4, padding = 0):
        super().__init__()

        self.residual_1      = nn.Conv2d(in_channels  = input_size, 
                                         out_channels = input_size, 
                                         kernel_size  = (7, 7),
                                         padding      = 3)
        
        self.residual_2      = nn.Conv2d(in_channels  = input_size, 
                                         out_channels = input_size, 
                                         kernel_size  = (7, 7),
                                         padding      = 3)
        
        self.residual_3      = nn.Conv2d(in_channels  = input_size, 
                                         out_channels = input_size, 
                                         kernel_size  = (7, 7),
                                         padding      = 3)
        
        self.residual_4      = nn.Conv2d(in_channels  = input_size, 
                                         out_channels = input_size, 
                                         kernel_size  = (7, 7),
                                         padding      = 3)
    
        self.se              = SqueezeExcitationBlock(input_size,  2)
    
        self.reconstitution  = nn.ConvTranspose2d(in_channels  = input_size, 
                                                  out_channels = output_size, 
                                                  kernel_size  = kernel_size,
                                                  padding      = padding,
                                                  stride       = reconstitution_factor)
        
        self.res_1_act       = nn.BatchNorm2d(input_size)
        
        self.res_2_act       = nn.BatchNorm2d(input_size)

    def forward(self, x):
        y = self.residual_1(x)
        y = self.residual_2(y)
        y = self.res_1_act(torch.add(x, y))
        z = self.residual_3(y)
        z = self.residual_4(z)
        z = self.res_2_act(torch.add(y, z))
        w = self.se(z)
        w = self.reconstitution(w)
        return w
    
# ----------------------------------------------------------------------------------------------------------
#
#                                            Kaskade Bloc - Combination
#
# ---------------------------------------------------------------------------------------------------------- 
class KKB_COM(nn.Module):
    """
    -------------
    Documentation
    -------------
    This block is an architectural unit designed to perform a combination of the multiple features extracted at different scales

        (H, W, C) --> (H, W, C)

    """
    def __init__(self, input_size):
        super().__init__()

        self.extraction    = nn.ConvTranspose2d(in_channels  = input_size, 
                                                out_channels = input_size, 
                                                kernel_size  = (6, 6),
                                                padding      = 2,
                                                stride       = 2)
        
        self.combination_1 = nn.Conv2d(in_channels  = input_size, 
                                       out_channels = 12, 
                                       kernel_size  = (7, 7),
                                       padding      = 3)
        
        self.combination_2 = nn.Conv2d(in_channels  = 12, 
                                       out_channels = 6, 
                                       kernel_size  = (5, 5),
                                       padding      = 2)
        
        self.combination_3 = nn.Conv2d(in_channels  = 6, 
                                       out_channels = 1, 
                                       kernel_size  = (3, 3),
                                       padding      = 1)
        
    def forward(self, x):
        x = self.extraction(x)
        x = self.combination_1(x)
        x = self.combination_2(x)
        x = self.combination_3(x)
        return x