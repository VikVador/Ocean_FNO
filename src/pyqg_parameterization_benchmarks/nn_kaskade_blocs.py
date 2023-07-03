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
import numpy                                                    as np
import torch.nn                                                 as nn
import fourierflow.FactorizedFourierNeuralOperator._2D_.Base_V3 as FFNO_2D

from einops     import rearrange

#----------------------------------------------------------------------------------------------------------------
#
#                                               Filter (Pass - Band)
#
#----------------------------------------------------------------------------------------------------------------
class Filtering(nn.Module):
    def __init__(self, modes_x, modes_y):
        super().__init__()

        # Initialization
        self.modes_x        = modes_x
        self.modes_y        = modes_y

    def forward(self, x):
        return self.spectral_filtering(x)

    def spectral_filtering(self, x):
        x          = rearrange(x, 'b m n i -> b i m n')
        B, I, M, N = x.shape
        x_fty      = torch.fft.rfft(x, dim = -1, norm = 'ortho')
        out_ft     = x_fty.new_zeros(B, I, M, N // 2 + 1)
        out_ft[:, :, :, self.modes_y[0]:self.modes_y[1]] = x_fty[:, :, :, self.modes_y[0]:self.modes_y[1]]
        xy     = torch.fft.irfft(out_ft, n = N, dim = -1, norm = 'ortho')
        x_ftx  = torch.fft.rfft(x, dim = -2, norm = 'ortho')
        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        out_ft[:, :, self.modes_x[0]:self.modes_x[1], :] = x_ftx[:, :, self.modes_x[0]:self.modes_x[1], :]
        xx     = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        x      = xx + xy
        x      = rearrange(x, 'b i m n -> b m n i')
        return x

#----------------------------------------------------------------------------------------------------------------
#
#                                             Dilated BottleNeck
#
#----------------------------------------------------------------------------------------------------------------
class DilatedBottleNeck(nn.Module):
    def __init__(self, input_size, proj_size, output_size, normalize = True):
        super().__init__()    
        
        # Bottle Neck
        self.projection   = nn.Conv2d(in_channels  = input_size, 
                                      out_channels = proj_size, 
                                      kernel_size  = (1, 1),
                                      padding      = (0, 0))
        
        self.conv_dilated = nn.Conv2d(in_channels  = proj_size,
                                      out_channels = proj_size,
                                      kernel_size  = (3, 3),
                                      padding      = (2, 2),
                                      dilation     = 2)
        
        self.reduction    = nn.Conv2d(in_channels  = proj_size, 
                                      out_channels = output_size, 
                                      kernel_size  = (1, 1),
                                      padding      = (0, 0))
        
        # Side Convolution
        self.conv_side    = nn.Conv2d(in_channels  = input_size, 
                                      out_channels = output_size, 
                                      kernel_size  = (1, 1),
                                      padding      = (0, 0))
    
        # Other stuff
        self.normalization  = nn.BatchNorm2d(output_size)
        self.normalize      = normalize
        self.activation     = nn.GELU()
        
    def forward(self, x):
        
        # Bottlenecking
        y = self.projection(x)
        y = self.conv_dilated(y)
        y = self.reduction(y)
        
        # Side convolution
        z = self.conv_side(x)
        
        # Addition of residuals
        x = torch.add(y, z)
        
        # Activation
        x = self.activation(x)
        
        # Normalization
        x = self.normalization(x) if self.normalize else x
        
        return x

#----------------------------------------------------------------------------------------------------------------
#
#                                                    Residual Block
#
#----------------------------------------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, padding_size, activation = True, normalize = True):
        super().__init__()

        # Residual Convolutions (right branch)
        self.conv_1 = nn.Conv2d(in_channels  = input_size, 
                                out_channels = output_size, 
                                kernel_size  = kernel_size,
                                padding      = padding_size)
        
        self.conv_2 = nn.Conv2d(in_channels  = output_size, 
                                out_channels = output_size, 
                                kernel_size  = kernel_size,
                                padding      = padding_size)
        
        # Skip connection (left branch)
        self.conv_3 = nn.Conv2d(in_channels  = input_size, 
                                out_channels = output_size, 
                                kernel_size  = kernel_size,
                                padding      = padding_size)
        
        
        # Other stuff
        self.normalization  = nn.BatchNorm2d(output_size)
        self.use_normalize  = normalize
        self.use_activation = activation
        self.activation     = nn.GELU()
        
    def forward(self, x):
        
        # Left branch
        l = self.conv_1(x)
        l = self.conv_2(l)

        # Right branch
        r = self.conv_3(x)
        
        # Addition of results
        x = r + l
        x = self.activation(x) if self.use_activation else x
        x = self.normalization(x) if self.use_normalize else x
        
        return x
        
#----------------------------------------------------------------------------------------------------------------
#
#                                                    Extractor
#
#----------------------------------------------------------------------------------------------------------------
class Extractor(nn.Module):

    def __init__(self, input_size, output_size, modes, width, number_layers):
        super().__init__()
        
        # Stores FFNO pass-band blocks
        self.ffno_blocks = nn.ModuleList([])
        
        # Stores residual blocks using input in time domain
        self.res_blocks = nn.ModuleList([])
        
        # Stores projection blocks used in input in time domain
        self.proj_blocks = nn.ModuleList([])
        
        # Stores normalization functions
        self.normal_blocs = nn.ModuleList([])
        
        # Used to filter initial input in time domain
        self.filter = Filtering(modes_x = modes , modes_y = modes)
                 
        # Adding all the corresponding blocks
        for i in range(number_layers) : 
            
            # Computing output dimension
            out_dim = input_size if i < number_layers - 1 else output_size
            
            self.ffno_blocks.append(FFNO_2D.FFNO(input_dim      = input_size,
                                                 output_dim     = out_dim,
                                                 modes_x        = modes,
                                                 modes_y        = modes,
                                                 width          = width,
                                                 n_layers       = 1,
                                                 share_weight   = False,
                                                 factor         = 4,
                                                 n_ff_layers    = 2,
                                                 ff_weight_norm = False,
                                                 layer_norm     = False))
            
            self.res_blocks.append(ResidualBlock(input_size     = input_size, 
                                                 output_size    = out_dim,
                                                 kernel_size    = (7, 7), 
                                                 padding_size   = (3, 3), 
                                                 activation     = True, 
                                                 normalize      = True))
            
            self.proj_blocks.append(DilatedBottleNeck(input_size  = input_size, 
                                                      proj_size   = input_size * 2, 
                                                      output_size = out_dim, 
                                                      normalize   = True))
            
            self.normal_blocs.append(nn.BatchNorm2d(out_dim))
        
    def forward(self, x):
                
        for i, (ffno_block, res_block, norm) in enumerate(zip(self.ffno_blocks, self.res_blocks, self.normal_blocs)):
            
            # ---- Initialization ----
            if i == 0:
                
                # ---- FREQUENCY DOMAIN ----
                f = ffno_block(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                
                # ------ TIME DOMAIN ------
                r = res_block(self.filter(x))
                
                # ------ Saving ------
                f_prime = f
                
            # ----- ODD STEP -----
            elif i + 1 % 2 == 0:
                
                # ---- FREQUENCY DOMAIN ----
                f = ffno_block(r.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                
                # ------ TIME DOMAIN ------
                r = res_block(f_prime)

            # ----- EVEN STEP -----
            else:
                
                # ---- FREQUENCY DOMAIN ----
                f = ffno_block(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                
                # ------ TIME DOMAIN ------
                r = res_block(r)
                
                # ------ Saving ------
                f_prime = f
                
        # Finalization       
        x = torch.cat((f, r), dim = 1)

        return x
    
#----------------------------------------------------------------------------------------------------------------
#
#                                                    Decoder
#
#----------------------------------------------------------------------------------------------------------------
class Decoder(nn.Module):

    def __init__(self, input_size, output_size, channels = 1):
        super().__init__()
        
        # Pre-decoding
        self.decoder_pointlike = ResidualBlock(input_size, channels, kernel_size = (1, 1), padding_size = (0, 0), activation = True, normalize = True)
        self.decoder_small     = ResidualBlock(input_size, channels, kernel_size = (3, 3), padding_size = (1, 1), activation = True, normalize = True)
        self.decoder_medium    = ResidualBlock(input_size, channels, kernel_size = (5, 5), padding_size = (2, 2), activation = True, normalize = True)
        self.decoder_large     = ResidualBlock(input_size, channels, kernel_size = (7, 7), padding_size = (3, 3), activation = True, normalize = True)
        
        # Ultimate merge
        self.decoder_merging   = ResidualBlock(channels * 4, output_size, kernel_size = (3, 3), padding_size = (1, 1), activation = False, normalize = False)
        
    def forward(self, x):
        
        # Pre-decoding
        p = self.decoder_pointlike(x)
        s = self.decoder_small(x)
        m = self.decoder_medium(x)
        l = self.decoder_large(x)
        
        # Ultimate merge
        x = self.decoder_merging(torch.cat((p, s, m, l), dim = 1))
        
        return x

    


