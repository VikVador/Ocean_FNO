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
import torch.nn as nn
from collections import OrderedDict

class UNet(nn.Module):

    def __init__(self, in_channels = 3, out_channels = 1, init_features = 32, zero_mean = True):
        super(UNet, self).__init__()
        
        # Storing more information about the data and parameters
        self.is_zero_mean = zero_mean
        features          = init_features
        
        # -------- ENCODER --------
        self.encoder1 = UNet._block(in_channels,  features,     name = "enc1")
        self.encoder2 = UNet._block(features,     features * 2, name = "enc2")
        self.encoder3 = UNet._block(features * 2, features * 4, name = "enc3")
        self.encoder4 = UNet._block(features * 4, features * 8, name = "enc4")
        
        self.pool1    = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool2    = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool3    = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.pool4    = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # ------- BOTTELNECK ------
        self.bottleneck = UNet._block(features * 8, features * 16, name = "bottleneck")

        # -------- DECODER --------
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name = "dec4")
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name = "dec3")
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name = "dec2")
        self.decoder1 = UNet._block( features * 2,      features,     name = "dec1")
        
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size = 2, stride = 2
        )
        self.upconv3 = nn.ConvTranspose2d(
            features * 8,  features * 4, kernel_size = 2, stride = 2
        )
        
        self.upconv2 = nn.ConvTranspose2d(
            features * 4,  features * 2, kernel_size = 2, stride = 2
        )
        
        self.upconv1 = nn.ConvTranspose2d(
            features * 2,  features, kernel_size = 2,     stride = 2
        )
        
        self.conv = nn.Conv2d(
            in_channels = features, out_channels = out_channels, kernel_size = 1
        )

        
    def forward(self, x):
        
        # -------- ENCODING --------
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # -------- BOTTLENECKING --------
        bottleneck = self.bottleneck(self.pool4(enc4))

        # -------- DECODING --------
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim = 1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim = 1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim = 1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim = 1)
        dec1 = self.decoder1(dec1)
        pred = self.conv(dec1)
        
        # Normalization of prediction
        if self.is_zero_mean:
            return pred - pred.mean(dim = (1,2,3), keepdim = True)
        else:
            return pred
    
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels  = in_channels,
                            out_channels = features,
                            kernel_size  = 3,
                            padding      = 1,
                            bias         = False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features = features)),
                    (name + "relu1", nn.ReLU(inplace = True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels  = features,
                            out_channels = features,
                            kernel_size  = 3,
                            padding      = 1,
                            bias         = False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features = features)),
                    (name + "relu2", nn.ReLU(inplace = True)),
                ]
            )
        )
    
    def count_parameters(self,):
        print("Model parameters  =", sum(p.numel() for p in self.parameters() if p.requires_grad))