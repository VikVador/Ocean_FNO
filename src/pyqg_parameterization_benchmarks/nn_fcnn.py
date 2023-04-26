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

class FullyCNN(nn.Sequential):

    def __init__(self, inputs, targets, padding = 'circular', zero_mean = True):

        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding in ['same', 'circular']:
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('FullyCNN - Unknow value for padding parameter, i.e. should be None or circular')

        # Dimension of input and output data
        n_in  = len(inputs)
        n_out = len(targets)

        # Storing more information about the data and parameters
        self.padding      = padding
        self.inputs       = inputs
        self.targets      = targets
        self.is_zero_mean = zero_mean
        self.n_in         = n_in
        kw = {}
        if padding == 'circular':
            kw['padding_mode'] = 'circular'

        #-----------------------------------------------------------------------------------
        #                                   Architecture
        #-----------------------------------------------------------------------------------
        block1 = self._make_subblock(nn.Conv2d(n_in, 128, 5, padding = padding_5, **kw))
        block2 = self._make_subblock(nn.Conv2d(128,   64, 5, padding = padding_5, **kw))
        block3 = self._make_subblock(nn.Conv2d(64,    32, 3, padding = padding_3, **kw))
        block4 = self._make_subblock(nn.Conv2d(32,    32, 3, padding = padding_3, **kw))
        block5 = self._make_subblock(nn.Conv2d(32,    32, 3, padding = padding_3, **kw))
        block6 = self._make_subblock(nn.Conv2d(32,    32, 3, padding = padding_3, **kw))
        block7 = self._make_subblock(nn.Conv2d(32,    32, 3, padding = padding_3, **kw))
        conv8  =                     nn.Conv2d(32, n_out, 3, padding = padding_3)

        # Combining everything together
        super().__init__(*block1, *block2, *block3, *block4, *block5, *block6, *block7, conv8)

    def _make_subblock(self, conv):
        return [conv, nn.ReLU(), nn.BatchNorm2d(conv.out_channels)]

    #-----------------------------------------------------------------------------------
    #                                     Forward
    #-----------------------------------------------------------------------------------
    def forward(self, x):
        
        # Final prediction
        pred = super().forward(x)
        
        # Normalization of prediction
        if self.is_zero_mean:
            return pred - pred.mean(dim = (1,2,3), keepdim = True)
        else:
            return pred