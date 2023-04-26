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
import pyqg
import numpy as np

# --------- PYQG Benchmark ---------
from pyqg_parameterization_benchmarks.utils import FeatureExtractor, Parameterization

# ---------------------
#   Parameterizations
# ---------------------
class BackscatterBiharmonic(Parameterization):
    
    def __init__(self, Cs, Cb): 
        self._param = pyqg.BackscatterBiharmonic(Cs, Cb)
        
    def predict(self, m): 
        return dict(q_subgrid_forcing = self._param(m))
    
    @property
    def targets(self): 
        return ['q_subgrid_forcing']

class Smagorinsky(Parameterization):
    def __init__(self, Cs): 
        self._param = pyqg.Smagorinsky(Cs)
        
    def predict(self, m):
        Su, Sv = self._param(m)
        return dict(u_subgrid_forcing = Su, v_subgrid_forcing = Sv)
    
    @property
    def targets(self): 
        return ['u_subgrid_forcing','v_subgrid_forcing']
    
class HybridSymbolic(Parameterization):
    
    # Exact weights learned in the paper (L. Zanne & Al. 2020)
    weights = np.array([
        [ 1.4077349573135765e+07,  1.9300721349777748e+15,
          2.3311494532833229e+22,  1.1828024430000000e+09,
          1.1410567621344224e+17, -6.7029178551956909e+10,
          8.9901990193476257e+10],
        [ 5.196460289865505e+06,  7.031351150824246e+14,
          1.130130768679029e+11,  8.654265196250000e+08,
          7.496556547888773e+16, -8.300923156070618e+11,
          9.790139405295905e+11]
    ]).T[:,:,np.newaxis,np.newaxis]
    
    def terms(self, m):
        return FeatureExtractor(m)([
            'laplacian(advected(q))',
            'laplacian(laplacian(advected(q)))',
            'laplacian(laplacian(laplacian(advected(q))))',
            'laplacian(laplacian(q))',
            'laplacian(laplacian(laplacian(q)))',
            'advected(advected(ddx(laplacian(v))))',
            'advected(advected(ddy(laplacian(u))))'
        ])
    
    def predict(self, m):
        return dict(q_subgrid_forcing = (self.weights * self.terms(m)).sum(axis = 0))
    
    @property
    def targets(self):
        return ['q_subgrid_forcing']