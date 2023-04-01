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
import gcm_filters
import numpy       as np

from scipy.stats   import pearsonr
from tqdm.notebook import tqdm, trange
from functools     import cached_property

# --------- PYQG ---------
import pyqg

# --------- PYQG Benchmark ---------
from pyqg_parameterization_benchmarks.utils_TFE import *

# ----------------------------------------------------------------------------------------------------------
#
#                                                    Coarsener
#
# ----------------------------------------------------------------------------------------------------------
class Coarsener:

    def __init__(self, high_res_model, low_res_nx):

        # Security
        assert low_res_nx < high_res_model.nx
        assert low_res_nx % 2 == 0

        # Saving the high resolution simulation
        self.m1 = high_res_model
        self.m1._invert()

        # Initialization of a low resolution PYQG model using
        # the parameters coming from the high resolution (to make them match)
        self.m2 = pyqg.QGModel(nx = low_res_nx, **config_for(high_res_model))

        # Filtering high resolution model and placing results in low resolution model
        self.m2.q = self.coarsen(self.m1.q)
        
        # Compute psi_bar, u_bar, and v_bar
        self.m2._invert()
        self.m2._calc_derived_fields()

    def to_real(self, var):
        """
        Convert variable to real space, if needed.
        """
        for m in [self.m1, self.m2]:
            if var.shape == m.qh.shape:
                return m.ifft(var)
        return var

    def to_spec(self, var):
        """
        Convert variable to spectral space, if needed.
        """
        for m in [self.m1, self.m2]:
            if var.shape == m.q.shape:
                return m.fft(var)
        return var
    
    @property
    def q_forcing_total(self):
        """
        Compute total forcing between two models.
        """
        for m in [self.m1, self.m2]:
            m._invert()
            m._do_advection()
            m._do_friction()

        dqdt_bar = self.coarsen(self.m1.dqhdt)
        dqbar_dt = self.to_real(self.m2.dqhdt)
        sq_tot   = dqdt_bar - dqbar_dt
        
        return dqdt_bar, dqbar_dt, sq_tot

    def subgrid_forcing(self, var):
        """
        Compute subgrid forcing of a given `var` (as string).
        """
        q1 = getattr(self.m1, var)
        q2 = getattr(self.m2, var)
        
        # Compute non-linear advection
        adv_HR = self.coarsen(self.m1._advect(q1))
        adv_LR = self.to_real(self.m2._advect(q2))
        
        # Compute Reynolds stresses + div
        return adv_HR - adv_LR

    def subgrid_fluxes(self, var):
        """
        Compute subgrid fluxes (wrt. u and v) of a given `var`.
        """
        q1 = getattr(self.m1, var)
        q2 = getattr(self.m2, var)
        
        # Compute
        u_flux = self.coarsen(self.m1.ufull * q1) - self.m2.ufull * q2
        v_flux = self.coarsen(self.m1.vfull * q1) - self.m2.vfull * q2
        
        # Compute Reynolds stresses
        return u_flux, v_flux

    def coarsen(self, var):
        """
        Filter and coarse-grain a variable (as array).
        """
        raise NotImplementedError()
        
    @property
    def ratio(self):
        """
        Ratio of high-res to low-res grid length.
        """
        return self.m1.nx / self.m2.nx

    @cached_property
    def ds1(self):
        """
        xarray representation of the high-res model.
        """
        return self.m1.to_dataset()

# ----------------------------------------------------------------------------------------------------------
#
#                                                    Filtering
#
# ----------------------------------------------------------------------------------------------------------
class SpectralCoarsener(Coarsener):
    """
    Spectral truncation with a configurable filter.
    """
    def coarsen(self, var):

        # Truncate high-frequency indices & filter
        vh    = self.to_spec(var)
        nk    = self.m2.qh.shape[1]//2
        trunc = np.hstack((vh[:,  :nk, :nk+1],
                           vh[:, -nk:, :nk+1]))

        # Filtered data using spectral_filter defined afterwards
        filtered = trunc * self.spectral_filter / self.ratio**2
        return self.to_real(filtered)

    @property
    def spectral_filter(self):
        raise NotImplementedError()

# ----------------------------------------------------------------------------------------------------------
#                                                    Operators
# ----------------------------------------------------------------------------------------------------------
class Operator1(SpectralCoarsener):
    """
    Spectral truncation with a sharp filter.
    """
    @property
    def spectral_filter(self):
        return self.m2.filtr

class Operator2(SpectralCoarsener):
    """
    Spectral truncation with a softer Gaussian filter.
    """
    @property
    def spectral_filter(self):
        return np.exp(-self.m2.wv**2 * (2*self.m2.dx)**2 / 24)

class Operator3(Coarsener):
    """
    Diffusion-based filter, then real-space coarsening.
    """
    def coarsen(self, var):
        f = gcm_filters.Filter(dx_min=1,
            filter_scale=self.ratio,
            filter_shape=gcm_filters.FilterShape.GAUSSIAN,
            grid_type=gcm_filters.GridType.REGULAR)
        d = self.m1.to_dataset().isel(time=-1)
        q = d.q*0 + self.to_real(var) # hackily convert to data array
        r = int(self.ratio)
        assert r == self.ratio
        return f.apply(q, dims=['y','x']).coarsen(y=r, x=r).mean().data
