# ----------------------------------------------------------
#  Copyright (c) .
#   All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------


try:
    from numba_stats import norm
except ImportError:
    from scipy.stats import norm

__all__ = ["distribution_norm"]

distribution_norm = norm