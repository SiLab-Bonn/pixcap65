# ----------------------------------------------------------
#  Copyright (c) .
#   All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------
"""
Utility module to simplify the access to the different statistics functions depending on the presence of installed
modules.
It makes sure that if `numba_stats` is installed these optimized functions will be used instead of the `scipy`
implementation.
But one of these two must be installed.
"""


try:
    from numba_stats import norm
except ImportError:
    from scipy.stats import norm

__all__ = ["distribution_norm"]

distribution_norm = norm