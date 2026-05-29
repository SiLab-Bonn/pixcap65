import numpy as np
try:
    import numba as nb
except ImportError:
    class nb:
        @classmethod
        def njit(cls, **kwargs):
            pass

SILICON_V_BIAS = 0.7
EPS_SILICON = 11.7


@nb.njit(parallel=True, fastmath=True)
def full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1):
    # ignores the reference voltage for now
    return (u0 * c * freq + i) / (1 + r * c * freq)


@nb.njit(parallel=True, fastmath=True)
def grad_full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1):
    return np.array([
        (u0 * freq * (1 + r * c * freq) - (u0 * c * freq + i) * r * freq) / (1 + r * c * freq) ** 2,
        ((u0 * c * freq + i) * c * freq) / (1 + r * c * freq) ** 2,
        1 / (1 + r * c * freq),
        c * freq / (1 + r * c * freq)
    ])


@nb.njit(parallel=True, fastmath=True)
def simple_capacitance_model(freq, c=1e-6, i=0, u0=1):
    return u0 * c * freq + i


# noinspection PyUnusedLocal
@nb.njit(parallel=True, fastmath=True)
def grad_simple_capacitance_model(freq, c=1e-6, i=0, u0=1):
    return np.array([u0 * freq, 1, c * freq])


@nb.njit(parallel=True, fastmath=True)
def depletion_model(x, a=1, b=0):
    return a * x + b


# noinspection PyUnusedLocal
@nb.njit(parallel=True, fastmath=True)
def grad_depletion_model(x, a=1, b=0):
    return np.array([x, 1])


# noinspection PyPep8Naming
def model_depletion(voltages, NAD=5e15, V=SILICON_V_BIAS, dep=-10, sat=1):
    import scipy.constants as constants
    # modified the sign as the bias voltages are saved with correct sign assigned to them.
    return np.where(voltages > dep, np.sqrt(2 * constants.epsilon_0 * EPS_SILICON / (constants.e * NAD) * (
            V - np.array(voltages))) * 1e3, sat)


def semi_bias_model(voltages, bias=SILICON_V_BIAS, thermic=1, i=1):
    return np.where(voltages >= 0, i * (np.exp((voltages - bias) / thermic) - 1), i)


@nb.njit(parallel=True, fastmath=True)
def gauss_model(x, u=0, s=1, a=1):
    return a / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((x - u) / s) ** 2)


@nb.njit(parallel=True, fastmath=True)
def gauss_model_s(x, u=0, s=1):
    return 1 / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((x - u) / s) ** 2)


@nb.njit(parallel=True, fastmath=True)
def log_gauss_model(x, u=0, s=1):
    return -0.5 * ((x - u) / s) ** 2


def extended_gauss_model(x, b=1, u=0, s=1):
    from .stats import distribution_norm as norm
    return b, np.log(b) + norm.logpdf(x, loc=u, scale=s)


def extended_gauss_integral(xe, b=1, u=0, s=1):
    from .stats import distribution_norm as norm
    return b * norm.cdf(xe, loc=u, scale=s)
