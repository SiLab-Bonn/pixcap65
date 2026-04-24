import numpy as np

SILICON_V_BIAS = 0.7
EPS_SILICON = 11.7


def full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1):
    # ignores the reference voltage for now
    return (u0 * c * freq + i) / (1 + r * c * freq)


def simple_capacitance_model(freq, c=1e-6, i=0, u0=1):
    return u0 * c * freq + i


def depletion_model(x, a=1, b=0):
    return a * x + b


def model_depletion(voltages, NA=1e16, ND=1e16, V=SILICON_V_BIAS):
    import scipy.constants as constants
    # modified the sign as the bias voltages are saved with correct sign assigned to them.
    return np.sqrt(2 * constants.epsilon_0 * EPS_SILICON / constants.e * (NA + ND) / (NA * ND) * (
            V - np.array(voltages)))


def semi_bias_model(voltages, bias=SILICON_V_BIAS, thermic=1, i=1):
    return np.where(voltages >= 0, i * (np.exp((voltages - bias) / thermic) - 1), i)
