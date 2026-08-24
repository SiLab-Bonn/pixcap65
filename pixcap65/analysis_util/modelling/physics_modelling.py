"""
Collection of physical models used e.g. for analysing the capacitance measurements or the distribution of the
capacitance over a (full) sensor.
"""
# ----------------------------------------------------------
#  Copyright (c) 2026. SiLab, Institute of Physics, University of Bonn.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ----------------------------------------------------------

import numpy as np
try:
    import numba as nb
except ImportError:
    class nb:
        @classmethod
        def njit(cls, **kwargs):
            """Placeholder function for numba to not break code if numba is not installed"""
            pass

SILICON_V_BIAS = 0.7
EPS_SILICON = 11.7


@nb.njit(parallel=True, fastmath=True)
def full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1):
    """
    full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1)

    @author Dominik Fischer
    last update: 2026-08-24

    Full model of the frequency dependence of the measured current in order to determine the capacitance.
    This model in particular should correct for a voltage-drop over the switching-transistors on-resistance.
    :param freq: switching frequency for which the current should be predicted.
    :param c: capacitance of the circuit tested.
    :param r: resistance of the circuit tested and the measurement circuit (most likely the on-resistance).
    :param i: leakage current of the measurement.
    :param u0: full but constant charging voltage of the capacitance.
    :return: predicition of the current which should be measured.
    """
    # ignores the reference voltage for now
    return (u0 * c * freq + i) / (1 + r * c * freq)

@nb.njit(parallel=True, fastmath=True)
def extended_full_capacitance_model(freql, c=1e-6, r=1e6, i=0, u0=1, tau=1.0):
    """
    extended_full_capacitance_model(freql, c=1e-6, r=1e6, i=0, u0=1)

    @author Dominik Fischer
    last update: 2026-08-24

    Extension of the modelling implementation
    :func:`pixcap65.analysis_util.modelling.physics_modelling.full_capacitance_model`.
    This particular model also accounts for the exponential dependence of the voltage over the capacitance
    depending on the charging time.
    :param freql: switching frequency for which the current should be predicted.
    :param c: capacitance of the circuit tested.
    :param r: resistance of the circuit tested and the measurement circuit (most likely the on-resistance).
    :param i: leakage current of the measurement.
    :param u0: full but constant charging voltage of the capacitance.
    :param tau: time constant of the circuit charging the capacitance.
    :return: predicition of the current which should be measured.
    """
    return full_capacitance_model(freql, c, r, i, u0=u0*(1 - np.exp(-1/(2*tau*freql))))


@nb.njit(parallel=True, fastmath=True)
def enhanced_full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1):
    """
    enhanced_full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1)

    @author Dominik Fischer
    last update: 2026-08-24

    Extension of the modelling implementation
    :func:`pixcap65.analysis_util.modelling.physics_modelling.full_capacitance_model`.
    For this particular model it is assumed that the combination of the switching-transistors on-resistance and the
    RC-low pass filter on the PCB result in behaviour like a second-order low-pass filter.

    :param freq: switching frequency for which the current should be predicted.
    :param c: capacitance of the circuit tested.
    :param r: resistance of the circuit tested and the measurement circuit (most likely the on-resistance).
    :param i: leakage current of the measurement.
    :param u0: full but constant charging voltage of the capacitance.
    :return: predicition of the current which should be measured.
    """
    # ignores the reference voltage for now
    return (u0 * c * freq + i) / (1 + r * c * freq)**2


@nb.njit(parallel=True, fastmath=True)
def grad_full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1):
    """
    Utility to estimate the gradient of the full capacitance model
    :func:`pixcap65.analysis_util.modelling.physics_modelling.full_capacitance_model`
    in parameter space.
    This could be useful to boost fitting algorithms with an analytic expression for the gradient when using gradient
    based procedures.
    :param freq: switching frequency for which the current should be predicted.
    :param c: capacitance of the circuit tested.
    :param r: resistance of the circuit tested and the measurement circuit (most likely the on-resistance).
    :param i: leakage current of the measurement.
    :param u0: full but constant charging voltage of the capacitance.
    :return: gradient of predicted current in parameter space.
    """
    return np.array([
        (u0 * freq * (1 + r * c * freq) - (u0 * c * freq + i) * r * freq) / (1 + r * c * freq) ** 2,
        ((u0 * c * freq + i) * c * freq) / (1 + r * c * freq) ** 2,
        1 / (1 + r * c * freq),
        c * freq / (1 + r * c * freq)
    ])


@nb.njit(parallel=True, fastmath=True)
def simple_capacitance_model(freq, c=1e-6, i=0, u0=1):
    """
    Simplified linear model of the averaged charging current in dependence on the switching frequency,
    characterized by the capacitance.
    This model neglects higher-order effects, e.g. by the finite/small charging time, and voltage-drops over components
    of the measurement circuit.
    The model will fail for sufficiently high switching-frequencies.

    :param freq: switching frequency for which the current should be predicted.
    :param c: capacitance of the circuit tested.
    :param i: leakage current of the measurement.
    :param u0: full but constant charging voltage of the capacitance.
    :return: predicition of the current which should be measured.
    """
    return u0 * c * freq + i


# noinspection PyUnusedLocal
@nb.njit(parallel=True, fastmath=True)
def grad_simple_capacitance_model(freq, c=1e-6, i=0, u0=1):
    """
    Utility to estimate the gradient of the simplified capacitance model
    :func:`pixcap65.analysis_util.modelling.physics_modelling.simple_capacitance_model`
    in parameter space.
    This could be useful to boost fitting algorithms with an analytic expression for the gradient when using gradient
    based procedures.
    :param freq: switching frequency for which the current should be predicted.
    :param c: capacitance of the circuit tested.
    :param i: leakage current of the measurement.
    :param u0: full but constant charging voltage of the capacitance.
    :return: gradient of predicted current in parameter space.
    """
    return np.array([u0 * freq, 1, c * freq])


@nb.njit(parallel=True, fastmath=True)
def depletion_model(x, a=1, b=0):
    """
    Linear model to estimate the depletion voltage by intersection of two straigth-line fits in asymptotic regions.

    :param x: bias voltage of the $1/C^2$
    :param a: slope parameter
    :param b: offset parameter
    :return: prediction value for $1/C^2$
    """
    return a * x + b


# noinspection PyUnusedLocal
@nb.njit(parallel=True, fastmath=True)
def grad_depletion_model(x, a=1, b=0):
    """
    Utility function for the gradient of a Linear model to estimate the depletion voltage by intersection of
    two straigth-line fits in asymptotic regions.
    This could be useful to boost fitting algorithms with an analytic expression for the gradient when using gradient
    based procedures.

    :param x: bias voltage of the $1/C^2$
    :param a: slope parameter
    :param b: offset parameter
    :return: gradient of prediction value for $1/C^2$ in parameter space.
    """
    return np.array([x, 1])


# noinspection PyPep8Naming
def model_depletion(voltages, NAD=5e15, V=SILICON_V_BIAS, dep=-10, sat=1):
    """
    Modelling the dependence of the depletion width/depth in dependence on the bias voltage when applying reversed-bias.
    :param voltages: bias voltages
    :param NAD: combined donator-acceptor-density.
    :param V: bias voltage/threshold voltage of the semiconductor
    :param dep: maximum depletion voltage, as the modelling is changing if it is exceeded.
    :param sat: saturation value of the depletion depth, meaning the full depth of the sensors substrate.
    :return: prediction for the depletion depth.
    """
    import scipy.constants as constants
    # modified the sign as the bias voltages are saved with correct sign assigned to them.
    return np.where(voltages > dep, np.sqrt(2 * constants.epsilon_0 * EPS_SILICON / (constants.e * NAD) * (
            V - np.array(voltages))) * 1e3, sat)


def semi_bias_model(voltages, bias=SILICON_V_BIAS, thermic=1, i=1):
    """
    Implements the Shottky-Model for the current through a semiconducting diode
    :param voltages: voltage(s) used for biasing the sensor
    :param bias: bias voltage/threshold voltage of the semiconductor
    :param thermic: thermic energy at the time of the measurement (given by the room temperature and the boltzmann constant)
    :param i: leakage current
    :return: prediction for the leakage current of the sensor.
    """
    return np.where(voltages >= 0, i * (np.exp((voltages - bias) / thermic) - 1), i)


@nb.njit(parallel=True, fastmath=True)
def gauss_model(x, u=0, s=1, a=1):
    """
    Implementation of a normal distribution to model the histogram of the capacitance values over a (full) sensor in
    order to estimate the spread of the capacitances' over a sensor.
    :param x: capacitance bin
    :param u: average value/central value of the capacitance of the sensor
    :param s: spread of the capacitance of the sensor
    :param a: normalization/scaling factor to correct for the fact that the histogram is not normalized to a density.
    :return: prediction of a quantity.
    """
    return a / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((x - u) / s) ** 2)


@nb.njit(parallel=True, fastmath=True)
def gauss_model_s(x, u=0, s=1):
    """
    Implementation of a normal distribution to model the histogram of the capacitance values over a (full) sensor in
    order to estimate the spread of the capacitances' over a sensor.
    This is particular implementation is for usage for normalized data.
    :param x: capacitance bin
    :param u: average value/central value of the capacitance of the sensor
    :param s: spread of the capacitance of the sensor
    :return: prediction of a quantity.
    """
    return 1 / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((x - u) / s) ** 2)


@nb.njit(parallel=True, fastmath=True)
def log_gauss_model(x, u=0, s=1):
    """
    Implementation of a normal distribution (its natural logarithm) to model the histogram of the capacitance values over a (full) sensor in
    order to estimate the spread of the capacitances' over a sensor.
    :param x: capacitance bin
    :param u: average value/central value of the capacitance of the sensor
    :param s: spread of the capacitance of the sensor
    :param a: normalization/scaling factor to correct for the fact that the histogram is not normalized to a density.
    :return: log of prediction of a quantity.
    """
    return -0.5 * ((x - u) / s) ** 2


def extended_gauss_model(x, b=1, u=0, s=1):
    """
    Implementation of a normal distribution (its natural logarithm) to model the histogram of the capacitance values over a (full) sensor in
    order to estimate the spread of the capacitances' over a sensor.
    Parameterisation is intended for extended NLL fit.
    :param x: capacitance bin
    :param u: average value/central value of the capacitance of the sensor
    :param s: spread of the capacitance of the sensor
    :param b: normalization/scaling factor to correct for the fact that the histogram is not normalized to a density.
    :return: log of prediction of a quantity.
    """
    from pixcap65.analysis_util.stats import distribution_norm as norm
    return b, np.log(b) + norm.logpdf(x, loc=u, scale=s)


def extended_gauss_integral(xe, b=1, u=0, s=1):
    """
    implementation of the cumulative distribution of a normal distribution to model the histogram of the capacitance
    values over a (full) sensor by an extended NLL fit.
    :param xe: bin and lower edge for the cumulative distribution
    :param b: normalisation factor for the extended NLL fit.
    :param u: average value/central value of the capacitance of the sensor.
    :param s: spread of the capacitance of the sensor.
    :return: prediction of a quantity.
    """
    from pixcap65.analysis_util.stats import distribution_norm as norm
    return b * norm.cdf(xe, loc=u, scale=s)
