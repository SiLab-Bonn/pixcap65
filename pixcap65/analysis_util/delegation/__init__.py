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
"""
Module for delegation functions to perform the actual fits in order to determine the capacitances from current
measurements.
"""
import numpy as np
try:
    # noinspection PyCompatibility
    from collections.abc import Callable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from typing import Callable
from typing import Any

from pixcap65.analysis_util.modelling.configuration_constants import ANALYSIS_INITIAL_CAPACITANCE, \
    ANALYSIS_INITIAL_RESISTANCE, ANALYSIS_INITIAL_LEAKAGE, ANALYSIS_REFERENCE_VOLTAGE
from pixcap65.analysis_util.modelling.physics_modelling import extended_full_capacitance_model, \
    enhanced_full_capacitance_model, full_capacitance_model, simple_capacitance_model
from pixcap65.analysis_util.utility import FULL_MODEL_LABEL, FULL_MODEL_EXPRESSION, FULL_MODEL_PARAMETER_DICT, \
    SIMPLE_MODEL_LABEL, SIMPLE_MODEL_EXPRESSION

ANALYSIS_FIT_Y_LABEL = "$I$ in A"
ANALYSIS_FIT_X_LABEL = "$\\nu$ in MHz"
ANALYSIS_FIT_CONTOUR_LEGEND = "Contour profiles for pixel ({col}, {row})"
ANALYSIS_FIT_PLOT_LEGEND = "Fit of the frequency dependence for pixel ({col}, {row})"
ANALYSIS_GROUP_NAME = "analysis"
ANALYSIS_CORRECTED_GROUP_NAME = "analysis_correction"
PERFORM_PIXEL_TYPE = np.dtype([('C', np.float64), ('Cerr', np.float64),
                               ('I', np.float64), ('Ierr', np.float64),
                               ('R', np.float64), ('Rerr', np.float64),
                               ('fit', Any), ('cov', np.ndarray)])


def __declare_fit_model(full_model) -> tuple[int, Callable[..., Any], str, str, dict[str, str], dict[str, float]]:
    """
    __declare_fit_model

    @author: Dominik Fischer
    last update: 2026-08-27

    Select the implementation to model the dependency of the measured currents onto the switching-frequency and provide
    parameter names for this model and their corresponding default/initial values.
    Some special fit models could be requested by providing a str key to `full_model`.
    By providing `extended`, the model which using the exponential approximation of the charging voltage at the
    capacitance is used.
    By providing `quad` a frequency behaviour like for a second-order low-pass filter is assumed for modelling of the
    frequency dependence.


    :param full_model: whether to apply the full capacitance model to the data. Instead of a boolean value also the
        values `extended` and `quad` are allowed.
    :type full_model: str | bool
    :return: tuple of the target dimension of the covariance matrix, the fitting model, its latex expression, its label,
        a mapping of the parameter names to their latex expressions and a mapping of the parameter names to their
        initial guesses.
    :rtype: tuple[int, Callable[..., Any], str, str, dict[str, str]]
    """
    if isinstance(full_model, str) and full_model == "extended":
        effective_model = extended_full_capacitance_model
        effective_label = FULL_MODEL_LABEL
        effective_expression = FULL_MODEL_EXPRESSION
        effective_parameter_dict = FULL_MODEL_PARAMETER_DICT
        effective_parameter_dict["tau"] = r"\tau"
        param_defaults = {'c': ANALYSIS_INITIAL_CAPACITANCE, 'r': ANALYSIS_INITIAL_RESISTANCE,
                          'i': ANALYSIS_INITIAL_LEAKAGE, 'u0': ANALYSIS_REFERENCE_VOLTAGE}
        param_defaults['tau'] = 1
        cov_array_limit = 4
    elif isinstance(full_model, str) and full_model == "quad":
        effective_model = enhanced_full_capacitance_model
        effective_label = FULL_MODEL_LABEL
        effective_expression = FULL_MODEL_EXPRESSION
        effective_parameter_dict = FULL_MODEL_PARAMETER_DICT
        param_defaults = {'c': ANALYSIS_INITIAL_CAPACITANCE, 'r': ANALYSIS_INITIAL_RESISTANCE,
                          'i': ANALYSIS_INITIAL_LEAKAGE, 'u0': ANALYSIS_REFERENCE_VOLTAGE}
        cov_array_limit = 3
    elif full_model:
        effective_model = full_capacitance_model
        effective_label = FULL_MODEL_LABEL
        effective_expression = FULL_MODEL_EXPRESSION
        effective_parameter_dict = FULL_MODEL_PARAMETER_DICT
        param_defaults = {'c': ANALYSIS_INITIAL_CAPACITANCE, 'r': ANALYSIS_INITIAL_RESISTANCE, 'i': ANALYSIS_INITIAL_LEAKAGE, 'u0': ANALYSIS_REFERENCE_VOLTAGE}
        cov_array_limit = 3
    else:
        effective_model = simple_capacitance_model
        effective_label = SIMPLE_MODEL_LABEL
        effective_expression = SIMPLE_MODEL_EXPRESSION
        effective_parameter_dict = {"c": r"C", "i": r"I", "u0": r"U_{0}", "freq": r"\nu"}
        param_defaults = {'c': ANALYSIS_INITIAL_CAPACITANCE, 'i': ANALYSIS_INITIAL_LEAKAGE, 'u0': ANALYSIS_REFERENCE_VOLTAGE}
        cov_array_limit = 2

    return cov_array_limit, effective_model, effective_expression, effective_label, effective_parameter_dict, param_defaults
