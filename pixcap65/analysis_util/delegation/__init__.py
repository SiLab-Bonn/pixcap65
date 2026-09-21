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
import tables as tb

from pixcap65.analysis_util.constants import SI_MOBILITY, BOUNDARY_TYPE, BIAS_VOLTAGE_ACCESS_IDX
from pixcap65.analysis_util.delegation.cv.doping import analyze_doping_profile
from pixcap65.analysis_util.delegation.depletion import depletion_delegation_impl
from pixcap65.analysis_util.modelling.data_store import DopingArrayStore, DepletionArrayStore
from pixcap65.analysis_util.multi_processing import get_manager_keywords
from pixcap65.utility.tables_util import group_get_file
from pixcap65.utility.utils_2 import create_carray
from .constants import MP_ACCELERATION_FLAG
from ..utility import GLOBAL_FILTERS, GENERAL_PIXCAP_SHAPE

try:
    # noinspection PyCompatibility
    from collections.abc import Callable, Sized, Iterable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Callable, Sized, Iterable
finally:
    from typing import Any, Optional, Tuple

from pixcap65.analysis_util.modelling.configuration_constants import ANALYSIS_INITIAL_CAPACITANCE, \
    ANALYSIS_INITIAL_RESISTANCE, ANALYSIS_INITIAL_LEAKAGE, ANALYSIS_REFERENCE_VOLTAGE
from pixcap65.analysis_util.modelling.physics_modelling import extended_full_capacitance_model, \
    enhanced_full_capacitance_model, full_capacitance_model, simple_capacitance_model, SILICON_V_BIAS, EPS_SILICON
from pixcap65.analysis_util.utility import FULL_MODEL_LABEL, FULL_MODEL_EXPRESSION, FULL_MODEL_PARAMETER_DICT, \
    SIMPLE_MODEL_LABEL, SIMPLE_MODEL_EXPRESSION, check_leaf_unit, HIST_CAP_UNIT, HIST_BIAS_MEAS_UNIT, \
    DepletionWidthData, fetch_bias_voltage

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


def _get_analyze(is_advanced: bool) -> Callable[..., None]:
    """
    _get_analyze

    @author: Dominik Fischer
    last update: 2026-08-13

    Internal helper function to fetch the analysis delegation depending on whether advanced analysis should be used or not.
    :param is_advanced: boolean indicating if the advanced analysis strategy should be used
        or not (may require additional keyword arguments)
    :return: callable to delegate the analysis to
    """
    if is_advanced:
        from pixcap65.analysis_util.delegation.enhanced_analysis import advanced_analysis_delegate
        perform_analysis = advanced_analysis_delegate
    else:
        from pixcap65.analysis_util.delegation.simplified_analysis import analyze_data_delegate
        perform_analysis = analyze_data_delegate
    return perform_analysis


def analyze_depletion_delegate(data_group: tb.Group, analysis_group: tb.Group,
                               first_boundaries: Optional[BOUNDARY_TYPE],
                               second_boundaries: Optional[BOUNDARY_TYPE], chip_group: Optional[tb.Group] = None,
                               apply_doping=False,
                               **kwargs):
    """
    analyse_depletion_delegate

    @author: Dominik Fischer
    last update: 2026-08-12

    Implementation of the investigation of the depletion behaviour of a pixel sensor.
    So first (this is the only non-optional functionality of this investigation) the C-V curve is used to obtain an
    estimator for the depletion voltage of each pixel on the investigated sensor module.
    Therefore, linear fits are applied to two distinc regions of the C-V curve.
    One in the large voltage limit (constant capacitances are expected as the depletion zone could not grow beyond the
    physical dimensions of the sensor) and one in the small.
    For the fits the capacitance is not used directly, but 1/C^2 as this quantity should be proportional to the
    (reversed) bias voltage.
    If no uncertainties for the capacitances are provided, a np.polyfit is used for this task, otherwise depending on
    the provided arguments either 'kafe2' or 'iminuit' is used.
    The depletion voltage is then estimated from the intersection of both straight line fits.
    For estimation of the uncertainty of the depletion voltage the full covariance matrix of the fits is used.

    Second, the depletion width and it's consequences are analysed if requested.
    This part will only be performed if it is requested by the 'apply_doping' parameter.
    This second analysis also requires the presence of the 'chip_group' parameter and within it the table/array
    'PhysicalDimensions' with the physical dimensions of the pixels on the module. Otherwise, it is not possible
    to determine the depletion widths from the measured capacitances, which will then be done by assuming a plate
    capacitator geometry.
    After this the effective doping profile is computed from the differential capacitance.


    This utility function is not using locks for synchroniztation, but forward them if provided.

    :param data_group: hierachy group of the opend hdf file containing the raw data (measurements).
    :param analysis_group: hierachy group of the opened hdf file to write the analysis results to.
    :param first_boundaries: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel.
    :param second_boundaries: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel.
    :param chip_group: HDF files hierarchy group with the data/specifications of the pixels on the current sensor.
    :param apply_doping: boolean, False, indicating whether to investigate the (effective) doping of the sensor.
    :keyword use_kafe2: boolean, False, indicates whether kafe2 is used for the fit.
    :keyword plot: boolean, False, indicates whether to plot the data. AN output PDF object could be submitted here
         instead of an explicitly created one.
    :keyword apply_contours: boolean, indicates whether to determine the contours and try to plot them.
    :keyword fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure)
    :keyword output_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure)
    :keyword verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    :keyword address: address of the socket of the multiprocessing.Manager object we want to connect to.
    :keyword authkey: authentication key necessary to connect to the socket. (It is recommended not to use this parameter as
        it is not pickable)
    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :keyword systematic_offset: enlargement in V for the fit range conditions applied for fitting. Needed to estiamte the
        systematic uncertainties by the fit range accurately. (default: 2)
    :keyword para_dist: standard deviation of the parasitic capacitance of the Pixcap chip on a single sensor.
    :keyword systematic_dispersion: spread of the dispersion of the parasitic capacitance between different Pixcap chip
        samples. This will induce a systematic effect on the accuracy of the capacitance's and the depletion voltage of
        the investigated sensor.
    :keyword systematic_offset: (default: 2)
    :type systematic_offset: float
    :keyword fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :keyword cv_fit_plot_pdf: analog to `fit_plot_pdf` to activate the plotting for c-v- and depletion fits independent from
        the plotting for capacitance estimation fits. If this keyword argument is present also the `plot` arguments will
        be set automatically.
    """
    from scipy.constants import epsilon_0
    from pixcap65.concurrency import get_context_manager

    # extract the additional parameters for advanced fitting procedures
    kwargs.setdefault('output_pdf', kwargs.get('fit_plot_pdf', None))
    manager_keywords = get_manager_keywords(**kwargs)
    if "authkey" in manager_keywords:
        del manager_keywords["authkey"]

    for key, value in kwargs.items():
        if 'lock' in key:
            print("Found a locking object", type(value), "in 'analyze_depletion_delegate'")
            try:
                print(value._serial, value._token)
            except:
                pass

    # verify and extract the raw data for further analysis
    cap_data = check_leaf_unit(analysis_group.UCHist, HIST_CAP_UNIT)
    cap_error_data = check_leaf_unit(analysis_group.UCErrHist, HIST_CAP_UNIT)
    voltage_data = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
    if len(voltage_data.shape) > 1:
        voltage_data = voltage_data[:, BIAS_VOLTAGE_ACCESS_IDX]

    # FIXME: provide here the correct manager arguments! (put it under investigation for now)
    with get_context_manager() as manager:
        if isinstance(first_boundaries, Iterable) and not isinstance(first_boundaries, Tuple):
            assert first_boundaries is not None
            assert second_boundaries is not None
            assert isinstance(first_boundaries, Sized)
            number_depletions = len(first_boundaries)

        else:
            number_depletions = 1

        if MP_ACCELERATION_FLAG:
            fit_result_storage = manager.DepletionArrayStorage(n_depletions=number_depletions)
        else:
            fit_result_storage = DepletionArrayStore(n_depletions=number_depletions)
        for k, (first_bound, second_bound) in enumerate(zip(np.atleast_2d(first_boundaries),
                                                                np.atleast_2d(second_boundaries))):
            first_lower, first_upper = first_bound
            second_lower, second_upper = second_bound
            fit_result_storage.set_depletion_region(k)
            depletion_delegation_impl(cap_data, cap_error_data, first_lower, first_upper, second_lower,
                                      second_upper,
                                      fit_result_storage, voltage_data, **kwargs)

        # save the depletion voltage data.
        file_h5 = group_get_file(analysis_group)
        create_carray(file_h5, where=analysis_group, name="DepletionHist",
                      title="Histogram of the depletion voltages", obj=fit_result_storage.depletion_voltage[:],
                      filters=GLOBAL_FILTERS, unit=HIST_BIAS_MEAS_UNIT)
        create_carray(file_h5, where=analysis_group, name="DepletionErrHist",
                      title="Histogram of the depletion voltage errors", obj=fit_result_storage.depletion_error[:],
                      filters=GLOBAL_FILTERS, unit=HIST_BIAS_MEAS_UNIT)
        create_carray(file_h5, where=analysis_group, name="DepFitParamHist",
                      title="Histogram of the depletion voltages fit parameters",
                      obj=fit_result_storage.fit_parameter_estimators[:],
                      filters=GLOBAL_FILTERS, unit="NONE")
        create_carray(file_h5, where=analysis_group, name="DepFitParamErrHist",
                      title="Histogram of the depletion voltages fit parameter errors",
                      obj=fit_result_storage.fit_parameter_errors[:],
                      filters=GLOBAL_FILTERS, unit="NONE")
        create_carray(file_h5, where=analysis_group, name="SystematicDispersionHist",
                      title="Histogram of the systamtic uncertainty by sensor dispersion",
                      obj=fit_result_storage.systematic_dispersion[:], filters=GLOBAL_FILTERS, unit=HIST_CAP_UNIT)
        create_carray(file_h5, where=analysis_group, name="SystematicGeneralHist",
                      title="Histogram of the systamtic uncertainty",
                      obj=fit_result_storage.systematic_errors[:], filters=GLOBAL_FILTERS, unit=HIST_CAP_UNIT)

        # we have to 2x2 matrices for each fit => overall there needs to be a 4x4 matrix per pixel!
        create_carray(file_h5, where=analysis_group, name="DepFitParamCovHist",
                      title="Matrix of the 2x2 covariance matrices for each pixel",
                      obj=fit_result_storage.fit_parameter_covariances[:], filters=GLOBAL_FILTERS, unit="None")

        # remove the fit storage object as it is no longer used anyway
        del fit_result_storage

    if (not apply_doping or chip_group is None or "PhysicalDimensions" not in chip_group or
            (chip_group.PhysicalDimensions.shape != (40, 40, 2) and chip_group.PhysicalDimensions.shape != (40, 41, 2))):
        return
    if chip_group.PhysicalDimensions.shape == (40, 40, 2):
        temp_data = np.full((40, 41, 2), fill_value=np.nan)
        temp_data[:, :40, :] = chip_group.PhysicalDimensions[:]
        temp_data[:, 40] = temp_data[:, 39, :]
        chip_group.PhysicalDimensions._f_remove()
        group_get_file(chip_group).create_array(where=chip_group, name="PhysicalDimensions", obj=temp_data)
        group_get_file(chip_group).flush()

    physical_dimensions_data = chip_group.PhysicalDimensions[:]
    pixel_areas = np.prod(physical_dimensions_data, axis=2)
    doping_shape = (40, 41, voltage_data.shape[0])
    doping_result_storage = DopingArrayStore(doping_shape, n_depletions=number_depletions)

    # some further definitions for the loop
    # this values will not be correct as I don't know the doping concentration the intrinsic bias voltage;
    # the intrinsic bias voltage could be estimated from a fit to the forward bias I-V characteristic.
    # noinspection PyPep8Naming
    NA = 1e16
    v_bi = SILICON_V_BIAS
    dep_table = file_h5.create_table(where=analysis_group, name="DepletionParamTable",
                                     description=DepletionWidthData,
                                     filters=GLOBAL_FILTERS)
    entry = dep_table.row

    _, bias_voltages, bias_voltage_errors = fetch_bias_voltage(data_group, BIAS_VOLTAGE_ACCESS_IDX)

    depletion_fit_parameters = analysis_group.DepFitParamHist[:]
    depletion_fit_parameter_errors = analysis_group.DepFitParamErrHist[:]

    for col, row in np.ndindex(GENERAL_PIXCAP_SHAPE):
        kwargs['fit_description_text'] = ' for Pixel ({col},{row})'.format(col=col, row=row)
        # make sure the provided data is useful for further investigation.
        # implies that the additonal row is not used at-all
        if row == 0 or not np.all(np.isfinite(physical_dimensions_data[col, row - 1])):
            continue
        pixel_cap_data = cap_data[col, row]
        pixel_cap_error_data = cap_error_data[col, row]
        pixel_area = pixel_areas[col, row - 1]  # should be calculated from the provded data in µm^2
        if not np.all(np.isfinite(pixel_cap_data)):
            continue
        doping_result_storage.set_pixel(row, col)
        entry["row"] = row
        entry["col"] = col

        n_eff, depletion_width_data, pos_min = analyze_doping_profile(bias_voltages, bias_voltage_errors,
                                                                      doping_result_storage, entry, NA / 2, v_bi,
                                                                      pixel_area, pixel_cap_data,
                                                                      pixel_cap_error_data, **kwargs)

        # could compute the resistivity from here!
        pixel_depletion_parameter = None
        try:
            pixel_depletion_parameter = np.atleast_2d(depletion_fit_parameters[col, row])
            pixel_depletion_parameter_errors = np.atleast_2d(depletion_fit_parameter_errors[col, row])
            doping_result_storage.store_data('res_mod', EPS_SILICON * epsilon_0 * pixel_area ** 2 / (2 * SI_MOBILITY) *
                                             pixel_depletion_parameter[:, 2] * 1e12)
            doping_result_storage.store_data('res_mod_err',
                                             EPS_SILICON * epsilon_0 * pixel_area ** 2 / (2 * SI_MOBILITY) *
                                             pixel_depletion_parameter_errors[:, 2] * 1e12)
        except:
            print(depletion_fit_parameters[col, row])
            print(pixel_depletion_parameter)
            raise

        if kwargs.get("verbose", False):
            msg = "The minimum concentration is {nm} and depth {dep_min} for pixel ({col}, {row})"
            print(msg.format(nm=n_eff[pos_min], dep_min=depletion_width_data[pos_min], col=col, row=row))

    # save the computed information about the depletion behaviour
    create_carray(file_h5, where=analysis_group, name="DepletionWidth",
                  title="Depletion width from the pixel capacitance",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.depletion_width_plate, unit="um")
    create_carray(file_h5, where=analysis_group, name="DepletionWidthErr",
                  title="Uncertainty of the depletion width from the pixel capacitance",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.depletion_width_plate_error, unit="um")

    create_carray(file_h5, where=analysis_group, name="DepletionParameters",
                  title="Depletion Parameters from fitting the depletion width",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.depletion_fit_parameter_table,
                  unit="cm^-3; cm^-3; V")
    create_carray(file_h5, where=analysis_group, name="DepletionErrors",
                  title="Depletion Parameters from fitting the depletion width",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.depletion_fit_parameter_error_table,
                  unit="cm^-3; cm^-3; V")
    create_carray(file_h5, where=analysis_group, name="DepletionCovariance",
                  title="Covariance matrices for Depletion Parameters from fitting the depletion width",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.depletion_fit_covariance_table,
                  unit="{{cm^-6, cm^-6, cm^-3 V},{cm^-6, cm^-6, cm^-3 V},{V cm^-3, V cm^-3, V^2}}")
    create_carray(file_h5, where=analysis_group, name="DepletionEffDoping",
                  title="Data for the effective doping from the cv-analysis",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.effective_doping_table, unit="cm^-3")
    create_carray(file_h5, where=analysis_group, name="DepletionResitivity",
                  title="Data for the specific resistivity from the cv-analysis", filters=GLOBAL_FILTERS,
                  obj=doping_result_storage.resistivity_table, unit="Ocm")
    create_carray(file_h5, where=analysis_group, name="ModDepletionResistivity",
                  title="Data for the specific resitivity from the slopes of the cv-depletion-analysis",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.second_resistivities, unit="Ocm")
    create_carray(file_h5, where=analysis_group, name="ModDepletionResistivityErr",
                  title="Data for the uncertainties of the specific resitivity from the slopes of the cv-depletion-analysis",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.second_resistivities_err, unit="Ocm")
    file_h5.flush()
