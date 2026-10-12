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

from pixcap65.analysis_util.constants import DOPING_RESULT_TYPE
from pixcap65.analysis_util.modelling.data_store import DopingArrayStore
from pixcap65.analysis_util.modelling.physics_modelling import EPS_SILICON, model_depletion
from pixcap65.analysis_util.utility import handle_kafe2_advanced_options, handle_minuit_advanced_options


def analyze_doping_profile(bias_voltages: np.ndarray,
                           bias_voltage_errors: np.ndarray,
                           doping_result_storage: DopingArrayStore, entry, n_a: float, v_bi: float,
                           pixel_area, pixel_cap_data: np.ndarray, pixel_cap_error_data: np.ndarray,
                           **kwargs) -> DOPING_RESULT_TYPE:
    """
    analyze:doping_profile

    @author Dominik Fischer
    @date 2026-05-07
    last update: 2026-08-12

    Extracts information about the doping profile from the C-V characterization provided.
    Also it is tried to model the doping profile in order to extract properties of the substrate.
    This utility function is not using locks for synchronization.

    :param bias_voltages: HV voltages used for the characterization.
    :param bias_voltage_errors: uncertainties/errors of the HV voltages used for the characterization.
    :param doping_result_storage: Data storage object for intermediate storage of doping results.
    :param entry: table row to store fit parameter results.
    :param n_a: guess for the doping concentration to perform the fit to the depletion width model
    :param v_bi: guess for the intrinsic bias voltage to perform the fit to the depletion width model
    :param pixel_area: area of the pixel diode to be investigated.
    :param pixel_cap_data: capacitance data from the C-V characterization.
    :param pixel_cap_error_data: uncertainties of the capacitance data from the C-V characterization.
    :param kwargs: further keyword arguments to be propagated to functions/implementations.
    :keyword use_kafe2: boolean, indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :keyword apply_contours: boolean, indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword plot: boolean, False, indicates whether to plot the data. An output PDF object could be submitted here instead of an explicitly created one. (Default: False)
    :type plot: bool
    :keyword fit_plot_pdf: PDF object to save the fit figures to.
    :keyword fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :type fit_description_text: str
    :return: tuple of doping_profile, depletion_width data, index of minimum doping concentration
    :rtype: tuple
    """
    from scipy import constants
    # Begin of the extraction part
    use_kafe2 = kwargs.pop("use_kafe2", False)
    apply_contours = kwargs.pop("apply_contours", False)
    plot = kwargs.pop("plot", False)
    output_pdf = kwargs.pop("fit_plot_pdf", None)
    fit_description_text = kwargs.pop("fit_description_text", "")

    # extract some further quantities from previous measurements and analysis.
    if np.all(np.isfinite(pixel_cap_error_data)):
        effective_cap_errors = pixel_cap_error_data
    else:
        effective_cap_errors = np.full_like(bias_voltages, 1)

    # compute the depletion width from the sensor properties.
    # What is the unit of this result
    # I assume this will result in wrong
    # provides the width in um
    depletion_width_data = np.asarray((constants.epsilon_0 * pixel_area * EPS_SILICON) / pixel_cap_data) * 1e-6
    depletion_width_error_data = (constants.epsilon_0 * pixel_area * effective_cap_errors * EPS_SILICON) / (np.array(
        pixel_cap_data) ** 2) * 1e-6

    if np.any(pixel_cap_data > 1e-5):
        with open("doping_handler.txt", 'a') as f:
            print("There was capacitance data much to large for F.", file=f)

    # fit the theoretical expected depletion width to determine some of the properties of the pixel diode
    parameter_guess = {
        "NAD": n_a,
        "V": v_bi,
        "dep": -10,
        "sat": np.max(depletion_width_data),
    }
    # this model fits may fail due to two indistinguishable parameters.
    if use_kafe2:
        from kafe2 import XYFit, XYContainer
        assert isinstance(bias_voltages, np.ndarray)
        xy_data = XYContainer(x_data=bias_voltages, y_data=depletion_width_data)
        xy_data.add_error(axis='y', err_val=depletion_width_error_data)
        if np.all(bias_voltage_errors[np.isfinite(bias_voltages)]):
            xy_data.add_error(axis='x', err_val=bias_voltage_errors)

        fitter = XYFit(xy_data, model_function=model_depletion, minimizer="iminuit")
        fitter.assign_parameter_latex_names(voltages="U_\\text{{bi}}", NA="N_\\text{{A}}", ND="N_\\text{{D}}",
                                            V="U_\\text{{th}}", )
        fitter.assign_model_function_latex_name("d_\\text{{depletion}}")
        fitter.assign_model_function_latex_expression(
            "\\sqrt{{\\frac{{2\\epsilon_0\\epsilon}}{{e}}\\cdot\\frac{{{NA}+{ND}}}{{{NA}\\cdot{ND}}}"
            "\\cdot ({V}+{voltages})}}")
        fitter.set_parameter_values(**parameter_guess)
        fitter.limit_parameter(name="V", lower=0.0, upper=10.0)
        fitter.limit_parameter(name="NAD", lower=0.0)
        fitter.limit_parameter(name="dep", upper=-0.5)
        fitter.limit_parameter(name="sat", lower=0.0)
        fitter.do_fit()
        assert fitter.did_fit
        depletion_fit_propagate_parameters = fitter.parameter_name_value_dict
        depletion_fit_params = fitter.parameter_values
        depletion_fit_errors = fitter.parameter_errors
        depletion_fit_cov = fitter.parameter_cov_mat
        if plot:
            assert isinstance(fitter, XYFit)
            handle_kafe2_advanced_options(fitter, apply_contours, "$U_\\text{{bi}}$ / \\unit{{\\volt}}",
                                          "$d$ / \\unit{{\\micro\\meter}}",
                                          "Depletion Width fit{}".format(fit_description_text),
                                          output_pdf,
                                          "Depletion Width contours{}".format(fit_description_text))
    else:
        from iminuit import Minuit
        # noinspection PyProtectedMember
        from iminuit.cost import LeastSquares, Model
        assert isinstance(model_depletion, Model)
        cost = LeastSquares(x=bias_voltages, y=depletion_width_data,
                            yerror=depletion_width_error_data, model=model_depletion)
        fitter = Minuit(cost, NAD=5e15, V=0.7, dep=-10, sat=np.max(depletion_width_data))
        fitter.limits["V"] = (0.0, 10)
        # noinspection PyTypeChecker
        fitter.limits["NAD", "sat"] = (0.0, None)
        fitter.limits["dep"] = (None, -0.5)
        fitter.migrad()
        fitter.hesse()
        depletion_fit_propagate_parameters = fitter.values.to_dict()
        depletion_fit_params = np.array(fitter.values)
        depletion_fit_errors = np.array(fitter.errors)
        depletion_fit_cov = np.asarray(fitter.covariance)
        if plot:
            assert isinstance(fitter, Minuit)
            handle_minuit_advanced_options(fitter, apply_contours, "$U_\\text{{bi}}$ / \\unit{{\\volt}}",
                                           "$d$ / \\unit{{\\micro\\meter}}",
                                           "Depletion Width fit{}".format(fit_description_text),
                                           output_pdf,
                                           "Depletion Width contours{}".format(fit_description_text))

    doping_result_storage.store_data("width", depletion_width_data)
    doping_result_storage.store_data("width_error", depletion_width_error_data)
    doping_result_storage.store_data("fit_parameters", depletion_fit_params)
    doping_result_storage.store_data("fit_parameters_error", depletion_fit_errors)
    doping_result_storage.store_data("fit_covariance", depletion_fit_cov)

    # calculate the effective doping profile of the sensor.
    n_eff = effective_doping(pixel_cap_data, -bias_voltages,
                             diode_area=pixel_area)
    doping_result_storage.store_data("doping", n_eff)
    resistivity = 1 / (constants.elementary_charge * n_eff * 1950)

    doping_result_storage.store_data("resistivity", resistivity)
    pos_min = int(np.argmin(n_eff))
    for key, value in depletion_fit_propagate_parameters.items():
        entry[key] = value
    entry.append()
    return n_eff, depletion_width_data, pos_min


def effective_doping(capacitance, bias_voltages, diode_area=None) -> np.ndarray:
    """
    effective doping

    @author Dominik Fischer
    last update: 2026-08-12

    Helper function to calculate the effective doping for every bias voltage/depletion depth.

    :param capacitance: measured (and corrected) capacitance of the CV characterization
    :param bias_voltages: bias voltages corresponding to the provided capacitance
    :param diode_area: area of the individual pixel.
    :return: effective doping concentration in cm^{-3}
    """
    from scipy import constants
    from findiff import Diff

    # make sure the data is provided as numpy arrays
    capacitance = np.asarray(capacitance)
    bias_voltages = np.asarray(bias_voltages)

    if diode_area is None:
        diode_area = 50 * 50  # measured in um^2
    temp_capacitance = np.reciprocal(capacitance ** 2)

    # noinspection PyTypeChecker
    try:
        # handle duplicates
        unique_voltage, cap_mask, voltage_counts = np.unique(bias_voltages, return_index=True, return_counts=True)
        duplicate_mask = voltage_counts > 1
        unique_temp_capacitance = temp_capacitance[cap_mask]
        derivative = np.full_like(bias_voltages, np.nan)
        d_du = Diff(0, unique_voltage, acc=4)
        derivative[cap_mask] = d_du(unique_temp_capacitance)
    except np.linalg.LinAlgError:
        print("bias_voltages", bias_voltages.shape)
        print(bias_voltages)
        print("capacitance", temp_capacitance.shape)
        print(temp_capacitance)
        print(capacitance)
        raise

    # need to manually broadcase the derivatives array if bias voltage duplicates were present.
    # only the duplicated values are still missing.
    if np.any(duplicate_mask):
        duplicate_values = unique_voltage[duplicate_mask]
        for dup_voltage in duplicate_values:
            indices = np.nonzero(bias_voltages == dup_voltage)
            actual_index = np.extract(unique_voltage == dup_voltage, cap_mask)[0]
            actual_doping = derivative[actual_index]
            derivative[indices] = actual_doping

    n_eff = 2 / (constants.elementary_charge * constants.epsilon_0 * EPS_SILICON * (diode_area ** 2) * np.asarray(derivative)) * 1e18
    return n_eff
