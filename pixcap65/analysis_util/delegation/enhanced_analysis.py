"""
Implements the enhanced analysis strategy by using non-linerar fitting algorithms to also estimate the parameter
uncertainties correctly.
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
import tables as tb
from typing import Union, Callable, Any

from pixcap65.analysis_util import TABLES_ARRAY_TYPE, TABLES_TABLE_TYPE, GENERAL_PIXCAP_SHAPE, transform_covariance, \
    GLOBAL_FILTERS, FARAD_CONVERSION_FACTOR, CURRENT_CONVERSION_FACTOR
from pixcap65.analysis_util.delegation import __declare_fit_model, ANALYSIS_FIT_X_LABEL, ANALYSIS_FIT_Y_LABEL, \
    ANALYSIS_FIT_PLOT_LEGEND, ANALYSIS_FIT_CONTOUR_LEGEND
from pixcap65.analysis_util.utility import HandleFitterStubClass, HandleFitterGeneral, HIST_CAP_UNIT, \
    HIST_LEAK_CURRENT_UNIT
from pixcap65.utility.tables_util import get_node_pathname
from pixcap65.utility.utils_2 import create_carray


def advanced_analysis_delegate(file: tb.File, group: tb.Group, current_hist: TABLES_ARRAY_TYPE,
                               scan_parameters: TABLES_TABLE_TYPE, **kwargs):
    """
    advanced_analysis_delegate

    @author: Dominik Fischer
    last update: 2026-08-12

    Implementation of the advanced analysis strategy for the capacitance measurement of a pixel sensor.
    For determination of the capacitance values non-linear fit algorithms are used.
    Depending on the choice of parameters either kafe2 or iminuit is used for least squares minimization.
    If requested the fit results will also be plotted to verify the convergence of the fit.
    If this mode is activated, additional keyword arguments must be present to specify the output of the plots.

    :param file: h5 file object containing the data to be analysed.
    :param group: hierarchy group of the opened hdf file to write the analysis results to.
    :param current_hist: 2D-Array for the current data to fit the model to.
    :param scan_parameters: table of the scan parameters used for each measurement point within the frequency and/or
        voltage scan.
    :key full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used.
    :key current_error_hist: 2D-Array for the errors of the current data. This keyword argument must be present
        for the advanced analysis strategy.
    :key use_kafe2: boolean, indicates whether kafe2 is used for the fit. (default: False)
    :key plot: boolean, indicates whether to plot the data. An output PDF object could be submitted here instead of an explicitly created one. (Default: False)
    :key apply_contour: boolean, indicates whether to determine the contours and try to plot them. (default: False)
    :key fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided)
    :key output_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided)
    :key is_inter_b: indicates whether this an analysis for the b-channel of the inter-pixel capacitances measurements, which would change the assumed voltage.
    """
    plot = kwargs.pop("plot", False)
    plot_fit = HandleFitterStubClass()
    if plot:
        from matplotlib.backends.backend_pdf import PdfPages

        # it must be possible to load the analysis as well when not plotting engine is installed.
        plot_fit = HandleFitterGeneral()
        output_format = "{}_{}_fit_results.pdf"
        output_pdf_name = output_format.format(file.filename[:-3],
                                             get_node_pathname(group).replace('/', '----'))
        fit_plot_pdf = kwargs.pop("fit_plot_pdf", None)
        if fit_plot_pdf is not None and isinstance(fit_plot_pdf, PdfPages):
            kwargs['output_pdf'] = fit_plot_pdf
        elif isinstance(plot, PdfPages):
            kwargs['output_pdf'] = plot
        else:
            with PdfPages(output_pdf_name) as pdf:
                assert isinstance(current_hist, np.ndarray)
                assert isinstance(scan_parameters, np.ndarray)
                _perform_advanced_fit(file, group, current_hist, plot_fit, scan_parameters, output_pdf=pdf, **kwargs)
                return

    # need 'to do' it this way for compatibility with python 2.7
    assert isinstance(plot_fit, HandleFitterStubClass)
    assert isinstance(current_hist, np.ndarray)
    assert isinstance(scan_parameters, np.ndarray)
    _perform_advanced_fit(file, group, current_hist, plot_fit, scan_parameters, **kwargs)


def _perform_advanced_fit(file: tb.File, group: tb.Group, current_hist: np.ndarray,
                          plot_fit: HandleFitterStubClass, scan_parameters: Union[tb.Table, np.ndarray],
                          **kwargs):
    """
    _perform_advanced_fit

    @author: Dominik Fischer
    last update: 2026-08-12

    Performs the actual fits (delegates it) for every pixel which has reasonable current measurements and saves
    the results back.

    :param file: .h5 file object containing the data to be analysed.
    :param group: hdf files' group containing the measurement data.
    :param current_hist: histogram/array-like of the measured currents.
    :param plot_fit: object handling the control plots for the fit.
    :param scan_parameters: mapping of scan-parameters used when measuring the capacitances'.
    :key full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used.
    :key current_error_hist: 2D-Array for the errors of the current data. This keyword argument must be present
        for the advanced analysis strategy.
        :key use_kafe2: boolean, indicates whether kafe2 is used for the fit. (default: False)
    :key plot: boolean, indicates whether to plot the data. An output PDF object could be submitted here instead of
        an explicitly created one. (Default: False)
    :key apply_contour: boolean, indicates whether to determine the contours and try to plot them. (default: False)
    :key output_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided)
    :key is_inter_b: indicates whether this an analysis for the b-channel of the inter-pixel capacitances measurements,
        which would change the assumed voltage.
    """
    # extract the additional keyword arguments
    full_model = kwargs.pop("full_model", True)
    assert "current_error_hist" in kwargs
    current_error_hist = kwargs['current_error_hist']
    assert isinstance(current_error_hist, TABLES_ARRAY_TYPE)
    use_kafe2 = kwargs.pop("use_kafe2", False)
    apply_contour = kwargs.pop("apply_contour", False)
    output_pdf = kwargs.pop("output_pdf", None)

    # create array_like objects to save the analysis results temporarily.
    cap_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    cap_error_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    leak_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    leak_error_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    resistor_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    resistor_error_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)

    # prepare the fit model
    (cov_array_limit, effective_model, effective_expression, effective_label,
     effective_parameter_dict, initial_guess) = __declare_fit_model(full_model)

    fit_cov = np.full(shape=tuple([*GENERAL_PIXCAP_SHAPE, cov_array_limit + 1, cov_array_limit + 1]), fill_value=np.nan)

    # Fit pixel data in order to extract capacitance for each pixel
    for ii, jj in np.ndindex(current_hist.shape[:2]):
        if np.count_nonzero(np.isfinite(current_hist[ii, jj, :])) < 3:
            continue
        mask = np.isfinite(current_hist[ii, jj, :])
        frequencies = scan_parameters['frequency'][mask]
        currents = current_hist[ii, jj, mask]
        current_errors = current_error_hist[ii, jj, mask]

        # make a first capacitance approximation using numpy; this needs to be done for each pixel individually.
        pre_result = np.polyfit(frequencies, currents, 1)
        initial_guess.update(c=pre_result[0], i=pre_result[1])

        cap, cap_error, fit_cov_temp, fitter, leakage, leakage_error, resistor, resistor_error = __perform_pixel_fit(
            currents, current_errors, frequencies, effective_expression, effective_label, effective_model,
            effective_parameter_dict, full_model, initial_guess, use_kafe2, kwargs.get('is_inter_b', False))

        plot_fit(fitter, apply_contour, ANALYSIS_FIT_X_LABEL, ANALYSIS_FIT_Y_LABEL,
                 ANALYSIS_FIT_PLOT_LEGEND.format(col=ii, row=jj), output_pdf,
                 ANALYSIS_FIT_CONTOUR_LEGEND.format(col=ii, row=jj))

        fit_cov[ii, jj, :cov_array_limit, :cov_array_limit] = transform_covariance(fit_cov_temp, )

        # temporarily save the results to memory
        cap_hist[ii, jj] = cap
        cap_error_hist[ii, jj] = cap_error
        leak_hist[ii, jj] = leakage
        leak_error_hist[ii, jj] = leakage_error
        resistor_hist[ii, jj] = resistor
        resistor_error_hist[ii, jj] = resistor_error

    # Store capacitance values and specify the used units as an attribute.
    create_carray(file, group, name=kwargs.get("cap_name", "HistCap"),
                  title=kwargs.get("cap_title", "Capacitance Histogram"), obj=cap_hist, filters=GLOBAL_FILTERS,
                  unit=HIST_CAP_UNIT)
    create_carray(file, group, name=kwargs.get("cap_err_name", "HistCapErr"),
                  title=kwargs.get("cap_err_title", "Capacitance Error Histogram"), obj=cap_error_hist,
                  filters=GLOBAL_FILTERS, unit=HIST_CAP_UNIT)

    create_carray(file, group, name=kwargs.get("leak_name", "HistLeak"),
                  title=kwargs.get("leak_title", "Leakage Current Histogram"), obj=leak_hist, filters=GLOBAL_FILTERS,
                  unit=HIST_LEAK_CURRENT_UNIT)
    create_carray(file, group, name=kwargs.get("leak_error_name", "HistLeakErr"),
                  title=kwargs.get("leak_error_title", "Leakage Current Error Histogram"), obj=leak_error_hist,
                  filters=GLOBAL_FILTERS, unit=HIST_LEAK_CURRENT_UNIT)
    create_carray(file, group, name=kwargs.get("resistor_name", "HistRes"),
                  title=kwargs.get("resistor_title", "On-Resistance Histogram"), obj=resistor_hist,
                  filters=GLOBAL_FILTERS, unit="O")
    create_carray(file, group, name=kwargs.get("resistor_error_name", "HistResErr"),
                  title=kwargs.get("resistor_error_title", "On-Resistance Error Histogram"), obj=resistor_error_hist,
                  filters=GLOBAL_FILTERS, unit="O")
    create_carray(file, group, name=kwargs.get("cov_name", "HistFitCov"),
                  title=kwargs.get("cov_title", 'Fit Covariance Matrix'), obj=fit_cov, filters=GLOBAL_FILTERS,
                  unit="{{F^2, F O, F nA, F V},{O F, O^2, O nA, O V},{nA F, nA O, nA^2, nA V}, {V F, V O, V nA, V^2}")


def __perform_pixel_fit(currents: np.ndarray, current_errors: np.ndarray, frequencies: np.ndarray,
                        effective_expression: str,
                        effective_label: str, effective_model: Callable[..., Any],
                        effective_parameter_dict: dict[str, str], full_model: bool, initial_guess: dict[str, float],
                        use_kafe2, inter_b=False) -> tuple[float, float, np.ndarray, Any, float, float, float, float]:
    """
    __perform_pixel_fit

    @author: Dominik Fischer
    last update: 2026-08-27

    Perform the actual model fit to the measured currents for a single pixel.
    To perform the actual fit either the `kafe2` framework or the `iminuit` framework is used.
    When using the `kafe2` framework also the uncertainties of the settled frequencies by the stability of the MIOs PLL
    are accounted for.

    Besides the parameter estimators and their uncertainties, the full covariance matrix of the parameters is extracted.

    :param currents: measured currents to perform the fit.
    :param current_errors: uncertainties of the measured currents to perform the fit.
    :param frequencies: frequencies corresponding to the measured currents to perform the fit.
    :param effective_expression: latex expression of the model function.
    :param effective_label: latex expression to label the model function.
    :param effective_model: model function to be used for the fit.
    :param effective_parameter_dict: mapping of all the parameter names to their latex expression
        (soley for visualization).
    :param full_model: Whether the full model is used. This information is necessary to decide whether to extract
        the on-resistance estimator from the fit.
    :type full_model: bool
    :param initial_guess: mapping of the models parameter names to their initial guess, which is necessary to start the
        iteration which fits the model.
    :param use_kafe2: whether to use Kafe2 framework or not.
    :type use_kafe2: bool
    :param inter_b: whether this is analysis for the B channel of the inter-pixel analysis. In this case the voltage
        over the capacitance is -2V instead of the usual +1V.
    :return: tuple of determined capacitance, its uncertainty, the covariance matrix, the fitting object itself,
        the leakage current estimator, its uncertainty, the on-resistance estimator and its uncertainty.
    """
    resistor = np.nan
    resistor_error = np.nan
    if use_kafe2:
        from kafe2 import XYContainer, XYFit
        xy_data = XYContainer(frequencies, currents)
        # errors?
        if np.all(np.isfinite(current_errors)):
            xy_data.add_error('y', err_val=current_errors)

        freq_errors = frequencies * 150e-6
        xy_data.add_error('x', err_val=freq_errors)
        fitter = XYFit(xy_data, model_function=effective_model)
        fitter.assign_model_function_latex_name(effective_label)
        fitter.assign_model_function_latex_expression(effective_expression)
        try:
            fitter.assign_parameter_latex_names(**effective_parameter_dict)
        except:
            print(effective_parameter_dict)
            raise
        fitter.set_parameter_values(**initial_guess)
        fitter.fix_parameter('u0', -2 if inter_b else 1)
        fitter.do_fit()

        # extract the fit parameters
        assert fitter.did_fit
        cap = fitter.parameter_values[0] * FARAD_CONVERSION_FACTOR  # convert to F
        cap_error = fitter.parameter_errors[0] * FARAD_CONVERSION_FACTOR
        leakage = fitter.parameter_values[2] * CURRENT_CONVERSION_FACTOR  # convert to nA
        leakage_error = fitter.parameter_errors[2] * CURRENT_CONVERSION_FACTOR  # convert to nA
        if full_model:
            # otherwise the requested information may not be present
            resistor = fitter.parameter_values[1]
            resistor_error = fitter.parameter_errors[1]
        fit_cov_temp = fitter.parameter_cov_mat
    else:
        from iminuit import Minuit
        from iminuit.cost import LeastSquares
        if np.all(np.isfinite(current_errors)):
            errors = current_errors
        else:
            errors = np.full_like(current_errors, fill_value=1)
        # noinspection PyTypeChecker
        cost = LeastSquares(x=frequencies, y=currents,
                            yerror=errors, model=effective_model)
        fitter = Minuit(cost, **initial_guess)
        fitter.fixto('u0', -2 if inter_b else 1)
        fitter.migrad()
        fitter.hesse()

        # extract the fit parameters
        cap = fitter.values['c'] * FARAD_CONVERSION_FACTOR  # convert to F
        cap_error = fitter.errors['c'] * FARAD_CONVERSION_FACTOR
        leakage = fitter.values['i'] * CURRENT_CONVERSION_FACTOR  # convert to nA
        leakage_error = fitter.errors['i'] * CURRENT_CONVERSION_FACTOR  # convert to nA
        if full_model:
            # otherwise the requested information may not be present.
            resistor = fitter.values['r']
            resistor_error = fitter.errors['r']
        fit_cov_temp = fitter.covariance

    assert isinstance(fit_cov_temp, np.ndarray)
    np.array([
        ('C', cap),
        ('Cerr', cap_error),
        ('I', leakage),
        ('Ierr', leakage_error),
        ('R', resistor),
        ('Rerr', resistor_error),
        ('fit', fitter)
    ])
    return cap, cap_error, fit_cov_temp, fitter, leakage, leakage_error, resistor, resistor_error
