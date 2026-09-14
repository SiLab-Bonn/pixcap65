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
General implementation of plotting for capacitance measurements using the CBCM method.
"""

import numpy as np
import tables as tb
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Iterable, Any

from pixcap65.analysis_util import CURRENT_CONVERSION_FACTOR
from pixcap65.analysis_util.utility import check_leaf_unit, HIST_CURRENT_MEAS_UNIT, HIST_CAP_UNIT, \
    HIST_LEAK_CURRENT_UNIT, extract_parasitic_capacitance, CVDistributionData
from pixcap65.pixcap.pixcap_structure import HIST_PIX_CAP_LABEL, COUNTS_HIST_LABEL, CAPACITANCE_CONVERSION_FACTOR, \
    DEFAULT_BIN_NUMBER
from pixcap65.plotting_util import global_interactive_lock, FREQUENCY_LABEL, CURRENT_LABEL
from pixcap65.plotting_util.constants import GENERATE_THESIS_PLOTS
from pixcap65.plotting_util.utility import advanced_figure_provider, evaluate_pixel_mask, figure_provider


def plot_1d_distribution(data: np.ndarray, label: str, bias_code: int, table: Optional[tb.Table], pdf,
                         group: tb.Group, **kwargs):
    """
    plot_1d_distribution

    @author: Dominik Fischer
    @date 2026-08-11

    Utility function to plot/graphically present the histogram of the capacitance distribution of a sensor.
    If a distribution plot is requested by the corresponding keyword argument, the necessary fits will be
    explicitly performed/re-performed.

    :param data: capacitance data from which the histogram will be plotted.
    :param label: label/title of the plot when it gets saved.
    :param bias_code: integer, determining which kind of measurement will be evaluated, a list of possible values is
        shipped with inter-pixel analysis functions within the source code (commented lines).
    :param table: pytables.Table containing the fit parameters and other results from the analysis of
        the capacitance's distribution.
    :type table: pytables.Table or numpy.ndarray
    :param pdf:
    :param group:
    :param kwargs: further keyword arguments to be propagated to sub-calls.
    :keyword plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a
        new figure at the same time as matplotlib is not necessarily thread-safe.
    :keyword hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :keyword unit: unit of the capacities presented within the plot.
    :type unit: str
    :keyword distribution: boolean, indicating whether to investigate the capacitance distribution over
     the whole sensor. (default: False)
    :type distribution: bool
    :keyword test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :keyword mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :keyword mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :keyword mask_upper: float, threshold to mask all pixels above this value.
    :type mask_upper: float
    :keyword capacitance: histogram of the capacitance to use instead of those extracted from the provided
     hdf files group.
    :keyword set_parasitic: boolean, whether to set the parasitic capacitance for this data set.
    :type set_parasitic: bool
    :keyword no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
    :keyword convert: boolean, whether to convert the capacitance to fF, or not (default: True)
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :keyword apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword fit_plot_pdf: PDF object to save the fit figures to.
    """
    unit = kwargs.pop("unit", "\\farad")
    interactive_lock = kwargs.get('plotting_lock', global_interactive_lock)
    lockless_propagation = {key: value for key, value in kwargs.items() if "lock" not in key}
    with advanced_figure_provider(interactive_lock) as (fig, ax):
        hist_cap_hist = evaluate_pixel_mask(data, **lockless_propagation)
        ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                bins=kwargs.get("hist_bins", DEFAULT_BIN_NUMBER))
        ax.set_ylabel(COUNTS_HIST_LABEL)
        ax.set_xlabel(HIST_PIX_CAP_LABEL)
        if not GENERATE_THESIS_PLOTS:
            ax.set_title(__get_1d_hist_label(bias_code, label, table, unit=unit))
        ax.grid()
        pdf.savefig(fig, bbox_inches='tight')
    if kwargs.pop("distribution", False):
        from pixcap65.analysis_util.delegation.distribution import analyze_capacitance_distribution_delegate

        analyze_capacitance_distribution_delegate(group, pdf, set_parasitic=False, **lockless_propagation)


def plot_data_delegate(data_group: tb.Group, analysis_group: tb.Group, output_pdf: PdfPages, **kwargs):
    """
    plot_data_delegate

    @authors: Dominik Fischer
    @date: 2026-08-11


    Actual implementation to plot the results of the analysis of simple pixel capacitance scan (total capacitance).
    Besides the naming it is not just plotting but also a bit of analysis as the distribution of the capacitance is
    investigated in this function. The current-frequency dependency will be plotted for each scanned pixel with finite
    currents. Also, the capacitance distribution over the whole sensor and the frequency of capacitance values are
    plotted.

    :param data_group: hdf files hierarchy group containing the raw measurement data.
    :param analysis_group: hdf files hierarchy group containing the analysis results.
    :param output_pdf: PDF object to write the created figures to for long-term saving.
    :keyword plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create
        a new figure at the same time as matplotlib is not necessarily thread-safe.
    :keyword hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :keyword test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :keyword mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :keyword mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :keyword mask_upper: float, threshold to mask all pixels above this value.
    :type mask_upper: float
    :keyword distribution: boolean, indicating whether to investigate the capacitance distribution
     over the whole sensor. (default: False)
    :type distribution: bool
    :keyword unit: unit of the capacities presented within the plot.
    :type unit: str
    :keyword capacitance: histogram of the capacitance to use instead of those extracted from the provided
     hdf files group.
    :keyword set_parasitic: boolean, whether to set the parasitic capacitance for this data set.
    :type set_parasitic: bool
    :keyword no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
    :keyword convert: boolean, whether to convert the capacitance to fF, or not (default: True)
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :keyword apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword fit_plot_pdf: PDF object to save the fit figures to.
    """
    interactive_lock = kwargs.get('plotting_lock', global_interactive_lock)

    # Read pixel map and verify units
    current_hist = check_leaf_unit(data_group.HistCurr, HIST_CURRENT_MEAS_UNIT)
    current_err_hist = check_leaf_unit(data_group.HistCurrErr, HIST_CURRENT_MEAS_UNIT)
    cap_hist = check_leaf_unit(analysis_group.HistCap, HIST_CAP_UNIT)
    leak_hist = check_leaf_unit(analysis_group.HistLeak, HIST_LEAK_CURRENT_UNIT)

    # Read scan parameters
    scan_parameters = data_group.scan_params[:]

    # 2D Pixel Capacitance Hist
    plot_2d_capacitance(cap_hist, "Capacitance Distribution", output_pdf)

    # another heat map which do not consider masked or boundary caps
    assert isinstance(cap_hist, np.ndarray)
    masked_cap_hist = evaluate_pixel_mask(cap_hist.copy(), **kwargs)
    if kwargs.get("exclude_test_cap", False) or kwargs.get("test_cap_exclusion", False)\
            or kwargs.get("exclude_cap_test", False) or kwargs.get("exclude_cap_hist", False):
        plot_2d_capacitance(masked_cap_hist, "Masked Pixel Capacitance distribution", output_pdf)

    # 1D Pixel Capacitance Hist
    distribution_table = analysis_group.DistResultfF if "DistResultfF" in analysis_group else None
    actual_unit = "\\femto\\farad"
    if distribution_table is None:
        actual_unit = "\\farad"
        distribution_table = analysis_group.DistResult if "DistResult" in analysis_group else None

    plot_1d_distribution(cap_hist, "Total Cap Distribution", 20000, distribution_table, output_pdf,
                         analysis_group, unit=actual_unit, **kwargs)

    # Current vs. frequency
    verify_pixel_mask = "mask_pixel" in kwargs and isinstance(kwargs["mask_pixel"], Iterable) and len(
        kwargs["mask_pixel"]) > 0
    effective_pixel_mask = [(i[0], i[1]) for i in kwargs.get("mask_pixel", [])]
    for col, row in np.ndindex(current_hist.shape[:2]):
        if verify_pixel_mask and (col, row) in effective_pixel_mask:
            continue
        if np.isfinite(current_hist[col, row, 0]):
            with figure_provider(interactive_lock, nrows=2, sharex=True, height_ratios=[4, 1]) as (fig, ax, _):
                # with advanced_figure_provider(interactive_lock) as (fig, ax):
                actual_cap = cap_hist[col, row] * CAPACITANCE_CONVERSION_FACTOR

                # need to make sure that the fitted line will not exceed the finite data to much.
                nan_mask = np.isfinite(current_hist[col, row, :])
                frequencies = scan_parameters['frequency'][nan_mask]
                f = np.arange(0, frequencies.max() * 1.1, 0.1)

                plot_current_model(ax[0], col, row, analysis_group, actual_cap, leak_hist, f,
                                   parasitic_correction=extract_parasitic_capacitance(analysis_group.HistCap))
                plot_current_data(ax[0], col, row, scan_parameters, current_hist, current_err_hist, marker='x', ls='')
                f_res, cap_pred = get_model_prediction(col, row, analysis_group, actual_cap, leak_hist, frequencies,
                                            parasitic_correction=extract_parasitic_capacitance(analysis_group.HistCap))
                residues = current_hist[col, row, nan_mask] * CURRENT_CONVERSION_FACTOR - cap_pred
                ax[1].errorbar(f_res, residues, fmt='o')

                ax[0].set_ylabel(CURRENT_LABEL)
                ax[0].set_xlabel(FREQUENCY_LABEL)
                ax[1].set_ylabel(CURRENT_LABEL)
                ax[1].set_xlabel(FREQUENCY_LABEL)
                ax[0].legend()
                ax[0].grid()
                ax[1].grid()
                output_pdf.savefig(fig, bbox_inches='tight')


def __get_1d_hist_label(bias_code: int, label: str, table: Optional[tb.Table], unit="F"):
    from matplotlib import rcParams
    if table is not None:
        temp_rec_result = [row[:] for row in
                           table.where("""(bias == {})""".format(bias_code))]
        try:
            def handle_nan(parameter):
                if np.isnan(parameter):
                    return 0.0
                return parameter
            data_rec_result = np.rec.array(temp_rec_result,
                                           dtype=tb.dtype_from_descr(CVDistributionData(),))

            if rcParams['text.usetex']:
                uncorrected_label = "\n$C=\\qty{{{:.2f}+-{:.2f}+-{:.2f}+-{:.2f}}}{{{}}}$".format(
                                                                    data_rec_result.capacitance[0],
                                                                    data_rec_result.cap_std[0],
                                                                    handle_nan(data_rec_result.cap_systematic_error[0]),
                                                                    handle_nan(data_rec_result.cap_systematic_dispersion[0]), unit)
                corrected_label = "\n$C_\\text{{corr}}=\\qty{{{:.2f}+-{:.2f}+-{:.2f}+-{:.2f}}}{{{}}}$".format(
                                                                       data_rec_result.cap_corrected[0],
                                                                       data_rec_result.cap_corrected_err[0],
                                                                       handle_nan(data_rec_result.cap_systematic_error[0]),
                                                                       handle_nan(data_rec_result.cap_systematic_dispersion[0]), unit)
            else:
                uncorrected_label = "\nC={:.2f}+-{:.2f}+-{:.2f}+-{:.2f}{}".format(
                    data_rec_result.capacitance[0],
                    data_rec_result.cap_std[0],
                    handle_nan(data_rec_result.cap_systematic_error[0]),
                    handle_nan(data_rec_result.cap_systematic_dispersion[0]), unit)
                corrected_label = "\nC = {:.2f}+-{:.2f}+-{:.2f}+-{:.2f}{}".format(
                    data_rec_result.cap_corrected[0],
                    data_rec_result.cap_corrected_err[0],
                    handle_nan(data_rec_result.cap_systematic_error[0]),
                    handle_nan(data_rec_result.cap_systematic_dispersion[0]), unit)
        except IndexError:
            print("Generation of the label failed", bias_code)
            print(temp_rec_result)
            if len(temp_rec_result) == 0:
                print("it seems like the requested table row does not exist.")
            print(table._v_pathname)
            print(table[:])
            uncorrected_label = "(Failed to extract data)"
            corrected_label = ""
    else:
        uncorrected_label = ""
        corrected_label = ""
    return "{}{}{}".format(label, uncorrected_label, corrected_label)


def plot_2d_capacitance(data, label, pdf, **kwargs):
    """
    plot_2d_capacitance

    @author: Dominik Fischer
    @date: 2026-08-11

    Utility function to plot the distribution of the capacitance's over a sensor matrix.

    :param data: capacitance data to plot.
    :type data: numpy.ndarray
    :param label: label/title of the plot/figure when saving it.
    :type label: str
    :param pdf: pdf object to save the final figure to.
    :keyword plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create
     a new figure at the same time as matplotlib is not necessarily thread-safe.
    :keyword test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :keyword mask_pixel: array of tuple of pixel positions to be masked.
    :keyword mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :keyword mask_upper: float, threshold to mask all pixels above this value.
    :type mask_upper: float
    """
    interactive_lock = kwargs.get("plotting_lock", global_interactive_lock)
    assert isinstance(data, np.ndarray)
    masked_cap_hist = evaluate_pixel_mask(data.copy(), **kwargs)
    data = masked_cap_hist.copy()

    with advanced_figure_provider(interactive_lock) as (fig, ax):
        im = ax.imshow(data * CAPACITANCE_CONVERSION_FACTOR)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, label=HIST_PIX_CAP_LABEL)
        ax.set_ylabel(COLUMN_LABEL)
        ax.set_xlabel(ROW_LABEL)
        if not GENERATE_THESIS_PLOTS:
            ax.set_title(label)
        pdf.savefig(fig, bbox_inches='tight')


def plot_current_data(ax: Axes, col, row, scan_parameters, current_hist, current_err_hist, color=0.2, prefix="",
                      **plot_args):
    """
    plot_current_data

    Helper function to plot the measured current data (used for determining the pixel capacitance).
    It will decide on its own whether to use an error bar plot or not.
    The decision will depend on whether uncertainty data for the currents is present or not.

    Parameters
    ----------------------
    :param ax: axes object to plot the model to.
    :param col: column coordinate of the pixel for which to plot the model.
    :param row: row coordinate of the pixel for which to plot the model.
    :param scan_parameters: table of the scan parameters used for measuring and investigating the pixel's capacitance.
    :param current_hist: table/data set with the current measurements used.
    :param current_err_hist: table/data set of the uncertainties of the used current measurements.
    :param color: colour code for the data points to be plotted.
    :param prefix: prefix for the plots legends to distinguish them from other kinds of current-frequency plots.
    :param plot_args: further arguments directly provided to the fit object.
    """
    assert "parasitic_correction" not in plot_args
    plot_args.setdefault('marker', 'x')
    plot_args.setdefault('ls', '')
    assert "prefix" not in plot_args
    if 'fmt' in plot_args:
        plot_args['marker'] = plot_args.pop('fmt')
    if np.all(np.isfinite(current_err_hist[col, row, :])):
        marker_code = plot_args.pop('marker', 'o')
        plot_args['fmt'] = marker_code
        ax.errorbar(scan_parameters['frequency'], current_hist[col, row] * CURRENT_CONVERSION_FACTOR,
                    yerr=current_err_hist[col, row] * 1e9,
                    label=SIMPLE_PIXEL_LABEL.format(i_col=col, i_row=row, prefix=prefix),
                    color=cmap(color), capsize=2., **plot_args)
    else:
        ax.plot(scan_parameters['frequency'], current_hist[col, row] * CURRENT_CONVERSION_FACTOR,
                label=SIMPLE_PIXEL_LABEL.format(i_col=col, i_row=row, prefix=prefix),
                color=cmap(color), **plot_args)


def get_model_prediction(col, row, analysis_group: tb.Group, actual_cap: Any, total_leak_hist, f: np.ndarray = None,
                         resistor_name="HistRes", **plot_args):
    """
    get_model_prediction

    @author: Dominik Fischer
    @date: 2026-08-11


    Utility function to compute the model predictions for capacitances depending on the used frequency
    for the measurement.

    :param col: PixCap65 measurement column for which to predict.
    :param row: PixCap65 measurement row for which to predict.
    :param analysis_group: hdf files group of the analysis results used for fetching the models parameter in order to
        compute the prediction.
    :param actual_cap:
    :param total_leak_hist: matrix of the estimated leakage currents by fitting the corresponding model.
    :param f: array of the frequencies for which a prediction is to be computed.
    :param resistor_name: name of the dataset containing the on-resistance estimators if such a dataset is present at
        all.
    :param plot_args: further keywords arguments to be propagated to a plotting utility function (unused?)
    :keyword parasitic_correction: parasitic capacitance for which the input values are already corrected (this needs to
        be accounted for by the model as the currents are not corrected at all).
    :return:
    """
    parasitic_correction = plot_args.pop('parasitic_correction', 0.0)
    assert "parasitic_correction" not in plot_args

    # noinspection PyUnresolvedReferences
    if resistor_name in analysis_group and np.isfinite(analysis_group[resistor_name][col, row]):
        hist_resistance = analysis_group[resistor_name]
        from pixcap65.analysis_util.modelling.physics_modelling import full_capacitance_model
        assert isinstance(hist_resistance, tb.Array) or isinstance(hist_resistance, np.ndarray)
        y = full_capacitance_model(f,
                                   c=(actual_cap + parasitic_correction) * ADVANCED_CAPACITANCE_CONVERSION_FACTOR,
                                   r=hist_resistance[col, row],
                                   i=total_leak_hist[col, row] / CURRENT_CONVERSION_FACTOR, u0=1
                                   ) * CURRENT_CONVERSION_FACTOR
        return f, y

    else:
        return f, (actual_cap + parasitic_correction) * f + total_leak_hist[col, row]


def plot_current_model(ax: Axes, col, row, analysis_group: tb.Group, actual_cap: Any, total_leak_hist, f: np.ndarray,
                       resistor_name="HistRes", prefix="", color=0.6, **plot_args):
    """
        plot_current_model

        Helper function to plot the current model from the specified parameters.
        It will decide on execution whether the complete advanced model has to be used depending on
        whether on-resistance estimators are present.

        :param ax: axes object to plot the model to.
        :param col: column coordinate of the pixel for which to plot the model.
        :param row: row coordinate of the pixel for which to plot the model.
        :param analysis_group: Group containing the analysis data.
        :param actual_cap: Actual capacitance (correction may already be applied).
        :param total_leak_hist: table with estimators for the detector/dut leakage current form CBCM method.
        :param f: Frequency (MHz) to use for the model functions plot.
        :param resistor_name: Name of resistance estimator to use for the model functions plot.
        :param color: colour code to use for plotted model.
        :param prefix: prefix for title and legend to use.
        :keyword parasitic_correction: if present, it should be the capacitance subtracted during capacitance correction
            procedure.
        """
    # fetch the model
    f, y = get_model_prediction(col, row, analysis_group, actual_cap, total_leak_hist, f=f, resistor_name=resistor_name,
                                **plot_args)
    if "parasitic_correction" in plot_args:
        del plot_args["parasitic_correction"]
    plot_args.setdefault('marker', '')
    plot_args.setdefault('ls', '--')
    assert 'prefix' not in plot_args

    # noinspection PyUnresolvedReferences
    if np.isnan(actual_cap):
        ax.plot(f, y, color=cmap(color), label="NAN Capacitance C d", **plot_args)
    else:
        ax.plot(f, y, color=cmap(color),
                label=SIMPLE_CAP_LABEL_PERCENT_FORMAT % (prefix, actual_cap), **plot_args)


def plot_compare_delegate(first_group: tb.Group, second_group: tb.Group, output_pdf: PdfPages):
    """
    plot_compare_delegate

    Actual implementation of the comparison of two pixel capacitance measurements.

    :param first_group: group containing the capacitance data of the first measurement to compare.
    :param second_group: group containing the capacitance data of the second measurement to compare.
    :param output_pdf: PDF file to write the figures to for long-term storage
    """
    # extract the data
    first_cap_hist = first_group.HistCurr[:]
    second_cap_hist = second_group.HistCurr[:]
    first_nan_mask = np.isfinite(first_cap_hist)
    second_nan_mask = np.isfinite(second_cap_hist)
    full_mask = np.logical_and(first_nan_mask, second_nan_mask)
    diff_cap_hist = np.where(full_mask, first_cap_hist - second_cap_hist, np.nan)

    # create the figure
    with advanced_figure_provider(global_interactive_lock) as (fig, ax):
        im = ax.imshow(diff_cap_hist * CAPACITANCE_CONVERSION_FACTOR, )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, label=HIST_PIX_CAP_LABEL)
        ax.set_ylabel(COLUMN_LABEL)
        ax.set_xlabel(ROW_LABEL)
        output_pdf.savefig(fig, bbox_inches='tight')


cmap = plt.get_cmap('viridis')
ROW_LABEL = 'Row'
COLUMN_LABEL = 'Column'
SIMPLE_CAP_LABEL_PERCENT_FORMAT = '%sFit to data:\n$C_d = \\qty{%.1f}{\\femto\\farad}$'
SIMPLE_PIXEL_LABEL = '{prefix}Pixel({i_col},{i_row})'
ADVANCED_CAPACITANCE_CONVERSION_FACTOR = 1.0e-9
