"""
Plotting of Pixcap65 data.
"""
import logging
import os.path
from typing import Any, Optional
from warnings import warn

from matplotlib.axes import Axes

from pixcap65.analysis_util.physics_modelling import model_depletion

NUMBER_DEPLETION_PLOT_POINTS = 1000

try:
    # noinspection PyCompatibility
    from collections.abc import Iterable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Iterable

import numpy as np
import tables as tb
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pixcap65.analysis_util.utility import check_leaf_unit, GENERAL_PIXCAP_SHAPE, HIST_BIAS_MEAS_UNIT, \
    HIST_LEAK_CURRENT_UNIT, \
    HIST_CAP_UNIT, HIST_CURRENT_MEAS_UNIT, extract_parasitic_capacitance, CURRENT_CONVERSION_FACTOR, get_base_group, \
    get_analysis_group, TABLES_LEAF_COMPAT_TYPE
from pixcap65.utility.utils_2 import GroupType

HISTOGRAM_SHAPE_FORMAT = "The histograms shape is {}"
ROW_LABEL = 'Row'
COLUMN_LABEL = 'Column'
HIST_PIX_CAP_LABEL = 'Pixel Capacitance / fF'
COUNTS_HIST_LABEL = 'Counts / \\#'
SIMPLE_CAP_LABEL_PERCENT_FORMAT = '%sFit to data:\n$C_d = %.1f\\,$fF'
FREQUENCY_LABEL = 'Frequency / MHz'
CURRENT_LABEL = 'Current / nA'
SIMPLE_PIXEL_LABEL = '{prefix}Pixel({i_col},{i_row})'
CAPACITANCE_CONVERSION_FACTOR = 1e15
ADVANCED_CAPACITANCE_CONVERSION_FACTOR = 1.0e-9
cmap = plt.get_cmap('viridis')
BIAS_CURVE_Y_LABEL = "I in nA"
BIAS_CURVE_X_LABEL = "U in V"
DEFAULT_BIN_NUMBER = 50
DEFAULT_TEST_CAP_EXCLUSION = True
logger = logging.getLogger(__name__)


def evaluate_pixel_mask(hist, perform_filter=False, **kwargs):
    result_hist = hist.copy()
    kargs = kwargs.copy()
    test_cap_exclusion = kargs.pop("test_cap_exclusion", DEFAULT_TEST_CAP_EXCLUSION)
    pixel_mask = np.asarray(kargs.pop("mask_pixel", []))
    mask_lower = kargs.pop("mask_lower", None)
    mask_upper = kargs.pop("mask_upper", None)
    if pixel_mask.shape[0] > 0:
        col_mask = pixel_mask[:, 0]
        row_mask = pixel_mask[:, 1]
        masks = (col_mask, row_mask)
        result_hist[masks] = np.nan
    if test_cap_exclusion:
        result_hist[:, 0] = np.nan
    if mask_lower:
        assert isinstance(mask_lower, float)
        result_hist[result_hist < mask_lower] = np.nan
    if mask_upper:
        assert isinstance(mask_upper, float)
        result_hist[result_hist > mask_upper] = np.nan

    if perform_filter:
        return result_hist[np.isfinite(result_hist)]
    return result_hist


# TODO: Implement the extraction of particular figures for certain pixels while performing the plotting for all pixels.
def get_pdf_name(base_path, interpreted_data, suffix: str, use_group: bool) -> str:
    """
    get_pdf_name

    Helper function to generate a PDF name for a given hdf file to export the created figures to.

    :param base_path: basic group descriptor for the analysis and measurements within the hdf file.
    :param interpreted_data: path to the hdf file used for measurement and analysis.
    :param suffix: actual suffix to use to identify the PDF file.
    :param use_group: boolean, False, indicates whether to append the group name to the PDF name.
    :return: name of the PDF file to use
    """
    if use_group and base_path is not None:
        _, group_component = os.path.split(base_path)
        pdf_name = "{file}_{s}_{group}.pdf".format(s=suffix, file=interpreted_data[:-3], group=group_component)
    else:
        pdf_name = "{file}_{s}.pdf".format(s=suffix, file=interpreted_data[:-3])
    return pdf_name


def plot_data(interpreted_data, base_path=None, suffix="general_data", use_group=False, **kwargs):
    """
    plot_data

    Helper function to graphical present/plot the analysis results of a simple pixel capacitance scan.
    The plotting is only performed for the pixels which contribute a usable capacitance measurement.
    In Addition to the fits for estimating the capacitance also the capacitance distribution and frequency is plotted.
    The name of the resulting PDF is derived from the file name with the measurement data.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    :key use_corrected: boolean, False, indicating whether to use the corrected capacitance for plotting.
    :key exclude_test_cap: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key extract_pixel: iterable of pixel positions on the grid to extract the figures from.
    :key distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor.
    """
    if kwargs.get("use_corrected", False):
        suffix = "{}_corrected".format(suffix)
    # determine the pdf file
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    data_file_modifier = 'a' if kwargs.get("distribution", False) else 'r'
    with PdfPages(pdf_name) as output_pdf:
        assert data_file_modifier != "w"
        with tb.open_file(interpreted_data, mode=data_file_modifier) as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            plot_data_delegate(base_group.total_cap.measurements, get_analysis_group(base_group.total_cap, **kwargs),
                               output_pdf, **kwargs)


def plot_inter_pix_data(interpreted_data, base_path=None, suffix="general_inter_pix_data", use_group=False,
                        total_path=None, total_data=None, **kwargs):
    """
    plot_data

    Helper function to graphical present/plot the analysis results of an inter-pixel capacitance scan.
    The plotting is only performed for the pixels which contribute a usable capacitance measurement.
    In Addition to the fits for estimating the capacitance also the capacitance distribution and frequency is plotted.
    The name of the resulting PDF is derived from the file name with the measurement data.

    Besides the naming it is not just plotting but also a bit of analysis as the distribution of the capacitance
    over the pixel and in general for all three currents is investigated, as well.
    The current-frequency dependency will plot for each scanned pixel with finite currents.
    Also, the capacitance distribution over the whole sensor and the frequency of capacitance values are plotted for all
    three current measurements. Besides the naming of the plots the capacitance are not directly the inter-pixel or
    total-pixel capacitance values.

    The modelling process is automatically corrected to use the bare capacitance if corrected capacitance values are
    supplied. If the correction is not applied by the functions from the analysis module then this may not work.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    :key exclude_test_cap: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key extract_pixel: iterable of pixel positions on the grid to extract the figures from.
    """
    # determine the pdf file
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    with PdfPages(pdf_name) as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            if total_data is None or not os.path.exists(total_data):
                plot_inter_pix_data_delegate(base_group.inter_cap.measurements,
                                             get_analysis_group(base_group.inter_cap, **kwargs), output_pdf,
                                             **kwargs)
            elif os.path.exists(total_data):
                with tb.open_file(total_data, mode='r') as total_file_h5:
                    total_base_group = get_base_group(total_path, total_file_h5)
                    plot_inter_pix_data_delegate(base_group.inter_cap.measurements,
                                                 get_analysis_group(base_group.inter_cap, **kwargs), output_pdf,
                                                 total_base_group.total_cap.analysis,
                                                 **kwargs)


def plot_bias_data(interpreted_data, base_path=None, suffix="bias_curve", use_group=False):
    """
    plot_bias_data

    Plot the data acquired for the pixel-diodes I-V characterization.

    :param interpreted_data: path to the hdf file which holds the raw data.
    :param base_path: path within the files hierarchy for the base group.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    """
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    with PdfPages(pdf_name) as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            plot_bias_delegate(base_group.biasing.measurements, output_pdf)


def plot_cv_data(interpreted_data, base_path=None, first_upper=None, first_lower=None, second_upper=None,
                 second_lower=None, suffix="C_V_characteristic", use_group=False, **kwargs):
    """
    plot_cv_data

    Plot the results of the C-V characterization of the scanned pixels.
    To achieve this we need the different c-v-data.
    Then the C-V curve is plotted for every pixel.
    If requested also fits to the boundary regions of the c-v-curve are performed to determine the depletion voltage of the pixel.
    To do so, two fit ranges for the two boundaries with physically distinct behaviour needs to be supplied.


    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param first_upper: upper limit of the first fit range
    :param first_lower: lower limit of the first fit range
    :param second_upper: upper limit of the second fit range
    :param second_lower: lower limit of the second fit range
    :param suffix:  additional suffix to use for naming the PDF containing the plots.
    :param use_group:   boolean, whether to append the group name of the measurements to the PDF name.
    :key verbose: boolean, indicating whether to use verbose output for depletion voltages
    :key distribution: boolean, indicating whether also the capacitance distribution of the whole sensor should be investigated.
    """
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    with PdfPages(pdf_name) as output_pdf:
        with tb.open_file(interpreted_data, mode='a') as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            plot_cv_data_delegate(base_group.biasing.measurements,
                                  get_analysis_group(base_group.biasing, **kwargs), output_pdf,
                                  first_upper, first_lower, second_upper, second_lower,
                                  apply_doping=kwargs.get('apply_doping', False))


def plot_combined_data(interpreted_data, base_path=None, first_upper=None, first_lower=None, second_upper=None,
                       second_lower=None, suffix="combined_bias_cv_curve", use_group=False, **kwargs):
    """
    plot_combined_data

    Plot the data acquired for the pixel-diodes I-V characterization and the C-V characterization of the pixels.
    Plot the results of the C-V characterization of the scanned pixels.
    To achieve this we need the different c-v-data.
    Then the C-V curve is plotted for every pixel.
    If requested also fits to the boundary regions of the c-v-curve are performed to determine the depletion voltage of the pixel.
    To do so, two fit ranges for the two boundaries with physically distinct behaviour needs to be supplied.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param first_upper: upper limit of the first fit range
    :param first_lower: lower limit of the first fit range
    :param second_upper: upper limit of the second fit range
    :param second_lower: lower limit of the second fit range
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    :key use_corrected: boolean, whether to use corrected data
    :key verbose: boolean, indicating whether to use verbose output for depletion voltages
    :key distribution: boolean, indicating whether also the capacitance distribution of the whole sensor should be investigated.
    """
    if kwargs.get("use_corrected", False):
        suffix = "{}_corrected".format(suffix)
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    with PdfPages(pdf_name) as output_pdf:
        # with PdfPages(interpreted_data[:-3] + '.pdf') as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            plot_bias_delegate(base_group.biasing.measurements, output_pdf)
            plot_cv_data_delegate(base_group.biasing.measurements, get_analysis_group(base_group.biasing, **kwargs),
                                  output_pdf, first_upper, first_lower, second_upper, second_lower, **kwargs)


def plot_bias_delegate(data_group, output_pdf: PdfPages):
    """
    plot_bias_delegate

    Actual implementation for presenting the results of the I-V characterization.
    It's just a simple plot with error bars for the different quantities.

    :param data_group: hdf file's hierarchy group containing the raw data.
    :param output_pdf: PDF object to write the plots to.
    """
    tabular = data_group.BiasTable
    assert isinstance(tabular, tb.Table)
    voltage_data = np.abs(tabular.col("U"))
    current_data = np.abs(tabular.col("I"))
    current_errors = tabular.col("DI")
    if not np.all(np.isfinite(voltage_data)):
        current_errors = None
    fig, ax = plt.subplots()
    ax.set(title="Bias data from the measurement", xlabel=BIAS_CURVE_X_LABEL, ylabel=BIAS_CURVE_Y_LABEL)
    ax.errorbar(voltage_data, current_data * CURRENT_CONVERSION_FACTOR,
                yerr=current_errors, xerr=None, fmt='o', label="Bias data")
    output_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def plot_cv_data_delegate(data_group, analysis_group, output_pdf, first_upper: Optional[float] = None, first_lower:
Optional[float] = None, second_upper: Optional[float] = None,
                          second_lower: Optional[float] = None, apply_doping=False, **kwargs):
    """
    plot_cv_data_delegate

    Actual implementation for presenting the results of the C-V characterization and if necessary the determination of
    the full depletion voltage. For each with a successful capacitance measurement for each bias voltage in use
    the C-V-curve will be plotted. If necessary, meaning if the boundaries for the two fit ranges for the
    two physically distinct regions of the c-v-curve are provided the full depletion voltage will be calculated and
    the necessary fits be plotted. If the analysis results provided already contain the necessary data sets for the
    estimation of the full depletion voltage, these will be used and the fit will be plotted, as well.

    :param data_group: hdf files hierarchy group containing the raw measurement data.
    :param analysis_group: HDF files hierarchy group containing the analysis results.
    :param output_pdf: PDF object to write the created figures to for long-term saving.
    :param first_upper: upper limit of the first fit range.
    :param first_lower: lower limit of the first fit range.
    :param second_upper: upper limit of the second fit range.
    :param second_lower: lower limit of the second fit range.
    :param apply_doping: boolean, False, indicates whether to plot the depletion data.
    :key verbose: boolean, indicating whether to use verbose output for depletion voltages.
    :key distribution: boolean, indicating whether also the capacitance distribution of the whole sensor should be investigated.
    """
    # extract the bias data
    voltage_data = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)

    approx_depletion = True
    if first_upper is None or first_lower is None or second_upper is None or second_lower is None:
        approx_depletion = False
    if approx_depletion and "DepletionHist" not in analysis_group:
        warn("The renew computation is now deperecated and will be removed in future version.")
        if 'chip_group' in kwargs:
            kwargs['apply_doping'] = apply_doping
        from pixcap65.analysis import analyze_depletion_delegate
        analyze_depletion_delegate(data_group, analysis_group, (first_lower, first_upper),
                                   (second_lower, second_upper), **kwargs)

    # investigate all the pixel for plotting
    for ii, jj in np.ndindex(GENERAL_PIXCAP_SHAPE):
        check_leaf_unit(analysis_group.UCHist, HIST_CAP_UNIT)
        check_leaf_unit(analysis_group.UCErrHist, HIST_CAP_UNIT)

        cap_data = analysis_group.UCHist[:, :, :]
        cap_errors = analysis_group.UCErrHist[:, :, :]
        if np.any(np.isnan(cap_data[ii, jj, :])):
            continue

        # cv_height, cv_width = rcParams['figure.figsize']
        # fig, ax = plt.subplots(ncols=2, figsize=(cv_width, cv_height))
        fig, ax = plt.subplots(ncols=2)

        title_str = ""
        # extract the information about the depletion voltage+
        if "DepletionHist" in analysis_group:
            depletion_fit_data = analysis_group.DepFitParamHist[:]
            depletion_hist = analysis_group.DepletionHist[:]
            assert isinstance(depletion_fit_data, np.ndarray)
            assert isinstance(depletion_hist, np.ndarray)
            if len(depletion_fit_data.shape) == 3:
                depletion_fit_data_temp = depletion_fit_data.reshape((40, 40, 1, 4))
                try:
                    assert np.allclose(depletion_fit_data_temp[:, :, 0, :], depletion_fit_data, equal_nan=True)
                except AssertionError:
                    finite_mask = np.isfinite(depletion_fit_data)
                    temp_data_reshape = np.full((40, 40, 1, 4), np.nan)
                    temp_data_reshape[:, :, 0, :] = depletion_fit_data
                    print(np.allclose(temp_data_reshape[:, :, 0, :][finite_mask], depletion_fit_data[finite_mask]))
                    print(np.abs(temp_data_reshape[:, :, 0, :][finite_mask] - depletion_fit_data[finite_mask]))
                    print(np.isclose(depletion_fit_data_temp[:, :, 0, :][finite_mask], depletion_fit_data[finite_mask]))
                    raise
                depletion_fit_data = depletion_fit_data_temp
                depletion_hist = depletion_hist.reshape((40, 40, 1))

            for dep_idx in range(depletion_fit_data.shape[2]):
                first_dep_parameters = depletion_fit_data[ii, jj, dep_idx, :2]
                second_dep_parameters = depletion_fit_data[ii, jj, dep_idx, 2:]
                dep_voltage_2 = depletion_hist[ii, jj, dep_idx]
                # directly plot these
                # first_voltage_x = np.linspace(-100, - dep_voltage_2 / 1.1, NUMBER_DEPLETION_PLOT_POINTS)
                # second_voltage_x = np.linspace(
                #     np.where(-dep_voltage_2 < second_lower, -dep_voltage_2, second_lower) * 1.1,
                #     second_upper, NUMBER_DEPLETION_PLOT_POINTS)
                first_voltage_x = np.linspace(np.min(voltage_data) - 10, dep_voltage_2 / 1.1,
                                              NUMBER_DEPLETION_PLOT_POINTS)
                second_voltage_x = np.linspace(dep_voltage_2 * 1.1, np.max(voltage_data) + 10,
                                               NUMBER_DEPLETION_PLOT_POINTS)
                first_cap_calc = first_dep_parameters[0] * first_voltage_x + first_dep_parameters[1]
                second_cap_calc = second_dep_parameters[0] * second_voltage_x + second_dep_parameters[1]
                ax[1].plot(first_voltage_x, first_cap_calc, '-', label="First section fit")
                ax[1].plot(second_voltage_x, second_cap_calc, '-', label="Second section fit")
                # TODO: What about the covariance matrix here!
                title_str += "U = {} V\n".format(dep_voltage_2)

        effective_capacitance_error_data = np.reciprocal(cap_data[ii, jj, :] * CAPACITANCE_CONVERSION_FACTOR) ** 3 * \
                                           cap_errors[
                                               ii, jj, :] if np.all(np.isfinite(cap_errors[ii, jj, :])) else None
        eff_cap_errors = cap_errors[ii, jj, :] if np.all(np.isfinite(cap_errors[ii, jj, :])) else None
        ax[0].set(title="Bias data from the \nmeasurement for pixel ({col},{row})".format(col=ii, row=jj),
                  xlabel=BIAS_CURVE_X_LABEL, ylabel="C in fF")
        ax[0].errorbar(voltage_data, cap_data[ii, jj, :] * CAPACITANCE_CONVERSION_FACTOR, yerr=eff_cap_errors,
                       fmt='o', label="Bias data")
        ax[1].set(title="Suited Bias data from the \nmeasurement for pixel ({col},{row})".format(col=ii, row=jj),
                  xlabel=BIAS_CURVE_X_LABEL, ylabel="$1/ C^2$ in $1/(fF)^2$")
        ax[1].errorbar(voltage_data, 1 / (cap_data[ii, jj, :] * CAPACITANCE_CONVERSION_FACTOR) ** 2,
                       yerr=effective_capacitance_error_data, fmt='o', label="Bias data", alpha=0.5)
        ax[1].set_xlim(np.min(voltage_data) - 10, 5 + np.max(voltage_data))
        ax[1].set_ylim(np.min(1 / (cap_data[ii, jj, :] * CAPACITANCE_CONVERSION_FACTOR) ** 2),
                       np.max(1 / (cap_data[ii, jj, :] * CAPACITANCE_CONVERSION_FACTOR) ** 2))
        ax[0].legend()
        ax[1].legend(title=title_str)
        ax[1].grid(True)
        fig.suptitle("C-V Characterization")
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        if apply_doping:
            depletion_width_plate = check_leaf_unit(analysis_group.DepletionWidth, "um")
            depletion_width_plate_error = check_leaf_unit(analysis_group.DepletionWidthErr, "um")
            effective_doping_table = check_leaf_unit(analysis_group.DepletionEffDoping, "cm^-3")
            bias_voltages = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
            table = analysis_group.DepletionParamTable
            plot_depletion_pixel_delegate(bias_voltages, ii, depletion_width_plate, depletion_width_plate_error,
                                          effective_doping_table, output_pdf, jj, table)
    if kwargs.pop("distribution", False):
        assert isinstance(voltage_data, Iterable)
        for k, bias_voltage in enumerate(voltage_data):
            if voltage_data.shape[0] > 10 and k % 10 != 0:
                continue
            fig, ax = plt.subplots()
            ax.set(title="Capacitance distribution for bias voltage {}".format(bias_voltage), xlabel="C in fF")
            ax.hist(analysis_group.UCHist[:, :, k].reshape(-1) * 1e15, bins=50)
            output_pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def plot_data_delegate(data_group: tb.Group, analysis_group: tb.Group, output_pdf: PdfPages, **kwargs):
    """
    plot_data_delegate

    Actual implementation to plot the results of the analysis of simple pixel capacitance scan (total capacitance).
    Besides the naming it is not just plotting but also a bit of analysis as the distribution of the capacitance is
    investigated in this function. The current-frequency dependency will be plotted for each scanned pixel with finite
    currents. Also, the capacitance distribution over the whole sensor and the frequency of capacitance values are
    plotted.

    :param data_group: hdf files hierarchy group containing the raw measurement data.
    :param analysis_group: hdf files hierarchy group containing the analysis results.
    :param output_pdf: PDF object to write the created figures to for long-term saving.
    :key exclude_test_cap: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key extract_pixel: iterable of pixel positions on the grid to extract the figures from.
    :key distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor.
    """
    # Read pixel map
    current_hist = check_leaf_unit(data_group.HistCurr, HIST_CURRENT_MEAS_UNIT)
    current_err_hist = check_leaf_unit(data_group.HistCurrErr, HIST_CURRENT_MEAS_UNIT)
    cap_hist = check_leaf_unit(analysis_group.HistCap, HIST_CAP_UNIT)
    leak_hist = check_leaf_unit(analysis_group.HistLeak, HIST_LEAK_CURRENT_UNIT)

    # Read scan parameters
    scan_parameters = data_group.scan_params[:]

    # 2D Pixel Capacitance Hist
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(cap_hist * CAPACITANCE_CONVERSION_FACTOR)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label=HIST_PIX_CAP_LABEL)
    ax.set_ylabel(COLUMN_LABEL)
    ax.set_xlabel(ROW_LABEL)
    output_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    if kwargs.get("exclude_test_cap", False):
        # another heat map which do not consider masked or boundary caps
        assert isinstance(cap_hist, np.ndarray)
        masked_cap_hist = cap_hist.copy()
        masked_cap_hist = evaluate_pixel_mask(masked_cap_hist, **kwargs)

        fig, ax = plt.subplots()
        im = ax.imshow(masked_cap_hist * CAPACITANCE_CONVERSION_FACTOR)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, label=HIST_PIX_CAP_LABEL)
        ax.set_ylabel(COLUMN_LABEL)
        ax.set_xlabel(ROW_LABEL)
        ax.set_title("Masked Pixel Capacitance distribution")
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # 1D Pixel Capacitance Hist
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    hist_cap_hist = evaluate_pixel_mask(cap_hist, **kwargs)
    ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
            bins=kwargs.get("hist_bins", DEFAULT_BIN_NUMBER))
    ax.set_ylabel(COUNTS_HIST_LABEL)
    ax.set_xlabel(HIST_PIX_CAP_LABEL)
    ax.grid()
    output_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    if kwargs.pop("distribution", False):
        from pixcap65.analysis import analyze_capacitance_distribution_delegate
        analyze_capacitance_distribution_delegate(analysis_group, output_pdf, **kwargs)

    # Current vs. frequency
    print(HISTOGRAM_SHAPE_FORMAT.format(current_hist.shape))
    verify_pixel_mask = "mask_pixel" in kwargs and isinstance(kwargs["mask_pixel"], Iterable) and len(
        kwargs["mask_pixel"]) > 0
    for col, row in np.ndindex(current_hist.shape[:2]):
        if verify_pixel_mask and (col, row) in kwargs["mask_pixel"]:
            continue
        if np.isfinite(current_hist[col, row, 0]):
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            # res = np.polyfit(scan_parameters['frequency'], current_hist[col, row] * 1e9, deg=1, cov=True)
            f = np.arange(0, scan_parameters['frequency'].max() * 1.1, 0.1)
            actual_cap = cap_hist[col, row] * CAPACITANCE_CONVERSION_FACTOR

            plot_current_model(ax, col, row, analysis_group, actual_cap, leak_hist, f,
                               parasitic_correction=extract_parasitic_capacitance(analysis_group.HistCap))
            plot_current_data(ax, col, row, scan_parameters, current_hist, current_err_hist, marker='o', ls='')
            ax.set_ylabel(CURRENT_LABEL)
            ax.set_xlabel(FREQUENCY_LABEL)
            ax.legend()
            ax.grid()
            output_pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # ax.plot(freq_sweep_array, fit_fn, label = 'a={a:.3E}, b={b:.3E}'.format(a=a, b=b))

        # #apply linear fit to measured current values; also returns covariance matrix.
        # matrix = np.polyfit(freq_sweep_array, current_array, 1, cov=True)

        # a, b = matrix[0][0], matrix[0][1]
        # #da = matrix[1][0][0] #squared fit error of a
        # #db = matrix[1][1][1] #squared fit error of b

        # #data structure in txt file: "slope, offset (y-intercept)"

        # # fit_fn = a*freq_sweep_array + b
        # # pl.plot(freq_sweep_array, current_array, 'o', label = 'COL({i_col})PIX(0)'.format(i_col=i_col))
        # # pl.plot(freq_sweep_array, fit_fn, label = 'a={a:.3E}, b={b:.3E}'.format(a=a, b=b))


def plot_inter_pix_data_delegate(data_group: tb.Group, analysis_group: tb.Group, output_pdf: PdfPages, total_group=None,
                                 **kwargs):
    """
    plot_inter_pix_data_delegate

    Actual implementation to plot the results of the analysis of the inter-pixel capacitance scan.
    Besides the naming it is not just plotting but also a bit of analysis as the distribution of the capacitance
    over the pixel and in general for all three currents is investigated, as well.
    The current-frequency dependency will plotted for each scanned pixel with finite currents.
    Also the capacitance distribution over the whole sensor and the frequency of capacitance values are plotted for all
    three current measurements. Besides the naming of the plots the capacitance are not directly the inter-pixel or
    total-pixel capacitance values.

    The modelling process is automatically corrected to use the bare capacitance if corrected capacitance values are
    supplied. If the correction is not applied by the functions from the analysis module then this may not work.


    :param data_group: hdf files hierarchy group containing the raw measurement data.
    :param analysis_group: hdf files hierarchy group containing the analysis results.
    :param output_pdf: PDF object to write the created figures to for long-term saving.
    :key exclude_test_cap: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key extract_pixel: iterable of pixel positions on the grid to extract the figures from.
    :key distribution: boolean, indicating whether to analyze also the capacitance distribution. TODO: implement it.
    """
    # Read pixel map
    total_current_hist = check_leaf_unit(data_group.TotalHistCurr, HIST_CURRENT_MEAS_UNIT)
    total_current_err_hist = check_leaf_unit(data_group.TotalHistCurrErr, HIST_CURRENT_MEAS_UNIT)
    total_cap_hist = check_leaf_unit(analysis_group.HistCap, HIST_CAP_UNIT)
    total_leak_hist = check_leaf_unit(analysis_group.HistLeak, HIST_LEAK_CURRENT_UNIT)
    inter_a_current_hist = check_leaf_unit(data_group.InterHistCurrA, HIST_CURRENT_MEAS_UNIT)
    inter_a_current_err_hist = check_leaf_unit(data_group.InterHistCurrErrA, HIST_CURRENT_MEAS_UNIT)
    inter_a_cap_hist = check_leaf_unit(analysis_group.HistCapInterA, HIST_CAP_UNIT)
    inter_a_leak_hist = check_leaf_unit(analysis_group.HistLeakInterA, HIST_LEAK_CURRENT_UNIT)
    inter_b_current_hist = check_leaf_unit(data_group.InterHistCurrB, HIST_CURRENT_MEAS_UNIT)
    inter_b_current_err_hist = check_leaf_unit(data_group.InterHistCurrErrB, HIST_CURRENT_MEAS_UNIT)
    inter_b_cap_hist = check_leaf_unit(analysis_group.HistCapInterB, HIST_CAP_UNIT)
    inter_b_leak_hist = check_leaf_unit(analysis_group.HistLeakInterB, HIST_LEAK_CURRENT_UNIT)
    total_ref_cap_hist = None if total_group is None else check_leaf_unit(total_group.HistCap, HIST_CAP_UNIT)

    need_distribution = kwargs.get("distribution", False)
    # Read scan parameters
    scan_parameters = data_group.scan_params[:]

    # 2D Pixel Capacitance Hist
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(total_cap_hist * CAPACITANCE_CONVERSION_FACTOR)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label=HIST_PIX_CAP_LABEL)
    ax.set_ylabel(COLUMN_LABEL)
    ax.set_xlabel(ROW_LABEL)
    ax.set_title("Total Pixel Capacitance")
    output_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    if total_ref_cap_hist is not None:
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        im = ax.imshow((total_ref_cap_hist - total_cap_hist) * CAPACITANCE_CONVERSION_FACTOR)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, label=HIST_PIX_CAP_LABEL)
        ax.set_ylabel(COLUMN_LABEL)
        ax.set_xlabel(ROW_LABEL)
        ax.set_title("Inter Pixel capacitance from In-Pix C")
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(inter_a_cap_hist * CAPACITANCE_CONVERSION_FACTOR)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label=HIST_PIX_CAP_LABEL)
    ax.set_ylabel(COLUMN_LABEL)
    ax.set_xlabel(ROW_LABEL)
    ax.set_title("Inter-Pixel Capacitance A")
    output_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(inter_b_cap_hist * CAPACITANCE_CONVERSION_FACTOR)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label=HIST_PIX_CAP_LABEL)
    ax.set_ylabel(COLUMN_LABEL)
    ax.set_xlabel(ROW_LABEL)
    ax.set_title("Inter-Pixel Capacitance B")
    output_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # 1D Pixel Capacitance Hist
    n_bins = kwargs.get("hist_bins", DEFAULT_BIN_NUMBER)
    if np.count_nonzero(np.isfinite(total_cap_hist)) > 2:
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        hist_cap_hist = evaluate_pixel_mask(total_cap_hist, **kwargs)
        ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                bins=n_bins)
        ax.set_ylabel(COUNTS_HIST_LABEL)
        ax.set_xlabel(HIST_PIX_CAP_LABEL)
        ax.set_title("Pixel Total Capacitance Distribution")
        ax.grid()
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        if need_distribution:
            from pixcap65.analysis import analyze_capacitance_distribution_delegate
            analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=total_cap_hist, **kwargs)

        if total_ref_cap_hist is not None:
            effective_inter_cap_hist = total_ref_cap_hist - total_cap_hist
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            hist_inter_cap_hist = evaluate_pixel_mask(effective_inter_cap_hist, **kwargs)
            ax.hist(hist_inter_cap_hist[~np.isnan(hist_inter_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                    bins=n_bins)
            ax.set_ylabel(COUNTS_HIST_LABEL)
            ax.set_xlabel(HIST_PIX_CAP_LABEL)
            ax.set_title("Pixel Inter Capacitance Distribution")
            ax.grid()
            output_pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            if need_distribution:
                from pixcap65.analysis import analyze_capacitance_distribution_delegate
                analyze_capacitance_distribution_delegate(analysis_group, output_pdf,
                                                          capacitance=effective_inter_cap_hist, **kwargs)

    if np.count_nonzero(np.isfinite(inter_a_current_hist)) > 2:
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        hist_cap_hist = evaluate_pixel_mask(inter_a_current_hist, **kwargs)
        ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                bins=n_bins)
        ax.set_ylabel(COUNTS_HIST_LABEL)
        ax.set_xlabel(HIST_PIX_CAP_LABEL)
        ax.set_title("Inter-Pixel A Capacitance Distribution")
        ax.grid()
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        if need_distribution:
            from pixcap65.analysis import analyze_capacitance_distribution_delegate
            analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=inter_a_cap_hist,
                                                      **kwargs)

    if np.count_nonzero(np.isfinite(inter_b_current_hist)) > 2:
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        hist_cap_hist = evaluate_pixel_mask(inter_b_current_hist, **kwargs)
        ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                bins=n_bins)
        ax.set_ylabel(COUNTS_HIST_LABEL)
        ax.set_xlabel(HIST_PIX_CAP_LABEL)
        ax.set_title("Inter-Pixel B Capacitance Distribution")
        ax.grid()
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        if need_distribution:
            from pixcap65.analysis import analyze_capacitance_distribution_delegate
            analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=inter_b_cap_hist,
                                                      **kwargs)

    # Current vs. frequency (Will try to plot all into just one coordinate system)
    print(HISTOGRAM_SHAPE_FORMAT.format(total_current_hist.shape))
    verify_mask_pixel = "mask_pixel" in kwargs and isinstance(kwargs["mask_pixel"], Iterable)
    for col, row in np.ndindex(total_current_hist.shape[:2]):
        if verify_mask_pixel and (col, row) in kwargs["mask_pixel"]:
            continue
        elif np.isfinite(total_current_hist[col, row, 0]):
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            f = np.arange(0, scan_parameters['frequency'].max() * 1.1, 0.1)
            actual_cap = total_cap_hist[col, row] * CAPACITANCE_CONVERSION_FACTOR
            plot_current_model(ax, col, row, analysis_group, actual_cap, total_leak_hist, f, prefix="Total ",
                               parasitic_correction=extract_parasitic_capacitance(analysis_group.HistCap))
            plot_current_data(ax, col, row, scan_parameters, total_current_hist, total_current_err_hist,
                              prefix="Total current for ", marker='o', ls='')
            if np.isfinite(inter_a_current_hist[col, row, 0]):
                plot_current_model(ax, col, row, analysis_group,
                                   inter_a_cap_hist[col, row] * CAPACITANCE_CONVERSION_FACTOR, inter_a_leak_hist, f,
                                   resistor_name="HistResInterA", prefix="Inter A ", ls='-.',
                                   parasitic_correction=extract_parasitic_capacitance(analysis_group.HistCapInterA))
                plot_current_data(ax, col, row, scan_parameters, inter_a_current_hist, inter_a_current_err_hist,
                                  prefix="Inter A current for", marker='v')
            if np.isfinite(inter_b_current_hist[col, row, 0]):
                plot_current_model(ax, col, row, analysis_group,
                                   inter_b_cap_hist[col, row] * CAPACITANCE_CONVERSION_FACTOR, inter_b_leak_hist, f,
                                   resistor_name="HistResInterB", prefix="Inter B ", ls=':',
                                   parasitic_correction=extract_parasitic_capacitance(analysis_group.HistCapInterB))
                plot_current_data(ax, col, row, scan_parameters, inter_b_current_hist, inter_b_current_err_hist,
                                  prefix="Inter B current for", marker='s')
            ax.set_ylabel(CURRENT_LABEL)
            ax.set_xlabel(FREQUENCY_LABEL)
            ax.legend()
            ax.grid()
            output_pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


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
    plot_args.setdefault('marker', 'o')
    plot_args.setdefault('ls', '')
    assert "prefix" not in plot_args
    if 'fmt' in plot_args:
        plot_args['marker'] = plot_args.pop('fmt')
    if np.all(np.isfinite(current_err_hist[col, row, :])):
        marker_code = plot_args.pop('marker', 'o')
        plot_args['fmt'] = marker_code
        ax.errorbar(scan_parameters['frequency'], current_hist[col, row] * 1e9,
                    yerr=current_err_hist[col, row] * 1e9,
                    label=SIMPLE_PIXEL_LABEL.format(i_col=col, i_row=row, prefix=prefix),
                    color=cmap(color), **plot_args)
    else:
        ax.plot(scan_parameters['frequency'], current_hist[col, row] * 1e9,
                label=SIMPLE_PIXEL_LABEL.format(i_col=col, i_row=row, prefix=prefix),
                color=cmap(color), **plot_args)


def plot_current_model(ax: Axes, col, row, analysis_group: Group, actual_cap: Any, total_leak_hist, f: np.ndarray,
                       resistor_name="HistRes", prefix="", color=0.6, **plot_args):
    """
        plot_current_model

        Helper function to plot the current model from the specified parameters.
        It will decide on execution whether the complete advanced model has to be used depending on whether on-resistance
        estimators are present.

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
        :key parasitic_correction: if present, it should be the capacitance subtracted during capacitance correction
            procedure.
        """
    parasitic_correction = plot_args.pop('parasitic_correction', 0.0)
    assert "parasitic_correction" not in plot_args
    plot_args.setdefault('marker', '')
    plot_args.setdefault('ls', '--')
    assert 'prefix' not in plot_args
    if resistor_name in analysis_group and np.isfinite(analysis_group[resistor_name][col, row]):
        hist_resistance = analysis_group[resistor_name]
        from pixcap65.analysis_util.physics_modelling import full_capacitance_model
        assert isinstance(hist_resistance, tb.Array) or isinstance(hist_resistance, np.ndarray)
        y = full_capacitance_model(f,
                                   c=(actual_cap + parasitic_correction) * ADVANCED_CAPACITANCE_CONVERSION_FACTOR,
                                   r=hist_resistance[col, row],
                                   i=total_leak_hist[col, row] * 1.e-9, u0=1) * 1e9
        ax.plot(f,
                y,
                color=cmap(color),
                label=SIMPLE_CAP_LABEL_PERCENT_FORMAT % (prefix, actual_cap), **plot_args)


    else:
        ax.plot(f, (actual_cap + parasitic_correction) * f + total_leak_hist[col, row],
                color=cmap(color),
                label=SIMPLE_CAP_LABEL_PERCENT_FORMAT % (prefix, actual_cap), **plot_args)


def plot_compare_delegate(first_group: GroupType, second_group: GroupType, output_pdf: PdfPages):
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
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(diff_cap_hist * CAPACITANCE_CONVERSION_FACTOR, )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label=HIST_PIX_CAP_LABEL)
    ax.set_ylabel(COLUMN_LABEL)
    ax.set_xlabel(ROW_LABEL)
    output_pdf.savefig(fig, bbox_inches='tight')

    # close the figures at last to not waste any memory resources
    plt.close(fig)


def plot_depletion_delegate(data_group: GroupType, analysis_group: GroupType, output_pdf: PdfPages):
    """
    plot_depletion_delegate

    Actual implementation of the depletion investigation plotting.
    This function will plot the depletion profile and the doping profile which was extracted from the
    C-V characterization previously.

    :param data_group: hdf files group where to find the measurement data
    :param analysis_group: hdf files group where to find the analysis results
    :param output_pdf: PDF file to write the figures to for long-term storage
    """
    assert isinstance(analysis_group, tb.Group)
    # extract the depletion parameters
    depletion_width_plate = check_leaf_unit(analysis_group.DepletionWidth, "um")
    depletion_width_plate_error = check_leaf_unit(analysis_group.DepletionWidthErr, "um")
    effective_doping_table = check_leaf_unit(analysis_group.DepletionEffDoping, "cm^-3")
    bias_voltages = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
    table = analysis_group.DepletionParamTable

    for col, row in np.ndindex(GENERAL_PIXCAP_SHAPE):
        plot_depletion_pixel_delegate(bias_voltages, col, depletion_width_plate, depletion_width_plate_error,
                                      effective_doping_table, output_pdf, row, table)


def plot_depletion_pixel_delegate(bias_voltages: TABLES_LEAF_COMPAT_TYPE, i_col,
                                  depletion_width_plate: TABLES_LEAF_COMPAT_TYPE,
                                  depletion_width_plate_error: TABLES_LEAF_COMPAT_TYPE,
                                  effective_doping_table: TABLES_LEAF_COMPAT_TYPE, output_pdf: PdfPages, i_row, table):
    if np.all(np.isfinite(depletion_width_plate[i_col, i_row])):
        condition = """(row == {}) & (col == {})""".format(i_row, i_col)
        for x in table.where(condition):
            depletion_fit_propagate_parameters = {p_key: x[p_key] for p_key in ["NAD", "V", "dep", "sat"]}
            break
        else:
            depletion_fit_propagate_parameters = {}
        doping_acceptor = depletion_fit_propagate_parameters["NAD"]
        effective_doping = effective_doping_table[i_col, i_row]
        fig, ax = plt.subplots(3)
        bias_mask = bias_voltages < -0.5
        if np.all(np.isfinite(depletion_width_plate_error[i_col, i_row])):
            ax[0].errorbar(bias_voltages[bias_mask], depletion_width_plate[i_col, i_row][bias_mask],
                           yerr=depletion_width_plate_error[i_col, i_row][bias_mask], label='d-measurement')
        else:
            ax[0].plot(bias_voltages[bias_mask], depletion_width_plate[i_col, i_row][bias_mask], label='d-measurement')

        # sample_voltage = -1 * np.linspace(np.min(-bias_voltages), np.max(-bias_voltages) * 1.1, 1000)
        # TODO: use here the correct values
        sample_voltage = np.linspace(np.min(bias_voltages[bias_mask]) * 1.1, np.max(bias_voltages[bias_mask]) / 1.1,
                                     1000)
        ax[0].plot(sample_voltage, model_depletion(sample_voltage, **depletion_fit_propagate_parameters),
                   label=f'd-theory for NAD = {doping_acceptor:4.2f}  and Ubi = {depletion_fit_propagate_parameters["V"]:.2f}')
        ax[0].set(xlabel='Bias Voltage [V]', ylabel='Depletion Width [µm]',
                  title=f"Analysis of the depletion width for pixel ({i_col}, {i_row}).")
        ax[0].grid(True)
        ax[0].legend(
            title=f"Saturating at {depletion_fit_propagate_parameters["dep"]} with {depletion_fit_propagate_parameters['sat']} saturation.")
        ax[1].plot(-bias_voltages, effective_doping)
        ax[1].set(xlabel='Bias Voltage [V]', ylabel='Effective \ndoping \nconcentration [cm-3]',
                  title="Analysis of the effective doping for pixel ({col}, {row}).")
        ax[1].grid(True)
        ax[1].set_yscale('log')
        ax[2].plot(depletion_width_plate[i_col, i_row], effective_doping)
        ax[2].set(xlabel='depletion width [um]', ylabel='Effective\ndoping\nconcentration [cm-3]',
                  title="Analysis of the effective doping")
        ax[2].set_yscale('log')
        output_pdf.savefig(fig, bbox_inches='tight')
        # close the figures at last, to not waste any memory resources
        plt.close(fig)


if __name__ == '__main__':
    from pixcap65.utility.homogenize_plots import set_params

    set_params(latex=True,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                           r"stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                           r"retain-zero-uncertainty}", fig_height=8.26772, fig_width=11.69291, )
    # plot_data(interpreted_data=os.path.expanduser('~/git/pixcap65/pixcap_LF_50x50_DC_R3_80V_HV.h5'))
    # plot_data(interpreted_data='Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1", suffix="unbiased_full_measurement", use_group=True)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data",
              use_group=False)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data",
              use_group=False,
              use_corrected=True, exclude_test_cap=True, distribution=True)
    # plot_inter_pix_data(interpreted_data='R13-Interpixel_Scan.h5',
    #                     base_path="Reference/R13/demo_measurement_65_unbiased_1_discharge",
    #                     use_group=True, suffix="inter_pix_65", total_data='Reference_R13_Scan.h5',
    #                     distribution=True, set_parasitic=False, total_path="Reference/R13/unbiased_12_full")
    # plot_inter_pix_data(interpreted_data='R13-Interpixel_Scan.h5', base_path="Reference/R13/demo_measurement_64_unbiased_1_discharge",
    #                     use_group=True, suffix="inter_pix_64")
    # plot_data(interpreted_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/unbiased_3_test", use_group=True)
    # plot_bias_data(interpreted_data='Data/r13-measurement/R13_BIAS_2.h5')
    # plot_combined_data(interpreted_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', first_lower=-100,first_upper=-40, second_lower=-10, second_upper=0)
    # plot_data(interpreted_data=os.path.expanduser('~/git/pixcap65/pixcap_LF_50x50_DC_R3_80V_HV.h5'))
    # plot_data(interpreted_data='Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1", suffix="unbiased_full_measurement", use_group=True)
    # plot_inter_pix_data(interpreted_data='R13-Interpixel_Scan.h5',
    #                     base_path="Reference/R13/demo_measurement_52_biased_80_V_1_charge",
    #                     use_group=True, suffix="inter_pix_52")
    # plot_data(interpreted_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/unbiased_1_test", use_group=True)
    # plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
    #           suffix="general_data", use_group=True, exclude_test_cap=True, mask_pixel=[[39, 39], [38, 39]],
    #           distribution=True)
    # plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
    #           suffix="general_data", use_group=True, use_corrected=True, mask_pixel=[[39, 39], [38, 39]],
    #           exclude_test_cap=True, distribution=True)
    # plot_bias_data(interpreted_data='Data/r13-measurement/R13_BIAS_2.h5')
    # plot_combined_data(interpreted_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', first_lower=-100,first_upper=-40, second_lower=-10, second_upper=0)

    # plot_data(interpreted_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
    #           suffix="general_bare_data_3", use_group=True,
    #           exclude_test_cap=True)

    plot_combined_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                       base_path="ATLAS ITk/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-60, second_upper=0,
                       use_corrected=True,
                       apply_doping=True)

    plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-60, second_upper=0,
                       use_corrected=True)
