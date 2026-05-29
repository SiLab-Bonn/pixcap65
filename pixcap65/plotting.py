"""
Plotting of Pixcap65 data.
"""
# ----------------------------------------------------------
#  Copyright (c) 2018. All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------

# TODO: update the documentation of the implementations

import os.path

import logging
import threading
import time
from contextlib import contextmanager
from matplotlib.axes import Axes
from types import NoneType
from typing import Any, Optional, Union, List, Tuple

from pixcap65.analysis_util.physics_modelling import model_depletion

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
NUMBER_DEPLETION_PLOT_POINTS = 1000
CAPACITANCE_LABEL = "C in fF"
X2_SCAN_2_FILE = "packaged/X2_2_Scan.h5"
E1_SCAN_FILE = "Reference_Evelyn_Scan.h5"
X2_SCAN_FILE = 'New_2_Scan.h5'
X1_SCAN_2_FILE = "packaged/data/X1_4_Renew_Scan.h5"
REFERENCE_TEST_FILE = "packaged/Reference_Demo.h5"
CV_DATA_FOR_ = "CV Data for {}"
NEW_PLOT_FILE_MODE = False

interactive_lock = threading.RLock()


def evaluate_pixel_mask(hist, perform_filter=False, **kwargs):
    """
    evaluate_pixel_mask

    @author Dominik Fischer
    @date 2026-05-11

    :param hist: histogram/data set to be masked for 'defect' pixels
    :param perform_filter: boolean, indicating whether the generated mask should be applied and only the filtered data
        returned.
    :param kwargs: further keyword arguments
    :key test_cap_exclusion: whether to exclude row 0 completely.
    :key mask_pixel: array of tuple of pixel positions to be masked.
    :key mask_lower: float, threshold to mask all pixels below this value.
    :key mask_upper: float, threshold to mask all pixels above this value.
    :return: masked histogram/data set (masked pixels values are replaced by np.nan)
    """
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
    file_mode = os.path.join(interpreted_data[:-3].split("_", 1)[0], interpreted_data[:-3])if NEW_PLOT_FILE_MODE else interpreted_data[:-3]
    if use_group and base_path is not None:
        _, group_component = os.path.split(base_path)
        pdf_name = "{file}_{s}_{group}.pdf".format(s=suffix, file=file_mode, group=group_component)
    else:
        pdf_name = "{file}_{s}.pdf".format(s=suffix, file=file_mode)
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
    with PdfPages(pdf_name) as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
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
    :param total_data: path to the hdf file which holds the analyzed data for the total capacitance scan.
    :param total_path: hdf group path inside the hdf file containing the total cap analysis results.
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


@contextmanager
def multi_sensor_file_handler_simple(interpreted_data, base_path, **kwargs):
    pdf_name = kwargs.pop("pdf_name", "I-V-Collection.pdf")
    # to simplify the operation we need a mapping of a file to all the group_mapping it should be used for,
    # and we need a mapping to the opened files
    group_mapping = {}
    files = {}
    groups = []
    try:
        for file_path, group_name in zip(interpreted_data, base_path):
            if file_path not in group_mapping:
                current_file = tb.open_file(file_path, mode='r')
                files[file_path] = current_file
                group = get_base_group(group_name, current_file)
                group_mapping[file_path] = [group]
            else:
                group = get_base_group(group_name, files[file_path])
                group_mapping[file_path].append(group)
            groups.append(group.biasing.measurements)
        with PdfPages(pdf_name) as output_pdf:
            yield groups, output_pdf
    finally:
        for file in files.values():
            assert isinstance(file, tb.File)
            file.flush()
            file.close()

@contextmanager
def multi_sensor_file_handler_advanced(interpreted_data, base_path, **kwargs):
    pdf_name = kwargs.pop("pdf_name", "I-V-Collection.pdf")
    # to simplify the operation we need a mapping of a file to all the group_mapping it should be used for,
    # and we need a mapping to the opened files
    group_mapping = {}
    files = {}
    groups = []
    analysis_groups = []
    try:
        for file_path, group_name in zip(interpreted_data, base_path):
            if file_path not in group_mapping:
                current_file = tb.open_file(file_path, mode='r')
                files[file_path] = current_file
                group = get_base_group(group_name, current_file)
                group_mapping[file_path] = [group]
            else:
                group = get_base_group(group_name, files[file_path])
                group_mapping[file_path].append(group)
            groups.append(group.biasing.measurements)
            analysis_groups.append(get_analysis_group(group.biasing, **kwargs))
        with PdfPages(pdf_name) as output_pdf:
            yield groups, analysis_groups, output_pdf
    finally:
        for file in files.values():
            assert isinstance(file, tb.File)
            file.flush()
            file.close()

def plot_bias_data(interpreted_data, base_path=None, suffix="bias_curve", use_group=False, **kwargs):
    """
    plot_bias_data

    Plot the data acquired for the pixel-diodes I-V characterization.

    :param interpreted_data: path to the hdf file which holds the raw data.
    :param base_path: path within the files hierarchy for the base group.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    """
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    if isinstance(interpreted_data, Union[List, Tuple, np.ndarray]):
        with multi_sensor_file_handler_simple(interpreted_data, base_path, **kwargs) as (groups, output_pdf):
            plot_bias_delegate(groups, output_pdf, **kwargs)
    else:
        with PdfPages(pdf_name) as output_pdf:
            with tb.open_file(interpreted_data, mode='r') as in_file_h5:
                base_group = get_base_group(base_path, in_file_h5)
                plot_bias_delegate(base_group.biasing.measurements, output_pdf)


def plot_cv_data(interpreted_data, base_path=None, suffix="C_V_characteristic", use_group=False, **kwargs):
    """
    plot_cv_data

    Plot the results of the C-V characterization of the scanned pixels.
    To achieve this we need the different c-v-data.
    Then the C-V curve is plotted for every pixel.
    If requested also fits to the boundary regions of the c-v-curve are performed to determine
    the depletion voltage of the pixel.
    To do so, two fit ranges for the two boundaries with physically distinct behaviour needs to be supplied.


    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix:  additional suffix to use for naming the PDF containing the plots.
    :param use_group:   boolean, whether to append the group name of the measurements to the PDF name.
    :key verbose: boolean, indicating whether to use verbose output for depletion voltages
    :key distribution: boolean, indicating whether also the capacitance distribution of the whole sensor
        should be investigated.
    """
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    if isinstance(interpreted_data, Iterable) and not isinstance(interpreted_data, str):
        with multi_sensor_file_handler_advanced(interpreted_data, base_path, **kwargs) as (groups, analysis_groups, output_pdf):
            plot_cv_data_delegate(groups, analysis_groups, output_pdf, **kwargs)
    else:
        with PdfPages(pdf_name) as output_pdf:
            with tb.open_file(interpreted_data, mode='a') as in_file_h5:
                base_group = get_base_group(base_path, in_file_h5)
                plot_cv_data_delegate(base_group.biasing.measurements,
                                      get_analysis_group(base_group.biasing, **kwargs), output_pdf,
                                      **kwargs)


def plot_combined_data(interpreted_data, base_path=None, suffix="combined_bias_cv_curve", use_group=False, **kwargs):
    """
    plot_combined_data

    Plot the data acquired for the pixel-diodes I-V characterization and the C-V characterization of the pixels.
    Plot the results of the C-V characterization of the scanned pixels.
    To achieve this we need the different c-v-data.
    Then the C-V curve is plotted for every pixel.
    If requested also fits to the boundary regions of the c-v-curve are performed to determine the
    depletion voltage of the pixel.
    To do so, two fit ranges for the two boundaries with physically distinct behaviour needs to be supplied.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    :key use_corrected: boolean, whether to use corrected data
    :key verbose: boolean, indicating whether to use verbose output for depletion voltages
    :key distribution: boolean, indicating whether also the capacitance distribution of the whole sensor
        should be investigated.
    """
    if kwargs.get("use_corrected", False):
        suffix = "{}_corrected".format(suffix)
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    with PdfPages(pdf_name) as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            plot_bias_delegate(base_group.biasing.measurements, output_pdf)
            plot_cv_data_delegate(base_group.biasing.measurements, get_analysis_group(base_group.biasing, **kwargs),
                                  output_pdf, **kwargs)


def plot_bias_delegate(data_group, output_pdf: PdfPages, **kwargs):
    """
    plot_bias_delegate

    Actual implementation for presenting the results of the I-V characterization.
    It's just a simple plot with error bars for the different quantities.

    :param data_group: hdf file's hierarchy group containing the raw data.
    :param output_pdf: PDF object to write the plots to.
    """
    fig, ax = plt.subplots()
    if isinstance(data_group, Iterable) and not isinstance(data_group, tb.Node):
        labels = kwargs.pop("labels", ["Bias_data"] * len(data_group))
        normalization = np.asarray(kwargs.pop("area_normalisation", np.full(len(data_group), 1.0)))
        print(labels)
        for group, label, norm in zip(data_group, labels, normalization):
            _bias_voltage_plotter(ax, group.BiasTable, label, norm=norm)
    else:
        tabular = data_group.BiasTable
        assert isinstance(tabular, tb.Table)
        _bias_voltage_plotter(ax, tabular, kwargs.pop("labels", "Bias data"))
    ax.legend()
    output_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _bias_voltage_plotter(ax, tabular: tb.Table, label, norm=1):
    voltage_data = np.abs(tabular.col("U"))
    current_data = np.abs(tabular.col("I"))
    current_errors = tabular.col("DI")
    try:
        voltage_error = np.abs(tabular.col("DU"))
    except (AttributeError, KeyError):
        voltage_error = voltage_data * 0.0002 + 0.1
    if not np.all(np.isfinite(voltage_data)):
        current_errors = None
    ax.set(title="Bias data from the measurement", xlabel=BIAS_CURVE_X_LABEL, ylabel=BIAS_CURVE_Y_LABEL)
    # currently we could not use the correct voltage range, but we assume the errors to be within
    ax.errorbar(voltage_data, current_data * CURRENT_CONVERSION_FACTOR / norm, xerr=voltage_error,
                yerr=current_errors / norm, fmt='o', label=label)
    if not np.isclose(norm, 1.0):
        ax.set_yscale('log')


SENSOR_ITERABLE = Union[List[tb.Group], Tuple[tb.Group, ...], np.ndarray[tb.Group]]


def __process_voltage_set(group: tb.Group):
    temp_hist = check_leaf_unit(group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
    if len(temp_hist.shape) > 1:
        return temp_hist[:, 0]
    else:
        return temp_hist

def plot_cv_data_delegate(data_group: Union[tb.Group, SENSOR_ITERABLE],
                          analysis_group: Union[tb.Group, SENSOR_ITERABLE], output_pdf,
                          apply_doping=False, **kwargs):
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
    :param apply_doping: boolean, False, indicates whether to plot the depletion data.
    :key verbose: boolean, indicating whether to use verbose output for depletion voltages.
    :key distribution: boolean, indicating whether also the capacitance distribution of the whole sensor
        should be investigated.
    """
    # extract the bias data
    voltage_data_sets = [__process_voltage_set(data_group)] if isinstance(data_group, tb.Node) else [__process_voltage_set(data_set) for data_set in data_group]
    approx_depletion = isinstance(data_group, tb.Node)

    labels = kwargs.pop('labels', [])
    # investigate all the pixel for plotting
    for ii, jj in np.ndindex(GENERAL_PIXCAP_SHAPE):
        with interactive_lock:
            fig, ax = plt.subplots(ncols=2)
        # will not only generate the title string of the figure but also the figure with the depletion fits.
        title_str = __plot_depletion_estimation(analysis_group, approx_depletion, ax, ii, jj, voltage_data_sets[0])

        if not _cv_plotter(analysis_group, jj, ii, ax, voltage_data_sets, labels):
            plt.close(fig)
            continue

        ax[0].legend()
        ax[1].legend(title=title_str)
        ax[1].grid(True)
        fig.suptitle("C-V Characterization for Pixel ({}, {})".format(ii, jj))
        output_pdf.savefig(fig, bbox_inches='tight')
        with interactive_lock:
            plt.close(fig)

        # Plot the doping analysis only for single-sensor samplings.
        if apply_doping and isinstance(data_group, tb.Group):
            # TODO: Refactor this part to support plotting for multiple sensors/data sets.
            depletion_width_plate = check_leaf_unit(analysis_group.DepletionWidth, "um")
            depletion_width_plate_error = check_leaf_unit(analysis_group.DepletionWidthErr, "um")
            effective_doping_table = check_leaf_unit(analysis_group.DepletionEffDoping, "cm^-3")
            bias_voltages = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
            table = analysis_group.DepletionParamTable
            plot_depletion_pixel_delegate(bias_voltages, ii, depletion_width_plate, depletion_width_plate_error,
                                          effective_doping_table, output_pdf, jj, table)

    if not (kwargs.pop("distribution", False) and approx_depletion):
        return
    assert isinstance(voltage_data_sets[0], Iterable)
    def iterator_filter(item):
        return voltage_data_sets[0].shape[0] < 10 or item[0] % 10 == 0

    iterator = filter(iterator_filter, enumerate(voltage_data_sets[0]))
    for k, bias_voltage in iterator:
        fig, ax = plt.subplots()
        ax.set(title="Capacitance distribution for bias voltage {}".format(bias_voltage), xlabel=CAPACITANCE_LABEL)
        ax.hist(analysis_group.UCHist[:, :, k].reshape(-1) * 1e15, bins=50)
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(ncols=2)
    # first prepare the datasets
    combiner = True
    if isinstance(analysis_group, tb.Group):
        group_handle = [analysis_group]
        label_handle = ["capacitance data"]
        combiner = False
    else:
        group_handle = analysis_group
        label_handle = kwargs.pop('labels', ['?'] * len(analysis_group))


    x_limits, y_limits = None, None
    title_str = ""
    for ana_group, label in zip(group_handle, label_handle):
        if "CVDistribution" not in ana_group:
            continue
        x_limits, y_limits, title_str = _plot_cv_distribution(ana_group, ax, x_limits, y_limits, label=label, is_combining=combiner, **kwargs)

    if x_limits is not None:
        print(type(x_limits))
        assert isinstance(x_limits, (tuple, list, set))
        assert isinstance(y_limits, (tuple, list, set))
        ax[1].set_xlim(*x_limits)
        ax[1].set_ylim(*y_limits)
        ax[1].legend()
        fig.suptitle(title_str)
        output_pdf.savefig(fig, bbox_inches='tight')

    plt.close(fig)

def _plot_cv_distribution(group: tb.Group, ax, x_limits=None, y_limits=None, **kwargs) -> Tuple[Optional[Tuple], Optional[Tuple], str]:
    title_str = ""
    corrected_data = kwargs.pop("use_corrected", False)
    is_combining = kwargs.pop("is_combining", False)
    # to solve it, it is only necessary to wrap it into an iterator
    if "SensorDepletionRaw" in group:
        depletion_data = group.SensorDepletionRaw[:]
        access_format_str = "_corrected" if corrected_data else ""
        try:
            depletion_fit_a = depletion_data["a{}".format(access_format_str)]
            depletion_fit_b = depletion_data["b{}".format(access_format_str)]
            depletion_fit_c = depletion_data["c{}".format(access_format_str)]
            depletion_fit_d = depletion_data["d{}".format(access_format_str)]
            dep_voltage = depletion_data["Ubi{}".format(access_format_str)]
        except (KeyError, TypeError, IndexError):
            print(depletion_data.coldescrs)
            print(depletion_data.description)
            raise
        voltage_data = group.CVDistribution[:]["bias"]
        assert isinstance(depletion_fit_a, np.ndarray)
        assert isinstance(depletion_fit_b, np.ndarray)
        assert isinstance(depletion_fit_c, np.ndarray)
        assert isinstance(depletion_fit_d, np.ndarray)

        for dep_idx in range(depletion_fit_a.shape[0]):
            dep_voltage_2 = dep_voltage[dep_idx]
            print("depletion voltage ...")
            print(dep_voltage_2)
            first_voltage_x = np.linspace(np.min(voltage_data) - 10, dep_voltage_2 / 1.1,
                                          NUMBER_DEPLETION_PLOT_POINTS)
            second_voltage_x = np.linspace(dep_voltage_2 * 1.1, np.max(voltage_data) + 10,
                                           NUMBER_DEPLETION_PLOT_POINTS)
            first_cap_calc = depletion_fit_a[dep_idx] * first_voltage_x + depletion_fit_b[dep_idx]
            second_cap_calc = depletion_fit_c[dep_idx] * second_voltage_x + depletion_fit_d[dep_idx]
            # skip plotting of this functions if the multiple C-V- is plotted
            if not is_combining:
                ax[1].plot(first_voltage_x, first_cap_calc, '-', label="First section fit")
                ax[1].plot(second_voltage_x, second_cap_calc, '-', label="Second section fit")
            # TODO: What about the covariance matrix here!
            title_str += "U = {} V\n".format(dep_voltage_2)

    # since distribution is selected we should assume that this condition is always fulfilled.
    assert "CVDistribution" in group
    # to solve it, it is only necessary to wrap it into an iterator.
    depletion_data = group.CVDistribution[:]
    voltage_data = depletion_data["bias"]
    from pixcap65.analysis import _extract_table_data

    cap_data = _extract_table_data("capacitance", corrected_data, depletion_data,)
    cap_data_errors = _extract_table_data("cap_std", corrected_data, depletion_data,)
    title_format = "sensor distribution"
    label = kwargs.pop("label", "capacitance data")

    if np.any(np.isnan(cap_data)):
        return None, None, ""

    x_limits, y_limits = __cv_plot_instance(ax, voltage_data, cap_data, cap_data_errors, label, title_format, x_limits, y_limits)
    return x_limits, y_limits, title_str


def __plot_depletion_estimation(analysis_group: Union[tb.Group, SENSOR_ITERABLE],
                                approx_depletion: bool, ax, ii, jj, voltage_data: np.ndarray) -> str:
    if not isinstance(analysis_group, tb.Group):
        return __plot_depletion_estimation(analysis_group[0], approx_depletion, ax, ii, jj, voltage_data)
    title_str = ""
    if approx_depletion and "DepletionHist" in analysis_group:
        depletion_fit_data = analysis_group.DepFitParamHist[:]
        depletion_hist = analysis_group.DepletionHist[:]
        assert isinstance(depletion_fit_data, np.ndarray)
        assert isinstance(depletion_hist, np.ndarray)
        if len(depletion_fit_data.shape) == 3:
            # could the reshape throw things althogether
            depletion_fit_data_temp = depletion_fit_data.reshape((40, 40, 1, 4))
            depletion_fit_data = depletion_fit_data_temp
            depletion_hist = depletion_hist.reshape((40, 40, 1))

        for dep_idx in range(depletion_fit_data.shape[2]):
            first_dep_parameters = depletion_fit_data[ii, jj, dep_idx, :2]
            second_dep_parameters = depletion_fit_data[ii, jj, dep_idx, 2:]
            dep_voltage_2 = depletion_hist[ii, jj, dep_idx]
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
    return title_str

def _cv_plotter(analysis, row, col, ax, voltage_data_sets, labels, **kwargs):
    title_format = "pixel ({col},{row})".format(col=col, row=row)
    # it should be quite simply to combine this two implementation branches into just a single one!
    # first get data iterators from the group iterators!
    if len(labels) == 0:
        labels = ["Bias Data"]

    if isinstance(analysis, tb.Group):
        analysis = [analysis]

    is_distribution_plot = kwargs.pop("is_distribution_plot", False)
    if is_distribution_plot and False:
        pass
    else:
        cap_data_sets = [check_leaf_unit(item.UCHist, HIST_CAP_UNIT)[col, row, :] for item in analysis]
        cap_data_errors_sets = [check_leaf_unit(item.UCErrHist, HIST_CAP_UNIT)[col, row, :] for item in analysis]


    eff_cap_data = None
    y_limits = None
    x_limits = None
    for cap_data, cap_data_errors, voltage_data, label in zip(cap_data_sets, cap_data_errors_sets, voltage_data_sets, labels):
        eff_cap_data = cap_data
        # could this be made common?
        if np.any(np.isnan(cap_data)):
            eff_cap_data = None
            continue

        # begin of the common part
        x_limits, y_limits = __cv_plot_instance(ax, voltage_data, cap_data, cap_data_errors, label, title_format,
                                                x_limits, y_limits)

    if eff_cap_data is None or isinstance(x_limits, NoneType) or isinstance(y_limits, NoneType):
        return False

    ax[1].set_xlim(*x_limits)
    ax[1].set_ylim(*y_limits)
    return True



def __cv_plot_instance(ax, voltage_data: np.ndarray, cap_data: np.ndarray, cap_data_errors: np.ndarray, label: str,
                       title_format, x_limits, y_limits) -> tuple[Iterable, Iterable]:
    effective_capacitance_error_data = np.reciprocal(cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 3 * cap_data_errors * CAPACITANCE_CONVERSION_FACTOR if np.all(np.isfinite(cap_data_errors)) else None
    eff_cap_errors = cap_data_errors * CAPACITANCE_CONVERSION_FACTOR if np.all(np.isfinite(cap_data_errors)) else None
    ax[0].set(title="Bias data from the \nmeasurement for {}".format(title_format), xlabel=BIAS_CURVE_X_LABEL,
              ylabel=CAPACITANCE_LABEL)
    ax[0].errorbar(voltage_data, cap_data * CAPACITANCE_CONVERSION_FACTOR, yerr=eff_cap_errors,
                   fmt='o', label=label)
    ax[1].set(title="Suited Bias data from the \nmeasurement for {}".format(title_format),
              xlabel=BIAS_CURVE_X_LABEL, ylabel="$1/ C^2$ in $1/(fF)^2$")
    ax[1].errorbar(voltage_data, 1 / (cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 2,
                   yerr=effective_capacitance_error_data, fmt='o', label=label, alpha=0.5)
    x_limits = get_x_limits(voltage_data, x_limits)
    y_limits = __get_y_limits(cap_data, y_limits)
    return x_limits, y_limits


def __get_y_limits(cap_data: np.ndarray, y_limits: Optional[Iterable], col=None, row=None) -> Iterable:
    if col and row:
        cap_data = cap_data[row, col, :]
    if y_limits is None:
        y_limits = [np.min(1 / (cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 2),
                    np.max(1 / (cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 2)]
    else:
        actual_lower_limit = np.min(1 / (cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 2)
        actual_upper_limit = np.max(1 / (cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 2)
        if y_limits[0] > actual_lower_limit:
            y_limits[0] = actual_lower_limit
        if y_limits[1] < actual_upper_limit:
            y_limits[1] = actual_upper_limit
    return y_limits


def get_x_limits(voltage_data, x_limits: Optional[Iterable]) -> Iterable:
    if x_limits is None:
        x_limits = [np.min(voltage_data) - 10, 5 + np.max(voltage_data)]
    else:
        actual_lower_limit = np.min(voltage_data) - 10
        actual_upper_limit = np.max(voltage_data) + 10
        if x_limits[0] > actual_lower_limit:
            x_limits[0] = actual_lower_limit
        if x_limits[1] < actual_upper_limit:
            x_limits[1] = actual_upper_limit
    return x_limits


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
    with interactive_lock:
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
    with interactive_lock:
        plt.close(fig)

    if kwargs.get("exclude_test_cap", False):
        # another heat map which do not consider masked or boundary caps
        assert isinstance(cap_hist, np.ndarray)
        masked_cap_hist = cap_hist.copy()
        masked_cap_hist = evaluate_pixel_mask(masked_cap_hist, **kwargs)
        with interactive_lock:
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
    with interactive_lock:
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
    with interactive_lock:
        plt.close(fig)
    if kwargs.pop("distribution", False):
        from pixcap65.analysis import analyze_capacitance_distribution_delegate

        analyze_capacitance_distribution_delegate(analysis_group, output_pdf, set_parasitic=False, **kwargs)

    # Current vs. frequency
    print(HISTOGRAM_SHAPE_FORMAT.format(current_hist.shape))
    verify_pixel_mask = "mask_pixel" in kwargs and isinstance(kwargs["mask_pixel"], Iterable) and len(
        kwargs["mask_pixel"]) > 0
    for col, row in np.ndindex(current_hist.shape[:2]):
        if verify_pixel_mask and (col, row) in kwargs["mask_pixel"]:
            continue
        if np.isfinite(current_hist[col, row, 0]):
            with interactive_lock:
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
            f = np.arange(0, scan_parameters['frequency'].max() * 1.1, 0.1)
            actual_cap = cap_hist[col, row] * CAPACITANCE_CONVERSION_FACTOR

            plot_current_model(ax, col, row, analysis_group, actual_cap, leak_hist, f,
                               parasitic_correction=extract_parasitic_capacitance(analysis_group.HistCap))
            plot_current_data(ax, col, row, scan_parameters, current_hist, current_err_hist, marker='x', ls='')
            ax.set_ylabel(CURRENT_LABEL)
            ax.set_xlabel(FREQUENCY_LABEL)
            ax.legend()
            ax.grid()
            output_pdf.savefig(fig, bbox_inches='tight')
            with interactive_lock:
                plt.close(fig)


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
    :param total_group: hdf files hierarchy group containing the total cap measurements (results).
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
    with interactive_lock:
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
    with interactive_lock:
        plt.close(fig)
    if total_ref_cap_hist is not None:
        with interactive_lock:
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
        with interactive_lock:
            plt.close(fig)

    with interactive_lock:
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
    with interactive_lock:
        plt.close(fig)

    with interactive_lock:
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
    with interactive_lock:
        plt.close(fig)

    # 1D Pixel Capacitance Hist
    n_bins = kwargs.get("hist_bins", DEFAULT_BIN_NUMBER)
    if np.count_nonzero(np.isfinite(total_cap_hist)) > 2:
        with interactive_lock:
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
        with interactive_lock:
            plt.close(fig)
        if need_distribution:
            from pixcap65.analysis import analyze_capacitance_distribution_delegate
            analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=total_cap_hist,
                                                      set_parasitic=False, **kwargs)

        if total_ref_cap_hist is not None:
            effective_inter_cap_hist = total_ref_cap_hist - total_cap_hist
            with interactive_lock:
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
            with interactive_lock:
                plt.close(fig)
            if need_distribution:
                from pixcap65.analysis import analyze_capacitance_distribution_delegate
                analyze_capacitance_distribution_delegate(analysis_group, output_pdf,
                                                            capacitance=effective_inter_cap_hist,
                                                            set_parasitic=False, **kwargs)

    if np.count_nonzero(np.isfinite(inter_a_current_hist)) > 2:
        with interactive_lock:
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
        with interactive_lock:
            plt.close(fig)
        if need_distribution:
            from pixcap65.analysis import analyze_capacitance_distribution_delegate
            analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=inter_a_cap_hist,
                                                      set_parasitic=False, **kwargs)

    if np.count_nonzero(np.isfinite(inter_b_current_hist)) > 2:
        with interactive_lock:
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
        with interactive_lock:
            plt.close(fig)
        if need_distribution:
            from pixcap65.analysis import analyze_capacitance_distribution_delegate
            analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=inter_b_cap_hist,
                                                      set_parasitic=False,
                                                      **kwargs)

    # Current vs. frequency (Will try to plot all into just one coordinate system)
    print(HISTOGRAM_SHAPE_FORMAT.format(total_current_hist.shape))
    verify_mask_pixel = "mask_pixel" in kwargs and isinstance(kwargs["mask_pixel"], Iterable)
    for col, row in np.ndindex(total_current_hist.shape[:2]):
        if verify_mask_pixel and (col, row) in kwargs["mask_pixel"]:
            continue
        elif np.isfinite(total_current_hist[col, row, 0]):
            with interactive_lock:
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
            with interactive_lock:
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
    plot_args.setdefault('marker', 'x')
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
                    color=cmap(color), capsize=2., **plot_args)
    else:
        ax.plot(scan_parameters['frequency'], current_hist[col, row] * 1e9,
                label=SIMPLE_PIXEL_LABEL.format(i_col=col, i_row=row, prefix=prefix),
                color=cmap(color), **plot_args)


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
        :key parasitic_correction: if present, it should be the capacitance subtracted during capacitance correction
            procedure.
        """
    parasitic_correction = plot_args.pop('parasitic_correction', 0.0)
    assert "parasitic_correction" not in plot_args
    plot_args.setdefault('marker', '')
    plot_args.setdefault('ls', '--')
    assert 'prefix' not in plot_args
    # noinspection PyUnresolvedReferences
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
    with interactive_lock:
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
    with interactive_lock:
        plt.close(fig)


def plot_depletion_delegate(data_group: tb.Group, analysis_group: GroupType, output_pdf: PdfPages):
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
    print("Called the pixel depletion plotter!")
    if np.all(np.isfinite(depletion_width_plate[i_col, i_row])):
        condition = """(row == {}) & (col == {})""".format(i_row, i_col)
        for x in table.where(condition):
            depletion_fit_propagate_parameters = {p_key: x[p_key] for p_key in ["NAD", "V", "dep", "sat"]}
            break
        else:
            depletion_fit_propagate_parameters = {}
        doping_acceptor = depletion_fit_propagate_parameters["NAD"]
        effective_doping = effective_doping_table[i_col, i_row]
        with interactive_lock:
            fig, ax = plt.subplots(3)
        logger.info("The type of bias_voltages is %s", type(bias_voltages))
        bias_mask = bias_voltages < -0.5
        if np.all(np.isfinite(depletion_width_plate_error[i_col, i_row])):
            ax[0].errorbar(bias_voltages[bias_mask], depletion_width_plate[i_col, i_row][bias_mask],
                           yerr=depletion_width_plate_error[i_col, i_row][bias_mask], label='d-measurement')
        else:
            ax[0].plot(bias_voltages[bias_mask], depletion_width_plate[i_col, i_row][bias_mask], label='d-measurement')

        # noqa: S125
        # sample_voltage = -1 * np.linspace(np.min(-bias_voltages), np.max(-bias_voltages) * 1.1, 1000)
        sample_voltage = np.linspace(np.min(bias_voltages[bias_mask]) * 1.1, np.max(bias_voltages[bias_mask]) / 1.1,
                                     1000)
        ax[0].plot(sample_voltage, model_depletion(sample_voltage, **depletion_fit_propagate_parameters),
                   label=f'd-theory for NAD = {doping_acceptor:4.2f}  and Ubi = {depletion_fit_propagate_parameters["V"]:.2f}')
        ax[0].set(xlabel='Bias Voltage [V]', ylabel='Depletion Width [µm]',
                  title=f"Analysis of the depletion width for pixel ({i_col}, {i_row}).")
        ax[0].grid(True)
        ax[0].legend(
            title=f"Saturating at {depletion_fit_propagate_parameters['dep']} with {depletion_fit_propagate_parameters['sat']} saturation.")
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
        with interactive_lock:
            plt.close(fig)


if __name__ == '__main__':
    # plot_data(interpreted_data=os.path.expanduser('~/git/pixcap65/pixcap_LF_50x50_DC_R3_80V_HV.h5'))

    # some usage examples
    from pixcap65.utility.homogenize_plots import set_params

    logging.basicConfig(level=logging.INFO)
    # plot_bias_data(interpreted_data="3D_Sensor_221_Scan.h5", base_path="Thesis/ATLAS_ITk/X3/I_V_Characteristic",
    #                use_group=True)
    # plot_bias_data(interpreted_data="3D_Sensor_221_W13_X_Scan.h5", base_path="Thesis/ATLAS_ITk/X3/I_V_Characteristic",
    #                use_group=True)
    # plot_combined_data(interpreted_data="3D_Sensor_221_W13_X_Scan.h5",
    #                    base_path="Thesis/ATLAS_ITk/X3/C_V_Characteristic", use_group=True)
    # plot_bias_data(interpreted_data="3D_Sensor_221_W6_j_Scan.h5", base_path="Thesis/ATLAS_ITk/X5/I_V_Characteristic",
    #                use_group=True)
    # plot_combined_data(interpreted_data="3D_Sensor_221_W6_j_Scan.h5",
    #                    base_path="Thesis/ATLAS_ITk/X5/C_V_Characteristic", use_group=True)
    try:
        from subprocess import run

        run_result = run(['pdflatex', '--version'], check=True, capture_output=True)
        has_latex = True
        logger.info("The latex compiler to use is: %s", run_result.stdout.decode("utf-8"))
    except (FileNotFoundError, ImportError):
        # proceed as if no latex exists
        logger.exception("Could not verify whether latex exists.")
        has_latex = False
    set_params(latex=False,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                           r"stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                           r"retain-zero-uncertainty}", fig_height=8.26772, fig_width=11.69291, )

    # some test evaluations
    # plot_data(interpreted_data=REFERENCE_TEST_FILE, base_path="Reference/TESTS/unbiased_30", use_group=True,
    #           exclude_test_cap=True)
    # plot_data(interpreted_data=REFERENCE_TEST_FILE, base_path="Reference/TESTS/unbiased_31", use_group=True,
    #           exclude_test_cap=True)

    # second try R13

    # some placeholder!

    # second try X1
    while True:
        try:
            with tb.open_file(X1_SCAN_2_FILE, mode="r") as h5_file:
                break
        except FileNotFoundError:
            raise
        except:
            print("Will try to access again, later.")
            time.sleep(120)




    print("Plot X1 Second Try.")
    x1_second_pixel_mask = [[39, 39], [38, 39]]
    # plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="ATLAS_ITk/X1/unbiased_61_full", use_group=True,
    #           exclude_test_cap=True, mask_pixel=x1_second_pixel_mask, distribution=True)
    # plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="ATLAS_ITk/X1/unbiased_61_full", use_group=True,
    #           exclude_test_cap=True, use_corrected=True, mask_pixel=x1_second_pixel_mask, distribution=True)
    # plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="ATLAS_ITk/X1/biased_80_V_full", use_group=True,
    #           exclude_test_cap=True, mask_pixel=x1_second_pixel_mask, distribution=True)
    # plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="ATLAS_ITk/X1/biased_80_V_full", use_group=True,
    #           exclude_test_cap=True, use_corrected=True, mask_pixel=x1_second_pixel_mask, distribution=True)
    # plot_bias_data(interpreted_data=X1_SCAN_2_FILE, base_path="ATLAS_ITk/X1/I_V_Characteristic", use_group=True, )
    plot_combined_data(interpreted_data=X1_SCAN_2_FILE, base_path="ATLAS_ITk/X1/C_V_Characteristic_refined",
                       use_group=True, mask_pixel=x1_second_pixel_mask, distribution=True)
    plot_combined_data(interpreted_data=X1_SCAN_2_FILE, base_path="ATLAS_ITk/X1/C_V_Characteristic_refined",
                       use_group=True, use_corrected=True,
                       apply_doping=False, distribution=True, mask_pixel=x1_second_pixel_mask)
    # plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE, base_path="Thesis/ATLAS_ITk/X1/inter_unbiased_full",
    #                     use_group=True, exclude_test_cap=True, distribution=True,
    #                     total_data=X1_SCAN_2_FILE, total_path="ATLAS_ITk/X1/unbiased_61_full")
    # plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE, base_path="Thesis/ATLAS_ITk/X1/inter_biased_M_80_V_full",
    #                     use_group=True, exclude_test_cap=True, distribution=True,
    #                     total_data=X1_SCAN_2_FILE, total_path="ATLAS_ITk/X1/unbiased_61_full")

    plot_combined_data(interpreted_data=X1_SCAN_2_FILE, base_path="Thesis/ATLAS_ITk/X1/C_V_Characteristic_Second_Extended",
                       use_group=True, mask_pixel=x1_second_pixel_mask, distribution=True)
    plot_combined_data(interpreted_data=X1_SCAN_2_FILE, base_path="Thesis/ATLAS_ITk/X1/C_V_Characteristic_Second_Extended",
                       use_group=True, use_corrected=True,
                       apply_doping=False, distribution=True, mask_pixel=x1_second_pixel_mask)

    # REMARK: This is not written back by now to the full analysis script.
    # E1 first run
    # print("Plot E1")
    # e1_pixel_mask = [[39, 1]]
    # plot_data(interpreted_data=E1_SCAN_FILE, base_path="Reference/E1/unbiased_4_full", use_group=True,
    #           exclude_test_cap=True, mask_pixel=e1_pixel_mask, )
    # plot_data(interpreted_data=E1_SCAN_FILE, base_path="Reference/E1/unbiased_4_full", use_group=True,
    #           use_corrected=True, distribution=True, exclude_test_cap=True, mask_pixel=e1_pixel_mask, )
    # plot_bias_data(interpreted_data=E1_SCAN_FILE, base_path="Reference/E1/I_V_Characteristic",
    #                use_group=True)
    # plot_combined_data(interpreted_data=E1_SCAN_FILE, base_path="Reference/E1/C_V_Characteristic",
    #                    use_group=True, distribution=True)
    # plot_combined_data(interpreted_data=E1_SCAN_FILE, base_path="Reference/E1/C_V_Characteristic",
    #                    use_group=True,
    #                    use_corrected=True,
    #                    apply_doping=False, distribution=False)
    # plot_data(interpreted_data="packaged/E1_Renew_Scan.h5", use_group=True, base_path="Reference/E1/unbiased_full",
    #           exclude_test_cap=True,
    #           mask_pixel=e1_pixel_mask,)
    # plot_data(interpreted_data="packaged/E1_Renew_Scan.h5", use_group=True, base_path="Reference/E1/unbiased_full",
    #           exclude_test_cap=True,
    #           mask_pixel=e1_pixel_mask, use_corrected=True)
    # plot_data(interpreted_data="packaged/E1_Renew_Scan.h5", use_group=True, base_path="Reference/E1/biased_80_V_full",
    #           mask_pixel=e1_pixel_mask, exclude_test_cap=True)
    # plot_data(interpreted_data="packaged/E1_Renew_Scan.h5", use_group=True, base_path="Reference/E1/biased_80_V_full",
    #           mask_pixel=e1_pixel_mask, use_corrected=True, exclude_test_cap=True)
    # plot_combined_data(interpreted_data="packaged/E1_Renew_Scan.h5",
    #                    base_path="Reference/E1/C_V_Characteristic_refined",
    #                    use_group=True, distribution=True, mask_pixel=e1_pixel_mask,)
    # plot_combined_data(interpreted_data="packaged/E1_Renew_Scan.h5",
    #                    base_path="Reference/E1/C_V_Characteristic_refined",
    #                    use_group=True, distribution=True, mask_pixel=e1_pixel_mask, use_corrected=True)
    # plot_bias_data(interpreted_data="packaged/E1_Renew_Scan.h5", base_path="Reference/E1/I_V_Characteristic",
    #                use_group=True)
    # plot_inter_pix_data(interpreted_data="packaged/E1_Renew_Scan.h5", base_path="Reference/E1/inter_unbiased_full",
    #                     use_group=True, total_data="packaged/E1_Renew_Scan.h5", total_path="Reference/E1/unbiased_full")
    # plot_inter_pix_data(interpreted_data="packaged/E1_Renew_Scan.h5", base_path="Reference/E1/inter_biased_M_80_V_full",
    #                     use_group=True, total_data="packaged/E1_Renew_Scan.h5", total_path="Reference/E1/biased_80_V_full")



    print("Plot Presentable")
    # noqa: S125
    # examples
    # plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data",
    #           use_group=False)
    # plot_inter_pix_data(interpreted_data='R13-Interpixel_Scan.h5',
    #                     base_path="Reference/R13/demo_measurement_65_unbiased_1_discharge",
    #                     use_group=True, suffix="inter_pix_65", total_data='Reference_R13_Scan.h5',
    #                     distribution=True, set_parasitic=False, total_path="Reference/R13/unbiased_12_full")
    # plot_bias_data(interpreted_data='Data/r13-measurement/R13_BIAS_2.h5')
    # plot_combined_data(interpreted_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', first_lower=-100,first_upper=-40, second_lower=-10, second_upper=0)

    # collect all our IV groups
    iv_file_names = [
        'pixcap65/Data/r13-measurement/R13_BIAS_2.h5',
        'pixcap65/Data/New_1_Initial_6_Scan.h5',
        X2_SCAN_FILE,
        X1_SCAN_2_FILE,
        "packaged/E1_Renew_Scan.h5",
        X2_SCAN_2_FILE,
        X2_SCAN_2_FILE,
        X1_SCAN_2_FILE,
        "packaged/R13_Renew_Scan.h5",
        X2_SCAN_FILE,
        X2_SCAN_FILE,
        'pixcap65/Data/New_1_Initial_6_Scan.h5',
    ]
    iv_group_names = [
        None,
        "ATLAS ITk/I_V_Characteristic",
        "ATLAS_Itk/X2/I_V_Characteristic",
        "ATLAS_ITk/X1/I_V_Characteristic",
        "Reference/E1/C_V_Characteristic_refined",
        "Thesis/ATLAS_ITk/X2/I_V_Characteristic_2",
        "ATLAS_ITk/X2/C_V_Characteristic_refined",
        "ATLAS_ITk/X1/C_V_Characteristic_refined",
        "Reference/R13/C_V_Characteristic_refined",
        "ATLAS_Itk/X2/C_V_Characteristic",
        "ATLAS_Itk/X2/C_V_Characteristic_refined",
        "ATLAS ITk/C_V_Characteristic",
    ]
    iv_labels = [
        'R13',
        "(HPK) X1",
        "(HPK) X2",
        "X1 (second)",
        "E1 (second, CV)",
        "X2 (third?)",
        "X2 (second, CV)",
        "X1 (second, CV)",
        "R13 (second, CV)",
        "X2 (CV)",
        "X2 (CV, refined)",
        "X1 (CV)",
    ]
    normalisation = [64*64, 384*400, 384*400, 384*400] * 30 * 30
    plot_bias_data(iv_file_names, iv_group_names, pdf_name="I-V Combination.pdf",
                   labels=["Bias Data for {}".format(item) for item in iv_labels])
    plot_bias_data(iv_file_names, iv_group_names, pdf_name="I-V Combination-2.pdf",
                   labels=["Bias Data for {}".format(item) for item in iv_labels], area_normalisation=normalisation)
    print("CV Combination")
    cv_file_names = [
        X2_SCAN_FILE,
        X2_SCAN_FILE,
        X2_SCAN_2_FILE,
        X1_SCAN_2_FILE,
        "packaged/R13_Renew_Scan.h5",
        X1_SCAN_2_FILE,
        "packaged/E1_Renew_Scan.h5",
        E1_SCAN_FILE,
        'pixcap65/Data/New_1_Initial_6_Scan.h5',
    ]
    cv_groups = [
        "ATLAS_Itk/X2/C_V_Characteristic",
        "ATLAS_Itk/X2/C_V_Characteristic_refined",
        "ATLAS_ITk/X2/C_V_Characteristic_refined",
        "ATLAS_ITk/X1/C_V_Characteristic_refined",
        "Reference/R13/C_V_Characteristic_refined",
        "Thesis/ATLAS_ITk/X1/C_V_Characteristic_Second_Extended",
        "Reference/E1/C_V_Characteristic_refined",
        "Reference/E1/C_V_Characteristic",
        "ATLAS ITk/C_V_Characteristic",
    ]
    cv_labels = [
        'X2_1_1',
        "X2_1_2",
        "X2_2",
        "X1 (second)",
        "R13 (second)",
        "X1 (second, extended)",
        "E1 (second)",
        "E1",
        "X1",
    ]
    plot_cv_data(cv_file_names, cv_groups,
                 pdf_name="C-V Combination.pdf", labels=[CV_DATA_FOR_.format(item) for item in cv_labels],
                 use_corrected=True)
    # print("CV Combination R13 only")
    # plot_cv_data([REFERENCE_TEST_FILE, REFERENCE_TEST_FILE],
    #              ["Reference/TESTS/cv_only_simple", "Reference/TESTS/cv_only_advanced"],
    #              pdf_name="C-V Combination 2.pdf", labels=[CV_DATA_FOR_.format(item) for item in ['simple', "advanced"]],
    #              use_corrected=True)
    # print("CV Combination R13 combined")
    # plot_cv_data([REFERENCE_TEST_FILE, REFERENCE_TEST_FILE],
    #              ["Reference/TESTS/cv_combined_simple", "Reference/TESTS/cv_combined_advanced"],
    #              pdf_name="C-V Combination 3.pdf",
    #              labels=[CV_DATA_FOR_.format(item) for item in ['simple', "advanced"]],
    #              use_corrected=True)
