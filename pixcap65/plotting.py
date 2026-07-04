"""
Plotting of Pixcap65 data.
"""
# ----------------------------------------------------------
#  Copyright (c) 2018-2026. All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------

# TODO: update the documentation of the implementations

import os.path

import logging
import threading
from contextlib import contextmanager
from matplotlib.axes import Axes
from types import NoneType
from typing import Any, Optional, Union, List, Tuple

from pixcap65.analysis_util.physics_modelling import model_depletion
from pixcap65.utility import synchronized_process_open_file, tables_lock as file_access_lock
from pixcap65.utility.homogenize_plots import enhanced_error_bar
from pixcap65.utility.tables_util import group_get_file

try:
    # noinspection PyCompatibility
    from collections.abc import Iterable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Iterable

import numpy as np
import tables as tb
from matplotlib import pyplot as plt, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pixcap65.analysis_util.utility import check_leaf_unit, GENERAL_PIXCAP_SHAPE, HIST_BIAS_MEAS_UNIT, \
    HIST_LEAK_CURRENT_UNIT, \
    HIST_CAP_UNIT, HIST_CURRENT_MEAS_UNIT, extract_parasitic_capacitance, CURRENT_CONVERSION_FACTOR, get_base_group, \
    get_analysis_group, TABLES_LEAF_COMPAT_TYPE, CVDistributionData
from pixcap65.utility.utils_2 import GroupType

HISTOGRAM_SHAPE_FORMAT = "The histograms shape is {}"
ROW_LABEL = 'Row'
COLUMN_LABEL = 'Column'
HIST_PIX_CAP_LABEL = '$C$ / \\unit{{\\femto\\farad}}'
COUNTS_HIST_LABEL = 'Counts / \\#'
SIMPLE_CAP_LABEL_PERCENT_FORMAT = '%sFit to data:\n$C_d = %.1f\\,$fF'
FREQUENCY_LABEL = '$f$ / \\unit{{\\mega\\hertz}}'
CURRENT_LABEL = '$I$ / \\unit{{\\nano\\ampere}}'
SIMPLE_PIXEL_LABEL = '{prefix}Pixel({i_col},{i_row})'
CAPACITANCE_CONVERSION_FACTOR = 1e15
ADVANCED_CAPACITANCE_CONVERSION_FACTOR = 1.0e-9
cmap = plt.get_cmap('viridis')
BIAS_CURVE_Y_LABEL = "$I$ / \\unit{{\\nano\\ampere}}"
BIAS_CURVE_X_LABEL = "$U$ / \\unit{{\\volt}}"
DEFAULT_BIN_NUMBER = 50
DEFAULT_TEST_CAP_EXCLUSION = True
logger = logging.getLogger(__name__)
NUMBER_DEPLETION_PLOT_POINTS = 1000
CAPACITANCE_LABEL = "$C$ / \\unit{{\\femto\\farad}}"
E1_SCAN_FILE = "Reference_Evelyn_Scan.h5"
X2_SCAN_FILE = 'New_2_Scan.h5'
X1_SCAN_2_FILE = "packaged/data/X1_4_Renew_Scan.h5"
REFERENCE_TEST_FILE = "packaged/Reference_Demo.h5"
CV_DATA_FOR_ = "CV Data for {}"
NEW_PLOT_FILE_MODE = True
GENERATE_THESIS_PLOTS = True
CV_USE_SEPARATE_PAGES = True

global_interactive_lock = threading.RLock()

replacement_order = {}
exclusion_list = ["Scan", "Full"]


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
    if NEW_PLOT_FILE_MODE:
        full_path = os.path.abspath(interpreted_data)
        directory, file = os.path.split(full_path)
        name, ext = os.path.splitext(file)
        file_components = name.split("_")
        file_components.append(ext)
        if file_components[0] == "3D":
            indices = [file_components.index(x) for x in exclusion_list if x in file_components]
            finish_idx = min(indices)
            intermediate_dir = '_'.join(file_components[2:finish_idx])
            intermediate_dir = replacement_order.get(intermediate_dir, intermediate_dir)
        else:
            intermediate_dir = file_components[0]

        if not os.path.isdir(os.path.join(directory, intermediate_dir)):
            os.makedirs(os.path.join(directory, intermediate_dir), exist_ok=True)
        file_mode = os.path.join(directory, intermediate_dir, file[:-3])
    else:
        file_mode = interpreted_data[:-3]

    if use_group and base_path is not None:
        _, group_component = os.path.split(base_path)
        pdf_name = "{file}_{s}_{group}.pdf".format(s=suffix, file=file_mode, group=group_component)
    else:
        pdf_name = "{file}_{s}.pdf".format(s=suffix, file=file_mode)
    return pdf_name

@contextmanager
def advanced_figure_provider(lock, output=None, **kwargs):
    with lock:
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
    yield fig, ax
    if output is not None:
        output.savefig(fig, **kwargs)
    with lock:
        plt.close(fig)

@contextmanager
def figure_provider(lock, *args, separate_plots=False, output=None, callback=None, **kwargs):
    call_all = kwargs.pop("call_all", False)
    back_inform = {'output': True}
    with lock:
        if separate_plots:
            args = list(args)
            try:
                n_rows = args.pop(0)
            except IndexError:
                n_rows = 1
            try:
                n_cols = args.pop(0)
            except IndexError:
                n_cols = 1
            n_rows = kwargs.pop('nrows', n_rows)
            n_cols = kwargs.pop('ncols', n_cols)
            fig = []
            ax = []
            for _ in range(n_rows * n_cols):
                temp_fig, temp_ax = plt.subplots(*args, **kwargs)
                ax.append(temp_ax)
                fig.append(temp_fig)

            if n_rows > 1 and n_cols > 1:
                ax = np.array(ax).reshape(n_rows, n_cols)
            else:
                ax = np.array(ax)
        else:
            fig, ax = plt.subplots(*args, **kwargs)

    yield fig, ax, back_inform
    figures = np.atleast_1d(fig)
    if callback is not None:
        if call_all:
            for figure in figures:
                callback(figure)
        else:
            callback(figures[0])

    if output is not None and back_inform['output']:
        for figure in figures:
            output.savefig(figure, bbox_inches='tight')
    with lock:
        for figure in np.atleast_1d(fig):
            plt.close(figure)

def plot_data(interpreted_data, base_path=None, suffix="general_data", use_group=False, lock=file_access_lock, **kwargs):
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
        with synchronized_process_open_file(interpreted_data, mode='r', lock=lock) as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            plot_data_delegate(base_group.total_cap.measurements, get_analysis_group(base_group.total_cap, **kwargs),
                               output_pdf, **kwargs)

@contextmanager
def inter_pix_data_fetch(path, group, lock, active_file: tb.File, type_name: str='total_cap'):
    if path is None or not os.path.exists(path):
        yield None
    elif os.path.abspath(path) == os.path.abspath(active_file.filename):
        total_base_group = get_base_group(group, active_file)
        yield total_base_group[type_name].analysis

    elif os.path.exists(path):
        with synchronized_process_open_file(path, mode='r', lock=lock) as total_file_h5:
            total_base_group = get_base_group(group, total_file_h5)
            yield total_base_group[type_name].analysis


def plot_inter_pix_data(interpreted_data, base_path=None, suffix="general_inter_pix_data", use_group=False,
                        total_path=None, total_data=None, inter_path=None, inter_data=None, lock=file_access_lock, **kwargs):
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
    :param inter_data: path to the hdf file which holds the in-pix measurement
    :param inter_path: hdf group path inside the hdf file containing the in-pix measurement
    :key exclude_test_cap: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key extract_pixel: iterable of pixel positions on the grid to extract the figures from.
    """
    # determine the pdf file
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    with PdfPages(pdf_name) as output_pdf:
        with synchronized_process_open_file(interpreted_data, mode='r', lock=lock) as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            # TODO: handle the in-pix provision data-group
            # we need to fetch the correcponding group
            with inter_pix_data_fetch(total_data, total_path, lock, in_file_h5) as total_group,\
                inter_pix_data_fetch(inter_data, inter_path, lock, in_file_h5, 'inter_cap') as inter_group:

                plot_inter_pix_data_delegate(base_group.inter_cap.measurements,
                                             get_analysis_group(base_group.inter_cap, **kwargs), output_pdf,
                                             total_group, inter_group, **kwargs)


            # if total_data is None or not os.path.exists(total_data):
            #     plot_inter_pix_data_delegate(base_group.inter_cap.measurements,
            #                                  get_analysis_group(base_group.inter_cap, **kwargs), output_pdf,
            #                                  **kwargs)
            #
            # elif os.path.abspath(total_data) == os.path.abspath(in_file_h5.filename):
            #     total_base_group = get_base_group(total_path, in_file_h5)
            #     plot_inter_pix_data_delegate(base_group.inter_cap.measurements,
            #                                  get_analysis_group(base_group.inter_cap, **kwargs), output_pdf,
            #                                  total_base_group.total_cap.analysis,
            #                                  **kwargs)
            # elif os.path.exists(total_data):
            #     with synchronized_process_open_file(total_data, mode='r', lock=lock) as total_file_h5:
            #         total_base_group = get_base_group(total_path, total_file_h5)
            #         # it will choose always the uncorrected data for the reference.
            #         plot_inter_pix_data_delegate(base_group.inter_cap.measurements,
            #                                      get_analysis_group(base_group.inter_cap, **kwargs), output_pdf,
            #                                      total_base_group.total_cap.analysis,
            #                                      **kwargs)


@contextmanager
def multi_sensor_file_handler_simple(interpreted_data, base_path, **kwargs):
    pdf_name = kwargs.pop("pdf_name", "I-V-Collection.pdf")
    file_lock = kwargs.pop("lock", file_access_lock)
    # to simplify the operation we need a mapping of a file to all the group_mapping it should be used for,
    # and we need a mapping to the opened files
    group_mapping = {}
    files = {}
    groups = []
    try:
        for file_path, group_name in zip(interpreted_data, base_path):
            if file_path not in group_mapping:
                current_file = synchronized_process_open_file(file_path, mode='r', lock=file_lock)
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
    file_lock = kwargs.pop("lock", file_access_lock)
    # to simplify the operation we need a mapping of a file to all the group_mapping it should be used for,
    # and we need a mapping to the opened files
    group_mapping = {}
    files = {}
    groups = []
    analysis_groups = []
    try:
        for file_path, group_name in zip(interpreted_data, base_path):
            if file_path not in group_mapping:
                current_file = synchronized_process_open_file(file_path, mode='r', lock=file_lock)
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

def plot_bias_data(interpreted_data, base_path=None, suffix="bias_curve", use_group=False, lock=file_access_lock, **kwargs):
    """
    plot_bias_data

    Plot the data acquired for the pixel-diodes I-V characterization.

    :param interpreted_data: path to the hdf file which holds the raw data.
    :param base_path: path within the files hierarchy for the base group.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    """
    if isinstance(interpreted_data, Union[List, Tuple, np.ndarray]):
        with multi_sensor_file_handler_simple(interpreted_data, base_path, lock=lock, **kwargs) as (groups, output_pdf):
            plot_bias_delegate(groups, output_pdf, **kwargs)
    else:
        pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
        with PdfPages(pdf_name) as output_pdf:
            with synchronized_process_open_file(interpreted_data, mode='r', lock=lock) as in_file_h5:
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
    file_lock = kwargs.get("lock", file_access_lock)
    if isinstance(interpreted_data, Iterable) and not isinstance(interpreted_data, str):
        with multi_sensor_file_handler_advanced(interpreted_data, base_path, **kwargs) as (groups, analysis_groups, output_pdf):
            plot_cv_data_delegate(groups, analysis_groups, output_pdf, **kwargs)
    else:
        pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
        with PdfPages(pdf_name) as output_pdf:
            with synchronized_process_open_file(interpreted_data, mode='a', lock=file_lock) as in_file_h5:
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

    file_lock = kwargs.get("lock", file_access_lock)
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    with PdfPages(pdf_name) as output_pdf:
        with synchronized_process_open_file(interpreted_data, mode='r', lock=file_lock) as in_file_h5:
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
    interactive_lock = kwargs.get("plotting_lock", global_interactive_lock)
    with rc_context(rc={'axes.prop_cycle': get_error_cycler()}), figure_provider(interactive_lock) as (fig, ax, _):
        if isinstance(data_group, Iterable) and not isinstance(data_group, tb.Node):
            labels = kwargs.pop("labels", ["Bias_data"] * len(data_group))
            norm_unit = "area_normalisation" in kwargs
            normalization = np.asarray(kwargs.pop("area_normalisation", np.full(len(data_group), 1.e8)), dtype=np.float64)
            normalization *= 1.0e-8
            if norm_unit and np.all(np.isclose(normalization, 1.0)):
                norm_unit = False
            print(labels)
            for group, label, norm in zip(data_group, labels, normalization):
                _bias_voltage_plotter(ax, group.BiasTable, label, norm=norm, apply_norm=norm_unit)
        else:
            tabular = data_group.BiasTable
            assert isinstance(tabular, tb.Table)
            _bias_voltage_plotter(ax, tabular, kwargs.pop("labels", "Bias data"))
        ax.legend()
        output_pdf.savefig(fig, bbox_inches='tight')


def _bias_voltage_plotter(ax, tabular: tb.Table, label, norm=1, apply_norm=False):
    voltage_data = np.abs(tabular.col("U"))
    current_data = np.abs(tabular.col("I"))
    current_errors = tabular.col("DI")
    # FIXME: R1/R11 seems to be missing i-v-errors!
    try:
        voltage_error = np.abs(tabular.col("DU"))
    except (AttributeError, KeyError):
        voltage_error = voltage_data * 0.0002 + 0.1
    if not np.all(np.isfinite(voltage_data)):
        current_errors = None
    if not GENERATE_THESIS_PLOTS:
        ax.set_title("Bias data from the measurement")
    ax.set(xlabel=BIAS_CURVE_X_LABEL, ylabel=BIAS_CURVE_Y_LABEL)
    # currently we could not use the correct voltage range, but we assume the errors to be within
    normalized_errors = None if current_errors is None else current_errors / norm
    if current_errors is None:
        from warnings import warn
        warn("The current sensor seems to be missing measurement uncertainties for the leakage current!")
    enhanced_error_bar(ax, voltage_data, current_data * CURRENT_CONVERSION_FACTOR / norm, xerr=voltage_error,
                       yerr=normalized_errors, label=label)
    # ax.errorbar(voltage_data, current_data * CURRENT_CONVERSION_FACTOR / norm, xerr=voltage_error,
    #             yerr=normalized_errors, fmt='o', label=label)
    if not np.isclose(norm, 1.0):
        ax.set_yscale('log')
        # FIXME: make the correct labels!
        # temp_current_label = ax.get_ylabel()
        # ax.set_ylabel(temp_current_label.replace("A ", "A / cm^2 "))
        ax.set_ylabel("I in \\unit{{\\nano\\ampere\\per\\centi\\meter\\squared}}")



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
    interactive_lock = kwargs.get('plotting_lock', global_interactive_lock)
    # extract the bias data
    voltage_data_sets = [__process_voltage_set(data_group)] if isinstance(data_group, tb.Node) else [__process_voltage_set(data_set) for data_set in data_group]
    approx_depletion = isinstance(data_group, tb.Node)

    labels = kwargs.pop('labels', [])
    # investigate all the pixel for plotting
    for ii, jj in np.ndindex(GENERAL_PIXCAP_SHAPE):
        response = lambda x: x.suptitle("C-V Characterization for Pixel ({}, {})".format(ii, jj))
        if GENERATE_THESIS_PLOTS:
            response = None
        with figure_provider(interactive_lock, ncols=2, callback=response, output=output_pdf, separate_plots=CV_USE_SEPARATE_PAGES) as (_, ax, back_pipe):
            # with interactive_lock:
            #     if CV_USE_SEPARATE_PAGES:
            #         fig_1, ax_1 = plt.subplots()
            #         fig_2, ax_2 = plt.subplots()
            #         ax = [ax_1, ax_2]
            #     else:
            #         fig, ax = plt.subplots(ncols=2)
            # will not only generate the title string of the figure but also the figure with the depletion fits.
            title_str = __plot_depletion_estimation(analysis_group, approx_depletion, ax, ii, jj, voltage_data_sets[0])

            # fixme:
            with rc_context(rc={'axes.prop_cycle': get_error_cycler()}):
                if jj == 40 or not _cv_plotter(analysis_group, jj, ii, ax, voltage_data_sets, labels):
                    # with interactive_lock:
                    #     if CV_USE_SEPARATE_PAGES:
                    #         plt.close(fig_1)
                    #         plt.close(fig_2)
                    #     else:
                    #         plt.close(fig)
                    back_pipe["output"] = False
                    continue

            ax[0].legend()
            ax[1].legend(title=title_str)
            ax[1].grid(True)
            if kwargs.get("use_log", False):
                ax[0].set_yscale('log')
                ax[1].set_yscale('log')

            # if CV_USE_SEPARATE_PAGES:
            #     fig_1.suptitle("C-V Characterization for Pixel ({}, {})".format(ii, jj))
            #     output_pdf.savefig(fig_1, bbox_inches="tight")
            #     output_pdf.savefig(fig_2, bbox_inches="tight")
            # else:
            #     fig.suptitle("C-V Characterization for Pixel ({}, {})".format(ii, jj))
            #     output_pdf.savefig(fig, bbox_inches='tight')
            # with interactive_lock:
            #     if CV_USE_SEPARATE_PAGES:
            #         plt.close(fig_1)
            #         plt.close(fig_2)
            #     else:
            #         plt.close(fig)

        # Plot the doping analysis only for single-sensor samplings.
        if apply_doping and isinstance(data_group, tb.Group):
            # TODO: Refactor this part to support plotting for multiple sensors/data sets.
            depletion_width_plate = check_leaf_unit(analysis_group.DepletionWidth, "um")
            depletion_width_plate_error = check_leaf_unit(analysis_group.DepletionWidthErr, "um")
            effective_doping_table = check_leaf_unit(analysis_group.DepletionEffDoping, "cm^-3")
            resistivity_table = check_leaf_unit(analysis_group.DepletionResitivity, "Ocm")
            origin_bias_voltages = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
            if len(origin_bias_voltages.shape) > 1:
                # TODO: better use the actual voltages here.
                bias_voltages = origin_bias_voltages[:, 0]
            else:
                bias_voltages = origin_bias_voltages
            table = analysis_group.DepletionParamTable
            plot_depletion_pixel_delegate(bias_voltages, ii, depletion_width_plate, depletion_width_plate_error,
                                          effective_doping_table, output_pdf, jj, table, resistivity_table)

    if not (kwargs.pop("distribution", False) and True):
        return

    # Why is this part here still not ready for multiple sensors?
    assert isinstance(voltage_data_sets[0], Iterable)
    def iterator_filter(item):
        return voltage_data_sets[0].shape[0] < 10 or item[0] % 10 == 0
    # first prepare the datasets
    combiner = True
    if isinstance(analysis_group, tb.Group):
        group_handle = [analysis_group]
        label_handle = ["capacitance data"]
        combiner = False
        iterator = filter(iterator_filter, enumerate(voltage_data_sets[0]))
        for k, bias_voltage in iterator:
            with figure_provider(interactive_lock, output=output_pdf) as (fig, ax, _):
                # TODO: we should use here the dynamic binning used at any other point to!
                if not GENERATE_THESIS_PLOTS:
                    ax.set_title("Capacitance distribution for bias voltage {}".format(bias_voltage))
                ax.set(xlabel=CAPACITANCE_LABEL)
                ax.hist(analysis_group.UCHist[:, :, k].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR, bins=50)
        # fig, ax = plt.subplots(ncols=2)
        # fig = [fig]
    else:
        group_handle = analysis_group
        n_items = len(analysis_group)
        label_handle = labels if len(labels) == n_items else ['?'] * n_items
        # FIXME: Additional swap of the actual dimensions here?
        # fig_cv, ax_cv = plt.subplots()
        # fig_dep, ax_dep = plt.subplots()
        # ax = [ax_cv, ax_dep]
        # fig = [fig_cv, fig_dep]

    title_str = ""
    def __callback_handler(fig):
        fig.suptitle(title_str)

    with figure_provider(interactive_lock, output=output_pdf, ncols=2, separate_plots=combiner, call_all=True,
                         callback=None if GENERATE_THESIS_PLOTS else __callback_handler) as (fig, ax, back_pipe):
        x_limits, y_limits = None, None
        for ana_group, label in zip(group_handle, label_handle):
            if "CVDistribution" not in ana_group:
                continue
            x_limits, y_limits, title_str = _plot_cv_distribution(ana_group, ax, x_limits, y_limits, label=label, is_combining=combiner, **kwargs)

        if x_limits is not None:
            assert isinstance(x_limits, (tuple, list, set))
            assert isinstance(y_limits, (tuple, list, set))
            if x_limits[0] < 0:
                x_limits = (-x_limits[1], x_limits[0])
            ax[1].set_xlim(*x_limits)
            ax[1].set_ylim(*y_limits)
            ax[1].legend()
            ax[0].legend()
            # for figure in fig:
            #     if not combiner:
            #         figure.suptitle(title_str)
            #     output_pdf.savefig(figure, bbox_inches='tight')
        else:
            back_pipe["output"] = False
        # for figure in fig:
        #     plt.close(figure)

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
            first_voltage_x = np.linspace(np.min(voltage_data) - 10, dep_voltage_2 / 1.1,
                                          NUMBER_DEPLETION_PLOT_POINTS)
            second_voltage_x = np.linspace(dep_voltage_2 * 1.1, np.max(voltage_data) + 10,
                                           NUMBER_DEPLETION_PLOT_POINTS)
            first_cap_calc = depletion_fit_a[dep_idx] * first_voltage_x + depletion_fit_b[dep_idx]
            second_cap_calc = depletion_fit_c[dep_idx] * second_voltage_x + depletion_fit_d[dep_idx]
            # skip plotting of this functions if the multiple C-V- is plotted
            if not is_combining:
                ax[1].plot(-first_voltage_x, first_cap_calc, '-', label="First section fit", marker=None)
                ax[1].plot(-second_voltage_x, second_cap_calc, '-', label="Second section fit", marker=None)
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
    try:
        x_limits, y_limits = __cv_plot_instance(ax, voltage_data, cap_data, cap_data_errors, label, title_format, x_limits, y_limits)
    except:
        print(depletion_data)
        print(group.CVDistribution.dtype)
        print(group_get_file(group).filename)
        raise
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
            depletion_fit_data_temp = depletion_fit_data.reshape((40, 41, 1, 4))
            depletion_fit_data = depletion_fit_data_temp
            depletion_hist = depletion_hist.reshape((40, 41, 1))

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
            ax[1].plot(-first_voltage_x, first_cap_calc, '-', label="First section fit", marker=None)
            ax[1].plot(-second_voltage_x, second_cap_calc, '-', label="Second section fit", marker=None)
            ax[1].vlines(-dep_voltage_2, 0, 1, linestyles="dashed")
            # TODO: What about the covariance matrix here!
            title_str += "U = {} V\n".format(dep_voltage_2)
    return title_str

def _cv_plotter(analysis, row, col, ax, voltage_data_sets, labels, **kwargs):
    title_format = "pixel ({col},{row})".format(col=col, row=row)
    # it should be quite simply to combine this two implementation branches into just a single one!
    # first get data iterators from the group iterators!
    if len(labels) == 0:
        labels = ["Bias Data"]

    analysis = np.atleast_1d(analysis)

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
        if len(voltage_data.shape) > 1:
            # FIXME: this might lead to biased results!
            voltage_data = voltage_data[:, 0]
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

    if x_limits[0] < 0:
        x_limits = (-x_limits[1], -x_limits[0])
    ax[1].set_xlim(*x_limits)
    ax[1].set_ylim(*y_limits)
    return True



def __cv_plot_instance(ax, voltage_data: np.ndarray, cap_data: np.ndarray, cap_data_errors: np.ndarray, label: str,
                       title_format, x_limits, y_limits) -> tuple[Iterable, Iterable]:
    effective_capacitance_error_data = np.reciprocal(cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 3 * cap_data_errors * CAPACITANCE_CONVERSION_FACTOR if np.all(np.isfinite(cap_data_errors)) else None
    eff_cap_errors = cap_data_errors * CAPACITANCE_CONVERSION_FACTOR if np.all(np.isfinite(cap_data_errors)) else None
    if not GENERATE_THESIS_PLOTS:
        ax[0].set_title("Bias data from the \nmeasurement for {}".format(title_format))
        ax[1].set_title("Suited Bias data from the \nmeasurement for {}".format(title_format))
    ax[0].set(xlabel=BIAS_CURVE_X_LABEL,
              ylabel=CAPACITANCE_LABEL)
    enhanced_error_bar(ax[0], -voltage_data, cap_data * CAPACITANCE_CONVERSION_FACTOR, yerr=eff_cap_errors, label=label)
    ax[1].set(xlabel=BIAS_CURVE_X_LABEL, ylabel="$1/ C^2$ / \\unit{{\\per\\femto\\farad\\squared}}")
    enhanced_error_bar(ax[1], -voltage_data, 1 / (cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 2, yerr=effective_capacitance_error_data, label=label, alpha=0.5)
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

def plot_1d_distribution(data: np.ndarray, label: str, bias_code: int, table: Optional[tb.Table], pdf, group: tb.Group, **kwargs):
    unit = kwargs.pop("unit", "F")
    interactive_lock = kwargs.get('plotting_lock', global_interactive_lock)
    with advanced_figure_provider(interactive_lock) as (fig, ax):
        hist_cap_hist = evaluate_pixel_mask(data, **kwargs)
        ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                bins=kwargs.get("hist_bins", DEFAULT_BIN_NUMBER))
        ax.set_ylabel(COUNTS_HIST_LABEL)
        ax.set_xlabel(HIST_PIX_CAP_LABEL)
        if not GENERATE_THESIS_PLOTS:
            ax.set_title(__get_1d_hist_label(bias_code, label, table, unit=unit))
        ax.grid()
        pdf.savefig(fig, bbox_inches='tight')
    if kwargs.pop("distribution", False):
        from pixcap65.analysis import analyze_capacitance_distribution_delegate

        analyze_capacitance_distribution_delegate(group, pdf, set_parasitic=False, **kwargs)

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
    interactive_lock = kwargs.get('plotting_lock', global_interactive_lock)
    # Read pixel map
    current_hist = check_leaf_unit(data_group.HistCurr, HIST_CURRENT_MEAS_UNIT)
    current_err_hist = check_leaf_unit(data_group.HistCurrErr, HIST_CURRENT_MEAS_UNIT)
    cap_hist = check_leaf_unit(analysis_group.HistCap, HIST_CAP_UNIT)
    leak_hist = check_leaf_unit(analysis_group.HistLeak, HIST_LEAK_CURRENT_UNIT)

    # Read scan parameters
    scan_parameters = data_group.scan_params[:]

    # 2D Pixel Capacitance Hist
    plot_2d_capacitance(cap_hist, "Capacitance Distribution", output_pdf)

    if kwargs.get("exclude_test_cap", False):
        # another heat map which do not consider masked or boundary caps
        assert isinstance(cap_hist, np.ndarray)
        masked_cap_hist = cap_hist.copy()
        masked_cap_hist = evaluate_pixel_mask(masked_cap_hist, **kwargs)
        plot_2d_capacitance(masked_cap_hist, "Masked Pixel Capacitance distribution", output_pdf)

    # 1D Pixel Capacitance Hist
    distribution_table = analysis_group.DistResultfF if "DistResultfF" in analysis_group else None
    actual_unit = "fF"
    if distribution_table is None:
        actual_unit = "F"
        distribution_table = analysis_group.DistResult if "DistResult" in analysis_group else None

    plot_1d_distribution(cap_hist, "Total Cap Distribution", 20000, distribution_table, output_pdf, analysis_group, unit=actual_unit, **kwargs)

    # Current vs. frequency
    verify_pixel_mask = "mask_pixel" in kwargs and isinstance(kwargs["mask_pixel"], Iterable) and len(
        kwargs["mask_pixel"]) > 0
    for col, row in np.ndindex(current_hist.shape[:2]):
        if verify_pixel_mask and (col, row) in kwargs["mask_pixel"]:
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
                # TODO: extract the residues from the predictions!
                f_res, cap_pred = get_model_prediction(col, row, analysis_group, actual_cap, leak_hist, frequencies)
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
    if table is not None:
        temp_rec_result = [row[:] for row in
                           table.where("""(bias == {})""".format(bias_code))]
        try:
            data_rec_result = np.rec.array(temp_rec_result,
                                       dtype=tb.dtype_from_descr(CVDistributionData(),))
            uncorrected_label = "\nC=({:.2f}+-{:.2f}+-{:.2f}+-{:.2f}) {}".format(data_rec_result.capacitance[0],
                                                                data_rec_result.cap_std[0],
                                                                data_rec_result.cap_systematic_error[0],
                                                                data_rec_result.cap_systematic_dispersion[0], unit)
            corrected_label = "\nC_corr=({:.2f}+-{:.2f}+-{:.2f}+-{:.2f}) {}".format(data_rec_result.cap_corrected[0],
                                                                   data_rec_result.cap_corrected_err[0],
                                                                   data_rec_result.cap_systematic_error[0],
                                                                   data_rec_result.cap_systematic_dispersion[0], unit)
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
    interactive_lock = kwargs.get("plotting_lock", global_interactive_lock)
    if kwargs.get("exclude_test_cap", False):
        assert isinstance(data, np.ndarray)
        masked_cap_hist = data.copy()
        masked_cap_hist = evaluate_pixel_mask(masked_cap_hist, **kwargs)
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


def plot_inter_pix_data_delegate(data_group: tb.Group, analysis_group: tb.Group, output_pdf: PdfPages, total_group=None,
                                 inter_group=None, **kwargs):
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
    interactive_lock = kwargs.get("plotting_lock", global_interactive_lock)
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
    in_ref_cap_hist = None if inter_group is None else check_leaf_unit(inter_group.HistCap, HIST_CAP_UNIT)

    need_distribution = kwargs.get("distribution", False)
    # Read scan parameters
    scan_parameters = data_group.scan_params[:]

    # 2D Pixel Capacitance Hist
    plot_2d_capacitance(total_cap_hist, "Total Pixel Capacitance", output_pdf, **kwargs)
    if in_ref_cap_hist is not None:
        plot_2d_capacitance(total_cap_hist - in_ref_cap_hist, "Grouped Inter-Pixel Capacitance", output_pdf, **kwargs)
    if total_ref_cap_hist is not None:
        plot_2d_capacitance(total_ref_cap_hist-total_cap_hist, "Inter Pixel Capacitance from In-Pix C", output_pdf, **kwargs)
    plot_2d_capacitance(inter_a_cap_hist, "Inter-Pixel Capacitance A", output_pdf, **kwargs)
    plot_2d_capacitance(inter_b_cap_hist, "Inter-Pixel Capacitance B", output_pdf, **kwargs)

    # 1D Pixel Capacitance Hist
    distribution_result_data = analysis_group.DistResultfF if "DistResultfF" in analysis_group else None
    actual_unit = "fF"
    if distribution_result_data is None:
        actual_unit = "F"
        distribution_result_data = analysis_group.DistResult if "DistResult" in analysis_group else None
    n_bins = kwargs.get("hist_bins", DEFAULT_BIN_NUMBER)
    if np.count_nonzero(np.isfinite(total_cap_hist)) > 2:
        with advanced_figure_provider(interactive_lock) as (fig, ax):
            hist_cap_hist = evaluate_pixel_mask(total_cap_hist, **kwargs)
            ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                    bins=n_bins)
            ax.set_ylabel(COUNTS_HIST_LABEL)
            ax.set_xlabel(HIST_PIX_CAP_LABEL)
            title_str = "Pixel Total Capacitance Distribution"
            if not GENERATE_THESIS_PLOTS:
                ax.set_title(__get_1d_hist_label(10000, title_str, distribution_result_data, unit=actual_unit))
            ax.grid()
            output_pdf.savefig(fig, bbox_inches='tight')
        if need_distribution:
            from pixcap65.analysis import analyze_capacitance_distribution_delegate
            analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=total_cap_hist,
                                                      set_parasitic=False, **kwargs)

        if in_ref_cap_hist is not None:
            effective_inter_cap_hist = total_cap_hist - in_ref_cap_hist
            with advanced_figure_provider(interactive_lock) as (fig, ax):
                hist_inter_cap_hist = evaluate_pixel_mask(effective_inter_cap_hist, **kwargs)
                ax.hist(hist_inter_cap_hist[~np.isnan(hist_inter_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                        bins=n_bins)
                ax.set_ylabel(COUNTS_HIST_LABEL)
                ax.set_xlabel(HIST_PIX_CAP_LABEL)
                if not GENERATE_THESIS_PLOTS:
                    # FIXME: need to be named correctly!
                    ax.set_title(
                        __get_1d_hist_label(kwargs.get('grouped_inter_pix_id', 18000), "Pixel Inter Capacitance Distribution (Grouped)", distribution_result_data,
                                            unit=actual_unit))
                ax.grid()
                output_pdf.savefig(fig, bbox_inches='tight')
            if need_distribution:
                from pixcap65.analysis import analyze_capacitance_distribution_delegate
                analyze_capacitance_distribution_delegate(analysis_group, output_pdf,
                                                          capacitance=effective_inter_cap_hist,
                                                          set_parasitic=False, **kwargs)

        if total_ref_cap_hist is not None:
            effective_inter_cap_hist = total_ref_cap_hist - total_cap_hist
            with advanced_figure_provider(interactive_lock) as (fig, ax):
                hist_inter_cap_hist = evaluate_pixel_mask(effective_inter_cap_hist, **kwargs)
                ax.hist(hist_inter_cap_hist[~np.isnan(hist_inter_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                        bins=n_bins)
                ax.set_ylabel(COUNTS_HIST_LABEL)
                ax.set_xlabel(HIST_PIX_CAP_LABEL)
                if not GENERATE_THESIS_PLOTS:
                    ax.set_title(__get_1d_hist_label(14000, "Pixel Inter Capacitance Distribution", distribution_result_data, unit=actual_unit))
                ax.grid()
                output_pdf.savefig(fig, bbox_inches='tight')
            if need_distribution:
                from pixcap65.analysis import analyze_capacitance_distribution_delegate
                analyze_capacitance_distribution_delegate(analysis_group, output_pdf,
                                                            capacitance=effective_inter_cap_hist,
                                                            set_parasitic=False, **kwargs)

    if np.count_nonzero(np.isfinite(inter_a_current_hist)) > 2:
        with advanced_figure_provider(interactive_lock) as (fig, ax):
            hist_cap_hist = evaluate_pixel_mask(inter_a_current_hist, **kwargs)
            ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                    bins=n_bins)
            ax.set_ylabel(COUNTS_HIST_LABEL)
            ax.set_xlabel(HIST_PIX_CAP_LABEL)
            if not GENERATE_THESIS_PLOTS:
                ax.set_title(__get_1d_hist_label(11000, "Inter-Pixel A Capacitance Distribution", distribution_result_data, unit=actual_unit))
            ax.grid()
            output_pdf.savefig(fig, bbox_inches='tight')
        if need_distribution:
            from pixcap65.analysis import analyze_capacitance_distribution_delegate
            analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=inter_a_cap_hist,
                                                      set_parasitic=False, **kwargs)

    if np.count_nonzero(np.isfinite(inter_b_current_hist)) > 2:
        with advanced_figure_provider(interactive_lock) as (fig, ax):
            hist_cap_hist = evaluate_pixel_mask(inter_b_current_hist, **kwargs)
            ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                    bins=n_bins)
            ax.set_ylabel(COUNTS_HIST_LABEL)
            ax.set_xlabel(HIST_PIX_CAP_LABEL)
            if not GENERATE_THESIS_PLOTS:
                ax.set_title(__get_1d_hist_label(12000, "Inter-Pixel B Capacitance Distribution", distribution_result_data, unit=actual_unit))
            ax.grid()
            output_pdf.savefig(fig, bbox_inches='tight')
        if need_distribution:
            from pixcap65.analysis import analyze_capacitance_distribution_delegate
            analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=inter_b_cap_hist,
                                                      set_parasitic=False,
                                                      **kwargs)

    # Current vs. frequency (Will try to plot all into just one coordinate system)
    verify_mask_pixel = "mask_pixel" in kwargs and isinstance(kwargs["mask_pixel"], Iterable)
    for col, row in np.ndindex(total_current_hist.shape[:2]):
        if verify_mask_pixel and (col, row) in kwargs["mask_pixel"]:
            continue
        elif np.isfinite(total_current_hist[col, row, 0]):
            with advanced_figure_provider(interactive_lock) as (fig, ax):
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

def get_model_prediction(col, row, analysis_group: tb.Group, actual_cap: Any, total_leak_hist, f: np.ndarray=None,
                         resistor_name="HistRes", **plot_args):
    parasitic_correction = plot_args.pop('parasitic_correction', 0.0)
    assert "parasitic_correction" not in plot_args
    # if no frequencies given, calculate them from the model
    if f is None:
        pass

    # noinspection PyUnresolvedReferences
    if resistor_name in analysis_group and np.isfinite(analysis_group[resistor_name][col, row]):
        hist_resistance = analysis_group[resistor_name]
        from pixcap65.analysis_util.physics_modelling import full_capacitance_model
        assert isinstance(hist_resistance, tb.Array) or isinstance(hist_resistance, np.ndarray)
        y = full_capacitance_model(f,
                                   c=(actual_cap + parasitic_correction) * ADVANCED_CAPACITANCE_CONVERSION_FACTOR,
                                   r=hist_resistance[col, row],
                                   i=total_leak_hist[col, row] * 1.e-9, u0=1) * CURRENT_CONVERSION_FACTOR
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
        :key parasitic_correction: if present, it should be the capacitance subtracted during capacitance correction
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
                                  effective_doping_table: TABLES_LEAF_COMPAT_TYPE, output_pdf: PdfPages, i_row, table, resistivity):
    if np.all(np.isfinite(depletion_width_plate[i_col, i_row])):
        condition = """(row == {}) & (col == {})""".format(i_row, i_col)
        for x in table.where(condition):
            depletion_fit_propagate_parameters = {p_key: x[p_key] for p_key in ["NAD", "V", "dep", "sat"]}
            break
        else:
            depletion_fit_propagate_parameters = {}
        doping_acceptor = depletion_fit_propagate_parameters["NAD"]
        effective_doping = effective_doping_table[i_col, i_row]
        effective_resistivity = resistivity[i_col, i_row]
        with figure_provider(global_interactive_lock, 3, output=output_pdf) as (fig, ax, _):
            logger.debug("The type of bias_voltages is %s", type(bias_voltages))
            logger.debug("The shape of the bias voltages is %s", bias_voltages.shape)
            bias_mask = bias_voltages < -0.5
            if np.all(np.isfinite(depletion_width_plate_error[i_col, i_row])):
                enhanced_error_bar(ax[0], bias_voltages[bias_mask], depletion_width_plate[i_col, i_row][bias_mask],
                                   yerr=depletion_width_plate_error[i_col, i_row][bias_mask], label='d-measurement')
                # ax[0].errorbar(bias_voltages[bias_mask], depletion_width_plate[i_col, i_row][bias_mask],
                #                yerr=depletion_width_plate_error[i_col, i_row][bias_mask], label='d-measurement',
                #                ls=None)
            else:
                ax[0].plot(bias_voltages[bias_mask], depletion_width_plate[i_col, i_row][bias_mask],
                           label='d-measurement', marker=None)

            # noqa: S125
            # sample_voltage = -1 * np.linspace(np.min(-bias_voltages), np.max(-bias_voltages) * 1.1, 1000)
            sample_voltage = np.linspace(np.min(bias_voltages[bias_mask]) * 1.1, np.max(bias_voltages[bias_mask]) / 1.1,
                                         1000)
            ax[0].plot(sample_voltage, model_depletion(sample_voltage, **depletion_fit_propagate_parameters),
                       label=f'd-theory for NAD = {doping_acceptor:4.2f}  and Ubi = {depletion_fit_propagate_parameters["V"]:.2f}',
                        marker = None)
            ax[0].set(xlabel='$U_\\text{{bi}}$ / \\unit{{\\volt}}', ylabel='$d$ / \\unit{{\\micro\\meter}}',)
            if not GENERATE_THESIS_PLOTS:
                ax[0].set_title(f"Analysis of the depletion width for pixel ({i_col}, {i_row}).")
                ax[1].set_title(f"Analysis of the effective doping for pixel ({i_col}, {i_row}).")
                ax[2].set_title("Analysis of the effective doping")
            ax[0].grid(True)
            ax[0].legend(
                title=f"Saturating at {depletion_fit_propagate_parameters['dep']} with {depletion_fit_propagate_parameters['sat']} saturation.")
            ax[1].plot(-bias_voltages, effective_doping, marker=None)
            ax[1].set(xlabel='$U_\\text{{bi}}$ / \\unit{{\\volt}}', ylabel='Effective \ndoping \nconcentration / \\unit{{\\per\\centi\\meter\\cubed}}')
            ax[1].grid(True)
            ax[1].set_yscale('log')
            ax[2].plot(depletion_width_plate[i_col, i_row], effective_doping)
            ax[2].set(xlabel='$d$ / \\unit{{\\micro\\meter}}', ylabel='Effective\ndoping\nconcentration / \\unit{{\\per\\centi\\meter\\cubed}}', marker=None)
            ax[2].set_yscale('log')
        with figure_provider(global_interactive_lock, 2, output=output_pdf) as (fig, ax, _):
            if not GENERATE_THESIS_PLOTS:
                ax[0].set_title(f"Analysis of the specific resistivity for pixel ({i_col}, {i_row}).")
                ax[1].set_title(f"Analysis of the specific resistivity for pixel ({i_col}, {i_row}).")

            ax[0].plot(-bias_voltages, effective_resistivity, marker=None)
            ax[0].set(xlabel='$U_\\text{{bi}}$ / \\unit{{\\volt}}', ylabel='$\\rho$ / \\unit{{\\ohm\\centi\\meter}}')
            ax[0].grid(True)
            ax[0].set_yscale('log')
            ax[1].plot(depletion_width_plate[i_col, i_row], effective_resistivity, marker=None)
            ax[1].set(xlabel='$d$ / \\unit{{\\micro\\meter}}', ylabel='$\\rho$ / \\unit{{\\ohm\\centi\\meter}}')
            ax[1].set_yscale('log')




def mp_plotting_init(backend, has_latex):
    from pixcap65.utility.homogenize_plots import set_params
    import matplotlib
    matplotlib.use(backend)
    set_params(latex=has_latex,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                           r"stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                           r"retain-zero-uncertainty}", fig_height=8.26772, fig_width=11.69291,
               minor=True, fontsize=25)


if __name__ == '__main__':
    # plot_data(interpreted_data=os.path.expanduser('~/git/pixcap65/pixcap_LF_50x50_DC_R3_80V_HV.h5'))

    # some usage examples
    from pixcap65.utility.homogenize_plots import set_params, get_error_cycler
    import matplotlib
    matplotlib.use('PDF')

    logging.basicConfig(level=logging.INFO)
    # plot_bias_data(interpreted_data="data/3D_Sensor_221_Scan.h5", base_path="Thesis/ATLAS_ITk/X3/I_V_Characteristic",
    #                use_group=True)
    # plot_bias_data(interpreted_data="data/3D_Sensor_221_W13_X_Scan.h5", base_path="Thesis/ATLAS_ITk/X3/I_V_Characteristic",
    #                use_group=True)
    # plot_combined_data(interpreted_data="data/3D_Sensor_221_W13_X_Scan.h5",
    #                    base_path="Thesis/ATLAS_ITk/X3/C_V_Characteristic", use_group=True)
    # plot_bias_data(interpreted_data="data/3D_Sensor_221_W6_j_Scan.h5", base_path="Thesis/ATLAS_ITk/X5/I_V_Characteristic",
    #                use_group=True)
    # plot_combined_data(interpreted_data="data/3D_Sensor_221_W6_j_Scan.h5",
    #                    base_path="Thesis/ATLAS_ITk/X5/C_V_Characteristic", use_group=True)
    # plot_bias_data(interpreted_data="data/3D_Sensor_I14_S24_Scan.h5", base_path="Thesis/ATLAS_ITk/X6/I_V_Characteristic",
    #                use_group=True)
    # plot_combined_data(interpreted_data="data/3D_Sensor_I14_S24_Scan.h5",
    #                    base_path="Thesis/ATLAS_ITk/X6/C_V_Characteristic", use_group=True)
    # plot_bias_data(interpreted_data="data/3D_Sensor_H23_S24_Scan.h5", base_path="Thesis/ATLAS_ITk/X7/I_V_Characteristic",
    #                use_group=True)
    # plot_combined_data(interpreted_data="data/3D_Sensor_H23_S24_Scan.h5",
    #                    base_path="Thesis/ATLAS_ITk/X7/C_V_Characteristic", use_group=True)
    # plot_bias_data(interpreted_data="data/argparser.h5", base_path="Reference/R11/I_V_Characteristic", use_group=True)
    # plot_combined_data(interpreted_data="data/argparser.h5", base_path="Reference/R11/C_V_Characteristic", use_group=True)
    # plot_combined_data(interpreted_data="data/3D_Sensor_221_W5_S_Scan.h5", base_path="Thesis/ATLAS_ITk/X4/C_V_Characteristic",
    #                    use_group=True)
    try:
        from subprocess import run

        run_result = run(['pdflatex', '--version'], check=True, capture_output=True)
        has_latex = True
        logger.info("The latex compiler to use is: %s", run_result.stdout.decode("utf-8"))
    except (FileNotFoundError, ImportError):
        # proceed as if no latex exists
        logger.exception("Could not verify whether latex exists.")
        has_latex = False
    set_params(latex=has_latex,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                           r"stat,sys,sys-disp.}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                           r"retain-zero-uncertainty}", fig_height=8.26772, fig_width=11.69291,
               minor=True, fontsize=20, dpi=300)

    # use this attempt to achieve a better performance when generating the plots
    import multiprocessing as mp
    from full_analysis import x7_plotter, presentation_plotter

    print(mp.current_process().name)
    print(mp.cpu_count())

    presentation_plotter(threading.RLock())

    with mp.Manager() as manager, mp.Pool(initializer=mp_plotting_init, initargs=("PDF", has_latex,)) as pool:
        tables_lock = manager.RLock()
        process_handles = [
            # bare_sample_plotter_second,
            # x1_plotter,
            # x2_plotter_second,
            # x5_plotter,
            # x6_plotter,
            x7_plotter,
            # r13_plotter_second,
            # e1_plotter_second,
            # r1_plotter,
        ]

        processes = [pool.apply_async(handle, (tables_lock,)) for handle in process_handles]

        for p in processes:
            p.wait()
            print("Finished the process; Was it sucessful?", p.successful())

    # r1_plotter(threading.RLock())
    # x2_plotter_second(threading.RLock())
