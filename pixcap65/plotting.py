"""
Plotting of Pixcap65 data.
"""
# ----------------------------------------------------------
#  Copyright (c) 2018-2026. All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------


import os.path

import logging
import threading
import time
from contextlib import contextmanager
from matplotlib.axes import Axes
from types import NoneType
from typing import Any, Optional, Union, List, Tuple

from pixcap65.analysis_util.physics_modelling import model_depletion
from pixcap65.utility import synchronized_process_open_file
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
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pixcap65.analysis_util.utility import check_leaf_unit, GENERAL_PIXCAP_SHAPE, HIST_BIAS_MEAS_UNIT, \
    HIST_LEAK_CURRENT_UNIT, \
    HIST_CAP_UNIT, HIST_CURRENT_MEAS_UNIT, extract_parasitic_capacitance, CURRENT_CONVERSION_FACTOR, get_base_group, \
    get_analysis_group, TABLES_LEAF_COMPAT_TYPE, CVDistributionData
from pixcap65.utility.utils_2 import GroupType
from pixcap65.utility.homogenize_plots import enhanced_error_bar

HISTOGRAM_SHAPE_FORMAT = "The histograms shape is {}"
ROW_LABEL = 'Row'
COLUMN_LABEL = 'Column'
HIST_PIX_CAP_LABEL = '$C$ / \\unit{{\\femto\\farad}}'
COUNTS_HIST_LABEL = 'Counts / \\#'
SIMPLE_CAP_LABEL_PERCENT_FORMAT = '%sFit to data:\n$C_d = \\qty{%.1f}{\\femto\\farad}$'
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
GENERATE_THESIS_PLOTS = False
CV_USE_SEPARATE_PAGES = True
SENSOR_ITERABLE = Union[List[tb.Group], Tuple[tb.Group, ...], np.ndarray[tb.Group]]
IS_PRESENTATION = False
IS_THESIS = False

global_interactive_lock = threading.RLock()

import atexit
def release_lock():
    global global_interactive_lock
    global_interactive_lock.acquire()
    global_interactive_lock.release()
    del global_interactive_lock
    global_interactive_lock = None
    import gc
    print("release the plotting lock")
    atexit.unregister(release_lock)
    gc.collect()

atexit.register(release_lock)

replacement_order = {}
exclusion_list = ["Scan", "Full"]


def evaluate_pixel_mask(hist, perform_filter=False, **kwargs):
    """
    evaluate_pixel_mask

    @author Dominik Fischer
    @date 2026-05-11
    last updated: 2026-08-11

    Helper function to evaluate the provided pixel mask and select only those pixels for further analysis or plotting
    which are not "deactivated" by the mask.
    This function could also handle upper and lower thresholds on the pixel capacitance' values to generate a mask on
    its own.


    :param hist: histogram/data set to be masked for 'defect' pixels
    :param perform_filter: boolean, indicating whether the generated mask should be applied and only the filtered data
        returned. (default: False)
    :type perform_filter: bool
    :param kwargs: further keyword arguments
    :key test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :key mask_pixel: array of tuple of pixel positions to be masked.
    :key mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :key mask_upper: float, threshold to mask all pixels above this value.
    :type mask_upper: float
    :return: masked histogram/data set (masked pixels values are replaced by np.nan). If `perform_filter` is True, the mask is already applied and the masked values are no longer included.
    """
    result_hist = hist.copy()
    kargs = kwargs.copy()
    warn_level = 2
    if "exclude_cap_hist" in kargs:
        from warnings import warn
        warn("Found the unexpected keyword-argument 'exclude_cap_hist'. The correct keyword should be 'test_cap_exclusion', but that might change again. If both are provided the first one will be ignored.", UserWarning, stacklevel=warn_level)
        kargs.setdefault("test_cap_exclusion", kargs.pop("exclude_cap_hist"))
    if "exclude_test_cap" in kargs:
        from warnings import warn
        warn(
            "Found the unexpected keyword-argument 'exclude_test_cap'. The correct keyword should be 'test_cap_exclusion', but that might change again. If both are provided the first one will be ignored.",
            UserWarning, stacklevel=warn_level)
        kargs.setdefault("test_cap_exclusion", kargs.pop("exclude_test_cap"))
    if "exclude_cap_test" in kargs:
        from warnings import warn
        warn(
            "Found the unexpected keyword-argument 'exclude_cap_test'. The correct keyword should be 'test_cap_exclusion', but that might change again. If both are provided the first one will be ignored.",
            UserWarning, stacklevel=warn_level)
        kargs.setdefault("test_cap_exclusion", kargs.pop("exclude_cap_test"))
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

    # mask further pixels by their capacitance thresholds
    if mask_lower:
        assert isinstance(mask_lower, float)
        result_hist[result_hist < mask_lower] = np.nan
    if mask_upper:
        assert isinstance(mask_upper, float)
        result_hist[result_hist > mask_upper] = np.nan

    # apply the mask selection immediately if requested instead of setting the values only to NaN.
    if perform_filter:
        return result_hist[np.isfinite(result_hist)]
    return result_hist


def get_pdf_name(base_path, interpreted_data, suffix: str, use_group: bool) -> str:
    """
    get_pdf_name

    @author: Dominik Fischer
    @date 2026-08-11

    Helper function to generate a PDF name for a given hdf file to export the created figures to.
    Within the current implementation there is global flag to distinguish two modes when constructing the name of output pdf files.
    With the new mode the sensor part in the naming of the files will be separated to put the pdf into a subdirectory.
    For 3D-Sensors the filenames should start with '3D_'.
    In general the different parts of the filenames should be separated by '_'.
    Except for 3D-Sensors the first component is considered to determine the actual sensor.

    When using the old naming scheme the file name is given SOURCEFILE_SUFFIX_GROUP.pdf.
    If group is not present, this part of the output file will be left out.

    :param base_path: basic group descriptor for the analysis and measurements within the hdf file.
    :param interpreted_data: path to the hdf file used for measurement and analysis.
    :param suffix: actual suffix to use to identify the PDF file.
    :type suffix: str
    :param use_group: boolean, indicates whether to append the group name to the PDF name. (default: False)
    :type use_group: bool
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
    """
    advanced_figure_provider

    @author: Dominik Fischer
    @date 2026-08-11

    Context manager helper function to create plots/figures using matplotlib and destroy/close them appropriately afterwards.
    The context manager will yield a tuple of a matplotlib.Figure and a matplotlib.Axes object.

    :param lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure at
        the same time as matplotlib is not necessarily thread-safe.
    :param output: (optional) pdf object to write the figure to before closing it. Figure will only be saved when
        this argument is not `None`(default: None)
    :param kwargs: further (keyword) arguments (directly) propagated when saving the figure.
    """
    from pixcap65.utility.homogenize_plots import close_figure
    with lock:
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
    yield fig, ax
    if output is not None:
        output.savefig(fig, **kwargs)
    with lock:
        close_figure(fig)


@contextmanager
def figure_provider(lock, *args, separate_plots=False, output=None, callback=None, **kwargs):
    """
    figure_provider

    @author: Dominik Fischer
    @date 2026-08-11

    Context manager helper function to create plots/figures using matplotlib.
    The context manager will yield a tuple of a matplotlib.Figure and a matplotlib.Axes object.
    Instead of an matplotlib.Axes object an iterable could yielded instead if creation of multiple axes within the
    figure is specified.


    :param lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure at
        the same time as matplotlib is not necessarily thread-safe.
    :param args: positional arguments to propagete to the matplotlib.pyplot handler to create new (sub-) figures.
    :param separate_plots: indicates whether the different axes should constructed within different matplotlib.Figure objects. (default: False)
    :type separate_plots: bool
    :param output: (optional) pdf object to write the figure to before closing it. Figure will only be saved when
        this argument is not `None`(default: None)
    :param callback: function reference (callback) which takes a matplotlib.Figure object as the only argument 'to do'
        something with the figure objects
    :param kwargs: further keyword arguments (directly) propagated when saving the figure or its creation.
    """
    from pixcap65.utility.homogenize_plots import close_figure
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
        close_figure(fig)


def plot_data(interpreted_data, base_path=None, suffix="general_data", use_group=False, **kwargs):
    """
    plot_data

    @author: Dominik Fischer
    @date 2026-08-11


    Graphical present/plot the analysis results of a simple pixel capacitance scan.
    The plotting is only performed for the pixels which contribute a usable capacitance measurement.
    In Addition to the fits for estimating the capacitance also the capacitance distribution and frequency is plotted.
    The name of the resulting PDF is derived from the file name with the measurement data.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :type interpreted_data: str
    :param base_path: path to the base group in the hdf files hierarchy.
    :type base_path: str
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :type suffix: str
    :param use_group: whether to append the group name of the measurements to the PDF name.
    :type use_group: bool
    :key use_corrected: boolean, indicating whether to use the corrected capacitance for plotting.
        (data corrected for parasitic capacitances of PixCap65, default: False)
    :type use_corrected: bool
    :key test_cap_exclusion: boolean, whether to exclude the test capacitator row from the histograms. (default: False)
    :type test_cap_exclusion: bool
    :key hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :key mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :key mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :key mask_upper: float, threshold to mask all pixels above this value.
    :type mask_upper: float
    :key distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor. (default: False)
    :type distribution: bool
    :key lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously write/read operations on the same file.
    :key plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure at
        the same time as matplotlib is not necessarily thread-safe.
    :key unit: unit of the capacities presented within the plot.
    :type unit: str
    :key capacitance: histogram of the capacitance to use instead of those extracted from the provided hdf files group.
    :key no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
        When investigating the capacitance distribution.
    :key use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :key apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :key fit_plot_pdf: PDF object to save the fit figures to.


    """
    lock = kwargs.pop("lock", None)

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
    """
    inter_pix_data_fetch

    @author Dominik Fischer
    @date 2026-08-11

    Utility function to fetch total-pixel-capacitance measurements reference data for plotting not only the 'in-pix' capacitance's but also the estimation for the inter-pixel capacitances'.

    :param path: path to h5 file containing the measurement data for the total-pix measurement.
    :type path: str
    :param group: hierarchical group of the total-pix measurement within the file.
    :type group: str
    :param lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
  write/read operations on the same file.
    :param active_file: active hdf file by the ongoing plotting handlers
    :param type_name: type of analysis results to be fetched, e.g. 'total_cap'
    """
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
                        total_path=None, total_data=None, inter_path=None, inter_data=None, **kwargs):
    """
    plot_inter_pix_data

    @author Dominik Fischer
    @date 2026-08-11

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
    :key lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :key plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :key distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor. (default: False) [boolean]
    :type distribution: bool
    :key hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :key test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :key mask_pixel: array of tuple of pixel positions to be masked.
    :key mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :key mask_upper: float, threshold to mask all pixels above this value.
    :type mask_upper: float
    :key no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
    :key use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :key apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :key fit_plot_pdf: PDF object to save the fit figures to.
    :key use_corrected: boolean, whether to use the corrected capacitance's for plotting.
    :key apply_correction: boolean, whether to use the corrected capacitance's for plotting/extraction.
    """
    lock = kwargs.pop("lock", None)
    # determine the pdf file
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)

    with PdfPages(pdf_name) as output_pdf:
        with synchronized_process_open_file(interpreted_data, mode='r', lock=lock) as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            # we need to fetch the correcponding group
            with inter_pix_data_fetch(total_data, total_path, lock, in_file_h5) as total_group,\
                inter_pix_data_fetch(inter_data, inter_path, lock, in_file_h5, 'inter_cap') as inter_group:

                plot_inter_pix_data_delegate(base_group.inter_cap.measurements,
                                             get_analysis_group(base_group.inter_cap, **kwargs), output_pdf,
                                             total_group, inter_group, **kwargs)


@contextmanager
def multi_sensor_file_handler_simple(interpreted_data, base_path, **kwargs):
    """
    multi_sensor_file_handler_simple

    @author: Dominik Fischer
    @date: 2026-08-11

    Utility function and context manager to get the file handles to the files containing the measurements and analysis
    data for a multiple sensors.
    In Addition to handling the access to the data files also the output pdf object managed by this functions
    context manager.

    This context manager yields a tuple of list of hdf files hierarchy groups and pdf object.

    :param interpreted_data: paths to the files containing the data to plot. (Iterable)
    :param base_path: hdf files hierarchy groups paths (Iterable)
    :key pdf_name: file name for the output pdf file.
    :key lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    """
    pdf_name = kwargs.pop("pdf_name", "I-V-Collection.pdf")
    file_lock = kwargs.pop("lock", None)

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
    """
    multi_sensor_file_handler_advanced

    @author: Dominik Fischer
    @date: 2026-08-11

    Utility function and context manager to get the file handles to the files containing the measurements and analysis
    data for a multiple sensors.
    In Addition to handling the access to the data files also the output pdf object managed by this functions
    context manager.

    This context manager yields a tuple of list of hdf files hierarchy groups and pdf object.

    In contrast to the simple implementation the yielded tuple contains at index 1 the analysis groups of the different sensors.

    :param interpreted_data: paths to the files containing the data to plot. (Iterable)
    :param base_path: hdf files hierarchy groups paths (Iterable)
    :key pdf_name: file name for the output pdf file.
    :key lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    """
    pdf_name = kwargs.pop("pdf_name", "I-V-Collection.pdf")
    file_lock = kwargs.pop("lock", None)

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


def plot_bias_data(interpreted_data, base_path=None, suffix="bias_curve", use_group=False, **kwargs):
    """
    plot_bias_data

    @author: Dominik Fischer
    @date: 2026-08-11

    Plot the data acquired for the pixel-diodes I-V characterization.

    :param interpreted_data: path to the hdf file which holds the raw data.
    :param base_path: path within the files hierarchy for the base group.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    :key lock: locking object used to synchronize the access to the file handles by the pytables library. It is highly
        encouraged to provide an explicit lock here.
    :key plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :key labels: required for multi-sensor plotting to label the plots from the different sensors correctly such that these could be identified. (Iterable)
    :key area_normalisation: areas of the individual pixel summed over all contributiong pixels. (Iterable)
    """
    lock = kwargs.pop("lock", None)
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

    @author: Dominik Fischer
    @date: 2026-08-11

    Plot the results of the C-V characterization of the scanned pixels.
    To achieve this we need the different c-v-data.
    Then the C-V curve is plotted for every pixel.
    If requested also fits to the boundary regions of the c-v-curve to determine
    the depletion voltage of the pixel are plotted.
    To do so, two fit ranges for the two boundaries with physically distinct behaviour needs to be supplied.

    On request also the doping profile, estimated from the differential capacitance and the depletion width (their dependence onto the applied voltage) is presented.
    In this case the estimated doping profile across the sensors thickness is plotted in dependence of the applied
    bias voltage and the depletion width.
    In Addition the corresponding fits to model these profiles are presented if their parameters are estimated before.
    Also the resistivity profile will be plotted.


    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix:  additional suffix to use for naming the PDF containing the plots.
    :param use_group:   boolean, whether to append the group name of the measurements to the PDF name.
    :key lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :key pdf_name: file name for the output pdf file.
    :key use_corrected: boolean, whether to use the corrected capacitance's for plotting.
    :type use_corrected: bool
    :key apply_correction: boolean, whether to use the corrected capacitance's for plotting/extraction.
    :type apply_correction: bool
    :key verbose: boolean, indicating whether to use verbose output for depletion voltages
    :key distribution: boolean, indicating whether also the capacitance distribution of the whole sensor
        should be investigated.
    :key plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :key labels: required for multi-sensor plotting to label the plots from the different sensors correctly such that these could be identified. (Iterable)
    :key mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :key hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    """
    file_lock = kwargs.get("lock", None)
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

    @author: Dominik Fischer
    @date: 2026-08-11

    Plot the data acquired for the pixel-diodes I-V characterization and the C-V characterization of the pixels.
    Plot the results of the C-V characterization of the scanned pixels.
    To achieve this we need the different c-v-data.
    Then the C-V curve is plotted for every pixel.
    If requested also fits to the boundary regions of the c-v-curve are performed to determine the
    depletion voltage of the pixel.
    To do so, two fit ranges for the two boundaries with physically distinct behaviour needs to be supplied.

    On request also the doping profile, estimated from the differential capacitance and the depletion width (their dependence onto the applied voltage) is presented.
    In this case the estimated doping profile across the sensors thickness is plotted in dependence of the applied
    bias voltage and the depletion width.
    In Addition the corresponding fits to model these profiles are presented if their parameters are estimated before.
    Also the resistivity profile will be plotted.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    :key lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :key pdf_name: file name for the output pdf file.
    :key use_corrected: boolean, whether to use the corrected capacitance's for plotting.
    :type use_corrected: bool
    :key apply_correction: boolean, whether to use the corrected capacitance's for plotting/extraction.
    :type apply_correction: bool
    :key verbose: boolean, indicating whether to use verbose output for depletion voltages
    :key distribution: boolean, indicating whether also the capacitance distribution of the whole sensor
        should be investigated.
    :key plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :key labels: required for multi-sensor plotting to label the plots from the different sensors correctly such that these could be identified. (Iterable)
    :key mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :key hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :key labels: required for multi-sensor plotting to label the plots from the different sensors correctly such that these could be identified. (Iterable)
    :key area_normalisation: areas of the individual pixel summed over all contributiong pixels. (Iterable)
    """
    if kwargs.get("use_corrected", False):
        suffix = "{}_corrected".format(suffix)

    file_lock = kwargs.get("lock", None)
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

    @author: Dominik Fischer
    @date: 2026-08-11

    Actual implementation for presenting the results of the I-V characterization.
    It's just a simple plot with error bars for the different quantities.

    :param data_group: hdf file's hierarchy group containing the raw data.
    :param output_pdf: PDF object to write the plots to.
    :key plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :key labels: required for multi-sensor plotting to label the plots from the different sensors correctly such that these could be identified. (Iterable)
    :key area_normalisation: areas of the individual pixel summed over all contributiong pixels. (Iterable)
    """
    interactive_lock = kwargs.get("plotting_lock", global_interactive_lock)
    with figure_provider(interactive_lock) as (fig, ax, _):
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
    """
    bias_voltage

    @author Dominik Fischer
    @date: 2026-08-11

    Internal utility function handling the actual plotting of single sensors i-v characteristics.
    The applied voltages and the corresponding leakage currents are to be extracted from the summary table, containing
    the set/design voltage, measured voltage and the measured leakage current.

    If not uncertainties for the leakage current are present, these will be recalculated assuming a Keithley 2410 SMU
    operating at the 1 kV sourcing range.

    :param ax: matplotlib.Axes object into which to plot the i-v-curve.
    :param tabular: pytables.Table object storing the leakage-current dependence from the scan.
    :param label: label of this measurement series within the figure. (only necessary when a correct legend is
        required.)
    :param norm: normalization factor for the currents (1 = no normalisation applied), to normalise the leakage current
        onto the sensors/pixel area.
    :param apply_norm: boolean, whether to apply area normalisation at all. (This parameter seems to be unused)
    """
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
        warn(f"The current sensor seems to be missing measurement uncertainties for the leakage current! The sensor is labeld by {label}")
    enhanced_error_bar(ax, voltage_data, current_data * CURRENT_CONVERSION_FACTOR / norm, xerr=voltage_error,
                       yerr=normalized_errors, label=label)
    # ax.errorbar(voltage_data, current_data * CURRENT_CONVERSION_FACTOR / norm, xerr=voltage_error,
    #             yerr=normalized_errors, fmt='o', label=label)
    if not np.isclose(norm, 1.0):
        ax.set_yscale('log')
        ax.set_ylabel("I in \\unit{{\\nano\\ampere\\per\\centi\\meter\\squared}}")


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

    @author: Dominik Fischer
    @date: 2026-08-11

    Actual implementation for presenting the results of the C-V characterization and if necessary the determination of
    the full depletion voltage. For each with a successful capacitance measurement for each bias voltage in use
    the C-V-curve will be plotted. If necessary, meaning if the boundaries for the two fit ranges for the
    two physically distinct regions of the c-v-curve are provided the full depletion voltage will be calculated and
    the necessary fits be plotted. If the analysis results provided already contain the necessary data sets for the
    estimation of the full depletion voltage, these will be used and the fit will be plotted, as well.

    For generating the plots of the C-V characterization for particular pixels, there are two modes available.
    Both plots (c-v and 1/c^2 - v) could be put into the same figure or into two different ones.
    The actual behaviour is controlled by the global flag 'CV_USE_SEPARATE_PAGES'

    :param data_group: hdf files hierarchy group containing the raw measurement data.
    :param analysis_group: HDF files hierarchy group containing the analysis results.
    :param output_pdf: PDF object to write the created figures to for long-term saving.
    :param apply_doping: boolean, False, indicates whether to plot the depletion data.
    :key plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :key labels: required for multi-sensor plotting to label the plots from the different sensors correctly such that these could be identified. (Iterable)
    :key mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :key verbose: boolean, indicating whether to use verbose output for depletion voltages.
    :key distribution: boolean, indicating whether also the capacitance distribution of the whole sensor
        should be investigated.
    :key hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :key use_corrected: boolean, indicating whether to use the corrected capacitance for plotting. (data corrected for parasitic capacitances of PixCap65, default: False)
    :type use_corrected: bool
    """
    interactive_lock = kwargs.get('plotting_lock', global_interactive_lock)
    # extract the bias data
    voltage_data_sets = [__process_voltage_set(data_group)] if isinstance(data_group, tb.Node) else [__process_voltage_set(data_set) for data_set in data_group]
    approx_depletion = isinstance(data_group, tb.Node)

    labels = kwargs.pop('labels', [])
    if "pixel_mask" in kwargs:
        from warnings import warn
        warn("Found a deprecated keyword argument 'pixel_mask' which will be removed in a future version.")
        kwargs.set_default('mask_pixel', kwargs.pop("pixel_mask"))
    masked_pixels = kwargs.get('mask_pixel', [])

    # investigate all the pixel for plotting
    for ii, jj in np.ndindex(GENERAL_PIXCAP_SHAPE):
        if (ii, jj) in masked_pixels:
            continue
        response = lambda x: x.suptitle("C-V Characterization for Pixel ({}, {})".format(ii, jj))
        if GENERATE_THESIS_PLOTS or CV_USE_SEPARATE_PAGES:
            response = None
        with figure_provider(interactive_lock, ncols=2, callback=response, output=output_pdf, separate_plots=CV_USE_SEPARATE_PAGES) as (_, ax, back_pipe):
            # will not only generate the title string of the figure but also the figure with the depletion fits.
            title_str = __plot_depletion_estimation(analysis_group, approx_depletion, ax, ii, jj, voltage_data_sets[0])

            if jj == 40 or not _cv_plotter(analysis_group, jj, ii, ax, voltage_data_sets, labels):
                back_pipe["output"] = False
                continue

            # the first legend was unnecessary as there are no labels specified.
            ax[1].legend(title=title_str, loc='lower right')
            ax[1].grid(True)
            if kwargs.get("use_log", False):
                ax[0].set_yscale('log')
                ax[1].set_yscale('log')

        # Plot the doping analysis only for single-sensor samplings.
        if apply_doping:
            # TODO: better use the actual voltages here.
            voltage_collection_idx = 0
            if isinstance(data_group, tb.Group):
                depletion_width_plate = check_leaf_unit(analysis_group.DepletionWidth, "um")
                depletion_width_plate_error = check_leaf_unit(analysis_group.DepletionWidthErr, "um")
                effective_doping_table = check_leaf_unit(analysis_group.DepletionEffDoping, "cm^-3")
                resistivity_table = check_leaf_unit(analysis_group.DepletionResitivity, "Ocm")
                origin_bias_voltages = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
                if len(origin_bias_voltages.shape) > 1:
                    bias_voltages = origin_bias_voltages[:, voltage_collection_idx]
                else:
                    bias_voltages = origin_bias_voltages
                table = analysis_group.DepletionParamTable
                view_bias_voltages = np.atleast_2d(bias_voltages)
                view_depletion_width_plate = depletion_width_plate[None, :]
                view_depletion_width_plate_error = depletion_width_plate_error[None, :]
                view_effective_doping_table = effective_doping_table[None, :]
                view_effective_resistivity_table = resistivity_table[None, :]
                view_table = [table]
            else:
                view_depletion_width_plate = np.array([check_leaf_unit(ana.DepletionWidth, "um") for ana in analysis_group])
                view_depletion_width_plate_error = np.array([check_leaf_unit(ana.DepletionWidthErr, "um") for ana in analysis_group])
                view_effective_doping_table = np.array([check_leaf_unit(ana.DepletionEffDoping, "cm^-3") for ana in analysis_group])
                view_effective_resistivity_table = np.array([check_leaf_unit(ana.DepletionResistivity, "Ocm") for ana in analysis_group])
                view_table = [gr.DepletionParamTable for gr in analysis_group]
                view_origin_bias_voltages = np.array([check_leaf_unit(gr.BiasVoltageHist, HIST_BIAS_MEAS_UNIT) for gr in data_group])
                if len(view_origin_bias_voltages.shape) > 2:
                    view_bias_voltages = view_origin_bias_voltages[:, :, voltage_collection_idx]
                else:
                    view_bias_voltages = view_origin_bias_voltages
            plot_depletion_pixel_delegate(view_bias_voltages, ii, view_depletion_width_plate, view_depletion_width_plate_error,
                                          view_effective_doping_table, output_pdf, jj, view_table, view_effective_resistivity_table)

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
                if not GENERATE_THESIS_PLOTS:
                    ax.set_title("Capacitance distribution for bias voltage {}".format(bias_voltage))
                ax.set(xlabel=CAPACITANCE_LABEL)
                ax.hist(analysis_group.UCHist[:, :, k].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                        bins=kwargs.get("hist_bins", DEFAULT_BIN_NUMBER))
    else:
        group_handle = analysis_group
        n_items = len(analysis_group)
        label_handle = labels if len(labels) == n_items else ['?'] * n_items

    title_str = ""
    def __callback_handler(fig):
        fig.suptitle(title_str)

    with figure_provider(interactive_lock, output=output_pdf, ncols=2, separate_plots=True, call_all=True,
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
                x_limits = (-x_limits[1], -x_limits[0])
            ax[1].set_xlim(*x_limits)
            ax[1].set_ylim(*y_limits)
            ax[1].legend()
            ax[0].legend()
        else:
            back_pipe["output"] = False


def _plot_cv_distribution(group: tb.Group, ax, x_limits=None, y_limits=None, **kwargs) -> Tuple[Optional[Tuple], Optional[Tuple], str]:
    """
    _plot_cv_distribution

    @author: Dominik Fischer
    @date: 2026-08-11

    (Internal) Utility Function.
    Fetches the results from the c-v characteriztation averaged/distributed accross the sensor and plots the analysis
    results for this case. (Including the C-V-curve and the estimation of the depletion voltage)

    Only if a single-sensor is processed also error-bands are drawn for the fits.

    :param group: hdf file group under which the analysis results are stored for the C-V characterization.
    :param ax: axes object(s) to use for plotting
    :param x_limits: tuple defining the x-axis plotting limits from the data points of the C-V-Curve.
    :param y_limits:tuple defining the y-axis plotting limits from the data points of the C-V-Curve.
    :key use_corrected: boolean, indicating whether to use the corrected capacitance for plotting. (data corrected for parasitic capacitances of PixCap65, default: False)
    :type use_corrected: bool
    :key is_combining: boolean, indicates whether multiple sensors are to be combined into a single figure. (default: False)
    :type is_combining: bool
    :return: tuple of the drawing limits for both axis and the final title string for the figure.
    """
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
            # skip plotting of this functions if the multiple C-V- is plotted
            if not is_combining:
                try:
                    from jacobi import propagate
                    # CHECK: would it now be present within the distribution data?
                    first_covariance = depletion_data["first_covariance"][dep_idx]
                    second_covariance = depletion_data["second_covariance"][dep_idx]
                    first_depletion_parameters = [depletion_fit_a[dep_idx], depletion_fit_b[dep_idx],]
                    second_depletion_parameters = [depletion_fit_c[dep_idx], depletion_fit_d[dep_idx],]
                    first_y, first_y_cov = propagate(lambda p: p[0] * first_voltage_x + p[1], first_depletion_parameters,
                                                     first_covariance)
                    second_y, second_y_cov = propagate(lambda p: p[0] * second_voltage_x + p[1],
                                                       second_depletion_parameters,
                                                       second_covariance)

                    ax[1].plot(-first_voltage_x, first_y, '-', label="First section fit")
                    ax[1].plot(-second_voltage_x, second_y, '-', label="Second section fit")

                    first_y_error_prop = np.diag(first_y_cov) ** 0.5
                    second_y_error_prop = np.diag(second_y_cov) ** 0.5

                    ax[1].fill_between(-first_voltage_x, first_y - first_y_error_prop, first_y + first_y_error_prop,
                                       facecolor="C1", alpha=0.5)
                    ax[1].fill_between(-second_voltage_x, second_y - second_y_error_prop,
                                       second_y + second_y_error_prop,
                                       facecolor="C1", alpha=0.5)
                except (ImportError, tb.exceptions.NoSuchNodeError, ValueError, KeyError):
                    from warnings import warn
                    warn("Something went wrong with the error bands of the depletion analysis.", stacklevel=1)
                    first_cap_calc = depletion_fit_a[dep_idx] * first_voltage_x + depletion_fit_b[dep_idx]
                    second_cap_calc = depletion_fit_c[dep_idx] * second_voltage_x + depletion_fit_d[dep_idx]
                    ax[1].plot(-first_voltage_x, first_cap_calc, '-', label="First section fit")
                    ax[1].plot(-second_voltage_x, second_cap_calc, '-', label="Second section fit")
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
    if approx_depletion and "DepletionHist" in analysis_group and jj < 40:
        depletion_fit_data = analysis_group.DepFitParamHist[:]
        depletion_hist = analysis_group.DepletionHist[:]
        assert isinstance(depletion_fit_data, np.ndarray)
        assert isinstance(depletion_hist, np.ndarray)
        if depletion_fit_data.shape[1] < 41:
            temp_shape = [*depletion_fit_data.shape]
            temp_shape[1] = 41
            temp_array = np.full(tuple(temp_shape), np.nan)
            temp_array[:, :40, :] = depletion_fit_data
            depletion_fit_data = temp_array

        if len(depletion_fit_data.shape) == 3:
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
            try:
                from jacobi import propagate
                full_covariance_matrix = analysis_group.DepFitParamCovHist[:]
                if len(full_covariance_matrix.shape) == 4:
                    full_covariance_matrix = full_covariance_matrix[:, :, None, :, :]
                first_covariance = full_covariance_matrix[ii, jj, dep_idx, :2, :2]
                second_covariance = full_covariance_matrix[ii, jj, dep_idx, 2:, 2:]
                first_y, first_y_cov = propagate(lambda p: p[0] * first_voltage_x + p[1], first_dep_parameters, first_covariance)
                second_y, second_y_cov = propagate(lambda p: p[0] * second_voltage_x + p[1], second_dep_parameters,
                                                 second_covariance)


                ax[1].plot(-first_voltage_x, first_y, '-', label="First section fit")
                ax[1].plot(-second_voltage_x, second_y, '-', label="Second section fit")

                first_y_error_prop = np.diag(first_y_cov) ** 0.5
                second_y_error_prop = np.diag(second_y_cov) ** 0.5

                ax[1].fill_between(-first_voltage_x, first_y - first_y_error_prop, first_y + first_y_error_prop, facecolor="C1", alpha=0.5)
                ax[1].fill_between(-second_voltage_x, second_y - second_y_error_prop, second_y + second_y_error_prop,
                                   facecolor="C1", alpha=0.5)

            except (ImportError, tb.exceptions.NoSuchNodeError) as e:
                first_cap_calc = first_dep_parameters[0] * first_voltage_x + first_dep_parameters[1]
                second_cap_calc = second_dep_parameters[0] * second_voltage_x + second_dep_parameters[1]
                ax[1].plot(-first_voltage_x, first_cap_calc, '-', label="First section fit")
                ax[1].plot(-second_voltage_x, second_cap_calc, '-', label="Second section fit")
                # CHECK: verify the new implementation!
                from warnings import warn
                # warn("Something went wrong with the error bands of the depletion analysis.")
                # print("handling the exception:", e)
            finally:
                ax[1].vlines(-dep_voltage_2, 0, 1, linestyles="dashed")
                ax[1].vlines(-dep_voltage_2, 0, 1, linestyles="dashed")
            title_str += "U = {:n} V\n".format(dep_voltage_2)
    return title_str

def _cv_plotter(analysis, row, col, ax, voltage_data_sets, labels, **kwargs):
    """
    _cv_plotter

    @author: Dominik Fischer
    @date: 2026-08-11

    (Internal) utility function handling the plotting of the c-v-curve of a single sensor for a particular pixel!

    :param analysis: hdf files group(s) containing the analysis results of the c-v-characterization for multiple sensors.
    :param row: row on the PixCap65 for which the c-v-curve should be plotted.
    :param col: column on the PixCap65 for which the c-v-curve should be plotted.
    :param ax: matplotlib.axes.Axes objects to plot into.
    :param voltage_data_sets: datasets of the applied bias voltages for (different) sensors. (could also contain the data for only a single sensor)
    :param labels: identifying names for the different sensors to use in the legend, when plotting for multiple sensors.
    :key is_distribution_plot: indicates wether we plot for the averaged sensor instead of a particular pixel (default: False)
    :type is_distribution_plot: bool
    :return:
    """
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
    # TODO: document the different meanings of this array!
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
    adjusted_cap_data = 1 / (cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 2
    if not GENERATE_THESIS_PLOTS:
        ax[0].set_title("Bias data from the \nmeasurement for {}".format(title_format))
        ax[1].set_title("Suited Bias data from the \nmeasurement for {}".format(title_format))
    ax[0].set(xlabel=BIAS_CURVE_X_LABEL,
              ylabel=CAPACITANCE_LABEL)
    enhanced_error_bar(ax[0], -voltage_data, cap_data * CAPACITANCE_CONVERSION_FACTOR, yerr=eff_cap_errors, label=label)
    ax[1].set(xlabel=BIAS_CURVE_X_LABEL, ylabel="$1 / C^2$ / \\unit{{\\per\\femto\\farad\\squared}}")
    enhanced_error_bar(ax[1], -voltage_data, adjusted_cap_data, yerr=np.abs(effective_capacitance_error_data), label=label, alpha=0.5)
    x_limits = __get_x_limits(voltage_data, x_limits)
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


def __get_x_limits(voltage_data, x_limits: Optional[Iterable]) -> Iterable:
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


# CHECK: Should we use such helper functions everywhere?
def plot_1d_distribution(data: np.ndarray, label: str, bias_code: int, table: Optional[tb.Table], pdf, group: tb.Group, **kwargs):
    """
    plot_1d_distribution

    @author: Dominik Fischer
    @date 2026-08-11

    Utility function to plot/graphically present the histogram of the capacitance distribution of a sensor.
    If a distribution plot is requested by the corresponding keyword argument, the necessary fits will be
    explicitly performed/re-performed.

    :param data: capacitance data from which the histogram will be plotted.
    :param label: label/title of the plot when it gets saved.
    :param bias_code: integer, determining which kind of measurement will be evaluated, a list of possible values is shipped with inter-pixel analysis functions within the source code (commented lines).
    :param table: pytables.Table containing the fit parameters and other results from the analysis of the capacitance's distribution.
    :type table: pytables.Table or numpy.ndarray
    :param pdf:
    :param group:
    :param kwargs: further keyword arguments to be propagated to sub-calls.
    :key plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure at
        the same time as matplotlib is not necessarily thread-safe.
    :key hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :key unit: unit of the capacities presented within the plot.
    :type unit: str
    :key distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor. (default: False)
    :type distribution: bool
    :key test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :key mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :key mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :key mask_upper: float, threshold to mask all pixels above this value.
    :type mask_upper: float
    :key capacitance: histogram of the capacitance to use instead of those extracted from the provided hdf files group.
    :key set_parasitic: boolean, whether to set the parasitic capacitance for this data set.
    :type set_parasitic: bool
    :key no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
    :key convert: boolean, whether to convert the capacitance to fF, or not (default: True)
    :key use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :key apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :key fit_plot_pdf: PDF object to save the fit figures to.
    """
    unit = kwargs.pop("unit", "\\farad")
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
    :key plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure at
        the same time as matplotlib is not necessarily thread-safe.
    :key hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :key test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :key mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :key mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :key mask_upper: float, threshold to mask all pixels above this value.
    :type mask_upper: float
    :key distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor. (default: False)
    :type distribution: bool
    :key unit: unit of the capacities presented within the plot.
    :type unit: str
    :key capacitance: histogram of the capacitance to use instead of those extracted from the provided hdf files group.
    :key set_parasitic: boolean, whether to set the parasitic capacitance for this data set.
    :type set_parasitic: bool
    :key no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
    :key convert: boolean, whether to convert the capacitance to fF, or not (default: True)
    :key use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :key apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :key fit_plot_pdf: PDF object to save the fit figures to.
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
    if kwargs.get("exclude_test_cap", False) or kwargs.get("test_cap_exclusion", False) or kwargs.get("exclude_cap_test", False) or kwargs.get("exclude_cap_hist", False):
        plot_2d_capacitance(masked_cap_hist, "Masked Pixel Capacitance distribution", output_pdf)

    # 1D Pixel Capacitance Hist
    distribution_table = analysis_group.DistResultfF if "DistResultfF" in analysis_group else None
    actual_unit = "\\femto\\farad"
    if distribution_table is None:
        actual_unit = "\\farad"
        distribution_table = analysis_group.DistResult if "DistResult" in analysis_group else None

    plot_1d_distribution(cap_hist, "Total Cap Distribution", 20000, distribution_table, output_pdf, analysis_group, unit=actual_unit, **kwargs)

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
                uncorrected_label = "\n$C=\\qty{{{:.2f}+-{:.2f}+-{:.2f}+-{:.2f}}}{{{}}}$".format(data_rec_result.capacitance[0],
                                                                    data_rec_result.cap_std[0],
                                                                    handle_nan(data_rec_result.cap_systematic_error[0]),
                                                                    handle_nan(data_rec_result.cap_systematic_dispersion[0]), unit)
                corrected_label = "\n$C_\\text{{corr}}=\\qty{{{:.2f}+-{:.2f}+-{:.2f}+-{:.2f}}}{{{}}}$".format(data_rec_result.cap_corrected[0],
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
    :key plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :key test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :key mask_pixel: array of tuple of pixel positions to be masked.
    :key mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :key mask_upper: float, threshold to mask all pixels above this value.
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


def plot_inter_pix_data_delegate(data_group: tb.Group, analysis_group: tb.Group, output_pdf: PdfPages, total_group=None,
                                 inter_group=None, **kwargs):
    """
    plot_inter_pix_data_delegate

    @author: Dominik Fischer
    @date 2026-08-11

    Actual implementation to plot the results of the analysis of the inter-pixel capacitance scan.
    Besides the naming it is not just plotting but also a bit of analysis as the distribution of the capacitance
    over the sensor and in general for all three currents is investigated, as well.
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
    :param inter_group: hdf files hierarchy group containing another inter-pixel measurements (results) for reference when extracting the individual contributions to the inter-pixel-capacitance. (default: None)
    :key plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :key distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor. (default: False) [boolean]
    :type distribution: bool
    :key hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :key test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :key mask_pixel: array of tuple of pixel positions to be masked.
    :key mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :key mask_upper: float, threshold to mask all pixels above this value.
    :type mask_upper: float
    :key no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
    :key use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :key apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :key fit_plot_pdf: PDF object to save the fit figures to.
    """
    interactive_lock = kwargs.get("plotting_lock", global_interactive_lock)
    need_distribution = kwargs.get("distribution", False)
    lockless_propagation = {key: value for key, value in kwargs.items() if "lock" not in key }

    # Read pixel map and verify that the assumed units are correct
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
    actual_unit = "\\femto\\farad"
    if distribution_result_data is None:
        actual_unit = "\\farad"
        distribution_result_data = analysis_group.DistResult if "DistResult" in analysis_group else None
    n_bins = kwargs.get("hist_bins", DEFAULT_BIN_NUMBER)

    # Investigate the counts of individual capacitance's
    if np.count_nonzero(np.isfinite(total_cap_hist)) > 2:
        # handle the in-pix capacitance and perform distribution fits if necessary
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

        # handle the inter-pix contributions by making use of the reference in-pix capacitance's
        if in_ref_cap_hist is not None:
            effective_inter_cap_hist = total_cap_hist - in_ref_cap_hist
            with advanced_figure_provider(interactive_lock) as (fig, ax):
                hist_inter_cap_hist = evaluate_pixel_mask(effective_inter_cap_hist, **lockless_propagation)
                ax.hist(hist_inter_cap_hist[~np.isnan(hist_inter_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
                        bins=n_bins)
                ax.set_ylabel(COUNTS_HIST_LABEL)
                ax.set_xlabel(HIST_PIX_CAP_LABEL)
                if not GENERATE_THESIS_PLOTS:
                    ax.set_title(
                        __get_1d_hist_label(kwargs.get('grouped_inter_pix_id', 18000), "Component pf Inter-Capacitance Distribution", distribution_result_data,
                                            unit=actual_unit))
                ax.grid()
                output_pdf.savefig(fig, bbox_inches='tight')
            if need_distribution:
                from pixcap65.analysis import analyze_capacitance_distribution_delegate
                analyze_capacitance_distribution_delegate(analysis_group, output_pdf,
                                                          capacitance=effective_inter_cap_hist,
                                                          set_parasitic=False, **lockless_propagation)

        # handle the inter-pixel capacitanes by making use of the provided total capacitance measurement
        if total_ref_cap_hist is not None:
            effective_inter_cap_hist = total_ref_cap_hist - total_cap_hist
            with advanced_figure_provider(interactive_lock) as (fig, ax):
                hist_inter_cap_hist = evaluate_pixel_mask(effective_inter_cap_hist, **lockless_propagation)
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

    if np.count_nonzero(np.isfinite(inter_a_cap_hist)) > 2:
        with advanced_figure_provider(interactive_lock) as (fig, ax):
            hist_cap_hist = evaluate_pixel_mask(inter_a_cap_hist, **kwargs)
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

    if np.count_nonzero(np.isfinite(inter_b_cap_hist)) > 2:
        with advanced_figure_provider(interactive_lock) as (fig, ax):
            hist_cap_hist = evaluate_pixel_mask(inter_b_cap_hist, **kwargs)
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
    effective_pixel_mask = [(i[0], i[1]) for i in kwargs.get("mask_pixel", [])]
    for col, row in np.ndindex(total_current_hist.shape[:2]):
        if verify_mask_pixel and (col, row) in effective_pixel_mask:
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
    """
    get_model_prediction

    @author: Dominik Fischer
    @date: 2026-08-11


    Utility function to compute the model predictions for capacitances depending on the used frequency for the measurement.

    :param col: PixCap65 measurement column for which to predict.
    :param row: PixCap65 measurement row for which to predict.
    :param analysis_group: hdf files group of the analysis results used for fetching the models parameter in order to compute the prediction.
    :param actual_cap:
    :param total_leak_hist: matrix of the estimated leakage currents by fitting the corresponding model.
    :param f: array of the frequencies for which a prediction is to be computed.
    :param resistor_name: name of the dataset containing the on-resistance estimators if such a dataset is present at all.
    :param plot_args: further keywords arguments to be propagated to a plotting utility function (unused?)
    :key parasitic_correction: parasitic capacitance for which the input values are already corrected (this needs to be accounted for by the model as the currents are not corrected at all).
    :return:
    """
    parasitic_correction = plot_args.pop('parasitic_correction', 0.0)
    assert "parasitic_correction" not in plot_args

    # noinspection PyUnresolvedReferences
    if resistor_name in analysis_group and np.isfinite(analysis_group[resistor_name][col, row]):
        hist_resistance = analysis_group[resistor_name]
        from pixcap65.analysis_util.physics_modelling import full_capacitance_model
        assert isinstance(hist_resistance, tb.Array) or isinstance(hist_resistance, np.ndarray)
        y = full_capacitance_model(f,
                                   c=(actual_cap + parasitic_correction) * ADVANCED_CAPACITANCE_CONVERSION_FACTOR,
                                   r=hist_resistance[col, row],
                                   i=total_leak_hist[col, row] / CURRENT_CONVERSION_FACTOR, u0=1) * CURRENT_CONVERSION_FACTOR
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
    from warnings import warn
    warn("Found the 'unused' additional function for depletion delegation!")
    data_groups = np.atleast_1d(data_group)
    analysis_groups = np.atleast_1d(analysis_group)
    # extract the depletion parameters
    view_depletion_width_plate = np.array([check_leaf_unit(group.DepletionWidth, "um") for group in analysis_groups])
    view_depletion_width_plate_error = np.array([check_leaf_unit(group.DepletionWidthErr, "um") for group in analysis_groups])
    view_effective_doping_table = np.array([check_leaf_unit(group.DepletionEffDoping, "cm^-3") for group in analysis_groups])
    view_bias_voltages = np.array([check_leaf_unit(group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT) for group in
                              data_groups])
    view_table = [group.DepletionParamTable for group in analysis_groups]
    # CHECK: what about here with handling multiple-sensors?

    # depletion_width_plate = check_leaf_unit(analysis_group.DepletionWidth, "um")
    # depletion_width_plate_error = check_leaf_unit(analysis_group.DepletionWidthErr, "um")
    # effective_doping_table = check_leaf_unit(analysis_group.DepletionEffDoping, "cm^-3")
    # bias_voltages = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
    # table = analysis_group.DepletionParamTable
    # view_depletion_width_plate = depletion_width_plate[None, :]
    # view_depletion_width_plate_error = depletion_width_plate_error[None, :]
    # view_effective_doping_table = effective_doping_table[None, :]
    # view_bias_voltages = np.atleast_2d(bias_voltages)
    # view_table = [table]

    for col, row in np.ndindex(GENERAL_PIXCAP_SHAPE):
        plot_depletion_pixel_delegate(view_bias_voltages, col, view_depletion_width_plate, view_depletion_width_plate_error,
                                      view_effective_doping_table, output_pdf, row, view_table)


def plot_depletion_pixel_delegate(bias_voltages: Iterable[TABLES_LEAF_COMPAT_TYPE], i_col,
                                  depletion_width_plates: Iterable[TABLES_LEAF_COMPAT_TYPE],
                                  depletion_width_plates_error: Iterable[TABLES_LEAF_COMPAT_TYPE],
                                  effective_doping_tables: Iterable[TABLES_LEAF_COMPAT_TYPE], output_pdf: PdfPages, i_row, tables, resistivities):
    """
    plot_depletion_pixel_delegate

    @author: Dominik Fischer
    @date: 2026-08-11

    Utility function to plot the dependence of the estimated resistivities and the doping-profile onto the applied
    bias voltage (reversed bias).

    :param bias_voltages: array of the applied bias voltages to use for plotting and investigation.
    :param i_col: column of the PixCap65 chip for which to perform the plotting of depletion data.
    :param depletion_width_plates: matrix of depletion widths for the measurement using PixCap65 and the connected sensor.
    :param depletion_width_plates_error: matrix of the uncertainties of the depletion widths.
    :param effective_doping_tables:
    :param output_pdf: matplotlib pdf object to write the (final) figures to.
    :param i_row: row of the PixCap65 chip for which to perform the plotting of depletion data.
    :param tables:
    :param resistivities:
    """
    # will need to perform this step for every sensor/depletion object provided.
    # How to transform this check here
    if np.any(np.array([np.all(entry[i_col, i_row]) for entry in depletion_width_plates], dtype=bool)):
        temp_depletion_fit_propagate_parameters = []
        effective_dopings = []
        effective_resistivities = []
        doping_acceptors = []
        condition = """(row == {}) & (col == {})""".format(i_row, i_col)
        for table, effective_doping_table, resistivity in zip(tables, effective_doping_tables, resistivities):
            for x in table.where(condition):
                depletion_fit_propagate_parameters = {p_key: x[p_key] for p_key in ["NAD", "V", "dep", "sat"]}
                break
            else:
                depletion_fit_propagate_parameters = {}
            temp_depletion_fit_propagate_parameters.append(depletion_fit_propagate_parameters)
            doping_acceptor = depletion_fit_propagate_parameters["NAD"]
            effective_doping = effective_doping_table[i_col, i_row]
            effective_resistivity = resistivity[i_col, i_row]
            doping_acceptors.append(doping_acceptor)
            effective_dopings.append(effective_doping)
            effective_resistivities.append(effective_resistivity)

        bias_masks = np.array([bias_voltage < 0.5 for bias_voltage in bias_voltages], dtype=bool)

        with figure_provider(global_interactive_lock, 3, output=output_pdf) as (fig, ax, _):
            logger.debug("The type of bias_voltages is %s", type(bias_voltages))
            logger.debug("The shape of the bias voltages is %s", bias_voltages.shape)
            legend_title_str = ""
            for bias_voltage, depletion_width_plate, depletion_width_plate_error, bias_mask, depletion_fit_propagate_parameters, doping_acceptor, effective_doping in zip(bias_voltages, depletion_width_plates, depletion_width_plates_error, bias_masks, temp_depletion_fit_propagate_parameters, doping_acceptors, effective_dopings):
                if np.all(np.isfinite(depletion_width_plate_error[i_col, i_row])):
                    enhanced_error_bar(ax[0], bias_voltage[bias_mask], depletion_width_plate[i_col, i_row][bias_mask],
                                       yerr=depletion_width_plate_error[i_col, i_row][bias_mask], label='d-measurement')
                else:
                    ax[0].plot(bias_voltage[bias_mask], depletion_width_plate[i_col, i_row][bias_mask],
                               label='d-measurement', marker=None)

                # noqa: S125
                # sample_voltage = -1 * np.linspace(np.min(-bias_voltage), np.max(-bias_voltage) * 1.1, 1000)
                sample_voltage = np.linspace(np.min(bias_voltage[bias_mask]) * 1.1, np.max(bias_voltage[bias_mask]) / 1.1,
                                             1000)
                ax[0].plot(sample_voltage, model_depletion(sample_voltage, **depletion_fit_propagate_parameters),
                           label=f'd-theory for NAD = {doping_acceptor:4.2n}  and Ubi = {depletion_fit_propagate_parameters["V"]:.2n}',
                            marker=None)
                if not GENERATE_THESIS_PLOTS:
                    ax[0].set_title(f"Analysis of the depletion width for pixel ({i_col}, {i_row}).")
                    ax[1].set_title(f"Analysis of the effective doping for pixel ({i_col}, {i_row}).")
                    ax[2].set_title("Analysis of the effective doping")
                legend_title_str += f"Saturating at {depletion_fit_propagate_parameters['dep']} with {depletion_fit_propagate_parameters['sat']} saturation.\n"
                ax[1].plot(-bias_voltage, effective_doping, marker=None)
                ax[2].plot(depletion_width_plate[i_col, i_row], effective_doping, marker=None)
            ax[0].legend(title =legend_title_str)
            ax[0].set(xlabel='$U_\\text{{bi}}$ / \\unit{{\\volt}}', ylabel='$d$ / \\unit{{\\micro\\meter}}', )
            ax[0].grid(True)
            ax[1].set(xlabel='$U_\\text{{bi}}$ / \\unit{{\\volt}}',
                      ylabel='Effective \ndoping \nconcentration / \\unit{{\\per\\centi\\meter\\cubed}}')
            ax[1].grid(True)
            ax[1].set_yscale('log')
            ax[2].set(xlabel='$d$ / \\unit{{\\micro\\meter}}', ylabel='Effective\ndoping\nconcentration / \\unit{{\\per\\centi\\meter\\cubed}}')
            ax[2].set_yscale('log')
        with figure_provider(global_interactive_lock, 2, output=output_pdf) as (fig, ax, _):
            if not GENERATE_THESIS_PLOTS:
                ax[0].set_title(f"Analysis of the specific resistivity for pixel ({i_col}, {i_row}).")
                ax[1].set_title(f"Analysis of the specific resistivity for pixel ({i_col}, {i_row}).")

            for bias_voltage, effective_resistivity in zip(bias_voltages, effective_resistivities):
                ax[0].plot(-bias_voltage, effective_resistivity, marker=None)
                ax[1].plot(depletion_width_plate[i_col, i_row], effective_resistivity, marker=None)
            ax[0].set(xlabel='$U_\\text{{bi}}$ / \\unit{{\\volt}}', ylabel='$\\rho$ / \\unit{{\\ohm\\centi\\meter}}')
            ax[0].grid(True)
            ax[0].set_yscale('log')
            ax[1].set(xlabel='$d$ / \\unit{{\\micro\\meter}}', ylabel='$\\rho$ / \\unit{{\\ohm\\centi\\meter}}')
            ax[1].set_yscale('log')


def mp_plotting_init(backend, has_latex):
    """
    mp_plotting_init

    @author: Dominik Fischer
    @date: 2026-08-11

    Helper function to initialize process for plotting when using multiprocessing to accelerate things.
    :param backend: matplotlib backend to use for plotting.
    :type backend: str
    :param has_latex: whether to use LaTeX for plotting (axis and plot labels etc.).
    :type has_latex: bool
    """
    from pixcap65.utility.homogenize_plots import set_params
    import matplotlib
    import locale
    locale.setlocale(locale.LC_ALL, "de_DE")
    matplotlib.use(backend)
    # adjustments for the general visualization
    set_params(latex=has_latex,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                           r"stat,sys,sys-disp.}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                           r"retain-zero-uncertainty}", fig_height=8.26772, fig_width=11.69291,
               minor=True, fontsize=25, dpi=300)

    if IS_PRESENTATION:
        # adjustment for the power point presentation plots
        set_params(latex=has_latex,
                   latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                               r"stat,sys,sys-disp.}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                               r"retain-zero-uncertainty}", fig_height=3.043307, fig_width=3.519,
                   minor=True, fontsize=25, dpi=300)

    # adjustment for articles written using latex!
    if IS_THESIS:
        set_params(latex=has_latex,
                   latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                               r"stat,sys,sys-disp.}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                               r"retain-zero-uncertainty}",
                   minor=True, fontsize=25, dpi=300)


def error_handler(exc):
    """
    error_handler

    @author: Dominik Fischer
    @date: 2026-08-11

    Callback function to log exceptions raised by functions within a multiprocessing worker pool when using multiprocessing.
    :param exc: exception information to log
    """
    logger.error("While performing the plotting in multiple processes an error occured.", exc_info=exc)

if __name__ == '__main__':
    # plot_data(interpreted_data=os.path.expanduser('~/git/pixcap65/pixcap_LF_50x50_DC_R3_80V_HV.h5'))

    # some usage examples
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

    mp_plotting_init('PDF', has_latex)

    # use this attempt to achieve a better performance when generating the plots
    import multiprocessing as mp
    from examples.full_analysis import presentation_plotter, \
    bare_sample_plotter_second, x1_plotter, r13_plotter_second, x2_plotter_second, x5_plotter, x6_plotter, x7_plotter, \
    e1_plotter_second, r1_plotter, x4_plotter

    print(mp.current_process().name)
    print(mp.cpu_count())

    start = time.time()
    with mp.Manager() as manager, mp.Pool(initializer=mp_plotting_init, initargs=("PDF", has_latex,)) as pool:
        tables_lock = manager.RLock()
        presentation_plotter(tables_lock)
        process_handles = [
            bare_sample_plotter_second,
            x1_plotter,
            x2_plotter_second,
            x5_plotter,
            x6_plotter,
            x7_plotter,
            r13_plotter_second,
            e1_plotter_second,
            r1_plotter,
            x4_plotter,
        ]
        processes = [pool.apply_async(handle, (tables_lock,), error_callback=error_handler) for handle in process_handles]

        for p in processes:
            p.wait()
            print("Finished the process; Was it sucessful?", p.successful())
        del tables_lock
    print("Time elapsed: ", time.time() - start)
