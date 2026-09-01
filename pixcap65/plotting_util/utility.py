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
Utility functions to simplify the plotting implementations and in particular the access to the data which should be
plotted.
"""

import numpy as np
import os
import tables as tb
from contextlib import contextmanager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from pixcap65.analysis_util.utility import get_base_group, get_analysis_group
from pixcap65.utility import synchronized_process_open_file

NEW_PLOT_FILE_MODE = True
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
    :return: masked histogram/data set (masked pixels values are replaced by np.nan). If `perform_filter` is True,
        he mask is already applied and the masked values are no longer included.
    """
    result_hist = hist.copy()
    kargs = kwargs.copy()
    warn_level = 5
    if "exclude_cap_hist" in kargs:
        from warnings import warn
        warn("Found the unexpected keyword-argument 'exclude_cap_hist'."
             "The correct keyword should be 'test_cap_exclusion', but that might change again. "
             "If both are provided the first one will be ignored.", UserWarning, stacklevel=warn_level)
        kargs.setdefault("test_cap_exclusion", kargs.pop("exclude_cap_hist"))
    if "exclude_test_cap" in kargs:
        from warnings import warn
        warn(
            "Found the unexpected keyword-argument 'exclude_test_cap'. "
            "The correct keyword should be 'test_cap_exclusion', but that might change again. "
            "If both are provided the first one will be ignored.",
            UserWarning, stacklevel=warn_level)
        kargs.setdefault("test_cap_exclusion", kargs.pop("exclude_test_cap"))
    if "exclude_cap_test" in kargs:
        from warnings import warn
        warn(
            "Found the unexpected keyword-argument 'exclude_cap_test'. "
            "The correct keyword should be 'test_cap_exclusion', but that might change again. "
            "If both are provided the first one will be ignored.",
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
    Within the current implementation there is global flag to distinguish two modes when constructing the name
    of output pdf files.
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

    Context manager helper function to create plots/figures using matplotlib and destroy/close them
    appropriately afterwards.
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
    :param separate_plots: indicates whether the different axes should constructed within different matplotlib.Figure
    objects. (default: False)
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

    In contrast to the simple implementation the yielded tuple contains at index 1 the analysis groups of the
    different sensors.

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


DEFAULT_TEST_CAP_EXCLUSION = True
