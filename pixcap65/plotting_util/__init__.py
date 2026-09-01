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
Plotting of Pixcap65 data.
"""
import atexit
import logging
import numpy as np
import tables as tb
import threading
from matplotlib.backends.backend_pdf import PdfPages
from typing import Union, List, Tuple, Iterable
from warnings import warn

from pixcap65.analysis_util import GENERAL_PIXCAP_SHAPE
from pixcap65.analysis_util.utility import get_base_group, get_analysis_group, check_leaf_unit, HIST_BIAS_MEAS_UNIT
from pixcap65.plotting_util.utility import get_pdf_name, multi_sensor_file_handler_simple, \
    multi_sensor_file_handler_advanced
from pixcap65.utility import synchronized_process_open_file
from pixcap65.utility.utils_2 import GroupType

LATEX_PREAMBLE = r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={" \
                 r"stat,sys,sys-disp.}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{" \
                 r"retain-zero-uncertainty}"

logger = logging.getLogger(__name__)


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
               latex_extra=LATEX_PREAMBLE, fig_height=8.26772, fig_width=11.69291,
               minor=True, fontsize=25, dpi=300)

    if IS_PRESENTATION:
        # adjustment for the power point presentation plots
        set_params(latex=has_latex,
                   latex_extra=LATEX_PREAMBLE, fig_height=3.043307, fig_width=3.519,
                   minor=True, fontsize=25, dpi=300)

    # adjustment for articles written using latex!
    if IS_THESIS:
        set_params(latex=has_latex,
                   latex_extra=LATEX_PREAMBLE,
                   minor=True, fontsize=25, dpi=300)


def error_handler(exc):
    """
    error_handler

    @author: Dominik Fischer
    @date: 2026-08-11

    Callback function to log exceptions raised by functions within a multiprocessing worker pool when
    using multiprocessing.
    :param exc: exception information to log
    """
    logger.error("While performing the plotting in multiple processes an error occured.", exc_info=exc)


IS_PRESENTATION = False
IS_THESIS = False
global_interactive_lock = threading.RLock()


def release_lock():
    """
    Release the lock and deinitialize the acquired lock to clear memory and prevent leakage of semaphore objects
    outside the process.
    """
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
    :keyword use_corrected: boolean, indicating whether to use the corrected capacitance for plotting.
        (data corrected for parasitic capacitances of PixCap65, default: False)
    :type use_corrected: bool
    :keyword test_cap_exclusion: boolean, whether to exclude the test capacitator row from the histograms. (default: False)
    :type test_cap_exclusion: bool
    :keyword hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :keyword mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :keyword mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :keyword mask_upper: float, threshold to mask all pixels above this value.
    :type mask_upper: float
    :keyword distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor.
        (default: False)
    :type distribution: bool
    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :keyword plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create
        a new figure at the same time as matplotlib is not necessarily thread-safe.
    :keyword unit: unit of the capacities presented within the plot.
    :type unit: str
    :keyword capacitance: histogram of the capacitance to use instead of those extracted from the provided hdf files group.
    :keyword no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
        When investigating the capacitance distribution.
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :keyword apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword fit_plot_pdf: PDF object to save the fit figures to.


    """
    from pixcap65.plotting_util.general import plot_data_delegate

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
    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :keyword plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :keyword distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor.
        (default: False) [boolean]
    :type distribution: bool
    :keyword hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :keyword test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :keyword mask_pixel: array of tuple of pixel positions to be masked.
    :keyword mask_lower: float, threshold to mask all pixels below this value.
    :type mask_lower: float
    :keyword mask_upper: float, threshold to mask all pixels above this value.
    :type mask_upper: float
    :keyword no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :keyword apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword fit_plot_pdf: PDF object to save the fit figures to.
    :keyword use_corrected: boolean, whether to use the corrected capacitance's for plotting.
    :keyword apply_correction: boolean, whether to use the corrected capacitance's for plotting/extraction.
    """
    from pixcap65.plotting_util.inter_pixel import inter_pix_data_fetch, plot_inter_pix_data_delegate

    lock = kwargs.pop("lock", None)
    # determine the pdf file
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)

    with PdfPages(pdf_name) as output_pdf:
        with synchronized_process_open_file(interpreted_data, mode='r', lock=lock) as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            # we need to fetch the correcponding group
            with inter_pix_data_fetch(total_data, total_path, lock, in_file_h5) as total_group, \
                    inter_pix_data_fetch(inter_data, inter_path, lock, in_file_h5, 'inter_cap') as inter_group:
                plot_inter_pix_data_delegate(base_group.inter_cap.measurements,
                                             get_analysis_group(base_group.inter_cap, **kwargs), output_pdf,
                                             total_group, inter_group, **kwargs)


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
    :keyword lock: locking object used to synchronize the access to the file handles by the pytables library. It is highly
        encouraged to provide an explicit lock here.
    :keyword plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :keyword labels: required for multi-sensor plotting to label the plots from the different sensors correctly
        such that these could be identified. (Iterable)
    :keyword area_normalisation: areas of the individual pixel summed over all contributiong pixels. (Iterable)
    """
    from pixcap65.plotting_util.biasing import plot_bias_delegate

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

    On request also the doping profile, estimated from the differential capacitance and the depletion width
    (their dependence onto the applied voltage) is presented.
    In this case the estimated doping profile across the sensors thickness is plotted in dependence of the applied
    bias voltage and the depletion width.
    In Addition the corresponding fits to model these profiles are presented if their parameters are estimated before.
    Also the resistivity profile will be plotted.


    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix:  additional suffix to use for naming the PDF containing the plots.
    :param use_group:   boolean, whether to append the group name of the measurements to the PDF name.
    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :keyword pdf_name: file name for the output pdf file.
    :keyword use_corrected: boolean, whether to use the corrected capacitance's for plotting.
    :type use_corrected: bool
    :keyword apply_correction: boolean, whether to use the corrected capacitance's for plotting/extraction.
    :type apply_correction: bool
    :keyword verbose: boolean, indicating whether to use verbose output for depletion voltages
    :keyword distribution: boolean, indicating whether also the capacitance distribution of the whole sensor
        should be investigated.
    :keyword plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :keyword labels: required for multi-sensor plotting to label the plots from the different sensors correctly such that
        these could be identified. (Iterable)
    :keyword mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :keyword hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    """
    from pixcap65.plotting_util.biasing import plot_cv_data_delegate, plot_depletion_pixel_delegate

    file_lock = kwargs.get("lock", None)
    if isinstance(interpreted_data, Iterable) and not isinstance(interpreted_data, str):
        with multi_sensor_file_handler_advanced(interpreted_data, base_path, **kwargs) as (groups, analysis_groups,
                                                                                           output_pdf):
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

    On request also the doping profile, estimated from the differential capacitance and the depletion width
    (their dependence onto the applied voltage) is presented.
    In this case the estimated doping profile across the sensors thickness is plotted in dependence of the applied
    bias voltage and the depletion width.
    In Addition the corresponding fits to model these profiles are presented if their parameters are estimated before.
    Also the resistivity profile will be plotted.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :keyword pdf_name: file name for the output pdf file.
    :keyword use_corrected: boolean, whether to use the corrected capacitance's for plotting.
    :type use_corrected: bool
    :keyword apply_correction: boolean, whether to use the corrected capacitance's for plotting/extraction.
    :type apply_correction: bool
    :keyword verbose: boolean, indicating whether to use verbose output for depletion voltages
    :keyword distribution: boolean, indicating whether also the capacitance distribution of the whole sensor
        should be investigated.
    :keyword plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :keyword labels: required for multi-sensor plotting to label the plots from the different sensors correctly such that
        these could be identified. (Iterable)
    :keyword mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :keyword hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :keyword labels: required for multi-sensor plotting to label the plots from the different sensors correctly such that
        these could be identified. (Iterable)
    :keyword area_normalisation: areas of the individual pixel summed over all contributiong pixels. (Iterable)
    """
    from pixcap65.plotting_util.biasing import plot_bias_delegate, plot_cv_data_delegate

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
    from pixcap65.plotting_util.biasing import plot_depletion_pixel_delegate

    warn("Found the 'unused' additional function for depletion delegation!")
    data_groups = np.atleast_1d(data_group)
    analysis_groups = np.atleast_1d(analysis_group)
    # extract the depletion parameters
    view_depletion_width_plate = np.array([check_leaf_unit(group.DepletionWidth, "um") for group in analysis_groups])
    view_depletion_width_plate_error = np.array(
        [check_leaf_unit(group.DepletionWidthErr, "um") for group in analysis_groups])
    view_effective_doping_table = np.array(
        [check_leaf_unit(group.DepletionEffDoping, "cm^-3") for group in analysis_groups])
    view_bias_voltages = np.array([check_leaf_unit(group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT) for group in
                                   data_groups])
    view_table = [group.DepletionParamTable for group in analysis_groups]
    view_effective_resistivity_table = np.array(
        [check_leaf_unit(group.DepletionResistivity, "Ocm") for group in analysis_groups])
    # CHECK: what about here with handling multiple-sensors? (under investigation)

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
        plot_depletion_pixel_delegate(view_bias_voltages, col, view_depletion_width_plate,
                                      view_depletion_width_plate_error,
                                      view_effective_doping_table, output_pdf, row, view_table,
                                      view_effective_resistivity_table)


CAPACITANCE_LABEL = "$C$ / \\unit{{\\femto\\farad}}"
FREQUENCY_LABEL = '$f$ / \\unit{{\\mega\\hertz}}'
CURRENT_LABEL = '$I$ / \\unit{{\\nano\\ampere}}'
BIAS_CURVE_Y_LABEL = "$I$ / \\unit{{\\nano\\ampere}}"
BIAS_CURVE_X_LABEL = "$U$ / \\unit{{\\volt}}"
