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
Implementation of the plotting of the data for all scans involving a ramp of the biasing voltage.
"""

import numpy as np
import tables as tb
from matplotlib.backends.backend_pdf import PdfPages
from types import NoneType
from typing import Iterable, Union, Tuple, Optional
from warnings import warn

from pixcap65.analysis_util import CURRENT_CONVERSION_FACTOR, GENERAL_PIXCAP_SHAPE
from pixcap65.analysis_util.modelling.physics_modelling import model_depletion
from pixcap65.analysis_util.utility import check_leaf_unit, HIST_BIAS_MEAS_UNIT, HIST_CAP_UNIT, TABLES_LEAF_COMPAT_TYPE
from pixcap65.pixcap.pixcap_structure import CAPACITANCE_CONVERSION_FACTOR, DEFAULT_BIN_NUMBER
from pixcap65.plotting_util import GENERATE_THESIS_PLOTS, CV_USE_SEPARATE_PAGES, SENSOR_ITERABLE, \
    global_interactive_lock, CAPACITANCE_LABEL, BIAS_CURVE_Y_LABEL, BIAS_CURVE_X_LABEL
from pixcap65.plotting_util import logger
from pixcap65.plotting_util.utility import figure_provider
from pixcap65.utility.homogenize_plots import enhanced_error_bar
from pixcap65.utility.tables_util import group_get_file

LABEL_RESISTIVITY = '$\\rho$ / \\unit{{\\ohm\\centi\\meter}}'

LABEL_EFFECTIVE_DOPING = 'Effective\ndoping\nconcentration / \\unit{{\\per\\centi\\meter\\cubed}}'

LABEL_BIASING = '$U_\\text{{bi}}$ / \\unit{{\\volt}}'

LABEL_DEPLETION_DEPTH = '$d$ / \\unit{{\\micro\\meter}}'


def plot_bias_delegate(data_group, output_pdf: PdfPages, **kwargs):
    """
    plot_bias_delegate

    @author: Dominik Fischer
    @date: 2026-08-11

    Actual implementation for presenting the results of the I-V characterization.
    It's just a simple plot with error bars for the different quantities.

    :param data_group: hdf file's hierarchy group containing the raw data.
    :param output_pdf: PDF object to write the plots to.
    :keyword plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :keyword labels: required for multi-sensor plotting to label the plots from the different sensors correctly such that
    these could be identified. (Iterable)
    :keyword area_normalisation: areas of the individual pixel summed over all contributiong pixels. (Iterable)
    """
    interactive_lock = kwargs.get("plotting_lock", global_interactive_lock)
    with figure_provider(interactive_lock) as (fig, ax, _):
        if isinstance(data_group, Iterable) and not isinstance(data_group, tb.Node):
            labels = kwargs.pop("labels", ["Bias_data"] * len(data_group))
            norm_unit = "area_normalisation" in kwargs
            normalization = np.asarray(kwargs.pop("area_normalisation", np.full(len(data_group), 1.e8)),
                                       dtype=np.float64)
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
    # looks like there are no usable data points for the uncertainties of the leakage current.
    try:
        voltage_error = np.abs(tabular.col("DU"))
    except (AttributeError, KeyError):
        voltage_error = np.abs(voltage_data) * 0.0002 + 0.1
    if not np.all(np.isfinite(voltage_data)):
        current_errors = None
    if not GENERATE_THESIS_PLOTS:
        ax.set_title("Bias data from the measurement")
    ax.set(xlabel=BIAS_CURVE_X_LABEL, ylabel=BIAS_CURVE_Y_LABEL)

    # currently we could not use the correct voltage range, but we assume the errors to be within
    normalized_errors = None if current_errors is None else current_errors * CURRENT_CONVERSION_FACTOR / norm
    if current_errors is None:
        msg = "The current sensor seems to be missing measurement uncertainties for the leakage current! "
        "The sensor is labeld by {label}"
        warn(msg.format(label=label))
    enhanced_error_bar(ax, voltage_data, current_data * CURRENT_CONVERSION_FACTOR / norm, xerr=voltage_error,
                       yerr=normalized_errors, label=label)
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
    :keyword plotting_lock: synchronization primitve/"lock" to make sure only one **process** is able to create a new figure
        at the same time as matplotlib is not necessarily thread-safe.
    :keyword labels: required for multi-sensor plotting to label the plots from the different sensors correctly such that
        these could be identified. (Iterable)
    :keyword mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation.
    :keyword verbose: boolean, indicating whether to use verbose output for depletion voltages.
    :keyword distribution: boolean, indicating whether also the capacitance distribution of the whole sensor
        should be investigated.
    :keyword hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :keyword use_corrected: boolean, indicating whether to use the corrected capacitance for plotting. (data corrected for
        parasitic capacitances of PixCap65, default: False)
    :type use_corrected: bool
    """
    interactive_lock = kwargs.get('plotting_lock', global_interactive_lock)
    # extract the bias data
    voltage_data_sets = [
        __process_voltage_set(data_group)
    ] if isinstance(data_group, tb.Node) else [
        __process_voltage_set(data_set) for data_set in data_group
    ]
    approx_depletion = isinstance(data_group, tb.Node)

    labels = kwargs.pop('labels', [])
    if "pixel_mask" in kwargs:
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
        with figure_provider(interactive_lock, ncols=2, callback=response,
                             output=output_pdf, separate_plots=CV_USE_SEPARATE_PAGES) as (_, ax, back_pipe):
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
                view_depletion_width_plate = np.array(
                    [check_leaf_unit(ana.DepletionWidth, "um") for ana in analysis_group])
                view_depletion_width_plate_error = np.array(
                    [check_leaf_unit(ana.DepletionWidthErr, "um") for ana in analysis_group])
                view_effective_doping_table = np.array(
                    [check_leaf_unit(ana.DepletionEffDoping, "cm^-3") for ana in analysis_group])
                view_effective_resistivity_table = np.array(
                    [check_leaf_unit(ana.DepletionResistivity, "Ocm") for ana in analysis_group])
                view_table = [gr.DepletionParamTable for gr in analysis_group]
                view_origin_bias_voltages = np.array(
                    [check_leaf_unit(gr.BiasVoltageHist, HIST_BIAS_MEAS_UNIT) for gr in data_group])
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
        """
        Filter function to only plot histograms for some of the bias voltages if sufficiently many voltages are
        available.

        :param item: acutal item of current iteration
        :return: whether to use this item or not.
        """
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
    :keyword use_corrected: boolean, indicating whether to use the corrected capacitance for plotting. (data corrected for parasitic capacitances of PixCap65, default: False)
    :type use_corrected: bool
    :keyword is_combining: boolean, indicates whether multiple sensors are to be combined into a single figure. (default: False)
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
                    first_covariance = depletion_data["first_covariance"][dep_idx]
                    second_covariance = depletion_data["second_covariance"][dep_idx]
                    first_depletion_parameters = [depletion_fit_a[dep_idx], depletion_fit_b[dep_idx],]
                    second_depletion_parameters = [depletion_fit_c[dep_idx], depletion_fit_d[dep_idx],]
                    first_y, first_y_cov = propagate(lambda p: p[0] * first_voltage_x + p[1],
                                                     first_depletion_parameters, first_covariance)
                    second_y, second_y_cov = propagate(lambda p: p[0] * second_voltage_x + p[1],
                                                       second_depletion_parameters, second_covariance)

                    ax[1].plot(-first_voltage_x, first_y, '-', label="First section fit")
                    ax[1].plot(-second_voltage_x, second_y, '-', label="Second section fit")

                    first_y_error_prop = np.diag(first_y_cov) ** 0.5
                    second_y_error_prop = np.diag(second_y_cov) ** 0.5

                    ax[1].fill_between(-first_voltage_x, first_y - first_y_error_prop, first_y + first_y_error_prop,
                                       facecolor="C1", alpha=0.5)
                    ax[1].fill_between(-second_voltage_x, second_y - second_y_error_prop,
                                       second_y + second_y_error_prop,
                                       facecolor="C1", alpha=0.5)
                except ImportError:
                    pass
                except (tb.exceptions.NoSuchNodeError, ValueError, KeyError) as e:
                    from warnings import warn
                    warn("Something went wrong with the error bands of the depletion analysis.", stacklevel=1)
                    print("Found the exception:", e)
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
        x_limits, y_limits = __cv_plot_instance(ax, voltage_data, cap_data, cap_data_errors, label,
                                                title_format, x_limits, y_limits)
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

                ax[1].fill_between(-first_voltage_x, first_y - first_y_error_prop, first_y + first_y_error_prop,
                                   facecolor="C1", alpha=0.5)
                ax[1].fill_between(-second_voltage_x, second_y - second_y_error_prop, second_y + second_y_error_prop,
                                   facecolor="C1", alpha=0.5)
            except ImportError:
                pass

            except tb.exceptions.NoSuchNodeError as e:
                first_cap_calc = first_dep_parameters[0] * first_voltage_x + first_dep_parameters[1]
                second_cap_calc = second_dep_parameters[0] * second_voltage_x + second_dep_parameters[1]
                ax[1].plot(-first_voltage_x, first_cap_calc, '-', label="First section fit")
                ax[1].plot(-second_voltage_x, second_cap_calc, '-', label="Second section fit")
                # Implementations seems to be missing for the E1 general CV Data!
                from warnings import warn
                warn("Something went wrong with the error bands of the depletion analysis.")
                print("handling the exception:", e)
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

    :param analysis: hdf files group(s) containing the analysis results of the c-v-characterization for
        multiple sensors.
    :param row: row on the PixCap65 for which the c-v-curve should be plotted.
    :param col: column on the PixCap65 for which the c-v-curve should be plotted.
    :param ax: matplotlib.axes.Axes objects to plot into.
    :param voltage_data_sets: datasets of the applied bias voltages for (different) sensors.
        (could also contain the data for only a single sensor)
    :param labels: identifying names for the different sensors to use in the legend, when plotting for multiple sensors.
    :keyword is_distribution_plot: indicates wether we plot for the averaged sensor instead of a particular pixel
        (default: False)
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
    # TODO: document the different meanings of this array! (refers to the voltage_data array)
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
    enhanced_error_bar(ax[1], -voltage_data, adjusted_cap_data, yerr=np.abs(effective_capacitance_error_data),
                       label=label, alpha=0.5)
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


def plot_depletion_pixel_delegate(bias_voltages: Iterable[TABLES_LEAF_COMPAT_TYPE], i_col,
                                  depletion_width_plates: Iterable[TABLES_LEAF_COMPAT_TYPE],
                                  depletion_width_plates_error: Iterable[TABLES_LEAF_COMPAT_TYPE],
                                  effective_doping_tables: Iterable[TABLES_LEAF_COMPAT_TYPE], output_pdf: PdfPages,
                                  i_row, tables, resistivities):
    """
    plot_depletion_pixel_delegate

    @author: Dominik Fischer
    @date: 2026-08-11

    Utility function to plot the dependence of the estimated resistivities and the doping-profile onto the applied
    bias voltage (reversed bias).

    :param bias_voltages: array of the applied bias voltages to use for plotting and investigation.
    :param i_col: column of the PixCap65 chip for which to perform the plotting of depletion data.
    :param depletion_width_plates: matrix of depletion widths for the measurement using PixCap65 and
    the connected sensor.
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
            for bias_voltage, depletion_width_plate, depletion_width_plate_error, bias_mask, \
                depletion_fit_propagate_parameters, doping_acceptor, effective_doping in \
                zip(bias_voltages, depletion_width_plates, depletion_width_plates_error, bias_masks,
                    temp_depletion_fit_propagate_parameters, doping_acceptors, effective_dopings):
                if np.all(np.isfinite(depletion_width_plate_error[i_col, i_row])):
                    enhanced_error_bar(ax[0], bias_voltage[bias_mask], depletion_width_plate[i_col, i_row][bias_mask],
                                       yerr=depletion_width_plate_error[i_col, i_row][bias_mask], label='d-measurement')
                else:
                    ax[0].plot(bias_voltage[bias_mask], depletion_width_plate[i_col, i_row][bias_mask],
                               label='d-measurement', marker=None)

                # noqa: S125
                # sample_voltage = -1 * np.linspace(np.min(-bias_voltage), np.max(-bias_voltage) * 1.1, 1000)
                sample_voltage = np.linspace(np.min(bias_voltage[bias_mask]) * 1.1,
                                             np.max(bias_voltage[bias_mask]) / 1.1,
                                             1000)
                ax[0].plot(sample_voltage, model_depletion(sample_voltage, **depletion_fit_propagate_parameters),
                           label='d-theory for NAD = {:4.2n}  and Ubi = {:.2n}'.format(
                               doping_acceptor, depletion_fit_propagate_parameters["V"]),
                           marker=None)
                if not GENERATE_THESIS_PLOTS:
                    ax[0].set_title("Analysis of the depletion width for pixel ({}, {}).".format(i_col, i_row))
                    ax[1].set_title("Analysis of the effective doping for pixel ({}, {}).".format(i_col, i_row))
                    ax[2].set_title("Analysis of the effective doping")
                legend_title_str += "Saturating at {} with {} saturation.\n".format(
                    depletion_fit_propagate_parameters['dep'], depletion_fit_propagate_parameters['sat'])
                ax[1].plot(-bias_voltage, effective_doping, marker=None)
                ax[2].plot(depletion_width_plate[i_col, i_row], effective_doping, marker=None)
            ax[0].legend(title=legend_title_str)
            ax[0].set(xlabel=LABEL_BIASING, ylabel=LABEL_DEPLETION_DEPTH, )
            ax[0].grid(True)
            ax[1].set(xlabel=LABEL_BIASING,
                      ylabel=LABEL_EFFECTIVE_DOPING)
            ax[1].grid(True)
            ax[1].set_yscale('log')
            ax[2].set(xlabel=LABEL_DEPLETION_DEPTH, ylabel=LABEL_EFFECTIVE_DOPING)
            ax[2].set_yscale('log')
        with figure_provider(global_interactive_lock, 2, output=output_pdf) as (fig, ax, _):
            if not GENERATE_THESIS_PLOTS:
                ax[0].set_title("Analysis of the specific resistivity for pixel ({}, {}).".format(i_col, i_row))
                ax[1].set_title("Analysis of the specific resistivity for pixel ({}, {}).".format(i_col, i_row))

            for bias_voltage, effective_resistivity in zip(bias_voltages, effective_resistivities):
                ax[0].plot(-bias_voltage, effective_resistivity, marker=None)
                ax[1].plot(depletion_width_plate[i_col, i_row], effective_resistivity, marker=None)
            ax[0].set(xlabel=LABEL_BIASING, ylabel=LABEL_RESISTIVITY)
            ax[0].grid(True)
            ax[0].set_yscale('log')
            ax[1].set(xlabel=LABEL_DEPLETION_DEPTH, ylabel=LABEL_RESISTIVITY)
            ax[1].set_yscale('log')


NUMBER_DEPLETION_PLOT_POINTS = 1000
