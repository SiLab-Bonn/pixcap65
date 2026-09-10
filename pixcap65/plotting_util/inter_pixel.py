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
Implements the plotting of the inter-pixel capacitance scan (data).
"""

import numpy as np
import os
import tables as tb
from contextlib import contextmanager
from matplotlib.backends.backend_pdf import PdfPages
from typing import Iterable

from pixcap65.analysis_util.utility import get_base_group, check_leaf_unit, HIST_CURRENT_MEAS_UNIT, HIST_CAP_UNIT, \
    HIST_LEAK_CURRENT_UNIT, extract_parasitic_capacitance
from pixcap65.pixcap.pixcap_structure import CAPACITANCE_CONVERSION_FACTOR, \
    DEFAULT_BIN_NUMBER
from pixcap65.plotting_util import global_interactive_lock, FREQUENCY_LABEL, CURRENT_LABEL
from pixcap65.plotting_util.general import plot_2d_capacitance, plot_current_data, \
    plot_current_model, plot_1d_distribution
from pixcap65.plotting_util.utility import advanced_figure_provider
from pixcap65.utility import synchronized_process_open_file


@contextmanager
def inter_pix_data_fetch(path, group, lock, active_file: tb.File, type_name: str = 'total_cap'):
    """
    inter_pix_data_fetch

    @author Dominik Fischer
    @date 2026-08-11

    Utility function to fetch total-pixel-capacitance measurements reference data for plotting not only
    the 'in-pix' capacitance's but also the estimation for the inter-pixel capacitances'.

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
    :param inter_group: hdf files hierarchy group containing another inter-pixel measurements (results) for reference
        when extracting the individual contributions to the inter-pixel-capacitance. (default: None)
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
    """
    interactive_lock = kwargs.get("plotting_lock", global_interactive_lock)
    need_distribution = kwargs.get("distribution", False)
    lockless_propagation = {key: value for key, value in kwargs.items() if "lock" not in key}

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
        plot_2d_capacitance(total_ref_cap_hist-total_cap_hist, "Inter Pixel Capacitance from In-Pix C",
                            output_pdf, **kwargs)
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
        plot_1d_distribution(total_cap_hist, "Total Pixel Capacitance Distribution", 10000, distribution_result_data, output_pdf, analysis_group, unit=actual_unit, capacitance=total_cap_hist, **kwargs)
        # with advanced_figure_provider(interactive_lock) as (fig, ax):
        #     hist_cap_hist = evaluate_pixel_mask(total_cap_hist, **kwargs)
        #     ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
        #             bins=n_bins)
        #     ax.set_ylabel(COUNTS_HIST_LABEL)
        #     ax.set_xlabel(HIST_PIX_CAP_LABEL)
        #     title_str = "Pixel Total Capacitance Distribution"
        #     if not GENERATE_THESIS_PLOTS:
        #         ax.set_title(__get_1d_hist_label(10000, title_str, distribution_result_data, unit=actual_unit))
        #     ax.grid()
        #     output_pdf.savefig(fig, bbox_inches='tight')
        # if need_distribution:
        #     from pixcap65.analysis import analyze_capacitance_distribution_delegate
        #     # FIXME: when using the replacement handler this here might break! (needs verification)
        #     analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=total_cap_hist,
        #                                               set_parasitic=False, **kwargs)

        # handle the inter-pix contributions by making use of the reference in-pix capacitance's
        if in_ref_cap_hist is not None:
            effective_inter_cap_hist = total_cap_hist - in_ref_cap_hist
            # FIXME: this is ignoring the special case of lockless propagation! (needs verification)
            plot_1d_distribution(effective_inter_cap_hist, "Component of Inter-Capacitance Distribution",
                                 kwargs.get('grouped_inter_pix_id', 18000),
                                 distribution_result_data, output_pdf, analysis_group, unit=actual_unit,
                                 capacitance=effective_inter_cap_hist, **kwargs)
            # with advanced_figure_provider(interactive_lock) as (fig, ax):
            #     hist_inter_cap_hist = evaluate_pixel_mask(effective_inter_cap_hist, **lockless_propagation)
            #     ax.hist(hist_inter_cap_hist[~np.isnan(hist_inter_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
            #             bins=n_bins)
            #     ax.set_ylabel(COUNTS_HIST_LABEL)
            #     ax.set_xlabel(HIST_PIX_CAP_LABEL)
            #     if not GENERATE_THESIS_PLOTS:
            #         ax.set_title(
            #             __get_1d_hist_label(kwargs.get('grouped_inter_pix_id', 18000),
            #                                 "Component of Inter-Capacitance Distribution",
            #                                 distribution_result_data,
            #                                 unit=actual_unit))
            #     ax.grid()
            #     output_pdf.savefig(fig, bbox_inches='tight')
            # if need_distribution:
            #     from pixcap65.analysis import analyze_capacitance_distribution_delegate
            #     analyze_capacitance_distribution_delegate(analysis_group, output_pdf,
            #                                               capacitance=effective_inter_cap_hist,
            #                                               set_parasitic=False, **lockless_propagation)

        # handle the inter-pixel capacitanes by making use of the provided total capacitance measurement
        if total_ref_cap_hist is not None:
            effective_inter_cap_hist = total_ref_cap_hist - total_cap_hist
            plot_1d_distribution(effective_inter_cap_hist, "Pixel Inter Capacitance Distribution",
                                 14000,
                                 distribution_result_data, output_pdf, analysis_group, unit=actual_unit,
                                 capacitance=effective_inter_cap_hist, **kwargs)
            # with advanced_figure_provider(interactive_lock) as (fig, ax):
            #     hist_inter_cap_hist = evaluate_pixel_mask(effective_inter_cap_hist, **lockless_propagation)
            #     ax.hist(hist_inter_cap_hist[~np.isnan(hist_inter_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
            #             bins=n_bins)
            #     ax.set_ylabel(COUNTS_HIST_LABEL)
            #     ax.set_xlabel(HIST_PIX_CAP_LABEL)
            #     if not GENERATE_THESIS_PLOTS:
            #         ax.set_title(__get_1d_hist_label(14000, "Pixel Inter Capacitance Distribution",
            #                                          distribution_result_data, unit=actual_unit))
            #     ax.grid()
            #     output_pdf.savefig(fig, bbox_inches='tight')
            # if need_distribution:
            #     from pixcap65.analysis import analyze_capacitance_distribution_delegate
            #     analyze_capacitance_distribution_delegate(analysis_group, output_pdf,
            #                                               capacitance=effective_inter_cap_hist,
            #                                               set_parasitic=False, **kwargs)

    if np.count_nonzero(np.isfinite(inter_a_cap_hist)) > 2:
        plot_1d_distribution(inter_a_cap_hist, "Inter-Pixel A Capacitance Distribution",
                             11000,
                             distribution_result_data, output_pdf, analysis_group, unit=actual_unit,
                             capacitance=inter_a_cap_hist, **kwargs)
        # with advanced_figure_provider(interactive_lock) as (fig, ax):
        #     hist_cap_hist = evaluate_pixel_mask(inter_a_cap_hist, **kwargs)
        #     ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
        #             bins=n_bins)
        #     ax.set_ylabel(COUNTS_HIST_LABEL)
        #     ax.set_xlabel(HIST_PIX_CAP_LABEL)
        #     if not GENERATE_THESIS_PLOTS:
        #         ax.set_title(__get_1d_hist_label(11000, "Inter-Pixel A Capacitance Distribution",
        #                                          distribution_result_data, unit=actual_unit))
        #     ax.grid()
        #     output_pdf.savefig(fig, bbox_inches='tight')
        # if need_distribution:
        #     from pixcap65.analysis import analyze_capacitance_distribution_delegate
        #     analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=inter_a_cap_hist,
        #                                               set_parasitic=False, **kwargs)

    if np.count_nonzero(np.isfinite(inter_b_cap_hist)) > 2:
        plot_1d_distribution(inter_b_cap_hist, "Inter-Pixel B Capacitance Distribution",
                             12000,
                             distribution_result_data, output_pdf, analysis_group, unit=actual_unit,
                             capacitance=inter_b_cap_hist, **kwargs)
        # with advanced_figure_provider(interactive_lock) as (fig, ax):
        #     hist_cap_hist = evaluate_pixel_mask(inter_b_cap_hist, **kwargs)
        #     ax.hist(hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR,
        #             bins=n_bins)
        #     ax.set_ylabel(COUNTS_HIST_LABEL)
        #     ax.set_xlabel(HIST_PIX_CAP_LABEL)
        #     if not GENERATE_THESIS_PLOTS:
        #         ax.set_title(__get_1d_hist_label(12000, "Inter-Pixel B Capacitance Distribution",
        #                                          distribution_result_data, unit=actual_unit))
        #     ax.grid()
        #     output_pdf.savefig(fig, bbox_inches='tight')
        # if need_distribution:
        #     from pixcap65.analysis import analyze_capacitance_distribution_delegate
        #     analyze_capacitance_distribution_delegate(analysis_group, output_pdf, capacitance=inter_b_cap_hist,
        #                                               set_parasitic=False,
        #                                               **kwargs)

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
