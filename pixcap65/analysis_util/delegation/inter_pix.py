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
import os
import tables as tb
import uuid

try:
    # noinspection PyCompatibility
    from collections.abc import Callable, Sized, Iterable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Callable, Sized, Iterable
finally:
    from typing import Optional

from contextlib import contextmanager

from pixcap65.analysis_util.constants import ADVANCED_PARAMETER_TYPE
from pixcap65.analysis_util.utility import check_leaf_unit, HIST_CURRENT_MEAS_UNIT

from pixcap65.utility import synchronized_process_open_file
from pixcap65.utility.utils_2 import walk_to_node, create_carray


@contextmanager
def perform_inter_pix_fetch(path, group, active_file, target: str, lock=None):
    """
    perform_inter_pix_fetch

    @author: Dominik Fischer
    last update: 2026-08-12

    For the analysis of the inter-pixel capacitance some reference data from total pixel capacitance runs is required.
    This helper function will load these these (independently whether they are in the same h5 file or in a different one.
    It will then yield (as a context manager) the corresponding group and the file object.
    The tuple item at index 2 indicates whether a new file handle was opened to fetch the data.

    :param path: path/filename from which to read the reference data.
    :param group: hdf files' group storing the reference data.
    :param active_file: .h5 file where the current iner-pixel analysis data is stored.
    :param target: kind of measurement to be fetched.
    :param lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    """
    if path is not None:
        general_h5_file = os.path.abspath(active_file.filename)
        inter_pix_h5_file = os.path.abspath(path)
        if general_h5_file == inter_pix_h5_file:
            yield group, active_file, False
        elif os.path.exists(path):
            with synchronized_process_open_file(path, mode='a', lock=lock) as total_h5_file:
                yield group, total_h5_file, True
        else:
            from warnings import warn
            warn("Failed to load the referenced cap data files. It looks like the provided path does not exist.")
            yield None, None, False
    else:
        yield None, None, False


def perform_inter_pix_deep_dive(reference_group, get_total_cap_group: Optional[str],
                                in_file_h5: tb.File, apply_correction_arg, parasitic: float,
                                parasitic_error: float, get_inter_cap_group: Optional[str], **kwargs):
    """
    perform_inter_pix_deep_dive

    @author: Dominik Fischer
    last update: 2026-08-13

    Utility function to perform the (actual) analysis of the **inter-pixel** capacitance.
    The only thing going in this direction done so or must be done upfront, is analysis the three smu channels and
    extract appropritate capacitance's from them.
    As the setup will measure the in-pix or backplane capacities by suitable triggering neighbour pixel, we need a
    total-cap measurement for reference to extract the inter-pixel capacitance (total contribution).
    If we want to extract the single contributions we also need a reference dataset for the total inter-pixel measurement data.

    If neither a total_cap_file is provided nor inter-pixel resolution is requested this will skip and return a tuple of None directly.

    If it is necessary to open a new file handle in order to read the reference data, a new temporary group will be created.
    (at top level)
    This temporary group will contain references to the fetched data and should be removed at some point.
    Anyway, references to the groups objects both for total-cap and inter-pix reference data will returned.
    If one of these was not read at all the value will be None.

    In the end the inter-pix data will be determined by the simple subtration total-cap - in-pix and the results will be
    written as a matrix to all the previous analysis results.
    As a last step the in-pixel/backplane capacities are corrected for the parasitic capacitance of the measurement, if
    a non-vanishing parasitic capacitance estimation is provided.

    :param reference_group: hdf files hierarchy group object containing the analysis results up to this point in
        a subgroup 'analysis'. Any data already corrected for parasitic effects will be ignored.
    :type reference_group: pytables.Group
    :param get_total_cap_group: hdf files' group storing the reference data for the total-cap measurements.
        The specified group must have an analysis sub-group but must already reference the inter-pix tree element.
    :type get_total_cap_group: str
    :param in_file_h5: .h5-file from which the 'main' data is read and where to write the results to.
    :type in_file_h5: pytables.File
    :param apply_correction_arg: boolean, apply_correction: boolean, indicates whether the measured capacitance should be corrected
        immediately; Will require the presence of further arguments as information about the parasitic capacitance needs to be
        submitted. (data corrected for parasitic capacitances of PixCap65, default: False)
    :type apply_correction_arg: bool
    :param parasitic: parasitic capcitance of the measurement circuit to be assumed
    :type parasitic: float
    :param parasitic_error: uncertainty/deviation of the parasitic capacitance of the circuit to be assumed.
    :type parasitic_error: float
    :param get_inter_cap_group: hdf files' group storing the reference data for the full inter-pix measurements.
    :type get_inter_cap_group: str
    :keyword total_ref_file: path/filename from which to read the reference data of the total-cap measurements.
    :keyword inter_ref_file: path/filename from which to read the reference data of the full inter-pixel measurements.
    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :keyword is_inter: indicates whether the single contributions to the inter-pixel capacitance shall
        resolved (default: False)
    :type is_inter: bool
    :return: tuple of hdf files group objects (or None) for the total-pix and inter-pix reference data.
    :rtype: tuple
    """
    # no necessity for accessing the manager!
    get_total_cap_file = kwargs.pop("total_ref_file", None)
    get_inter_cap_file = kwargs.pop("inter_ref_file", None)
    lock = kwargs.get("lock", None)
    if not kwargs.pop("is_inter", False) or get_total_cap_file is None:
        return None, None

    # In particular to also present the calculated systematic uncertainties.
    # without the additional data from the total capactiance measurement
    assert get_total_cap_group is not None
    with perform_inter_pix_fetch(get_total_cap_file, get_total_cap_group, in_file_h5, 'total_cap', lock=lock) as (total_h5_group,
                                                                                                         total_h5_file,
                                                                                                         total_temp), \
            perform_inter_pix_fetch(get_inter_cap_file, get_inter_cap_group, in_file_h5, 'inter_cap', lock=lock) as (
                    inter_h5_group, inter_h5_file, inter_temp):
        try:
            total_node, inter_node = __perform_inter_pix_deeper(apply_correction_arg, total_h5_group, in_file_h5, parasitic,
                                                                parasitic_error, reference_group, total_h5_file,
                                                                inter_h5_file, inter_h5_group)
        except:
            print("Test the information about the provided arguments")
            print(get_total_cap_file)
            print(total_h5_file)
            raise
        if any((total_temp, inter_temp)):
            if 'temporary' not in in_file_h5.root:
                in_file_h5.create_group(in_file_h5.root, name="temporary")
            if total_temp:
                total_node = total_h5_file.copy_node(where=total_node._v_parent, name=total_node._v_name,
                                                     newparent=in_file_h5.root.temporary, newname=uuid.uuid4().hex,
                                                     recursive=True)
            if inter_temp:
                inter_node = inter_h5_file.copy_node(where=inter_node._v_parent, name=inter_node._v_name,
                                                     newparent=in_file_h5.root.temporary, newname=uuid.uuid4().hex,
                                                     recursive=True)
        return total_node, inter_node


def __perform_inter_pix_deeper(apply_correction_arg, get_total_cap_group: str, in_file_h5: tb.File, parasitic: float,
                               parasitic_error: float, reference_group, total_h5_file: tb.File,
                               inter_h5_file: Optional[tb.File] = None,
                               inter_h5_group: Optional[str] = None) -> tb.Group:
    # this reference implementation has the drawback that always the uncorrected data is used.
    # this should not make any difference as we are taking the differences (BUT: the in-pix capacitances are still effected).
    # FIXME: Why is there no escape channel in the case there is no total_cap data provided
    total_cap_data, _ = walk_to_node(total_h5_file.root, get_total_cap_group, create=False, verify_create=True)
    # Why going here to the root layer?
    inter_cap_data, _ = (None, None) if inter_h5_file is None or inter_h5_group is None else walk_to_node(
        inter_h5_file.root, inter_h5_group, create=False, verify_create=True)
    try:
        total_cap = total_cap_data.analysis.HistCap[:]
    except:
        print(total_h5_file.filename)
        assert isinstance(total_cap_data, tb.Group)
        print(total_cap_data._f_list_nodes())
        raise
    total_cap_error = total_cap_data.analysis.HistCapErr[:]

    inter_pix_cap = total_cap - reference_group.analysis.HistCap[:]
    inter_pix_cap_err = total_cap_error - reference_group.analysis.HistCapErr[:]

    # account for systematic effects?
    in_pix_cap = reference_group.analysis.HistCap[:]
    in_pix_cap_err = reference_group.analysis.HistCapErr[:]
    create_carray(in_file_h5, where=reference_group.analysis, name="InPixHistCap", obj=in_pix_cap)
    create_carray(in_file_h5, where=reference_group.analysis, name="InPixHistCapErr", obj=in_pix_cap_err)
    create_carray(in_file_h5, where=reference_group.analysis, name="InterPixHistCap", obj=inter_pix_cap)
    create_carray(in_file_h5, where=reference_group.analysis, name="InterPixHistCapErr", obj=inter_pix_cap_err)

    # now account at last for modified inter_cap
    if inter_cap_data is not None:
        in_cap = inter_cap_data.analysis.HistCap[:]
        in_cap_error = inter_cap_data.analysis.HistCapErr[:]
        mod_inter_pix_cap = reference_group.analysis.HistCap[:] - in_cap
        mod_inter_pix_cap_error = reference_group.analysis.HistCapErr[:] - in_cap_error
        create_carray(in_file_h5, where=reference_group.analysis, name="ModInterPixHistCap", obj=mod_inter_pix_cap)
        create_carray(in_file_h5, where=reference_group.analysis, name="ModInterPixHistCapErr",
                      obj=mod_inter_pix_cap_error)

    if apply_correction_arg:
        # must retrieve the parasitics first.
        in_pix_cap -= parasitic
        in_pix_cap_err = np.where(np.isfinite(in_pix_cap), np.sqrt(in_pix_cap_err ** 2 + parasitic_error ** 2),
                                  np.nan)
        create_carray(in_file_h5, where=reference_group.analysis_correction, name="InPixHistCap", obj=in_pix_cap)
        create_carray(in_file_h5, where=reference_group.analysis_correction, name="InPixHistCapErr",
                      obj=in_pix_cap_err)
        in_file_h5.copy_node(where=reference_group.analysis, newparent=reference_group.analysis_correction,
                             newname="InterPixHistCap", name="InterPixHistCap")
        in_file_h5.copy_node(where=reference_group.analysis, newparent=reference_group.analysis_correction,
                             newname="InterPixHistCapErr", name="InterPixHistCapErr")
        if inter_cap_data is not None:
            in_file_h5.copy_node(where=reference_group.analysis, newparent=reference_group.analysis_correction,
                                 newname="ModInterPixHistCap", name="ModInterPixHistCap")
            in_file_h5.copy_node(where=reference_group.analysis, newparent=reference_group.analysis_correction,
                                 newname="ModInterPixHistCapErr", name="ModInterPixHistCapErr")

        # FIMXE: under this circumstances we are still missing the distribution information about these!
        in_file_h5.flush()

    return total_cap_data, inter_cap_data


def _handle_inter_pix_capacitance(file: tb.File, data_group: tb.Group, is_advanced: ADVANCED_PARAMETER_TYPE,
                                  perform_analysis: Callable[..., None], result_group: tb.Group, **kwargs):
    """
    _handle_inter_pix_capacitance

    @author: Dominik Fischer
    last update: 2026-08-12

    (Internal) utility function analyzing the different inter-pixel/in-pix capacitances.
    The PixCap65 chip and the setup allows for up to three current measurement series per pixel.
    Thus the (primary) analysis handler to determine the capacitances needs to run three times.
    In case of the analysis of the SMU channel connected to VM1 it is necessary to assume a voltage over the capacitance
    of -2 V and adjust the determination of the capacitance for this.

    The deviating names for the different datasets are handled automatically.


    :param file: h5 file object containing the data to be analysed.
    :param data_group: hdf files' group containing the measurement data for inter-pix/in-pix capacitance measurement.
    :param is_advanced: boolean indicating if the advanced analysis strategy should be used
        or not (may require additional keyword arguments)
    :param perform_analysis: function/callable to handle the analysis for individual pixels
    :param result_group: hdf files' group to write the analysis results of the individual pixels to.
    :keyword current_error_hist: histogram/array-like of the uncertainties of the current measurements for the pixels; 2D-Array for the errors of the current data. This keyword argument must be present
        for the advanced analysis strategy.
    :type current_error_hist: numpy.ndarray
    :keyword full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used.
    :keyword use_kafe2: boolean, indicates whether kafe2 is used for the fit. (default: False)
    :keyword plot: boolean, indicates whether to plot the data. An output PDF object could be submitted here instead of an explicitly created one. (Default: False)
    :keyword apply_contour: boolean, indicates whether to determine the contours and try to plot them. (default: False)
    :keyword fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided)
    :keyword output_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided)
    """
    current_hist = check_leaf_unit(data_group.TotalHistCurr, HIST_CURRENT_MEAS_UNIT)
    if is_advanced and "TotalHistCurr" in data_group:
        current_error_hist = check_leaf_unit(data_group.TotalHistCurr, HIST_CURRENT_MEAS_UNIT)
    else:
        current_error_hist = np.full_like(current_hist, fill_value=np.nan)
    kwargs["current_error_hist"] = current_error_hist

    # Read scan parameters
    scan_parameters = data_group.scan_params[:]
    assert isinstance(scan_parameters, tb.Table) or isinstance(scan_parameters, np.ndarray)
    assert isinstance(current_hist, tb.CArray) or isinstance(current_hist, np.ndarray)
    perform_analysis(file, result_group, current_hist, scan_parameters, **kwargs)

    # handle the first inter-pixel measurement.
    inter_a_output_dict = {
        "cap_name": "HistCapInterA",
        "cap_title": "Capacitance Histogram of inter pixel A",
        "cap_err_name": "HistCapErrInterA",
        "cap_err_title": "Capacitance Error Histogram of inter pixel A",
        "leak_name": "HistLeakInterA",
        "leak_title": "Leakage Current Histogram of inter pixel A",
        "leak_error_name": "HistLeakErrInterA",
        "leak_error_title": "Leakage Current Error Histogram of inter pixel A",
        "resistor_name": "HistResInterA",
        "resistor_title": "On-Resistance Histogram of inter pixel A",
        "resistor_error_name": "HistResErrInterA",
        "resistor_error_title": "On-Resistance Error Histogram of inter pixel A",
        "cov_name": "HistFitCovInterA",
        "cov_title": 'Fit Covariance Matrix of inter pixel A'
    }
    kwargs.update(inter_a_output_dict)
    current_hist_temp = check_leaf_unit(data_group.InterHistCurrA, HIST_CURRENT_MEAS_UNIT)
    back_current_hist = current_hist_temp - current_hist
    current_hist = current_hist_temp
    if is_advanced and "InterHistCurrErrA" in data_group:
        current_error_hist = check_leaf_unit(data_group.InterHistCurrErrA, HIST_CURRENT_MEAS_UNIT)
    else:
        current_error_hist = np.full_like(current_hist, fill_value=np.nan)

    kwargs["current_error_hist"] = current_error_hist
    assert isinstance(current_hist, tb.CArray) or isinstance(current_hist, np.ndarray)
    perform_analysis(file, result_group, current_hist, scan_parameters, **kwargs)
    inter_c_output_dict = {
        "cap_name": "HistCapInterC",
        "cap_title": "Capacitance Histogram of inter pixel C",
        "cap_err_name": "HistCapErrInterC",
        "cap_err_title": "Capacitance Error Histogram of inter pixel C",
        "leak_name": "HistLeakInterC",
        "leak_title": "Leakage Current Histogram of inter pixel C",
        "leak_error_name": "HistLeakErrInterC",
        "leak_error_title": "Leakage Current Error Histogram of inter pixel C",
        "resistor_name": "HistResInterC",
        "resistor_title": "On-Resistance Histogram of inter pixel C",
        "resistor_error_name": "HistResErrInterC",
        "resistor_error_title": "On-Resistance Error Histogram of inter pixel C",
        "cov_name": "HistFitCovInterC",
        "cov_title": 'Fit Covariance Matrix of inter pixel C'
    }
    kwargs.update(inter_c_output_dict)
    assert isinstance(current_hist, tb.CArray) or isinstance(current_hist, np.ndarray)
    perform_analysis(file, result_group, back_current_hist, scan_parameters, **kwargs)

    inter_b_output_dict = {
        "cap_name": "HistCapInterB",
        "cap_title": "Capacitance Histogram of inter pixel B",
        "cap_err_name": "HistCapErrInterB",
        "cap_err_title": "Capacitance Error Histogram of inter pixel B",
        "leak_name": "HistLeakInterB",
        "leak_title": "Leakage Current Histogram of inter pixel B",
        "leak_error_name": "HistLeakErrInterB",
        "leak_error_title": "Leakage Current Error Histogram of inter pixel B",
        "resistor_name": "HistResInterB",
        "resistor_title": "On-Resistance Histogram of inter pixel B",
        "resistor_error_name": "HistResErrInterB",
        "resistor_error_title": "On-Resistance Error Histogram of inter pixel B",
        "cov_name": "HistFitCovInterB",
        "cov_title": 'Fit Covariance Matrix of inter pixel B'
    }
    kwargs.update(inter_b_output_dict)
    current_hist = check_leaf_unit(data_group.InterHistCurrB, HIST_CURRENT_MEAS_UNIT)
    if "InterHistCurrErrB" in data_group:
        current_error_hist = check_leaf_unit(data_group.InterHistCurrErrB, HIST_CURRENT_MEAS_UNIT)
    else:
        current_error_hist = np.full_like(current_hist, fill_value=np.nan)
    kwargs["current_error_hist"] = current_error_hist
    assert isinstance(current_hist, tb.CArray) or isinstance(current_hist, np.ndarray)
    perform_analysis(file, result_group, current_hist, scan_parameters, is_inter_b=True, **kwargs)
