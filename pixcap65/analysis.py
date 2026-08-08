"""
Analysis of Pixcap65 data. Fits freq vs current to extract the capacitance. A 2D histogram containing the capacitance
for each pixel is stored.
"""
import os.path

import locale
import logging
import numpy as np
import tables as tb
import threading
import time
from contextlib import contextmanager
from tqdm import tqdm
from typing import Optional, Tuple, Union, Callable, Any, List
from warnings import deprecated, warn

from pixcap65.analysis_util.data_store import DepletionDataStore, DepletionTableStore, DepletionArrayStore, \
    DopingArrayStore, DepletionNumpyStore
from pixcap65.analysis_util.physics_modelling import SILICON_V_BIAS, depletion_model, model_depletion, EPS_SILICON, \
    gauss_model, \
    extended_gauss_integral
from pixcap65.analysis_util.utility import check_leaf_unit, str_join, ANALYSIS_CORRECTED_GROUP_NAME, \
    ANALYSIS_GROUP_NAME, \
    GENERAL_PIXCAP_SHAPE, GLOBAL_FILTERS, DepletionData, DepletionWidthData, handle_kafe2_advanced_options, \
    handle_minuit_advanced_options, \
    PARASITIC_SUBTRACTION, get_base_group, handle_analysis_mix_up, HIST_CAP_UNIT, HIST_CURRENT_MEAS_UNIT, \
    HIST_BIAS_MEAS_UNIT, get_analysis_group, investigate_fit_convergence, TABLES_LEAF_COMPAT_TYPE, \
    CVDistributionData, handle_fitter_advanced_options
from pixcap65.concurrency import get_context_manager
from pixcap65.plotting import CAPACITANCE_CONVERSION_FACTOR, evaluate_pixel_mask
from pixcap65.utility import synchronized_process_open_file
from pixcap65.utility.tables_util import get_groups, get_leaves, copy_node, list_attributes, group_get_file, \
    set_group_attribute, get_group_attribute, get_group_attributes, get_parent_group
from pixcap65.utility.utils_2 import walk_to_node, GroupType, create_carray, prevent_group_mix_up

# from tables import open_file as synchronized_process_open_file

# TODO: manually clean-up the constants and imports
# TODO: update the documentation of these implementations

try:
    # noinspection PyCompatibility
    from collections.abc import Sized, Iterable
except ImportError:
    # python 2.7
    import collections.Sized as Sized
    import collections.Iterable as Iterable

SI_MOBILITY = 1450
UNITS_ATTRIBUTE_KEY = "Units"
BOUNDARY_TYPE = Union[Tuple, Iterable[Tuple]]
ADVANCED_PARAMETER_TYPE = Union[bool, Iterable[bool]]
SYSTEMATICS_SAMPLE_SIZE = 500  # perhaps this should better be an keyword argument?
REDUCED_SYSTEMATICS_SAMPLE_SIZE = 20
RANDOM_SEED = 42
DISPERSION_PARASITIC_DEVIATION = 3.e-16
BIAS_VOLTAGE_ACCESS_IDX = 0
SLOPE_RESISTIVITY_CONVERSION = 1e12

logger = logging.getLogger(__name__)

global_rng = np.random.default_rng(RANDOM_SEED)


def get_manager_keywords(**kwargs):
    return {key: value for key, value in kwargs.items() if key in ("address", "authkey")}


def get_rng():
    return global_rng.spawn(1)[0]


from warnings import filterwarnings
from tables.exceptions import NaturalNameWarning
from iminuit.warnings import IMinuitWarning

filterwarnings("ignore", category=NaturalNameWarning)
filterwarnings("ignore", category=IMinuitWarning)
filterwarnings("ignore", category=np.exceptions.RankWarning, module="jacobi")


@deprecated("Please use analyze_data instead.")
def analyze_data_temporary_replacement(raw_data, base_path=None, is_advanced=False, is_cv=False,
                                       first_boundaries: Optional[BOUNDARY_TYPE] = None,
                                       second_boundaries: Optional[BOUNDARY_TYPE] = None,
                                       is_inter_pixel=False, **kwargs):
    """
    analyze_data

    Implementation of the analysis strategy for the capacitance measurement of a pixel sensor.
    But keep in mind that this function serves as a wrapper to handle file access and modification around
    the actual analysis implementation.
    The capacitance of the pixels are measured and investigated individually. For determination of the
    capacitance values either a linear fit or non-linear least square fit algorithms are used.
    Depending on the choice of the ´is_advanced` parameter non-linear techniques are used.
    In this case either 'kafe2' or 'iminuit' are used for the least-squares minimization depending on the choice
    of parameters.
    When using the advanced least-squares procedure the fit results will be plotted to verify the convergence of the
    fit.
    Thus, it is possible to use this wrapper to handle the PDF file to save fit-plot figures to instead of doing this
    individually for each analysis call.

    Afterwards, it is possible to directly correct the results for the capacitance by the connection and the
    measurement circuit.

    In Addition, there is the special case of an inter-pixel capacitance measurement.
    If the data to be analyzed comes from such a measurement this needs to be specified.
    Thus, in this case it will be checked which of the total current or the two inter-pix current data sets exist.
    The analysis will be done for each existing data set.
    Combinations with a C-V-Characterization might still be an issue.

    For measurements of the C-V-Characteristic of a sensor, the fits will be applied for every bias voltage measured.
    If additional fit boundaries are supplied, it will be tried to also determine the depletion behaviour of the
    pixel sensor including the depletion voltage and the corresponding capacitance.
    When doing this, also the doping profile and some intrinsic properties could be investigated.
    Currently, the C-V characterization for the inter-pixel measurements is not implemented yet.


    :param raw_data: path to the hdf file containing the raw data.
    :param base_path: path to the base group to look for the data.
    :param is_advanced: boolean indicating if the advanced analysis strategy should be used
        or not (may require additional keyword arguments)
    :param is_cv: boolean, indicates whether this is a C-V characterization.
    :param first_boundaries: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will only be estimated if this argument is provided.
    :param second_boundaries: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will be estimated if this argument is provided.
    :param is_inter_pixel: boolean, False, indicating whether the measurement to be analyzed is an inter-pixel
        capacitance measurement. In this case more fits will be applied, adjusted to the specific structure of this
        problem
    :key use_kafe2: boolean, indicates whether kafe2 is used for the fit. (Only used for the advanced procedure)
    :key plot: boolean, indicates whether to plot the data. An output PDF object could be submitted here
         instead of an explicitly created one. (Only used for the advanced procedure)
    :key apply_contour: boolean, indicates whether to determine the contours and try to plot them.
        (Only used for the advanced procedure)
    :key full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used. (Only used for the advanced procedure)
    :key apply_correction: boolean, False, indicates whether the measured capacitance should be
        corrected immediately; Will require the presence of further arguments as information about the
        parasitic capacitance needs to be submitted.
    :key use_corrected: boolean, False, indicates whether to use the corrected capacitance for the depletion
        analysis.
    :key fit_plot_pdf_name:  Name of the PDF file to save fitting figures from the advanced procedures to.
        (Only used for the advanced procedure)
    :key bare_file: hdf file containing the measurements and investigation of a bare pixcap sample to obtain
        information about intrinsic and parasitic capacitance. (Only required for the correction procedure, but in
        this case it must be present)
    :key bare_path: hdf files hierarchy path to the group containing the bare pixcap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present)
    :key chip_group: HDF files hierarchy group with the data/specifications of the pixels on the current sensor.
        (Will only be usd if the depletion behaviour is investigated)
    """
    analyze_data(raw_data, base_path, is_advanced, is_cv, first_boundaries, second_boundaries, is_inter_pixel, **kwargs)


def analyze_data(raw_data, base_path=None, is_advanced=False, is_cv=False,
                 first_boundaries: Optional[BOUNDARY_TYPE] = None,
                 second_boundaries: Optional[BOUNDARY_TYPE] = None,
                 is_inter_pixel=False, **kwargs) -> Optional[List]:
    """
    analyze_data

    Implementation of the analysis strategy for the capacitance measurement of a pixel sensor.
    But keep in mind that this function serves as a wrapper to handle file access and modification around
    the actual analysis implementation.
    The capacitance of the pixels are measured and investigated individually. For determination of the
    capacitance values either a linear fit or non-linear least square fit algorithms are used.
    Depending on the choice of the ´is_advanced` parameter non-linear techniques are used.
    In this case either 'kafe2' or 'iminuit' are used for the least-squares minimization depending on the choice
    of parameters.
    When using the advanced least-squares procedure the fit results will be plotted to verify the convergence of the
    fit.
    Thus, it is possible to use this wrapper to handle the PDF file to save fit-plot figures to instead of doing this
    individually for each analysis call.

    Afterwards, it is possible to directly correct the results for the capacitance by the connection and the
    measurement circuit.

    In Addition, there is the special case of an inter-pixel capacitance measurement.
    If the data to be analyzed comes from such a measurement this needs to be specified.
    Thus, in this case it will be checked which of the total current or the two inter-pix current data sets exist.
    The analysis will be done for each existing data set.
    Combinations with a C-V-Characterization might still be an issue.

    For measurements of the C-V-Characteristic of a sensor, the fits will be applied for every bias voltage measured.
    If additional fit boundaries are supplied, it will be tried to also determine the depletion behaviour of the
    pixel sensor including the depletion voltage and the corresponding capacitance.
    When doing this, also the doping profile and some intrinsic properties could be investigated.
    Currently, the C-V characterization for the inter-pixel measurements is not implemented yet.


    :param raw_data: path to the hdf file containing the raw data.
    :param base_path: path to the base group to look for the data.
    :param is_advanced: boolean indicating if the advanced analysis strategy should be used
        or not (may require additional keyword arguments)
    :param is_cv: boolean, indicates whether this is a C-V characterization.
    :param first_boundaries: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will only be estimated if this argument is provided.
    :param second_boundaries: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will be estimated if this argument is provided.
    :param is_inter_pixel: boolean, False, indicating whether the measurement to be analyzed is an inter-pixel
        capacitance measurement. In this case more fits will be applied, adjusted to the specific structure of this
        problem
    :param lock: IT IS STRONGLY RECOMMENDED TO EXPLICITLY SUPPLY A LOCK for synchronization
    :key use_kafe2: boolean, indicates whether kafe2 is used for the fit. (Only used for the advanced procedure)
    :key plot: boolean, indicates whether to plot the data. An output PDF object could be submitted here
         instead of an explicitly created one. (Only used for the advanced procedure)
    :key apply_contour: boolean, indicates whether to determine the contours and try to plot them.
        (Only used for the advanced procedure)
    :key full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used. (Only used for the advanced procedure)
    :key apply_correction: boolean, False, indicates whether the measured capacitance should be
        corrected immediately; Will require the presence of further arguments as information about
        the parasitic capacitance needs to be submitted.
    :key use_corrected: boolean, False, indicates whether to use the corrected capacitance for the depletion
        analysis. Will be removed in the future.
    :key fit_plot_pdf_name:  Name of the PDF file to save fitting figures from the advanced procedures to.
        (Only used for the advanced procedure)
    :key bare_file: hdf file containing the measurements and investigation of a bare pix cap sample to obtain
        information about intrinsic and parasitic capacitance. (Only required for the correction procedure, but in
        this case it must be present)
    :key bare_hdf_path: hdf files hierarchy path to the group containing the bare pix cap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present)
    :key chip_group_name: HDF files hierarchy group with the data/specifications of the pixels on the current sensor.
        (Will only be usd if the depletion behaviour is investigated)
    :returns: optional list of figure holder objects to be processed later on.
    """
    manager_kargs = get_manager_keywords(**kwargs)
    if "lock" not in kwargs:
        from mp_analysis import processed_manager
        with processed_manager(**manager_kargs) as (manager, lock):
            kwargs["lock"] = lock
            manager_kargs.update(address=manager.address)
            kwargs.update(**manager_kargs)
            _analyze_data(raw_data, base_path, is_advanced, is_cv, first_boundaries, second_boundaries, is_inter_pixel,
                         **kwargs)
        return
    lock = kwargs.get("lock", None)
    if lock is not None:
        print(type(lock))
        if hasattr(lock, "_token"):
            print("Token of the proxy object:", lock._token)
    correction_key_value = kwargs.pop("use_corrected", None)
    if correction_key_value is not None:
        msg = "keyword argument `use_corrected` is deprecated, use `apply_correction` instead. If `apply_correction` is also present this value will take precedence, otherwise the provided value will be used. This keyword argument will be removed in the future."
        warn(msg, DeprecationWarning, stacklevel=2)
        kwargs.setdefault("apply_correction", correction_key_value)

    # handle the additional PDF file in case of plotting enabled.
    fit_plot_pdf_name = kwargs.pop('fit_plot_pdf_name', None)
    if "plot" in kwargs and kwargs["plot"] and fit_plot_pdf_name is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        assert "fit_plot_pdf_name" not in kwargs
        with PdfPages(fit_plot_pdf_name) as pdf:
            kwargs['fit_plot_pdf'] = pdf
            _analyze_data(raw_data, base_path, is_advanced, is_cv, first_boundaries,
                         second_boundaries, is_inter_pixel, **kwargs)
            return

    _analyze_data(raw_data, base_path, is_advanced, is_cv, first_boundaries, second_boundaries, is_inter_pixel, **kwargs)

def _analyze_data(raw_data, base_path=None, is_advanced=False, is_cv=False,
                     first_boundaries: Optional[BOUNDARY_TYPE] = None,
                     second_boundaries: Optional[BOUNDARY_TYPE] = None,
                     is_inter_pixel=False, **kwargs) -> Optional[List]:
    # CHECK: does this splitting of the primary analysis method works at all?
    # extract further keyword arguments for further processing and propagting them to subroutines.
    lockless_kargs = {key: value for key, value in kwargs.items() if key != "lock"}
    get_distribution = kwargs.get("distribution", False)
    get_total_cap_file = kwargs.get("total_cap_file", None)
    get_total_cap_group = kwargs.get("total_cap_group", None)
    get_inter_cap_file = kwargs.get("in_cap_file", None)
    get_inter_cap_group = kwargs.get("in_cap_group", None)
    get_modified_inter_pix_id = kwargs.get("inter_pix_id", 18000)
    propagate_key_args = kwargs.copy()
    apply_correction_arg = kwargs.pop("apply_correction", False)
    bare_file_arg = kwargs.pop("bare_file", None)
    bare_path_arg = kwargs.pop("bare_hdf_path", None)
    # handle the real analysis.
    lock = kwargs.get("lock", None)
    with synchronized_process_open_file(raw_data, mode='a', lock=lock) as in_file_h5:
        # extract the hdf file groups to perform the analysis on.
        base_group = get_base_group(base_path, in_file_h5)

        # get capacitance to correct for.
        if apply_correction_arg:
            # extract the parasitic capacitance right here!
            parasitic, parasitic_error = _handle_mp_parasitic_cap(bare_path_arg, bare_file_arg, lock=lock)
        else:
            parasitic = 0
            parasitic_error = 0

        # special handling for C-V characterization.
        if is_cv:
            reference_group = _cv_analysis(in_file_h5, bare_file_arg, bare_path_arg, base_group, apply_correction_arg,
                                           first_boundaries, second_boundaries, is_advanced, is_inter_pixel,
                                           get_distribution, parasitic, parasitic_error, **kwargs)
        else:
            if is_inter_pixel:
                reference_group = base_group.inter_cap
            else:
                reference_group = base_group.total_cap

            handle_analysis_mix_up(reference_group)
            ana_group, _ = walk_to_node(reference_group, "analysis", create=True, verify_create=True)
            assert isinstance(ana_group, tb.Group)

            # this not change our 'ana_group' at all even if we providing some options for applying the parasitic
            # correction
            analysis_data_handle(in_file_h5, reference_group.measurements, ana_group,
                                 is_advanced=is_advanced,
                                 is_inter_pixel=is_inter_pixel, **propagate_key_args)

            total_reference_group, inter_reference_group = perform_inter_pix_deep_dive(reference_group,
                                                                                       get_total_cap_group, in_file_h5,
                                                                                       apply_correction_arg, parasitic,
                                                                                       parasitic_error,
                                                                                       get_inter_cap_group,
                                                                                       is_inter=is_inter_pixel,
                                                                                       total_ref_file=get_total_cap_file,
                                                                                       inter_ref_file=get_inter_cap_file,
                                                                                       lock=lock)
            if get_distribution:
                if is_inter_pixel:
                    # in case of the inter-pixel capacitance the bias voltage field will be used to encode the special case
                    # 10000: in-pix / total measure
                    # 11000: inter-A
                    # 12000: inter-B
                    # 13000: in-pix
                    # 14000: inter-pix-ana
                    # 15000: ?
                    # 18000: inter-pix grouped diagonal
                    # 19000: inter-pix grouped top
                    # 20000: inter-pix grouped sides
                    # but it is necessary to re-perform this whole thing here also for the corrected capacitances!
                    total_cap_data = ana_group.HistCap[:]
                    inter_a_cap_data = ana_group.HistCapInterA[:]
                    inter_b_cap_data = ana_group.HistCapInterB[:]
                    dist_table = in_file_h5.create_table(where=reference_group.analysis, name="DistResult",
                                                         description=CVDistributionData, filters=GLOBAL_FILTERS)
                    dist_entry = dist_table.row
                    # the activation of the parasitic correction would not change these tables anyway
                    assert "apply_correction" not in kwargs
                    # using this distribution handlers, will introduce a bias as it will also compute corrected values;
                    # but we will see what this is ending.
                    _get_sensor_distribution(ana_group, 10000, total_cap_data, dist_entry, parasitic, parasitic_error,
                                             **lockless_kargs)
                    if np.any(np.isfinite(inter_a_cap_data)):
                        _get_sensor_distribution(ana_group, 11000, inter_a_cap_data, dist_entry, parasitic,
                                                 parasitic_error,
                                                 **lockless_kargs)
                    if np.any(np.isfinite(inter_b_cap_data)):
                        _get_sensor_distribution(ana_group, 12000, inter_b_cap_data, dist_entry, parasitic,
                                                 parasitic_error,
                                                 **lockless_kargs)

                    # these are not useful if applied onto the corrected data!
                    if get_total_cap_group is not None and get_total_cap_file is not None:
                        _get_sensor_distribution(ana_group, 13000, ana_group.InPixHistCap[:], dist_entry, parasitic,
                                                 parasitic_error,
                                                 **lockless_kargs)

                        # The inter-pix capacitance is already the difference of two measurements and therefore free from any
                        # parasitic effects
                        _get_sensor_distribution(ana_group, 14000, ana_group.InterPixHistCap[:], dist_entry, 0,
                                                 0,
                                                 **lockless_kargs)

                    if get_inter_cap_group is not None and get_inter_cap_file is not None:
                        try:
                            _get_sensor_distribution(ana_group, get_modified_inter_pix_id,
                                                     ana_group.ModInterPixHistCap[:], dist_entry,
                                                     parasitic, parasitic_error, **lockless_kargs)
                        except tb.NoSuchNodeError:
                            assert isinstance(ana_group, tb.Group)
                            print(ana_group._f_list_nodes())
                            raise

                    # at last calculate the inter-pix distribution parameter also from the distribution data
                    dist_table.flush()
                    temp_rec_result = [row[:] for row in
                                       dist_table.where("""(bias == {})""".format(10000))]
                    total_inter_cap_result = np.rec.array(temp_rec_result,
                                                          dtype=tb.dtype_from_descr(CVDistributionData(), ))
                    second_inter_pixel_cap = total_inter_cap_result.copy()
                    third_inter_pixel_cap = total_inter_cap_result.copy()

                    # also here: this not useful for corrected data application
                    if total_reference_group is not None:
                        # need to extract the total capacitance
                        total_cap_distribution_result = total_reference_group.analysis.DistResult

                        total_cap_distribution_result_data = total_cap_distribution_result[:]

                        assert isinstance(total_cap_distribution_result, tb.Table)
                        second_inter_pixel_cap.bias[:] = 15000
                        for key in total_cap_distribution_result.dtype.names:
                            # but the errors will need a separate handling!
                            if "bias" in key:
                                continue
                            elif "std" in key or "err" in key:
                                second_inter_pixel_cap[key] = np.sqrt(
                                    total_cap_distribution_result_data[key][0] ** 2 + total_inter_cap_result[key] ** 2)
                            else:
                                second_inter_pixel_cap[key] = total_cap_distribution_result_data[key][0] - \
                                                              total_inter_cap_result[key]

                    # also here: this not useful for corrected data application
                    if inter_reference_group is not None:
                        # need to extract the total capacitance
                        inter_cap_distribution_result = inter_reference_group.analysis.DistResult
                        temp_rec_result = [row[:] for row in
                                           inter_cap_distribution_result.where("""(bias == {})""".format(10000))]
                        inter_cap_distribution_result_data = np.rec.array(temp_rec_result,
                                                                          dtype=tb.dtype_from_descr(
                                                                              CVDistributionData(),))

                        assert isinstance(inter_cap_distribution_result, tb.Table)
                        third_inter_pixel_cap.bias[:] = get_modified_inter_pix_id + 3000
                        for key in inter_cap_distribution_result.dtype.names:
                            # but the errors will need a separate handling!
                            if "bias" in key:
                                continue
                            elif "std" in key or "err" in key:
                                third_inter_pixel_cap[key] = np.sqrt(
                                    inter_cap_distribution_result_data[key][0] ** 2 + total_inter_cap_result[key] ** 2)
                            else:
                                third_inter_pixel_cap[key] = total_inter_cap_result[key] - \
                                                             inter_cap_distribution_result_data[key][0]

                    dist_table.append(second_inter_pixel_cap)
                    dist_table.append(third_inter_pixel_cap)
                else:
                    # extract the capacitance data for tabular value; will also need corrected data.
                    cap_data = ana_group.HistCap[:]
                    dist_table = in_file_h5.create_table(where=reference_group.analysis, name="DistResult",
                                                         description=CVDistributionData, filters=GLOBAL_FILTERS)
                    dist_entry = dist_table.row
                    _get_sensor_distribution(ana_group, 20000, cap_data, dist_entry, parasitic, parasitic_error,
                                             **lockless_kargs)
                    if apply_correction_arg:
                        dist_table.flush()
                        dist_table.copy(reference_group.analysis_correction)

        adjust_dist_table(reference_group.analysis, CAPACITANCE_CONVERSION_FACTOR)
        if apply_correction_arg:
            adjust_dist_table(reference_group.analysis_correction, CAPACITANCE_CONVERSION_FACTOR)

    global cap_counter
    print("The last correction counter is", cap_counter)


def _cv_analysis(in_file_h5: tb.File, bare_file_arg, bare_path_arg, base_group: tb.Group,
                 apply_correction_arg, first_boundaries: Optional[Union[tuple, Iterable[tuple]]],
                 second_boundaries: Optional[Union[tuple, Iterable[tuple]]], is_advanced: bool, is_inter_pixel: bool,
                 get_distribution, parasitic: float, parasitic_error: float,
                 **kwargs) -> Optional[tb.Group]:
    lockless_kargs = {key: value for key, value in kwargs.items() if "lock" not in key }
    lock = kwargs.get("lock", None)
    # need to perform the analysis for every bias voltage
    reference_group = base_group.biasing
    cv_data = np.full(shape=(40, 41, reference_group.measurements.BiasVoltageHist.shape[0]),
                      fill_value=np.nan)
    cv_err_data = np.full(shape=(40, 41, reference_group.measurements.BiasVoltageHist.shape[0]),
                          fill_value=np.nan)
    cv_data_corrected = np.full(shape=(40, 41, reference_group.measurements.BiasVoltageHist.shape[0]),
                                fill_value=np.nan)
    cv_err_data_corrected = np.full(shape=(40, 41, reference_group.measurements.BiasVoltageHist.shape[0]),
                                    fill_value=np.nan)

    # make sure to not mix-up with previous analysis results
    handle_analysis_mix_up(reference_group)

    # Why check this only for cv data and total cap?
    if get_distribution:
        in_file_h5.create_group(reference_group, "analysis")
        dist_table = in_file_h5.create_table(where=reference_group.analysis, name="CVDistribution",
                                             description=CVDistributionData, filters=GLOBAL_FILTERS)
        dist_entry = dist_table.row
    else:
        dist_entry = {}

    # no progressbar as the overhead for this is much too large in most cases.
    origin_bias_voltage_data = reference_group.measurements.BiasVoltageHist[:]
    if len(origin_bias_voltage_data.shape) > 1:
        # this wont use available measured voltage data but the settings instead. This may lower accuracy.
        bias_voltage_data = origin_bias_voltage_data[:, BIAS_VOLTAGE_ACCESS_IDX]
    else:
        bias_voltage_data = origin_bias_voltage_data

    bias_voltage_mapping = {}
    for k, bias_voltage in enumerate(tqdm(bias_voltage_data)):
        if np.isnan(bias_voltage) or (
                len(origin_bias_voltage_data.shape) > 1 and np.isnan(origin_bias_voltage_data[k, 1])):
            cv_data[:, :, k] = np.nan
            cv_err_data[:, :, k] = np.nan
        else:
            try:
                bias_name = "bias_{volt}_V".format(volt=bias_voltage).replace('-', "M_").replace(".", "__")
                bias_voltage_mapping[bias_voltage] = bias_name
                data_group = base_group.biasing.measurements[bias_name]
            except:
                print("Encountered a error when trying to read")
                print(origin_bias_voltage_data[k])
                bias_name = "bias_{volt}_V".format(volt=origin_bias_voltage_data[k, 2]).replace('-', "M_").replace(".", "__")
                bias_voltage_mapping[bias_voltage] = bias_name
                try:
                    data_group = base_group.biasing.measurements[bias_name]
                except:
                    print("The error still exists")
                else:
                    print("LOOKS LIKE THE ERROR WAS SOLVED!")
                raise
            ana_group, _ = walk_to_node(base_group.biasing,
                                        str_join("/", ANALYSIS_GROUP_NAME, bias_name), create=True,
                                        verify_create=True)
            assert isinstance(ana_group, tb.Group)
            analysis_data_handle(in_file_h5, data_group, ana_group, is_advanced=is_advanced,
                                 is_inter_pixel=is_inter_pixel, **kwargs)

            # extract the capacitance data for tabular value; will also need corrected data.
            cap_data = ana_group.HistCap[:]
            cap_error_data = ana_group.HistCapErr[:]
            cv_data[:, :, k] = cap_data[:, :]
            cv_err_data[:, :, k] = cap_error_data[:, :]

            if get_distribution:
                _get_sensor_distribution(ana_group, bias_voltage, cap_data, dist_entry, parasitic,
                                         parasitic_error,
                                         **lockless_kargs)

    in_file_h5.flush()

    # we should not write down the excluded pixels here as they would be a impurity to the biasing summary data!
    for entry in kwargs.get('mask_pixel', []):
        exclude_col, exclude_row = entry
        cv_data[exclude_col, exclude_row] = np.nan
        cv_err_data[exclude_col, exclude_row] = np.nan
    create_carray(in_file_h5, reference_group.analysis, name="UCHist", title="Histogram of the U-C-curve",
                  filters=GLOBAL_FILTERS, obj=cv_data, unit=HIST_CAP_UNIT)
    create_carray(in_file_h5, reference_group.analysis, name="UCErrHist",
                  title="Error Histogram of the U-C-curve",
                  filters=GLOBAL_FILTERS,
                  obj=cv_err_data,
                  unit=HIST_CAP_UNIT)

    # in case of active correction, also the summary capacitance tables needs to be corrected, but it should
    # also an uncorrected table present.
    if apply_correction_arg:
        # perform the transfer
        apply_correction_simple(bare_file_arg, bare_path_arg, reference_group.analysis, lock=lock)

        for k, bias_voltage in enumerate(bias_voltage_data):
            ana_group_correction, _ = walk_to_node(reference_group,
                                                   str_join("/", ANALYSIS_CORRECTED_GROUP_NAME,
                                                            bias_voltage_mapping[bias_voltage]),
                                                   create=True, verify_create=True)
            cap_data = ana_group_correction.HistCap[:]
            cap_error_data = ana_group_correction.HistCapErr[:]
            cv_data_corrected[:, :, k] = cap_data[:, :]
            cv_err_data_corrected[:, :, k] = cap_error_data[:, :]

        # we should not write down the excluded pixels here as they would be a impurity to the biasing summary data!
        for entry in kwargs.get('mask_pixel', []):
            exclude_col, exclude_row = entry
            cv_data_corrected[exclude_col, exclude_row] = np.nan
            cv_err_data_corrected[exclude_col, exclude_row] = np.nan
        # save also the corrected capacitance data
        create_carray(in_file_h5, reference_group.analysis_correction, name="UCHist",
                      title="Histogram of the U-C-curve",
                      filters=GLOBAL_FILTERS,
                      obj=cv_data_corrected, unit=HIST_CAP_UNIT)
        create_carray(in_file_h5, reference_group.analysis_correction, name="UCErrHist",
                      title="Error Histogram of the U-C-curve",
                      filters=GLOBAL_FILTERS,
                      obj=cv_err_data_corrected, unit=HIST_CAP_UNIT)

        create_carray(in_file_h5, reference_group.analysis_correction, name="UCSystematicHist",
                      title="Error Histogram of the U-C-curve (systematic uncertainty)",
                      filters=GLOBAL_FILTERS,
                      obj=np.full_like(cv_data_corrected, fill_value=parasitic_error), unit=HIST_CAP_UNIT)
        create_carray(in_file_h5, reference_group.analysis_correction, name="UCSystematicDispersionHist",
                      title="Error Histogram of the U-C-curve (uncertainty by systematic dispersion)",
                      filters=GLOBAL_FILTERS,
                      obj=np.full_like(cv_data_corrected, fill_value=DISPERSION_PARASITIC_DEVIATION),
                      unit=HIST_CAP_UNIT)

    if first_boundaries is not None and second_boundaries is not None:
        # We will need all the usages as here might be a inconsitency with the data systems.
        dep_ana_group = get_analysis_group(reference_group, use_corrected=apply_correction_arg)
        assert isinstance(dep_ana_group, tb.Group)
        chip_name = lockless_kargs.pop("chip_group_name", None)
        # we need to put the plotting arguments back in
        if chip_name is not None:
            chip_spec_group, _ = walk_to_node(in_file_h5.root, chip_name, create=False, verify_create=True)
            lockless_kargs['chip_group'] = chip_spec_group

        # Anyway it is necessary to perform this analysis also on a sensor-average basis!
        start_time = time.time()
        # perhaps this function should be jit compiled.
        if apply_correction_arg:
            analyze_depletion_delegate(reference_group.measurements,
                                       reference_group.analysis,
                                       first_boundaries, second_boundaries,
                                       para=parasitic,
                                       para_dist=parasitic_error, **lockless_kargs)
        analyze_depletion_delegate(reference_group.measurements, dep_ana_group,
                                   first_boundaries, second_boundaries, para=parasitic,
                                   para_dist=parasitic_error, **lockless_kargs)
        print(f"The depletion analysis of the scanned pixels took {time.time() - start_time} seconds.")

        if get_distribution:
            # the appropriate options are in this case: create one combination table for each of the
            # fit boundaries or save numpy arrays within the table?
            dist_table.flush()
            # Why reloading the full table here.
            dist_table = reference_group.analysis.CVDistribution[:]
            depletion_data_table_raw = in_file_h5.create_table(where=reference_group.analysis,
                                                               name="SensorDepletionRaw1",
                                                               description=DepletionData,
                                                               filters=GLOBAL_FILTERS)
            corrected_depletion_data_table = in_file_h5.create_table(where=reference_group.analysis,
                                                                     name="SensorDepletionRaw2",
                                                                     description=DepletionData,
                                                                     filters=GLOBAL_FILTERS)

            # will need to perform the operation with two different tables which only differ by their entries
            __distribution_depletion_estimation(dist_table, first_boundaries,
                                                second_boundaries, depletion_data_table_raw, False,
                                                **lockless_kargs)

            depletion_data_table_raw.flush()
            depletion_data_table = depletion_data_table_raw.copy(reference_group.analysis,
                                                                 "SensorDepletionRaw")
            depletion_data_table.flush()

            # FIXME: for correct resistivity estimation it is necessary to use the correct pixel area/size
            # perhaps we should move this here to another position to also fetch the pixel geometry data.
            # this would require detailed information about the sensor geometry!
            # we would need to know which pixels contribute to give an estimate
            # could we fetch the correct geometry from the corresponding data set?
            from scipy.constants import epsilon_0

            try:
                raw_pixel_areas = np.prod(chip_spec_group.PhysicalDimensions[:], axis=2)
                sensor_pixel_mask = np.isfinite(cv_data[:, :, 0])
                distribution_pixel_area = np.mean(raw_pixel_areas[sensor_pixel_mask])
            except:
                distribution_pixel_area = 50 * 50
            resistivity_estimator = EPS_SILICON * epsilon_0 * distribution_pixel_area ** 2 * depletion_data_table.cols.c[
                :] / (2 * SI_MOBILITY) * SLOPE_RESISTIVITY_CONVERSION
            resistivity_err_estimator = EPS_SILICON * epsilon_0 * distribution_pixel_area ** 2 * depletion_data_table.cols.c_error[
                :] / (2 * SI_MOBILITY) * SLOPE_RESISTIVITY_CONVERSION
            depletion_data_table.cols.rho[:] = resistivity_estimator
            depletion_data_table.cols.rho_error[:] = resistivity_err_estimator
            depletion_data_table.flush()

            if apply_correction_arg:
                # we need to further repeat these calculations to estimate the systematic uncertainties
                # of all the depletion voltages.
                # But how to perform this step for ALL the pixels?
                print("Will try to apply the distribution data.")
                __distribution_depletion_estimation(dist_table, first_boundaries, second_boundaries,
                                                    corrected_depletion_data_table, True, **lockless_kargs)
                corrected_depletion_data_table.flush()
                depletion_data_table.cols.Ubi_corrected[:] = corrected_depletion_data_table.cols.Ubi[:]
                depletion_data_table.cols.Ubi_corrected_error[
                    :] = corrected_depletion_data_table.cols.Ubi_error[:]
                depletion_data_table.cols.a_corrected[:] = corrected_depletion_data_table.cols.a[:]
                depletion_data_table.cols.a_corrected_error[:] = corrected_depletion_data_table.cols.a_error[:]
                depletion_data_table.cols.b_corrected[:] = corrected_depletion_data_table.cols.b[:]
                depletion_data_table.cols.b_corrected_error[:] = corrected_depletion_data_table.cols.b_error[:]
                depletion_data_table.cols.c_corrected[:] = corrected_depletion_data_table.cols.c[:]
                depletion_data_table.cols.c_corrected_error[:] = corrected_depletion_data_table.cols.c_error[:]
                depletion_data_table.cols.d_corrected[:] = corrected_depletion_data_table.cols.d[:]
                depletion_data_table.cols.d_corrected_error[:] = corrected_depletion_data_table.cols.d_error[:]
                depletion_data_table.cols.Ubi_systematic_corrected[
                    :] = corrected_depletion_data_table.cols.Ubi_systematic[:]
                depletion_data_table.cols.Ubi_systematic_dispersion_corrected_error[
                    :] = corrected_depletion_data_table.cols.Ubi_systematic_dispersion[:]

                resistivity_estimator = EPS_SILICON * epsilon_0 * (
                        50 * 50) ** 2 * corrected_depletion_data_table.cols.c[:] / (2 * 1450)
                resistivity_err_estimator = EPS_SILICON * epsilon_0 * (
                        50 * 50) ** 2 * corrected_depletion_data_table.cols.c_error[:] / (2 * 1450)
                depletion_data_table.cols.rho_corrected[:] = resistivity_estimator
                depletion_data_table.cols.rho_corrected_error[:] = resistivity_err_estimator

                # What about the table in the corrected analysis group?
            else:
                placeholder_data = np.full_like(depletion_data_table_raw.cols.Ubi[:], fill_value=np.nan)
                depletion_data_table.cols.Ubi_corrected[:] = placeholder_data[:]
                depletion_data_table.cols.Ubi_corrected_error[:] = placeholder_data[:]
                depletion_data_table.cols.a_corrected[:] = placeholder_data[:]
                depletion_data_table.cols.a_corrected_error[:] = placeholder_data[:]
                depletion_data_table.cols.b_corrected[:] = placeholder_data[:]
                depletion_data_table.cols.b_corrected_error[:] = placeholder_data[:]
                depletion_data_table.cols.c_corrected[:] = placeholder_data[:]
                depletion_data_table.cols.c_corrected_error[:] = placeholder_data[:]
                depletion_data_table.cols.d_corrected[:] = placeholder_data[:]
                depletion_data_table.cols.d_corrected_error[:] = placeholder_data[:]
                depletion_data_table.cols.Ubi_systematic_corrected[:] = placeholder_data[:]

            depletion_data_table.flush()

            if apply_correction_arg:
                in_file_h5.copy_node(where=reference_group.analysis, newparent=dep_ana_group,
                                     name="SensorDepletionRaw", newname="SensorDepletionRaw")

            # remove the additonal tables
            depletion_data_table_raw.remove()
            corrected_depletion_data_table.remove()
    return reference_group


def adjust_dist_table(group: tb.Group, conversion_factor: float, name="DistResultfF"):
    if "DistResult" in group:
        standard_table = group.DistResult
        assert isinstance(standard_table, tb.Table)
        old_table_data = standard_table[:]
        new_table_data = np.rec.array(old_table_data, dtype=tb.dtype_from_descr(CVDistributionData()))
        assert isinstance(new_table_data, np.recarray)
        assert new_table_data.dtype == standard_table.dtype
        new_table_data.capacitance *= conversion_factor
        new_table_data.cap_err *= conversion_factor
        new_table_data.cap_std *= conversion_factor
        new_table_data.cap_std_err *= conversion_factor
        new_table_data.cap_corrected *= conversion_factor
        new_table_data.cap_corrected_err *= conversion_factor
        new_table_data.cap_systematic_error *= conversion_factor
        new_table_data.cap_corrected_est_error *= conversion_factor
        new_table_data.cap_corrected_std_error *= conversion_factor
        new_table_data.cap_systematic_dispersion *= conversion_factor

        new_table = group_get_file(group).create_table(where=group, name=name, title=standard_table.title,
                                                       description=new_table_data, filters=standard_table.filters)
        new_table.flush()


@contextmanager
def perform_inter_pix_fetch(path, group, active_file, target: str, lock=None):
    if path is not None:
        general_h5_file = os.path.abspath(active_file.filename)
        inter_pix_h5_file = os.path.abspath(path)
        if general_h5_file == inter_pix_h5_file:
            yield group, active_file, False
        elif os.path.exists(path):
            with synchronized_process_open_file(path, mode='a', lock=lock) as total_h5_file:
                yield group, total_h5_file, True
        else:
            yield None, None, False
    else:
        yield None, None, False


def perform_inter_pix_deep_dive(reference_group, get_total_cap_group: Optional[str],
                                in_file_h5: tb.File, apply_correction_arg, parasitic: float,
                                parasitic_error: float, get_inter_cap_group: Optional[str], **kwargs):
    # no necessity for accessing the manager!
    import uuid
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
        total_node, inter_node = __perform_inter_pix_deeper(apply_correction_arg, total_h5_group, in_file_h5, parasitic,
                                                            parasitic_error, reference_group, total_h5_file,
                                                            inter_h5_file, inter_h5_group)
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


def __generate_gaussian_samples(loc: np.ndarray, scale: float, size: int, rng: np.random.Generator) -> np.ndarray:
    result_shape = tuple((*loc.shape, size))
    result_data = np.zeros(result_shape, dtype=np.float64)
    for indices in np.ndindex(*loc.shape):
        result_data[indices] = rng.normal(loc[indices], scale, size)

    return result_data


def __second_generate_gaussian_samples(loc: np.ndarray, scale: float, size: int,
                                       rng: np.random.Generator) -> np.ndarray:
    result_shape = tuple((*loc.shape, size))
    result_data = np.zeros(result_shape, dtype=np.float64)
    for indices in np.ndindex(*loc.shape):
        result_data[indices] = rng.normal(loc[indices], scale, size)

    return np.moveaxis(result_data, -1, 0)


def __mp_init_distribution_delegate(cap_data, kargs):
    global mp_shared_cap_ref, mp_shared_keyword_args
    mp_shared_cap_ref = cap_data
    mp_shared_keyword_args = kargs
    if "lock" in kargs:
        lock = kargs["lock"]
        print("found a lock:", type(lock))
        if hasattr(lock, "_token"):
            print(lock._token)
        if hasattr(lock, "_serial"):
            print(lock._serial)


def __mp_handle_distribution_delegate(offset):
    global mp_shared_cap_ref, mp_shared_keyword_args
    return analyze_capacitance_distribution_delegate(None, None, capacitance=mp_shared_cap_ref + offset, convert=False,
                                                     **mp_shared_keyword_args)


def _get_sensor_distribution(ana_group: tb.Group, bias_voltage, cap_data, dist_entry, parasitic: float,
                             parasitic_error: float, **kwargs):
    """
    _get_sensor_distribution

    @author Dominik Fischer
    @date 2026-05-31

    Internal helper function to perform the fits to the capacitance distribution over the measured part of the sensor.
    The handler is introduced to also handle the estimation of systematic uncertainties on the capacitance values
    in the fits.
    Will first compute the distribution of the capacitance over the full sensor approximated by a gaussian pdf.
    The values from this fit will be used as the average values later on.
    The procedure is repeated with the corrected capacitance values, if necessary. Otherwise the whole fit is recomputed
    and treated as a corrected fit for all further computations.

    To extract the systematic uncertainties on this distribution, it is necessary to repeat the fit multiple times.
    As the uncertainties of the data points are not used when applying a binned nll fit, the systematic effects could
    only be approximated on a statistical basis.

    For the spread of the parasitics two fits at the extremes are used with appropriately modified capacitance arrays.
    For the dispersion effects random samples of the dispersion are generated and added to the corrected capacitance
    data.



    :param ana_group: hdf files' analysis group which stores the capacitances' for the pixels on which to compute the
        capacitance distribution.
    :param bias_voltage: hv voltage applied as reversed bias to the sensor for this capacitance measurement. Will be
        propagated without further processing to the table row.
    :type cap_data: numpy.ndarray
    :param cap_data: numpy array of the capacitance for each pixel (nan if the capacitance for a particular pixel is
        not measured)
    :param dist_entry: tables row to append the extracted information about the fit result and parasitic capacitance,
        as well, as some other data about systematic uncertainties to.
    :param parasitic: parasitic capacitance of the pixcap chip and the bump-bonds below the sensor.
    :param parasitic_error: spread of the parasitic capacitance over a pixcap chip
    :param kwargs: further keyword arguments (but which only used for the resistance distribution if estimators
        are present)
    :key no_plot:
    :key hist_res_key: name/identifiert of the array/table which contains the on-resistance estimators if present.
    """
    # remove all unnecessary keywords
    kwargs.pop("plot", None)
    kwargs.pop("fit_plot_pdf", None)
    kwargs.pop("use_kafe2", None)
    kwargs.pop("output_pdf", None)
    kwargs['no_plot'] = True

    hist_res_key = kwargs.pop("hist_res_key", "HistRes")
    # Will continue using tuples for performance!
    # first get a histogram with all the data
    dist_result_list = []
    dist_result_list.append(analyze_capacitance_distribution_delegate(None,
                                                                      None,
                                                                      capacitance=cap_data,
                                                                      convert=False,
                                                                      **kwargs))
    # we need an independent fit for the corrected distributions
    # interestingly only the parasitic negatives are present here.
    cap_data_para = np.where(np.isfinite(cap_data), cap_data - parasitic, np.nan)
    rng = get_rng()
    sensor_distribution_type = np.dtype([
        ("n", np.uint64),
        ("C", np.float64),
        ("C_err", np.float64),
        ("C_std", np.float64),
        ("C_std_err", np.float64),
    ])
    dist_result_list.append(analyze_capacitance_distribution_delegate(None, None,
                                                                      capacitance=cap_data_para,
                                                                      convert=False,
                                                                      **kwargs))
    # this should not change the computation time.
    parasitic_advanced_samples = rng.normal(loc=0, scale=parasitic_error, size=REDUCED_SYSTEMATICS_SAMPLE_SIZE)
    # we could also optimise here by using multiprocessing iterators!
    dispersion_cap_sample = rng.normal(loc=0, scale=DISPERSION_PARASITIC_DEVIATION,
                                       size=REDUCED_SYSTEMATICS_SAMPLE_SIZE, )
    # FIXME: prevent this mp worker pool from leaking semaphore objects all around!
    if True:
        __mp_init_distribution_delegate(cap_data_para, kwargs)
        parasitic_adv_cap_est = np.rec.array([__mp_handle_distribution_delegate(item) for item in parasitic_advanced_samples], dtype=sensor_distribution_type)
        dispersion_estimator = np.rec.array([__mp_handle_distribution_delegate(item) for item in dispersion_cap_sample], dtype=sensor_distribution_type)
    else:
        # for current size of these iterables there is no improvement in time by using pooled execution!
        with mp.Pool(processes=8, initializer=__mp_init_distribution_delegate, initargs=(cap_data_para, kwargs)) as pool:
            try:
                parasitic_adv_cap_est = np.rec.array(pool.map(__mp_handle_distribution_delegate, parasitic_advanced_samples,
                                                              chunksize=None), dtype=sensor_distribution_type)
                dispersion_estimator = np.rec.array(pool.map(__mp_handle_distribution_delegate, dispersion_cap_sample,
                                                             chunksize=None), dtype=sensor_distribution_type)
            finally:
                pool.close()

    # with concurrent_futures.ProcessPoolExecutor(max_workers=5, mp_context=ctx) as executor:
    #     parasitic_futures = [executor.submit(analyze_capacitance_distribution_delegate, None, None, capacitance=cap_data_para + offset, convert=False, **kargs) for offset in parasitic_advanced_samples]
    #     dispersion_futures = [executor.submit(analyze_capacitance_distribution_delegate, None, None, capacitance=cap_data_para + iterat, convert=False, **kargs) for iterat in dispersion_cap_sample]
    #     parasitic_adv_cap_est = np.rec.array([future.result() for future in parasitic_futures], dtype=sensor_distribution_type)
    #     dispersion_estimator = np.rec.array([future.result() for future in dispersion_futures], dtype=sensor_distribution_type)

    # for usage of the map function it would be necessary to wrap the executable appropriately.
    # parasitic_adv_cap_est = np.rec.array([analyze_capacitance_distribution_delegate(None, None, capacitance=cap_data_para+offset,convert=False, **kargs) for offset in parasitic_advanced_samples], dtype=sensor_distribution_type)
    # dispersion_estimator = np.rec.array([analyze_capacitance_distribution_delegate(None, None, capacitance=cap_data_para+iterat, convert=False, **kargs) for iterat in dispersion_cap_sample], dtype=sensor_distribution_type)

    distribution_result = np.rec.array(dist_result_list, dtype=sensor_distribution_type)

    # Fehler von Fehlern werden i.d.R. unterdrückt.
    def assemble_systematic_propagation(data: np.recarray):
        assert data.dtype == sensor_distribution_type
        weights = np.reciprocal(data.C_err ** 2 + data.C_std ** 2)
        avg = np.average(data.C, weights=weights, keepdims=True)
        systematic_propagation = np.nanstd(data.C, mean=avg, dtype=np.float64)
        return systematic_propagation

    cap_corr_systematic = assemble_systematic_propagation(parasitic_adv_cap_est)

    # the procedure must also be performed for the literature value of chip dispersion effects for the
    # parasitic capacitances.
    # this function is not able to distinguish between corrected and uncorrected iterations!
    cap_dispersion_systematic = assemble_systematic_propagation(dispersion_estimator)

    # try to get to the on-resistance datasets to perform the same distribution handler!
    # only issue with this attempt the capacitance will be saved as a parasitic one!
    if hist_res_key in ana_group and np.any(np.isfinite(ana_group[hist_res_key])):
        resistance_array = ana_group[hist_res_key]
        assert isinstance(resistance_array, (tb.Array, np.ndarray, tb.Table))
        resistance_data = resistance_array[:]
        temp_tuple = analyze_capacitance_distribution_delegate(None,
                                                               kwargs.get(
                                                                   "distribution_output_pdf",
                                                                   None),
                                                               capacitance=resistance_data,
                                                               **kwargs)
        _, res, res_err, res_std, res_std_err = temp_tuple
    else:
        res = 0.0
        res_err = 0.0
        res_std_err = 0.0
        res_std = 0.0

    dist_entry['bias'] = kwargs.pop("alter_spec", bias_voltage)
    dist_entry['n_pixel'] = distribution_result.n[0]
    dist_entry['capacitance'] = distribution_result.C[0]
    dist_entry['cap_err'] = distribution_result.C_err[0]
    dist_entry['cap_std'] = distribution_result.C_std[0]
    dist_entry['cap_std_err'] = distribution_result.C_std_err[0]
    dist_entry['r_on'] = res
    dist_entry['r_on_err'] = res_err
    dist_entry['r_on_std'] = res_std
    dist_entry['r_on_std_err'] = res_std_err
    dist_entry['cap_systematic_dispersion'] = cap_dispersion_systematic
    dist_entry['cap_systematic_error'] = cap_corr_systematic
    dist_entry['cap_corrected'] = distribution_result.C[1]
    dist_entry['cap_corrected_err'] = distribution_result.C_std[1]
    dist_entry['cap_parasitic'] = parasitic
    dist_entry['cap_corrected_est_error'] = distribution_result.C_err[1]
    dist_entry['cap_corrected_std_error'] = distribution_result.C_std_err[1]

    if isinstance(dist_entry, tb.tableextension.Row):
        dist_entry.append()
    else:
        print("No append of the distribution!")


def _extract_table_data(key: str, is_corrected: bool, table: np.ndarray, ):
    """
    extract_table_data

    Wrapper for the extraction of table column data. Only relevant to simplify the implementation for the usage of
    corrected capacitance data. It replaces some of the keys to extract capacitance data from the table by the corrected
    table columns.

    :type key: str
    :param key: name of the original column to extract data from
    :type is_corrected: bool
    :param is_corrected: boolean, whether to use the corrected capacitance data.
    :param table: table (as a structured numpy array) to extract data from
    :return: ??
    """
    if is_corrected:
        match (key):
            case 'capacitance':
                return table['cap_corrected']
            case 'cap_std':
                return table['cap_corrected_err']
            case _:
                return table[key]
    else:
        return table[key]

first_run_dist = True

# perhaps extract the result table as an parameter in order to put the correct depletion values also in the table!
def __distribution_depletion_estimation(dist_table,
                                        first_boundaries: Union[tuple, Iterable[tuple]],
                                        second_boundaries: Union[tuple, Iterable[tuple]], depletion_data_table,
                                        is_corrected, **kwargs) -> Optional[List]:
    """
    distribution_depletion_estimation

    NO LOCKS IN USE HERE!

    Internal handler function to estimate the depletion voltage / reversed bias for full depletion of the sensor.
    This handler function uses the capacitance distribution over the full sensor (or at least the measured part)
    assuming a gaussian pdf.
    It takes the gaussian centered capacitances (more accuratley: the distributions c-v-curve), fits two straight lines
    to low voltage limit for the capacitance behaviour of the undepleted sensor and the high voltage limit for the
    capacitance behaviour for a depletion zone extending over the full physical dimensions of the sensor.
    The depletion voltage is estimated as the intersection of these two straight lines.
    Both the c-v-data is taken from a table and the results are written back to another table.

    These estimation attempt is redone with slightly adjust fit ranges to estimate the systematic uncertainty.
    Here we account for the measured distribution of parasitic capacitance over the Pixcap Chip (its spread), for the
    dispersion of the parasitic capacitance between multiple sensors/chips and the particular choice of the fit range.
    For the additional fits to extract the systematic uncertainties no plotting of the fits is available.
    The Estimation of the different systematic uncertainties is performed by again fitting the voltage ranges but with
    either varied fit range or with the systematic uncertainties provided as the measurement error submitted to the fit
    algorithm. In the latter two cases the extracted fit errors on the depletion voltage is further propagated through
    the analysis.

    For the systmatic estimation another depeltion table will be temporarily created and then removed.

    :type dist_table: numpy.ndarray, numpy.rec.array
    :param dist_table: table of the measures capacitance data in dependence on the applied reverse bias. The data must be
        provided by a structured numpy rec-array and not by a pytables table.
    :param first_boundaries: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will only be estimated if this argument is provided.
        It is also possible to provide an Iterable of boundaries for multiple depletion regions to fit.
    :param second_boundaries: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will be estimated if this argument is provided.
        It is also possible to provide an Iterable of boundaries for multiple depletion regions to fit.
    :type depletion_data_table: pytables.Table
    :param depletion_data_table: table to write the depletion voltage estimation data to.
    :type is_corrected: bool
    :param is_corrected: whether to use the corrected data for this operation.
    :param kwargs: further keyword arguments for the fits to estimate the depletion voltage.
    :key fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :key use_kafe2: boolean, False, indicates whether kafe2 is used for the fit.
    :key apply_contours: boolean, indicates whether to determine the contours and try to plot them. (No effect for
        the fit to estimate systematic effects)
    :type apply_contours: bool
    :key plot: boolean, False, indicates whether to plot the data. AN output PDF object could be submitted here
         instead of an explicitly created one. (No effect for the fit to estimate systematic effects)
    :key fit_plot_pdf: PDF object to save the fit figures to. (No effect for
        the fit to estimate systematic effects.)
    :key verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    :key cv_fit_plot_pdf: analog to `fit_plot_pdf` to activate the plotting for c-v- and depletion fits independent from
        the plotting for capacitance estimation fits. If this keyword argument is present also the `plot` arguments will
        be set automatically. (This keyword argument will be ignored for the fits to estimate systematic effects)
    :returns: optional list of figure holder objects.
    """
    pixel_cap_data = _extract_table_data('capacitance', is_corrected, dist_table)
    pixel_cap_error_data = _extract_table_data('cap_std', is_corrected, dist_table)
    voltage_data = _extract_table_data('bias', is_corrected, dist_table)
    systematic_offset = kwargs.pop("systematic_offset", 2)
    systematic_key_args = kwargs.copy()
    systematic_key_args.pop("plot", False)
    systematic_key_args.pop("fit_plot_pdf", None)
    systematic_key_args.pop("output_pdf", None)
    systematic_key_args.pop("cv_fit_plot_pdf", None)

    # is dist_table a ordinary numpy array instead of an rec-array?
    if is_corrected:
        try:
            temp_systematic = dist_table['cap_corr_systematic_error']
            temp_dispersion = dist_table['cap_dispersion_error']
            if np.all(temp_systematic == 0) or not np.any(np.isfinite(temp_systematic)):
                systematic_error = dist_table['cap_systematic_error']
            else:
                systematic_error = temp_systematic
            if np.all(temp_dispersion == 0) or not np.any(np.isfinite(temp_dispersion)):
                dispersion_error = dist_table['cap_systematic_dispersion']
            else:
                dispersion_error = temp_dispersion
        except (KeyError, ValueError):
            # just a safe-guard in case we encounter a table entry before the update!
            systematic_error = dist_table['cap_systematic_error']
            dispersion_error = dist_table['cap_systematic_dispersion']
    else:
        systematic_error = dist_table['cap_systematic_error']
        dispersion_error = dist_table['cap_systematic_dispersion']
    second_systematic_result_storage = DepletionNumpyStore(depletion_data_table.dtype)
    third_systematic_result_storage = DepletionNumpyStore(depletion_data_table.dtype)
    fourth_systematic_result_storage = DepletionNumpyStore(depletion_data_table.dtype)

    if isinstance(first_boundaries, tuple):
        first_boundaries = [first_boundaries]
        second_boundaries = [second_boundaries]

    assert isinstance(first_boundaries, Sized)
    n_depletion_regions = len(first_boundaries)
    fit_result_storage = DepletionTableStore(depletion_data_table, n_depletions=n_depletion_regions, )
    for k, (first_bound, second_bound) in enumerate(zip(first_boundaries, second_boundaries)):
        first_lower, first_upper = first_bound
        second_lower, second_upper = second_bound

        # maybe this could be speed up by a different implementation!
        # Now the real implementation starts.
        analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, pixel_cap_error_data,
                                second_lower,
                                second_upper, voltage_data, fit_result_storage,
                                fit_description_text=" Sensor Distribution", **kwargs)

        # What about the systematic uncertainties of the depletion voltage?
        # 1. fit range
        # 2. fit initial guess
        # 3. systematic uncertainties of the capacitances used for the fit
        # 4. systematic dispersion of the capacitances used for the fit (must be treaded separately from the others)
        # the later two are quite simple to implement, but the first two might be problem,
        # in particular when combining them!
        # Currently it is not even possible to determine the second effect at all.
        __extract_systematic_depletion_effects((first_lower, first_upper),
                                               (second_lower, second_upper), voltage_data,
                                               pixel_cap_data, pixel_cap_error_data, systematic_error, dispersion_error,
                                               systematic_offset, second_systematic_result_storage,
                                               third_systematic_result_storage,
                                               fourth_systematic_result_storage,
                                               systematic_key_args)

    depletion_data_table.flush()
    depletion_data_table.cols.Ubi_systematic[:] = np.sqrt(np.abs(
        depletion_data_table.cols.Ubi[:] - second_systematic_result_storage.table.Ubi[
            :]) ** 2 + third_systematic_result_storage.table.Ubi_error[:] ** 2)
    depletion_data_table.cols.Ubi_systematic_dispersion[:] = fourth_systematic_result_storage.table.Ubi_error[:]
    depletion_data_table.flush()
    group_get_file(depletion_data_table).flush()


def __extract_systematic_depletion_effects(lower_boundary: Tuple, upper_boundary: Tuple, voltage_data: np.ndarray,
                                           pixel_cap_data: np.ndarray,
                                           pixel_cap_error_data: np.ndarray, systematic_error: np.ndarray,
                                           dispersion_error: np.ndarray, systematic_offset: float,
                                           second_systematic_result_storage: DepletionNumpyStore,
                                           third_systematic_result_storage: DepletionNumpyStore,
                                           fourth_systematic_result_storage: DepletionNumpyStore,
                                           systematic_key_args: dict[str, Any]):
    """
    __extract_systematic_depletion_effects

    @author Dominik Fischer
    @date 2026-05-30

    NO LOCKS IN USE HERE

    Private/Internal helper function to compute the systematic uncertainties of the depletion voltage by the used fit
    range, the spread of the parasitic capacitance over a Pixcap Chip and the dispersion of parasitic effects over
    different Pixcap Chip samples.

    These estimation attempt is redone with slightly adjust fit ranges to estimate the systematic uncertainty.
    Here we account for the measured distribution of parasitic capacitance over the Pixcap Chip (its spread), for the
    dispersion of the parasitic capacitance between multiple sensors/chips and the particular choice of the fit range.
    For the additional fits to extract the systematic uncertainties no plotting of the fits is available.
    The Estimation of the different systematic uncertainties is performed by again fitting the voltage ranges but with
    either varied fit range or with the systematic uncertainties provided as the measurement error submitted to the fit
    algorithm. In the latter two cases the extracted fit errors on the depletion voltage is further propagated through
    the analysis.

    :param lower_boundary: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will only be estimated if this argument is provided.
        It is also possible to provide an Iterable of boundaries for multiple depletion regions to fit.
    :param upper_boundary: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will be estimated if this argument is provided.
        It is also possible to provide an Iterable of boundaries for multiple depletion regions to fit.
    :param voltage_data: array of the biasing HV voltages (with the correct sign)
    :param pixel_cap_data: capacitance data for one particular pixel on the sensor
    :param pixel_cap_error_data: uncertainties of the capacitance data for one particular pixel on the sensor.
    :param systematic_error: standard deviation of the parasitic capacitance of the Pixcap chip on a single sensor.
    :param dispersion_error: spread of the dispersion of the parasitic capacitance between different Pixcap chip
        samples. This will induce a systematic effect on the accuracy of the capacitance's and the depletion voltage of
        the investigated sensor.
    :param systematic_offset: enlargement in V for the fit range conditions applied for fitting. Needed to estiamte the
        systematic uncertainties by the fit range accurately. (default: 2)
    :param second_systematic_result_storage: numpy based result storage object for the results
        from varying the fit range.
    :param third_systematic_result_storage: numpy based result storage object for the results
        from accounting for the spread of the parasitic capacitance on the same pixcap chip.
    :param fourth_systematic_result_storage: numpy based result storage object for the results
        from accouting for the dispersion of the parasitic capacitance between different Pixcap chips.
    :param systematic_key_args: further keyword arguments to be propagated to the fits.
    """

    first_lower, first_upper = lower_boundary
    second_lower, second_upper = upper_boundary

    analyze_pixel_depletion(first_lower - systematic_offset, first_upper + systematic_offset, pixel_cap_data,
                            pixel_cap_error_data,
                            second_lower - systematic_offset,
                            second_upper + systematic_offset, voltage_data, second_systematic_result_storage,
                            fit_description_text=" Sensor Distribution Systematic", **systematic_key_args)
    analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, systematic_error, second_lower, second_upper,
                            voltage_data,
                            third_systematic_result_storage, fit_description_text=" Sensor Distribution Systematic",
                            **systematic_key_args)
    analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, dispersion_error, second_lower, second_upper,
                            voltage_data,
                            fourth_systematic_result_storage, fit_description_text=" Sensor Distribution Systematic",
                            **systematic_key_args)


def _handle_mp_parasitic_cap(bare_path: Optional[str], bare_file: Optional[str], **kwargs) -> tuple[Any, Any]:
    if bare_file is None:
        return 0, 0
    lock = kwargs.get('lock', None)
    keywords = {}
    if lock is not None:
        keywords["lock"] = lock
    with synchronized_process_open_file(bare_file, mode='r', **keywords) as correction_h5:
        assert isinstance(bare_path, (str, None))
        parasitic, parasitic_error = _handle_parasitic_cap(bare_path, correction_h5)
        if parasitic < 1.0:
            print("Unfortunatley the parasitic capacitance vanishs.")
        parasitic /= CAPACITANCE_CONVERSION_FACTOR
        parasitic_error /= CAPACITANCE_CONVERSION_FACTOR

        return parasitic, parasitic_error


def _handle_parasitic_cap(bare_path: Optional[str], in_file_h5: tb.File) -> tuple[Any, Any]:
    if bare_path is None:
        bare_group = in_file_h5.root
    else:
        bare_group, _ = walk_to_node(in_file_h5.root, bare_path, verify_create=True)

    assert "parasitic" in get_group_attributes(bare_group.analysis)
    assert "parasitic_error" in get_group_attributes(bare_group.analysis)
    parasitic = get_group_attribute(bare_group.analysis, "parasitic")
    parasitic_error = get_group_attribute(bare_group.analysis, "parasitic_error")
    return parasitic, parasitic_error


def analysis_data_handle(file: tb.File, data_group: GroupType, result_group: GroupType,
                         is_advanced=False, is_inter_pixel=False, **kwargs):
    """
    analysis_data_handle

    Implementation of the analysis strategy for the capacitance measurement of a pixel sensor.
    The capacitance of the pixels are measured and investigated individually. For determination of the
    capacitance values either a linear fit or non-linear least square fit algorithms are used.
    Depending on the choice of the ´is_advanced` parameter non-linear techniques are used.
    In this case either 'kafe2' or 'iminuit' are used for the least-squares minimization depending on the choice
    of parameters.
    When using the advanced least-squares procedure the fit results will be plotted to verify the convergence of the
    fit.

    Afterwards, it is possible to directly correct the results for the capacitance by the connection and the
    measurement circuit.

    In Addition, there is the special case of an inter-pixel capacitance measurement.
    If the data to be analysed comes from such a measurement this needs to be specified.
    Thus, in this case it will be checked which of the total current or the two inter-pix current data sets exist.
    The analysis will be done for each existing data set.
    Combinations with a C-V-Characterization might still be an issue.

    :param file: h5 file object containing the data to be analysed.
    :param data_group: hierarchy group of the opened hdf file containing the raw data (measurements).
    :param result_group: hierarchy group of the opened hdf file to write the analysis results to.
    :param is_advanced: boolean indicating if the advanced analysis strategy should be used
        or not (may require additional keyword arguments)
    :param is_inter_pixel: boolean, False, indicating whether the measurement to be analysed is an inter-pixel
        capacitance measurement. In this case more fits will be applied, adjusted to the specific structure of this
        problem
    :key full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used. (Only used for the advanced procedure)
    :key use_kafe2: boolean, indicates whether kafe2 is used for the fit. (Only used for the advanced procedure)
    :key plot: boolean, indicates whether to plot the data. An output PDF object could be submitted here
         instead of an explicitly created one. (Only used for the advanced procedure)
    :key apply_contour: boolean, indicates whether to determine the contours and try to plot them.
        (Only used for the advanced procedure)
    :key fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure)
    :key apply_correction: boolean, False, indicates whether the measured capacitance should be
        corrected immediately; Will require the presence of further arguments as information about
        the parasitic capacitance needs to be submitted.
    :key bare_file: hdf file containing the measurements and investigation of a bare pix cap sample to obtain
        information about intrinsic and parasitic capacitance. (Only required for the correction procedure, but in
        this case it must be present)
    :key bare_path: hdf files hierarchy path to the group containing the bare pix cap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present)
    """
    # get the correct analysis function
    perform_analysis = _get_analyze(is_advanced)

    # Read pixel map
    assert isinstance(data_group, tb.Group)
    assert isinstance(result_group, tb.Group)
    if is_inter_pixel:
        _handle_inter_pix_capacitance(file, data_group, is_advanced, perform_analysis, result_group, **kwargs)

    else:
        current_hist = check_leaf_unit(data_group.HistCurr, HIST_CURRENT_MEAS_UNIT)
        if np.any(current_hist > 1e30):
            data_group.HistCurr[data_group.HistCurr[:] > 1e30] = np.nan
            current_hist = check_leaf_unit(data_group.HistCurr, HIST_CURRENT_MEAS_UNIT)
        if is_advanced and "HistCurrErr" in data_group:
            check_leaf_unit(data_group.HistCurrErr, HIST_CURRENT_MEAS_UNIT)
            current_error_hist = data_group.HistCurrErr[:]
        else:
            current_error_hist = np.full_like(current_hist, fill_value=np.nan)

        if current_hist.shape[1] <= 40:
            temp_shape = [*current_hist.shape]
            temp_shape[1] = 41
            temp_current_hist = np.full(tuple(temp_shape), fill_value=np.nan)
            temp_current_hist[:, :40] = current_hist
            temp_current_hist_error = np.full(tuple(temp_shape), fill_value=np.nan)
            temp_current_hist_error[:, :40] = current_error_hist
            current_hist = temp_current_hist
            current_error_hist = temp_current_hist_error
            # maybe these changes should also be written back?
        # Read scan parameters
        scan_parameters = data_group.scan_params[:]
        assert isinstance(scan_parameters, tb.Table) or isinstance(scan_parameters, np.ndarray)
        assert isinstance(current_hist, tb.CArray) or isinstance(current_hist, np.ndarray)
        kwargs["current_error_hist"] = current_error_hist
        perform_analysis(file, result_group, current_hist, scan_parameters, **kwargs)

    # if necessary: directly apply the correction of the capacitance values
    # FIXME: this is explicitly using a lock!
    _handle_cap_correction(result_group, **kwargs)


def _handle_inter_pix_capacitance(file: tb.File, data_group: tb.Group, is_advanced: ADVANCED_PARAMETER_TYPE,
                                  perform_analysis: Callable[..., None], result_group: tb.Group, **kwargs):
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


cap_counter = 0


def _handle_cap_correction(result_group: tb.Group, **kwargs):
    global cap_counter
    cap_counter += 1
    if "lock" not in kwargs:
        try:
            lock = threading.RLock()
            kwargs["lock"] = lock
            _handle_cap_correction(result_group, **kwargs)
            return
        finally:
            del lock

    if "apply_correction" in kwargs and kwargs["apply_correction"]:
        assert 'bare_file' in kwargs
        assert 'bare_hdf_path' in kwargs
        apply_correction_simple(kwargs['bare_file'], kwargs['bare_hdf_path'], result_group, lock=kwargs['lock'])


def _get_analyze(is_advanced: bool) -> Callable[..., None]:
    if is_advanced:
        from pixcap65.advanced_analysis import advanced_analysis_delegate
        perform_analysis = advanced_analysis_delegate
    else:
        from pixcap65.analysis_util import analyze_data_delegate
        perform_analysis = analyze_data_delegate
    return perform_analysis

def fetch_bias_voltage(data_group, selection, scan_parameters=None):
    bias_voltages = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
    if scan_parameters is None:
        try:
            scan_parameters = data_group.scan_params[:]
        except:
            pass
    if len(bias_voltages.shape) > 1:
        bias_voltage_errors = bias_voltages[:, 2]
        return bias_voltages, bias_voltages[:, selection], bias_voltage_errors
    else:
        renew_bias_voltages = np.full((bias_voltages.shape[0], 3,), fill_value=np.nan)
        if np.any(np.isfinite(bias_voltages)):
            voltages = bias_voltages
            voltage_errors = np.full_like(voltages, np.nan)
            voltage_settings = voltages.copy()
        elif scan_parameters is not None:
            voltages = scan_parameters["hv_voltage"]
            voltage_errors = np.full_like(voltages, np.nan)
        else:
            warn("The scan parameters required to recalculate the bias voltage errors were not found", stacklevel=2)

        if np.all(~np.isfinite(voltage_errors)):
            from pixcap65.configs.config_handler import extract_smu_voltage_error
            import yaml
            try:
                voltage_settings = scan_parameters["bias_voltage"]
                with open("pixcap65/configs/keithley_2410_range.yaml") as f:
                    range_config = yaml.safe_load(f)
                    voltage_errors = extract_smu_voltage_error(range_config, voltages, 1000)
            except:
                voltage_errors = np.full_like(voltages, np.nan)

        renew_bias_voltages[:, 0] = voltage_settings
        renew_bias_voltages[:, 1] = voltages
        renew_bias_voltages[:, 2] = voltage_errors
        # file = group_get_file(data_group)
        # temp = data_group.BiasVoltageHist
        # assert isinstance(temp, tb.CArray)
        # try:
        #     import uuid
        #     file.copy_node(where=data_group, name="BiasVoltageHist",
        #                    newname="BiasVoltageHist_{}.backing".format(uuid.uuid4()))
        #     temp._f_remove()
        #     time.sleep(30)
        #     create_carray(file, data_group, "BiasVoltageHist", obj=renew_bias_voltages,
        #                   filters=tb.Filters(complib='blosc', fletcher32=False, complevel=5), unit=HIST_BIAS_MEAS_UNIT)
        # except Exception as e:
        #     warn("Encountered the error: " + repr(e), stacklevel=1)

        return renew_bias_voltages, voltages, voltage_errors


def analyze_depletion_delegate(data_group: tb.Group, analysis_group: tb.Group,
                               first_boundaries: Optional[BOUNDARY_TYPE],
                               second_boundaries: Optional[BOUNDARY_TYPE], chip_group: Optional[tb.Group] = None,
                               apply_doping=False,
                               **kwargs):
    """
    analyse_depletion_delegate

    NO LOCKS HERE except for no-op popping!

    Implementation of the investigation of the depletion behaviour of a pixel sensor.
    So first (this is the only non-optional functionality of this investigation) the C-V curve is used to obtain an
    estimator for the depletion voltage of each pixel on the investigated sensor module.
    Therefore, linear fits are applied to two distinc regions of the C-V curve.
    One in the large voltage limit (constant capacitances are expected as the depletion zone could not grow beyond the
    physical dimensions of the sensor) and one in the small.
    For the fits the capacitance is not used directly, but 1/C^2 as this quantity should be proportional to the
    (reversed) bias voltage.
    If no uncertainties for the capacitances are provided, a np.polyfit is used for this task, otherwise depending on
    the provided arguments either 'kafe2' or 'iminuit' is used.
    The depletion voltage is then estimated from the intersection of both straight line fits.
    For estimation of the uncertainty of the depletion voltage the full covariance matrix of the fits is used.

    Second, the depletion width and it's consequences are analysed if requested.
    This part will only be performed if it is requested by the 'apply_doping' parameter.
    This second analysis also requires the presence of the 'chip_group' parameter and within it the table/array
    'PhysicalDimensions' with the physical dimensions of the pixels on the module. Otherwise, it is not possible
    to determine the depletion widths from the measured capacitances, which will then be done by assuming a plate
    capacitator geometry.
    After this the effective doping profile is computed from the differential capacitance.


    :param data_group: hierachy group of the opend hdf file containing the raw data (measurements).
    :param analysis_group: hierachy group of the opened hdf file to write the analysis results to.
    :param first_boundaries: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel.
    :param second_boundaries: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel.
    :param chip_group: HDF files hierarchy group with the data/specifications of the pixels on the current sensor.
    :param apply_doping: boolean, False, indicating whether to investigate the (effective) doping of the sensor.
    :key use_kafe2: boolean, False, indicates whether kafe2 is used for the fit.
    :key plot: boolean, False, indicates whether to plot the data. AN output PDF object could be submitted here
         instead of an explicitly created one.
    :key apply_contours: boolean, indicates whether to determine the contours and try to plot them.
    :key fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure)
    :key verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    """
    from scipy.constants import epsilon_0
    # extract the additional parameters for advanced fitting procedures
    kwargs.setdefault('output_pdf', kwargs.get('fit_plot_pdf', None))
    manager_keywords = get_manager_keywords(**kwargs)
    if "authkey" in manager_keywords:
        del manager_keywords["authkey"]

    for key, value in kwargs.items():
        if 'lock' in key:
            print("Found a locking object", type(value), "in 'analyze_depletion_delegate'")
            try:
                print(value._serial, value._token)
            except:
                pass

    # verify and extract the raw data for further analysis
    cap_data = check_leaf_unit(analysis_group.UCHist, HIST_CAP_UNIT)
    cap_error_data = check_leaf_unit(analysis_group.UCErrHist, HIST_CAP_UNIT)
    voltage_data = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
    if len(voltage_data.shape) > 1:
        voltage_data = voltage_data[:, BIAS_VOLTAGE_ACCESS_IDX]

    # FIXME: provide here the correct manager arguments!
    with get_context_manager() as manager:
        if isinstance(first_boundaries, Iterable) and not isinstance(first_boundaries, Tuple):
            assert first_boundaries is not None
            assert second_boundaries is not None
            assert isinstance(first_boundaries, Sized)
            # fit_result_storage = DepletionArrayStore(n_depletions=len(first_boundaries))
            number_depletions = len(first_boundaries)
            fit_result_storage = manager.DepletionArrayStorage(n_depletions=number_depletions)
            for k, (first_bound, second_bound) in enumerate(zip(first_boundaries, second_boundaries)):
                first_lower, first_upper = first_bound
                second_lower, second_upper = second_bound
                fit_result_storage.set_depletion_region(k)
                depletion_delegation_impl(cap_data, cap_error_data, first_lower, first_upper, second_lower,
                                          second_upper,
                                          fit_result_storage, voltage_data, **kwargs)

        else:
            # extract the required data and create arrays for temporary storage.
            first_lower, first_upper = first_boundaries
            second_lower, second_upper = second_boundaries
            # fit_result_storage = DepletionArrayStore()
            number_depletions = 1
            fit_result_storage = manager.DepletionArrayStorage(**manager_keywords)
            depletion_delegation_impl(cap_data, cap_error_data, first_lower, first_upper, second_lower, second_upper,
                                      fit_result_storage, voltage_data, **kwargs)

        # save the depletion voltage data.
        file_h5 = group_get_file(analysis_group)
        create_carray(file_h5, where=analysis_group, name="DepletionHist",
                      title="Histogram of the depletion voltages", obj=fit_result_storage.depletion_voltage[:],
                      filters=GLOBAL_FILTERS, unit=HIST_BIAS_MEAS_UNIT)
        create_carray(file_h5, where=analysis_group, name="DepletionErrHist",
                      title="Histogram of the depletion voltage errors", obj=fit_result_storage.depletion_error[:],
                      filters=GLOBAL_FILTERS, unit=HIST_BIAS_MEAS_UNIT)
        create_carray(file_h5, where=analysis_group, name="DepFitParamHist",
                      title="Histogram of the depletion voltages fit parameters",
                      obj=fit_result_storage.fit_parameter_estimators[:],
                      filters=GLOBAL_FILTERS, unit="NONE")
        create_carray(file_h5, where=analysis_group, name="DepFitParamErrHist",
                      title="Histogram of the depletion voltages fit parameter errors",
                      obj=fit_result_storage.fit_parameter_errors[:],
                      filters=GLOBAL_FILTERS, unit="NONE")
        create_carray(file_h5, where=analysis_group, name="SystematicDispersionHist",
                      title="Histogram of the systamtic uncertainty by sensor dispersion",
                      obj=fit_result_storage.systematic_dispersion[:], filters=GLOBAL_FILTERS, unit=HIST_CAP_UNIT)
        create_carray(file_h5, where=analysis_group, name="SystematicGeneralHist",
                      title="Histogram of the systamtic uncertainty",
                      obj=fit_result_storage.systematic_errors[:], filters=GLOBAL_FILTERS, unit=HIST_CAP_UNIT)

        # CHECK: for correct implementation and then remove the try wrapper again!
        try:
            # we have to 2x2 matrices for each fit => overall there needs to be a 4x4 matrix per pixel!
            create_carray(file_h5, where=analysis_group, name="DepFitParamCovHist",
                          title="Matrix of the 2x2 covariance matrices for each pixel",
                          obj=fit_result_storage.fit_parameter_covariances[:], filters=GLOBAL_FILTERS, unit="None")
        except:
            pass

        # remove the fit storage object as it is no longer used anyway
        del fit_result_storage

    if (not apply_doping or chip_group is None or "PhysicalDimensions" not in chip_group or
            (chip_group.PhysicalDimensions.shape != (40, 40, 2) and chip_group.PhysicalDimensions.shape != (40, 41, 2))):
        return
    if chip_group.PhysicalDimensions.shape == (40, 40, 2):
        temp_data = np.full((40, 41, 2), fill_value=np.nan)
        temp_data[:, :40, :] = chip_group.PhysicalDimensions[:]
        temp_data[:, 40] = temp_data[:, 39, :]
        chip_group.PhysicalDimensions._f_remove()
        group_get_file(chip_group).create_array(where=chip_group, name="PhysicalDimensions", obj=temp_data)
        group_get_file(chip_group).flush()

    physical_dimensions_data = chip_group.PhysicalDimensions[:]
    pixel_areas = np.prod(physical_dimensions_data, axis=2)
    doping_shape = (40, 41, voltage_data.shape[0])
    doping_result_storage = DopingArrayStore(doping_shape, n_depletions=number_depletions)

    # some further definitions for the loop
    # this values will not be correct as I don't know the doping concentration the intrinsic bias voltage;
    # the intrinsic bias voltage could be estimated from a fit to the forward bias I-V characteristic.
    # noinspection PyPep8Naming
    NA = 1e16
    v_bi = SILICON_V_BIAS
    dep_table = file_h5.create_table(where=analysis_group, name="DepletionParamTable",
                                     description=DepletionWidthData,
                                     filters=GLOBAL_FILTERS)
    entry = dep_table.row

    _, bias_voltages, bias_voltage_errors = fetch_bias_voltage(data_group, BIAS_VOLTAGE_ACCESS_IDX)

    depletion_fit_parameters = analysis_group.DepFitParamHist[:]
    depletion_fit_parameter_errors = analysis_group.DepFitParamErrHist[:]

    for col, row in np.ndindex(GENERAL_PIXCAP_SHAPE):
        kwargs['fit_description_text'] = ' for Pixel ({col},{row})'.format(col=col, row=row)
        # make sure the provided data is useful for further investigation.
        # implies that the additonal row is not used at-all
        if row == 0 or not np.all(np.isfinite(physical_dimensions_data[col, row - 1])):
            continue
        pixel_cap_data = cap_data[col, row]
        pixel_cap_error_data = cap_error_data[col, row]
        pixel_area = pixel_areas[col, row - 1]  # should be calculated from the provded data in µm^2
        if not np.all(np.isfinite(pixel_cap_data)):
            continue
        doping_result_storage.set_pixel(row, col)
        entry["row"] = row
        entry["col"] = col

        n_eff, depletion_width_data, pos_min = analyze_doping_profile(bias_voltages, bias_voltage_errors,
                                                                      doping_result_storage, entry, NA / 2, v_bi,
                                                                      pixel_area, pixel_cap_data,
                                                                      pixel_cap_error_data, **kwargs)

        # could compute the resistivity from here!
        try:
            pixel_depletion_parameter = np.atleast_2d(depletion_fit_parameters[col, row])
            pixel_depletion_parameter_errors = np.atleast_2d(depletion_fit_parameter_errors[col, row])
            doping_result_storage.store_data('res_mod', EPS_SILICON * epsilon_0 * pixel_area ** 2 / (2 * SI_MOBILITY) *
                                             pixel_depletion_parameter[:, 2] * 1e12)
            doping_result_storage.store_data('res_mod_err',
                                             EPS_SILICON * epsilon_0 * pixel_area ** 2 / (2 * SI_MOBILITY) *
                                             pixel_depletion_parameter_errors[:, 2] * 1e12)
        except:
            print(depletion_fit_parameters[col, row])
            print(pixel_depletion_parameter)
            raise

        if kwargs.get("verbose", False):
            msg = "The minimum concentration is {nm} and depth {dep_min} for pixel ({col}, {row})"
            print(msg.format(nm=n_eff[pos_min], dep_min=depletion_width_data[pos_min], col=col, row=row))

    # save the computed information about the depletion behaviour
    create_carray(file_h5, where=analysis_group, name="DepletionWidth",
                  title="Depletion width from the pixel capacitance",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.depletion_width_plate, unit="um")
    create_carray(file_h5, where=analysis_group, name="DepletionWidthErr",
                  title="Uncertainty of the depletion width from the pixel capacitance",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.depletion_width_plate_error, unit="um")

    create_carray(file_h5, where=analysis_group, name="DepletionParameters",
                  title="Depletion Parameters from fitting the depletion width",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.depletion_fit_parameter_table,
                  unit="cm^-3; cm^-3; V")
    create_carray(file_h5, where=analysis_group, name="DepletionErrors",
                  title="Depletion Parameters from fitting the depletion width",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.depletion_fit_parameter_error_table,
                  unit="cm^-3; cm^-3; V")
    create_carray(file_h5, where=analysis_group, name="DepletionCovariance",
                  title="Covariance matrices for Depletion Parameters from fitting the depletion width",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.depletion_fit_covariance_table,
                  unit="{{cm^-6, cm^-6, cm^-3 V},{cm^-6, cm^-6, cm^-3 V},{V cm^-3, V cm^-3, V^2}}")
    create_carray(file_h5, where=analysis_group, name="DepletionEffDoping",
                  title="Data for the effective doping from the cv-analysis",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.effective_doping_table, unit="cm^-3")
    create_carray(file_h5, where=analysis_group, name="DepletionResitivity",
                  title="Data for the specific resistivity from the cv-analysis", filters=GLOBAL_FILTERS,
                  obj=doping_result_storage.resistivity_table, unit="Ocm")
    create_carray(file_h5, where=analysis_group, name="ModDepletionResistivity",
                  title="Data for the specific resitivity from the slopes of the cv-depletion-analysis",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.second_resistivities, unit="Ocm")
    create_carray(file_h5, where=analysis_group, name="ModDepletionResistivityErr",
                  title="Data for the uncertainties of the specific resitivity from the slopes of the cv-depletion-analysis",
                  filters=GLOBAL_FILTERS, obj=doping_result_storage.second_resistivities_err, unit="Ocm")
    file_h5.flush()


def __depletion_iterator_mp_implementation(storage, index, **kwargs):
    global mp_shared_cap_data, mp_shared_cap_error_data, mp_shared_voltage_data, mp_shared_keywords
    global mp_shared_lower_limit, mp_shared_upper_limit
    key_args = mp_shared_keywords.copy()
    key_args.update(kwargs)
    __depletion_iterator_implementation(index, mp_shared_cap_data, mp_shared_cap_error_data, mp_shared_lower_limit,
                                        mp_shared_upper_limit, mp_shared_voltage_data, storage, **key_args)


def __init_mo_depletion_iterator_implementation(cap_data, cap_error_data, voltage_data, keywords, lower, upper):
    global mp_shared_cap_data, mp_shared_cap_error_data, mp_shared_voltage_data, mp_shared_keywords
    mp_shared_cap_data = cap_data
    mp_shared_cap_error_data = cap_error_data
    mp_shared_voltage_data = voltage_data
    mp_shared_keywords = keywords.copy()
    global mp_shared_lower_limit, mp_shared_upper_limit
    mp_shared_lower_limit = lower
    mp_shared_upper_limit = upper
    if "lock" in mp_shared_keywords:
        print("Found a locking/semaphore object within the keywords; this might be part of the leakage issue!")


def __depletion_iterator_implementation(index, cap_data, cap_error_data, lower_boundary,
                                        upper_boundary, voltage_data, storage, **kwargs):
    try:
        # this will require the usage of additonal arrays! But we could reuse the implementation for
        systematic_offset = kwargs.pop("systematic_offset", 2)
        systematic_key_args = kwargs.copy()
        systematic_key_args.pop("plot", False)
        systematic_key_args.pop("fit_plot_pdf", None)
        systematic_key_args.pop("output_pdf", None)
        systematic_key_args.pop("cv_fit_plot_pdf", None)

        ii, jj = index
        first_lower, first_upper = lower_boundary
        second_lower, second_upper = upper_boundary
        pixel_cap_data = cap_data[ii, jj, :]
        pixel_cap_error_data = cap_error_data[ii, jj, :]
        # could only perform the analysis for pixels with trustable measurements.
        if np.all(np.isfinite(pixel_cap_data)):
            storage.set_pixel(jj, ii)
            analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, pixel_cap_error_data, second_lower,
                                    second_upper, voltage_data, storage,
                                    fit_description_text=" for Pixel ({col},{row})".format(col=ii, row=jj),
                                    **kwargs)

            # needs to be fixed by the correct value!
            systematic_error = np.full_like(pixel_cap_data, fill_value=kwargs.get("para_dist", 1.e-18))
            dispersion_error = np.full_like(pixel_cap_data,
                                            fill_value=kwargs.get("systematic_dispersion",
                                                                  DISPERSION_PARASITIC_DEVIATION))

            second_systematic_result_storage = DepletionNumpyStore(tb.dtype_from_descr(DepletionData))
            third_systematic_result_storage = DepletionNumpyStore(tb.dtype_from_descr(DepletionData))
            fourth_systematic_result_storage = DepletionNumpyStore(tb.dtype_from_descr(DepletionData))
            assert isinstance(voltage_data, np.ndarray)
            __extract_systematic_depletion_effects((first_lower, first_upper),
                                                   (second_lower, second_upper), voltage_data,
                                                   pixel_cap_data, pixel_cap_error_data, systematic_error,
                                                   dispersion_error, systematic_offset,
                                                   second_systematic_result_storage,
                                                   third_systematic_result_storage,
                                                   fourth_systematic_result_storage,
                                                   systematic_key_args)
            dispersion_result = fourth_systematic_result_storage.table.Ubi_error[-1]
            if storage.depletion_reg is None:
                systematic_result = np.sqrt(np.abs(
                    storage.depletion_voltage[ii, jj] - second_systematic_result_storage.table.Ubi[-1]) ** 2 +
                                            third_systematic_result_storage.table.Ubi_error[-1] ** 2)
            else:
                systematic_result = np.sqrt(np.abs(
                    storage.depletion_voltage[ii, jj, storage.depletion_reg] -
                    second_systematic_result_storage.table.Ubi[-1]) ** 2 +
                                            third_systematic_result_storage.table.Ubi_error[
                                                -1] ** 2)

            storage.store_data("systematic", systematic_result)
            storage.store_data("dispersion", dispersion_result)
    except:
        print("Exception occured")
        import sys
        print(sys.exc_info())
        raise


def init_pool(storage):
    global fit_result_mp_storage
    fit_result_mp_storage = storage[0]


def depletion_delegation_impl(cap_data, cap_error_data, first_lower, first_upper, second_lower, second_upper,
                              fit_result_storage: DepletionArrayStore, voltage_data: Union[np.ndarray, tb.CArray],
                              **kwargs) -> Optional[List]:
    """
    depletion_delegate_impl

    @author Dominik Fischer
    @date 2026-05-07

    Implementation of the pixel-wise depletion voltage estimation from a provided C-V characterization.
    For pixel the depletion voltage of the sensor is estimated from two fits to the C-V curve (more precise: 1/C^2)
    The intersection of this two straight lines then provides the depletion voltage.

    Also the systematic uncertainties of the depletion voltage are estimated.
    Here we account for the measured distribution of parasitic capacitance over the Pixcap Chip (its spread), for the
    dispersion of the parasitic capacitance between multiple sensors/chips and the particular choice of the fit range.
    For the additional fits to extract the systematic uncertainties no plotting of the fits is available.
    The Estimation of the different systematic uncertainties is performed by again fitting the voltage ranges but with
    either varied fit range or with the systematic uncertainties provided as the measurement error submitted to the fit
    algorithm. In the latter two cases the extracted fit errors on the depletion voltage is further propagated through
    the analysis.


    :param cap_data: capacitance data from the characterization.
    :param cap_error_data: uncertainties of the capacitance data from the characterization.
    :param first_lower: lower bound of the first fit section (high voltage limit) to estimate the depletion voltage of
        the pixel.
    :param first_upper: upper bound of the first fit section (high voltage limit) to estimate the depletion voltage of
        the pixel.
    :param second_lower: lower bound of the second fit section (low voltage limit) to estimate the depletion voltage of
        the pixel.
    :param second_upper: upper bound of the second fit section (low voltage limit) to estimate the depletion voltage of
        the pixel.
    :param fit_result_storage: DataStorage object for intermediate storage of fit results and depletion parameters
    :param voltage_data: data of the applied HV voltages
    :param kwargs: further keyword arguments to be propagated to functions/implementations.
    :key use_kafe2: boolean, False, indicates whether kafe2 is used for the fit.
    :key apply_contours: boolean, indicates whether to determine the contours and try to plot them.
    :key plot: boolean, False, indicates whether to plot the data. AN output PDF object could be submitted here
         instead of an explicitly created one.
    :key fit_plot_pdf: PDF object to save the fit figures to.
    :key verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    :key systematic_offset: enlargement in V for the fit range conditions applied for fitting. Needed to estiamte the
        systematic uncertainties by the fit range accurately. (default: 2)
    :key para_dist: standard deviation of the parasitic capacitance of the Pixcap chip on a single sensor.
    :key systematic_dispersion: spread of the dispersion of the parasitic capacitance between different Pixcap chip
        samples. This will induce a systematic effect on the accuracy of the capacitance's and the depletion voltage of
        the investigated sensor.
    :return: optionally list of figure holder objects; the list elements could also be None themselves.
    """
    kwargs.setdefault("verbose", False)
    # here we have the largest optimisation potential by using multiprocessing, but it would be necessary to share
    # the data storage class among the different processes. We could not be sure that there wont be a data-race.

    lower_hv_boundary = (first_lower, first_upper)
    upper_hv_boundary = (second_lower, second_upper)
    iterator = tqdm(np.ndindex(GENERAL_PIXCAP_SHAPE), total=1600)

    # need to make sure that we will not leake any locks or other synchronization primitives!
    # in this branch, there seems to be no leakage of semaphore objects!
    # this is the old standard mode
    [__depletion_iterator_implementation(index, cap_data, cap_error_data,
                                         lower_hv_boundary, upper_hv_boundary, voltage_data,
                                         fit_result_storage, **kwargs) for index in iterator]


DOPING_RESULT_TYPE = Tuple[np.ndarray, np.ndarray, int]


def analyze_doping_profile(bias_voltages: np.ndarray,
                           bias_voltage_errors: np.ndarray,
                           doping_result_storage: DopingArrayStore, entry, n_a: float, v_bi: float,
                           pixel_area, pixel_cap_data: np.ndarray, pixel_cap_error_data: np.ndarray,
                           **kwargs) -> DOPING_RESULT_TYPE:
    """
    analyze:doping_profile

    NO USAGE OF LOCKS HERE!

    @author Dominik Fischer
    @date 2026-05-07

    Extracts information about the doping profile from the C-V characterization provided.

    :param bias_voltages: HV voltages used for the characterization.
    :param bias_voltage_errors: uncertainties/errors of the HV voltages used for the characterization.
    :param doping_result_storage: Data storage object for intermediate storage of doping results.
    :param entry: table row to store fit parameter results.
    :param n_a: guess for the doping concentration to perform the fit to the depletion width model
    :param v_bi: guess for the intrinsic bias voltage to perform the fit to the depletion width model
    :param pixel_area: area of the pixel diode to be investigated.
    :param pixel_cap_data: capacitance data from the C-V characterization.
    :param pixel_cap_error_data: uncertainties of the capacitance data from the C-V characterization.
    :param kwargs: further keyword arguments to be propagated to functions/implementations.
    :key use_kafe2: boolean, False, indicates whether kafe2 is used for the fit.
    :key apply_contours: boolean, indicates whether to determine the contours and try to plot them.
    :key plot: boolean, False, indicates whether to plot the data. AN output PDF object could be submitted here
         instead of an explicitly created one.
    :key fit_plot_pdf: PDF object to save the fit figures to.
    :return: doping_profile, depletion_width data, index of minimum doping concentration
    """
    from scipy import constants
    # Begin of the extraction part
    use_kafe2 = kwargs.pop("use_kafe2", False)
    apply_contours = kwargs.pop("apply_contours", False)
    plot = kwargs.pop("plot", False)
    output_pdf = kwargs.pop("fit_plot_pdf", None)
    fit_description_text = kwargs.pop("fit_description_text", "")

    # extract some further quantities from previous measurements and analysis.
    if np.all(np.isfinite(pixel_cap_error_data)):
        effective_cap_errors = pixel_cap_error_data
    else:
        effective_cap_errors = np.full_like(bias_voltages, 1)

    # compute the depletion width from the sensor properties.
    # What is the unit of this result
    # I assume this will result in wrong
    # provides the width in um
    depletion_width_data = np.asarray((constants.epsilon_0 * pixel_area * EPS_SILICON) / pixel_cap_data) * 1e-6
    depletion_width_error_data = (constants.epsilon_0 * pixel_area * effective_cap_errors * EPS_SILICON) / (np.array(
        pixel_cap_data) ** 2) * 1e-6

    if np.any(pixel_cap_data > 1e-5):
        with open("doping_handler.txt", 'a') as f:
            print("There was capacitance data much to large for F.", file=f)

    # fit the theoretical expected depletion width to determine some of the properties of the pixel diode
    parameter_guess = {
        "NAD": n_a,
        "V": v_bi,
        "dep": -10,
        "sat": np.max(depletion_width_data),
    }
    # this model fits may fail due to two indistinguishable parameters.
    if use_kafe2:
        from kafe2 import XYFit, XYContainer
        assert isinstance(bias_voltages, np.ndarray)
        xy_data = XYContainer(x_data=bias_voltages, y_data=depletion_width_data)
        xy_data.add_error(axis='y', err_val=depletion_width_error_data)
        if np.all(bias_voltage_errors[np.isfinite(bias_voltages)]):
            xy_data.add_error(axis='x', err_val=bias_voltage_errors)

        fitter = XYFit(xy_data, model_function=model_depletion, minimizer="iminuit")
        fitter.assign_parameter_latex_names(voltages="U_\\text{{bi}}", NA="N_\\text{{A}}", ND="N_\\text{{D}}",
                                            V="U_\\text{{th}}", )
        fitter.assign_model_function_latex_name("d_\\text{{depletion}}")
        fitter.assign_model_function_latex_expression(
            "\\sqrt{{\\frac{{2\\epsilon_0\\epsilon}}{{e}}\\cdot\\frac{{{NA}+{ND}}}{{{NA}\\cdot{ND}}}"
            "\\cdot ({V}+{voltages})}}")
        fitter.set_parameter_values(**parameter_guess)
        fitter.limit_parameter(name="V", lower=0.0, upper=10.0)
        fitter.limit_parameter(name="NAD", lower=0.0)
        fitter.limit_parameter(name="dep", upper=-0.5)
        fitter.limit_parameter(name="sat", lower=0.0)
        fitter.do_fit()
        assert fitter.did_fit
        depletion_fit_propagate_parameters = fitter.parameter_name_value_dict
        depletion_fit_params = fitter.parameter_values
        depletion_fit_errors = fitter.parameter_errors
        depletion_fit_cov = fitter.parameter_cov_mat
        if plot:
            assert isinstance(fitter, XYFit)
            handle_kafe2_advanced_options(fitter, apply_contours, "$U_\\text{{bi}}$ / \\unit{{\\volt}}",
                                          "$d$ / \\unit{{\\micro\\meter}}",
                                          "Depletion Width fit{}".format(fit_description_text),
                                          output_pdf,
                                          "Depletion Width contours{}".format(fit_description_text))
    else:
        from iminuit import Minuit
        # noinspection PyProtectedMember
        from iminuit.cost import LeastSquares, Model
        assert isinstance(model_depletion, Model)
        cost = LeastSquares(x=bias_voltages, y=depletion_width_data,
                            yerror=depletion_width_error_data, model=model_depletion)
        fitter = Minuit(cost, NAD=5e15, V=0.7, dep=-10, sat=np.max(depletion_width_data))
        fitter.limits["V"] = (0.0, 10)
        # noinspection PyTypeChecker
        fitter.limits["NAD", "sat"] = (0.0, None)
        fitter.limits["dep"] = (None, -0.5)
        fitter.migrad()
        fitter.hesse()
        depletion_fit_propagate_parameters = fitter.values.to_dict()
        depletion_fit_params = np.array(fitter.values)
        depletion_fit_errors = np.array(fitter.errors)
        depletion_fit_cov = np.asarray(fitter.covariance)
        if plot:
            assert isinstance(fitter, Minuit)
            handle_minuit_advanced_options(fitter, apply_contours, "$U_\\text{{bi}}$ / \\unit{{\\volt}}",
                                           "$d$ / \\unit{{\\micro\\meter}}",
                                           "Depletion Width fit{}".format(fit_description_text),
                                           output_pdf,
                                           "Depletion Width contours{}".format(fit_description_text))

    doping_result_storage.store_data("width", depletion_width_data)
    doping_result_storage.store_data("width_error", depletion_width_error_data)
    doping_result_storage.store_data("fit_parameters", depletion_fit_params)
    doping_result_storage.store_data("fit_parameters_error", depletion_fit_errors)
    doping_result_storage.store_data("fit_covariance", depletion_fit_cov)

    # calculate the effective doping profile of the sensor.
    n_eff = effective_doping(pixel_cap_data, -bias_voltages,
                             diode_area=pixel_area)
    doping_result_storage.store_data("doping", n_eff)
    resistivity = 1 / (constants.elementary_charge * n_eff * 1950)

    doping_result_storage.store_data("resistivity", resistivity)
    pos_min = int(np.argmin(n_eff))
    for key, value in depletion_fit_propagate_parameters.items():
        entry[key] = value
    entry.append()
    return n_eff, depletion_width_data, pos_min


# For performace optimisation we must make sure that the implementation is thread-safe.
def analyze_pixel_depletion(first_lower, first_upper,
                            pixel_cap_data: np.ndarray,
                            pixel_cap_error_data: np.ndarray, second_lower, second_upper,
                            voltage_data: TABLES_LEAF_COMPAT_TYPE,
                            result: DepletionDataStore, **kwargs):
    """
    analyze_pixel_depletion

    @author Dominik Fischer
    @date 2026-05-07

    NO LOCKS IN USE HERE!

    Helper function to perform the investigation of the depletion voltage for a single pixel.
    The function does not need to know which pixel is currently investigated which enables the usage of
    for distributions of the capacitance over the whole sensor.

    :param first_upper: upper limit of the first fit range for the high voltage limit of the capacitance behaviour
        to estimate the depletion voltage of the pixel.
    :param first_lower: lower limit of the first fit range for the high voltage limit of the capacitance behaviour
        to estimate the depletion voltage of the pixel.
    :param second_upper: upper limit of the second fit range for the low voltage limit of the capacitance behaviour
        to estimate the depletion voltage of the pixel.
    :param second_lower: lower limit of the second fit range for the low voltage limit of the capacitance behaviour
        to estimate the depletion voltage of the pixel.
    :param pixel_cap_data: capacitance data for one particular pixel on the sensor
    :param pixel_cap_error_data: uncertainties of the capacitance data for one particular pixel on the sensor.
    :param voltage_data: array of the biasing HV voltages (with the correct sign)
    :param result: data store container to write the results back
    :param kwargs: further keyword arguments for fitting and output.
    :key fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :key use_kafe2: boolean, False, indicates whether kafe2 is used for the fit.
    :key apply_contours: boolean, indicates whether to determine the contours and try to plot them.
    :key plot: boolean, False, indicates whether to plot the data. AN output PDF object could be submitted here
         instead of an explicitly created one.
    :key fit_plot_pdf: PDF object to save the fit figures to.
    :key verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    :key cv_fit_plot_pdf: analog to `fit_plot_pdf` to activate the plotting for c-v- and depletion fits independent from
        the plotting for capacitance estimation fits. If this keyword argument is present also the `plot` arguments will
        be set automatically.
    """
    # extract the additional parameters for advanced fitting procedures
    verbose_output = kwargs.pop("verbose", False)
    # FIXME: What is about the enhanced keyword argument available? The readout does not support the additonal fields here!
    if kwargs.get("enhanced", False):
        raise KeyError("Enhanced mode not implemented yet")
    # perhaps the implemenation could be improved in general if the data is first collected by a structured numpy array.
    (dep_voltage_2, dep_voltage_error_2, first_dep_errors, first_dep_parameters, first_covariance,
     second_dep_errors, second_dep_parameters, second_covariance) = _analyze_pixel_depletion_fit(
        pixel_cap_data, pixel_cap_error_data, voltage_data, first_lower, first_upper, second_lower, second_upper,
        **kwargs)
    if verbose_output:
        print("The depletion voltage is {voltage}+-{error}".format(voltage=dep_voltage_2,
                                                                   error=dep_voltage_error_2))

    result.store_data("depletion", dep_voltage_2)
    result.store_data("depletion_error", dep_voltage_error_2)
    result.store_data("fit_result_first", first_dep_parameters)
    result.store_data("fit_result_second", second_dep_parameters)
    result.store_data("fit_error_first", first_dep_errors)
    result.store_data("fit_error_second", second_dep_errors)
    result.store_data("fit_result_cov_first", first_covariance)
    result.store_data("fit_result_cov_second", second_covariance)
    result.flush_data()


def _analyze_pixel_depletion_fit(pixel_cap_data: np.ndarray,
                                 pixel_cap_error_data: np.ndarray,
                                 voltage_data: TABLES_LEAF_COMPAT_TYPE,
                                 first_lower, first_upper, second_lower, second_upper, **kwargs) -> tuple:
    # TODO: missing docstring
    # no locks in use here!
    cv_fit_pdf = kwargs.pop("cv_fit_plot_pdf", None)
    if cv_fit_pdf:
        kwargs["fit_plot_pdf"] = cv_fit_pdf
        kwargs["plot"] = True
    # extract the information about the depletion voltage
    assert not isinstance(voltage_data, tb.Leaf)
    first_section_upper_mask = voltage_data <= first_upper
    first_section_lower_mask = voltage_data >= first_lower
    first_section_mask = np.logical_and(first_section_upper_mask, first_section_lower_mask)

    # print(second_upper, second_lower)
    second_section_upper_mask = voltage_data <= second_upper
    second_section_lower_mask = voltage_data >= second_lower
    second_section_mask = np.logical_and(second_section_upper_mask, second_section_lower_mask)

    # extract and select the usable voltage and capacitance data for the two fit ranges.
    effective_capacitance_data = np.reciprocal(pixel_cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 2
    effective_capacitance_error_data = np.reciprocal(
        pixel_cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 3 * pixel_cap_error_data * CAPACITANCE_CONVERSION_FACTOR if np.all(
        np.isfinite(pixel_cap_error_data)) else pixel_cap_error_data
    first_voltage_data = voltage_data[first_section_mask]
    second_voltage_data = voltage_data[second_section_mask]
    first_cap_data = effective_capacitance_data[first_section_mask]
    second_cap_data = effective_capacitance_data[second_section_mask]
    first_cap_error_data = effective_capacitance_error_data[first_section_mask]
    second_cap_error_data = effective_capacitance_error_data[second_section_mask]

    # perform fits to the two boundary regions specified to estimate the two distinct behaviours.
    is_first = True
    try:
        first_dep_cov, first_dep_errors, first_dep_parameters = get_depletion_fit(first_cap_data, first_cap_error_data,
                                                                                  first_voltage_data, "First",
                                                                                  **kwargs)

        is_first = False

        second_dep_cov, second_dep_errors, second_dep_parameters = get_depletion_fit(second_cap_data,
                                                                                     second_cap_error_data,
                                                                                     second_voltage_data, "Second",
                                                                                     **kwargs)
    except AssertionError:
        print(first_lower, first_upper, second_lower, second_upper)
        print(is_first)
        if kwargs.get("enhanced", False):
            return np.nan, np.nan, np.nan, np.nan, \
                np.nan, np.nan, np.nan, np.nan, \
                np.nan, np.nan
        else:
            raise

    # estimate the depletion voltage
    d = first_dep_parameters[1]
    b = second_dep_parameters[1]
    c = first_dep_parameters[0]
    a = second_dep_parameters[0]
    dep_voltage_2 = (d - b) / (a - c)

    dep_voltage_jacobian = np.array([
        (b - d) / ((a - c) ** 2),
        1 / (c - a),
        (d - b) / ((a - c) ** 2),
        1 / (a - c)
    ])

    # combine both cov matrices into a single one:
    full_cov_2 = np.zeros((4, 4))  # cross correlations between the two fits are not known
    assert full_cov_2 is not None
    assert first_dep_cov is not None
    assert second_dep_cov is not None
    full_cov_2[:2, :2] = second_dep_cov
    full_cov_2[2:, 2:] = second_dep_cov

    # perform the full error calculation with matrix methods:
    dep_voltage_error_2 = np.sqrt(dep_voltage_jacobian @ full_cov_2 @ dep_voltage_jacobian)

    if kwargs.get("enhanced", False):
        return dep_voltage_2, dep_voltage_error_2, first_dep_parameters[0], first_dep_errors[0], first_dep_parameters[
            1], first_dep_errors[1], second_dep_parameters[0], second_dep_errors[0], second_dep_parameters[1], \
            second_dep_errors[1]

    return dep_voltage_2, dep_voltage_error_2, first_dep_errors, first_dep_parameters, first_dep_cov, second_dep_errors, second_dep_parameters, second_dep_cov


def get_depletion_fit(cap_data: np.ndarray, cap_error_data: np.ndarray, voltage_data: np.ndarray,
                      fit_reference, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    get_depletion_fit

    @author Dominik Fischer
    @date 2026-05-07

    NO LOCKS IN USE ANYWHERE

    performs the necessary fits to estimate the depletion voltage from to linear fits to the capacitance
    characterization.


    :param cap_data: capacitance data from the characterization for this fit section.
    :param cap_error_data: uncertainties of the capacitance data from the characterization for this fit section.
    :param voltage_data: data of the applied HV voltages for this fit section.
    :param fit_reference: identifying the fit section for which the fit is performed.
    :key fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :key use_kafe2: boolean, False, indicates whether kafe2 is used for the fit.
    :key apply_contours: boolean, indicates whether to determine the contours and try to plot them.
    :key plot: boolean, False, indicates whether to plot the data. AN output PDF object could be submitted here
         instead of an explicitly created one.
    :key fit_plot_pdf: PDF object to save the fit figures to.
    :return: covariance_matrix, fit parameter errors, fit parameter values
    """
    # extract the additional parameters for advanced fitting procedures
    use_kafe2 = kwargs.pop("use_kafe2", False)
    apply_contours = kwargs.pop("apply_contours", False)
    plot = kwargs.pop("plot", False)
    output_pdf = kwargs.pop("fit_plot_pdf", None)
    fit_description_text = kwargs.pop("fit_description_text", "")
    if np.all(np.isfinite(cap_error_data)):
        if use_kafe2:
            # perform the fit
            from kafe2 import XYContainer, XYFit
            data_container = XYContainer(voltage_data, cap_data)
            data_container.add_error(axis='y', err_val=cap_error_data)
            m = XYFit(data_container, model_function=depletion_model)
            m.do_fit()

            # extract the fit parameters
            assert m.did_fit
            first_dep_parameters = np.array([m.parameter_values[0], m.parameter_values[1]])
            first_dep_errors = np.array([m.parameter_errors[0], m.parameter_errors[1]])
            first_dep_cov = m.parameter_cov_mat
            assert first_dep_cov is not None
            if plot:
                handle_kafe2_advanced_options(m, apply_contours, "U in V", "\\frac{{1}}{{C^2}}",
                                              "{} Fit{}".format(fit_reference, fit_description_text),
                                              output_pdf,
                                              "{} Contour{}".format(fit_reference, fit_description_text))

        else:
            # perform the fit
            from iminuit import Minuit
            # noinspection PyProtectedMember
            from iminuit.cost import LeastSquares, Model
            assert isinstance(depletion_model, Model)
            cost = LeastSquares(voltage_data, cap_data, cap_error_data, depletion_model)
            m = Minuit(cost, a=1, b=0)
            m.migrad()
            m.hesse()

            # extract the parameters
            first_dep_parameters = np.array([m.values['a'], m.values['b']])
            first_dep_errors = np.array([m.errors['a'], m.errors['b']])
            first_dep_cov = m.covariance
            if plot:
                from pixcap65.threaded_plotting import threading_lock
                with threading_lock:
                    handle_minuit_advanced_options(m, apply_contours, "$U$ / \\unit{{\\volt}}",
                                                   "$\\frac{{1}}{{C^2}}$ / \\unit{{\\per\\femto\\farad\\squared}}",
                                                   "{} Fit{}".format(fit_reference, fit_description_text),
                                                   output_pdf,
                                                   "{} Contour{}".format(fit_reference, fit_description_text))
            try:
                assert first_dep_cov is not None
            except AssertionError:
                print(m.valid)
                print(m.fmin)
                print(voltage_data)
                from matplotlib import pyplot as plt
                plt.close('all')
                plt.errorbar(voltage_data, cap_data, yerr=cap_error_data, )
                plt.show()
                print(voltage_data.shape)
                print(cap_data.shape)
                print(cap_error_data.shape)
                raise
    else:
        first_result = np.polyfit(voltage_data, cap_data, deg=1, cov=True)

        # extract the parameters and fit results.
        first_dep_parameters = np.asarray(first_result[0])
        first_dep_cov = np.asarray(first_result[1])
        first_dep_errors = np.sqrt(np.diag(first_dep_cov))
        assert first_dep_cov is not None
    assert isinstance(first_dep_parameters, np.ndarray)
    assert isinstance(first_dep_errors, np.ndarray)
    assert isinstance(first_dep_cov, np.ndarray)
    return first_dep_cov, first_dep_errors, first_dep_parameters


def effective_doping(capacitance, bias_voltages, diode_area=None) -> np.ndarray:
    """
    effective doping

    Helper function to calculate the effective doping for every bias voltage/depletion depth.

    :param capacitance: measured (and corrected) capacitance of the CV characterization
    :param bias_voltages: bias voltages corresponding to the provided capacitance
    :param diode_area: area of the individual pixel.
    :return: effective doping concentration in cm^{-3}
    """
    from scipy import constants
    from findiff import Diff

    # make sure the data is provided as numpy arrays
    capacitance = np.asarray(capacitance)
    bias_voltages = np.asarray(bias_voltages)

    if diode_area is None:
        diode_area = 50 * 50  # measured in um^2
    temp_capacitance = np.reciprocal(capacitance ** 2)

    # noinspection PyTypeChecker
    try:
        # handle duplicates
        unique_voltage, cap_mask, voltage_counts = np.unique(bias_voltages, return_index=True, return_counts=True)
        duplicate_mask = voltage_counts > 1
        unique_temp_capacitance = temp_capacitance[cap_mask]
        derivative = np.full_like(bias_voltages, np.nan)
        d_du = Diff(0, unique_voltage, acc=4)
        derivative[cap_mask] = d_du(unique_temp_capacitance)
    except np.linalg.LinAlgError:
        print("bias_voltages", bias_voltages.shape)
        print(bias_voltages)
        print("capacitance", temp_capacitance.shape)
        print(temp_capacitance)
        print(capacitance)
        raise

    # need to manually broadcase the derivatives array if bias voltage duplicates were present.
    # only the duplicated values are still missing.
    if np.any(duplicate_mask):
        duplicate_values = unique_voltage[duplicate_mask]
        for dup_voltage in duplicate_values:
            indices = np.nonzero(bias_voltages == dup_voltage)
            actual_index = np.extract(unique_voltage == dup_voltage, cap_mask)[0]
            actual_doping = derivative[actual_index]
            derivative[indices] = actual_doping

    n_eff = 2 / (constants.elementary_charge * constants.epsilon_0 * EPS_SILICON * (diode_area ** 2) * np.asarray(derivative)) * 1e18
    return n_eff


def analyze_capacitance_distribution(raw_data, base_path=None, corrected_distribution=False, **kwargs):
    """
    analyze_capacitance_distribution

    Helper function to describe how the capacitance is distributed over the sensor.
    We are in particular interested in the average capacitance of the pixel and their spread/dispersion.
    This is just a wrapper function to perform the file handles and delegate the whole analysis to another
    function.

    :param raw_data: hdf file containing the measurements and investigation of a pixcap sample to obtain
        information about the capacitances.
    :param corrected_distribution: boolean, False, indicates whether the corrected capacitance distribution
        should be analyzed.
    :param base_path: hdf files group witht the measurement data.
    :key fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure).
    :key output_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure).
    :key fit_plot_pdf_name:  Name of the PDF file to save fitting figures from the advanced procedures to
        (Only used for the advanced procedure).
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key exclude_cap_hist: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    """
    temp_plot_name = kwargs.pop("fit_plot_pdf_name", None)
    if temp_plot_name is not None:
        assert "fit_plot_pdf_name" not in kwargs
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(temp_plot_name) as pdf:
            analyze_capacitance_distribution(raw_data, base_path, corrected_distribution,
                                             fit_plot_pdf=pdf, **kwargs)
    else:
        output_pdf = kwargs.pop("output_pdf", None)
        if output_pdf is None:
            output_pdf = kwargs.pop("fit_plot_pdf", None)
        with tb.open_file(raw_data, mode='a') as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            if corrected_distribution:
                analyze_capacitance_distribution_delegate(base_group.total_cap.analysis_correction, output_pdf,
                                                          **kwargs)
            else:
                analyze_capacitance_distribution_delegate(base_group.total_cap.analysis, output_pdf, **kwargs)


def analyze_capacitance_distribution_delegate(analysis_group: Optional[tb.Group], output_pdf, **kwargs):
    """
    analyse_capacitance_distribution_delegate

    Implementation of the investigation in the distribution of the capacitance on the chip.
    It should predominantly be used to determine the intrinsic and parasitic capacitance of a bare pix cap chip.
    To achieve the distribution the data is first binned and presented into a histogram.
    Next, a gaussian shape is fitted to the histogram to match its shape.

    Last the main results are written back to the analysis group as an attribute.

    :param analysis_group: hdf file's group where to find the capacitance to be analysed.
    :param output_pdf: PDF object to write the created figures to for long-term saving.
    :key use_kafe2: boolean, False, indicates whether kafe2 is used for the fit.
    :key apply_contours: boolean, indicates whether to determine the contours and try to plot them.
    :key fit_plot_pdf: PDF object to save the fit figures to.
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key mask_lower: float, threshold to mask all pixels below this value.
    :key mask_upper: float, threshold to mask all pixels above this value.
    :key test_cap_exclusion: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    :key capacitance: histogram of the capacitance to use instead of those extracted from the provided hdf files group.
    :key set_parasitic: boolean, whether to set the parasitic capacitance for this data set.
    :key no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
    :key convert: boolean, whether to convert the capacitance to fF, or not (default: True)
    """
    # none to expect, as these function runs fine without leaks when not called from an mp processing pool!
    from pixcap65.plotting import DEFAULT_BIN_NUMBER, COUNTS_HIST_LABEL, HIST_PIX_CAP_LABEL
    from matplotlib import pyplot as plt
    # extract further arguments for the performance of the fitting
    use_kafe2 = kwargs.pop("use_kafe2", False)
    if output_pdf is None:
        output_pdf = kwargs.pop("fit_plot_pdf", None)
    set_parasitic = kwargs.pop("set_parasitic", analysis_group is not None)
    assert not set_parasitic or analysis_group is not None

    # we want to fit a binned distribution; but some bins might be empty
    cap_hist = kwargs.pop("capacitance", None)
    if cap_hist is None or not isinstance(cap_hist, np.ndarray):
        assert analysis_group is not None
        cap_hist = check_leaf_unit(analysis_group.HistCap, HIST_CAP_UNIT)

    fig, ax = plt.subplots()
    hist_cap_hist = evaluate_pixel_mask(cap_hist, **kwargs)
    temp_hist_back_data = hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR
    hist_data, bins, _ = ax.hist(temp_hist_back_data,
                                 bins=kwargs.get("hist_bins", DEFAULT_BIN_NUMBER), density=False, )
    bin_positions = bins[:-1] + np.diff(bins)

    # now fit this to a gauss function
    initial_estimator = {'u': float(np.mean(temp_hist_back_data)), 's': float(np.std(temp_hist_back_data))}
    extended_fitter = None
    if use_kafe2:
        from kafe2 import HistContainer, HistFit
        data_container = HistContainer(bin_edges=bins, fill_data=temp_hist_back_data)
        fitter = HistFit(data=data_container, density=False, model_function=gauss_model)
        fitter.set_parameter_values(**initial_estimator)
        fitter.do_fit()
        assert fitter.did_fit
        mean_value = fitter.parameter_values[0]
        mean_error = fitter.parameter_errors[0]
        std_value = fitter.parameter_values[1]
        std_error = fitter.parameter_errors[1]
        norm = fitter.parameter_values[2]
        fit_results = fitter.parameter_values
        fit_cov = fitter.parameter_cov_mat
    else:
        from iminuit import Minuit
        # noinspection PyProtectedMember
        from iminuit.cost import Model, BinnedNLL, ExtendedBinnedNLL
        from pixcap65.analysis_util.stats import distribution_norm as norm
        assert isinstance(gauss_model, Model)
        assert isinstance(extended_gauss_integral, Model)
        initial_estimator_2 = initial_estimator.copy()
        initial_estimator_2["b"] = 1.0
        cost = BinnedNLL(hist_data, bins, cdf=norm.cdf, name=('u', 's'))
        cost_2 = ExtendedBinnedNLL(hist_data, bins, scaled_cdf=extended_gauss_integral, name=("b", 'u', 's'))
        extended_fitter = Minuit(cost_2, **initial_estimator_2)
        extended_fitter.migrad()
        extended_fitter.hesse()
        fitter = Minuit(cost, **initial_estimator)
        fitter.migrad()
        fitter.hesse()
        fit_results = fitter.values
        fit_cov = fitter.covariance
        norm = np.sum(hist_data)
        mean_value = fitter.values["u"]
        mean_error = fitter.errors["u"]
        std_value = fitter.values["s"]
        std_error = fitter.errors["s"]

    if output_pdf is not None:
        if extended_fitter is not None:
            handle_fitter_advanced_options(extended_fitter, False, "$C$ / \\unit{{\\femto\\farad}}", COUNTS_HIST_LABEL,
                                           "Capacitance distribution (EXTENDED)", output_pdf,
                                           "Contours for the capacitance distribution (EXTENDED)")
        handle_fitter_advanced_options(fitter, True, "$C$ / \\unit{{\\femto\\farad}}", COUNTS_HIST_LABEL,
                                       "Capacitance distribution", output_pdf,
                                       "Contours for the capacitance distribution")
    if set_parasitic:
        assert analysis_group is not None
        set_group_attribute(analysis_group, "parasitic", mean_value)
        set_group_attribute(analysis_group, "parasitic_error", std_value)
        set_group_attribute(analysis_group, UNITS_ATTRIBUTE_KEY, "fF")
    binning_span = np.linspace(bin_positions[0], bin_positions[-1], 1000)
    normalisation = norm * np.mean(np.diff(bin_positions))
    model_data = gauss_model(binning_span, mean_value, std_value, normalisation)
    ax.plot(binning_span, model_data, ":",
            label="C =  \\qty{{{c:.2f}\\pm{error:.2f}}}{{\\femto\\farad}}".format(c=mean_value, error=std_value))

    try:
        from jacobi import propagate
        # noinspection PyTypeChecker
        y, y_cov = propagate(lambda p: gauss_model(binning_span, p[0], p[1], normalisation), fit_results,
                             fit_cov)
        y_err_prop = np.diag(y_cov) ** 0.5
        ax.fill_between(binning_span, y - y_err_prop, y + y_err_prop, facecolor="C1", alpha=0.5)

    except ImportError:
        # noinspection PyUnusedLocal
        def propagate(*args):
            """
            This is just a template.
            :param args:
            """
            # This just a template
            pass
    except np.linalg.LinAlgError:
        pass
    except TypeError as e:
        if str(e).startswith("unsupported operand type(s) for *"):
            print("THE ERROR ESTIMATION FAILED!")
        else:
            raise

    # add some information about the model
    hypo_test = investigate_fit_convergence(fitter)
    ax.set_ylabel(COUNTS_HIST_LABEL)
    ax.set_xlabel(HIST_PIX_CAP_LABEL)
    ax.grid()
    if fit_cov is None:
        print(hist_data)
        print(bins)
        print(temp_hist_back_data)
        plt.show()
        raise ValueError("The fit failed by estimating the covariance matrix.")
    else:
        ax.legend(
            title=f"GoF = {hypo_test['x']:.4n}\nndf = {hypo_test['ndf']: .4n}\np = {hypo_test['p']: .4n}\nu = "
                  f"{mean_value:.6n}+-{mean_error:.6n}\ns = {std_value:.6n}+-{std_error:.6n}")
    if kwargs.get('no_plot', False):
        plt.close(fig)
    elif output_pdf is None:
        plt.show()
    else:
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    if not kwargs.pop("convert", True):
        mean_value /= CAPACITANCE_CONVERSION_FACTOR
        mean_error /= CAPACITANCE_CONVERSION_FACTOR
        std_value /= CAPACITANCE_CONVERSION_FACTOR
        std_error /= CAPACITANCE_CONVERSION_FACTOR

    return temp_hist_back_data.reshape(-1).shape[0], mean_value, mean_error, std_value, std_error


def apply_correction(raw_data, base_path=None, bare_data_path=None, bare_group=None, **tb_kwargs):
    """
    apply_correction

    Wrapper function to handle the file access, when correcting the capacitance data tables.

    :param raw_data: hdf file containing the measurements and investigation of a pix cap sample which needs correction
    :param base_path: hdf files group with the analysis data/capacitance data to be corrected.
    :param bare_data_path: file containing the analysis of the bare pix cap sample to be used for
        capacitance correction.
    :param bare_group: hdf files group for the bare analysis (holding the parasitic capacitance information)
    """
    # first extract the parasitic capacitance
    assert bare_data_path is not None
    with synchronized_process_open_file(raw_data, mode='a', **tb_kwargs) as in_file_h5_inner:
        base_group = get_base_group(base_path, in_file_h5_inner)
        apply_correction_simple(bare_data_path, bare_group, base_group.analysis)


def apply_correction_simple(bare_data_path: str, bare_path: str, analysis_group: tb.Group, **kwargs):
    """
    apply_correction_simple

    Minimal wrapper for the correction of the measured capacitance for the capacitance of the bump-bond and the
    switching circuit itself.
    It will open the data file and analysis group of the bare pix cap measurement, extract the parasitic capacitance
    (including their uncertainties which will be applied as systematic uncertainties to the measurement), correct for
    these effect and write all the results of the current analysis with the correction applied back.
    As the bump-bond capacitance and the intrinsic capacitance of the switching circuit can be considered as being
    parallel no further effects needs to be accounted for. Also, these capacitance could be considered being parallel
    to the pixel sensors capacitance. So the correction could be done by a simple subtraction.


    :param bare_data_path: path to the hdf file of the bare pix cap measurement to obtain information about parasitic
        capacitance.
    :param bare_path: hdf files hierarchy path to the group containing the bare pix cap analysis with the information
        about the parasitic after investigating the capacitance distribution.
    :param analysis_group: hdf files hierarchy group with the analysis results which needs to be corrected for the
        intrinsic and parasitic effects.
    :key lock:
    """
    with synchronized_process_open_file(bare_data_path, mode='r', **kwargs) as in_file_h5:
        parasitic, parasitic_error = _handle_parasitic_cap(bare_path, in_file_h5)
        # second perform correction of the capacitance data
        apply_correction_delegate(parasitic, parasitic_error, analysis_group)


def apply_correction_delegate(parasitic, parasitic_error, group: GroupType):
    """
    apply_correction_delegate

    Implementation of the capacitance correction for the parasitic capacitance of the bump-bond and the intrinsic
    capacitance of the switching circuit.
    As the bump-bond capacitance and the intrinsic capacitance of the switching circuit can be considered as being
    parallel no further effects needs to be accounted for. Also, these capacitance could be considered being parallel
    to the pixel sensors capacitance. So the correction could be done by a simple subtraction.

    To make it possible to still plot all the data later on, the new tables get also the parasitic capacitance
    subtracted as an attribute.

    :param parasitic: parasitic capacitance to correct for.
    :param parasitic_error: (systematic) error of the parasitic capacitance to correct for.
    :param group: hdf files group containing the capacitance data to be corrected.
    """
    assert isinstance(group, tb.Group)
    # define the corrected analysis group
    prevent_group_mix_up(get_parent_group(group), "analysis_correction")

    file_h5 = group_get_file(group)
    correction_group = file_h5.create_group(where=get_parent_group(group), name="analysis_correction")

    # first extract the capacitance histograms (errors shall be copied to the new group
    _correct_data(correction_group, file_h5, group, parasitic, parasitic_error)

    # will need to handle also the subgroups (only one level down)!
    for group_key, next_group in get_groups(group):
        next_correction_group = file_h5.create_group(where=correction_group, name=group_key)
        _correct_data(next_correction_group, file_h5, next_group, parasitic, parasitic_error)


def _correct_data(correction_group: tb.Group, file_h5: tb.File, group: tb.Group, parasitic, parasitic_error):
    for key, value in get_leaves(group):
        assert isinstance(value, tb.Leaf)
        if "Cap" in key:
            logger.debug("Found the key %s for transferring capacitance data.", key)
            if "Err" in key:
                copy_node(value, newparent=correction_group)
            else:
                assert isinstance(value, tb.Array) or isinstance(value, tb.Table)
                cap_data_hist = value[:]
                assert isinstance(cap_data_hist, np.ndarray)
                corrected_cap_data = np.where(np.isfinite(cap_data_hist), cap_data_hist - parasitic * 1.e-15, np.nan)
                corrected_cap_systematics = np.full_like(corrected_cap_data, fill_value=parasitic_error)
                temp_data_array = file_h5.create_carray(where=correction_group, name=key, title=value.title,
                                                        filters=value.filters, obj=corrected_cap_data)
                # careful the parasitic capacitance are provided in fF were all other capacitance are saved in F
                temp_data_array.attrs[PARASITIC_SUBTRACTION] = parasitic
                temp_error_array = file_h5.create_carray(where=correction_group, name="{}Systematics".format(key),
                                                         title=value.title,
                                                         filters=value.filters, obj=corrected_cap_systematics)
                assert isinstance(value, tb.Leaf)
                for name in list_attributes(value):
                    temp_data_array.attrs[name] = value.attrs[name]
                    temp_error_array.attrs[name] = value.attrs[name]
        else:
            copy_node(value, newparent=correction_group)


def get_test_capacitance_data(group: tb.Group, **kwargs):
    locale.setlocale(locale.LC_NUMERIC, "de_DE")
    test_cap = group.HistCap[:][:, 0]
    test_cap_error = group.HistCapErr[:][:, 0]
    if kwargs.get("print_result", True):
        for k, (cap, err) in enumerate(zip(test_cap, test_cap_error)):
            eff_cap = cap * 1e15
            eff_err = err * 1e15
            print(k, f"{eff_cap:.3n}+-{eff_err:.3n}")

    test_cap[16] = np.nan
    test_cap_error[16] = np.nan
    return test_cap[np.isfinite(test_cap)], test_cap_error[np.isfinite(test_cap_error)]


def bare_analysis_handler(tb_lock):
    with synchronized_process_open_file("packaged/Reference_Bare_renewed.h5", 'a', lock=tb_lock) as h5_file:
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_31_renew", newname="unbiased_31_renew_full_model",
                          overwrite=True, recursive=True)

    with synchronized_process_open_file("packaged/data/Bare_Sample_05_Extended_Scan.h5", 'a', lock=tb_lock) as h5_file:
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_model",
                          overwrite=True, recursive=True)
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_kafe2",
                          overwrite=True, recursive=True)
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_extended",
                          overwrite=True, recursive=True)
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_quad",
                          overwrite=True, recursive=True)

    analyze_data(raw_data="packaged/Reference_Bare_renewed.h5", base_path="Reference/Bare/unbiased_31_renew",
                 is_advanced=True, full_model=False, lock=tb_lock)
    analyze_data(raw_data="packaged/Reference_Bare_renewed.h5",
                 base_path="Reference/Bare/unbiased_31_renew_full_model", is_advanced=True,
                 full_model=True, lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5", base_path="Reference/Bare/unbiased_full",
                 is_advanced=True, full_model=False, lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_model", is_advanced=True,
                 full_model=True, lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_kafe2", is_advanced=True,
                 full_model=True, lock=tb_lock, use_kafe2=True)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_extended", is_advanced=True,
                 full_model='extended', lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_quad", is_advanced=True,
                 full_model='quad', lock=tb_lock)
    analyze_capacitance_distribution(raw_data="packaged/Reference_Bare_renewed.h5",
                                     base_path="Reference/Bare/unbiased_31_renew",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_renew_parasitic.pdf", lock=tb_lock)
    analyze_capacitance_distribution(raw_data="packaged/Reference_Bare_renewed.h5",
                                     base_path="Reference/Bare/unbiased_31_renew_full_model",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_renew_parasitic_full_model.pdf", lock=tb_lock)
    analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5',
                                     base_path="Reference/bare/unbiased_8_full_model",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_parasitic_full_model.pdf", lock=tb_lock)
    analyze_capacitance_distribution(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                                     base_path="Reference/Bare/unbiased_full",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_parasitic_extended_renew_full_model.pdf",
                                     lock=tb_lock)


if __name__ == '__main__':
    # analyse_data(raw_data='/home/silab/git/pixcap65/pixcap_full_data_image1.h5')
    # some usage examples
    from pixcap65.utility.homogenize_plots import set_params
    # hold this for now as it simplifies synchronization between the two devices!
    # analyze_data(raw_data="data/3D_Sensor_221_W13_X_Scan.h5", base_path="Thesis/ATLAS_ITk/X3/C_V_Characteristic",
    #              is_cv=True)
    # analyze_data(raw_data="data/3D_Sensor_221_W6_j_Scan.h5", base_path="Thesis/ATLAS_ITk/X5/C_V_Characteristic",
    #              is_cv=True)
    # analyze_data(raw_data="data/3D_Sensor_I14_S24_Scan.h5", base_path="Thesis/ATLAS_ITk/X6/C_V_Characteristic",
    #              is_cv=True)
    # analyze_data(raw_data="data/3D_Sensor_H23_S24_Scan.h5", base_path="Thesis/ATLAS_ITk/X7/C_V_Characteristic",
    #              is_cv=True)
    # analyze_data(raw_data="data/argparser.h5", base_path="Reference/R11/C_V_Characteristic", is_cv=True)
    # analyze_data(raw_data="data/3D_Sensor_221_W5_S_Scan.h5", base_path="Thesis/ATLAS_ITk/X4/C_V_Characteristic",
    #              is_cv=True)
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
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors="
                           r"{stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}"
                           r"\sisetup{retain-zero-uncertainty}")

    bare_correction_args = {
        "apply_correction": True,
        "bare_file": "Bare_Repeat_2_Scan.h5",
        "bare_hdf_path": "Reference/bare/unbiased_8/total_cap",
    }

    bare_correction_args = {
        "apply_correction": True,
        "bare_file": "packaged/Reference_Bare_renewed.h5",
        "bare_hdf_path": "Reference/Bare/unbiased_31_renew/total_cap",
    }
