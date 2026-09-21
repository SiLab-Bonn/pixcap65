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
import tables as tb
import time
from tqdm import tqdm

try:
    # noinspection PyCompatibility
    from collections.abc import Iterable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Iterable
finally:
    from typing import Optional, List, Union

from .constants import SI_MOBILITY, DISPERSION_PARASITIC_DEVIATION, BIAS_VOLTAGE_ACCESS_IDX, \
    SLOPE_RESISTIVITY_CONVERSION
from .delegation.depletion import __distribution_depletion_estimation
from .multi_processing import _handle_mp_parasitic_cap
from .delegation.distribution import _get_sensor_distribution
from .delegation import _get_analyze, analyze_depletion_delegate
from .correction import apply_correction_simple, _handle_cap_correction
from .delegation.inter_pix import perform_inter_pix_deep_dive, _handle_inter_pix_capacitance
from .utility import GLOBAL_FILTERS
from .constants import BOUNDARY_TYPE
from .modelling.physics_modelling import EPS_SILICON
from .utility import get_base_group, handle_analysis_mix_up, CVDistributionData, str_join, \
    ANALYSIS_GROUP_NAME, HIST_CAP_UNIT, ANALYSIS_CORRECTED_GROUP_NAME, get_analysis_group, DepletionData, \
    adjust_dist_table, check_leaf_unit, HIST_CURRENT_MEAS_UNIT
from pixcap65.pixcap.pixcap_structure import CAPACITANCE_CONVERSION_FACTOR
from pixcap65.utility import synchronized_process_open_file
from pixcap65.utility.utils_2 import walk_to_node, create_carray, GroupType


def _analyze_data(raw_data, base_path=None, is_advanced=False, is_cv=False,
                     first_boundaries: Optional[BOUNDARY_TYPE] = None,
                     second_boundaries: Optional[BOUNDARY_TYPE] = None,
                     is_inter_pixel=False, **kwargs) -> Optional[List]:
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

    # global cap_counter
    # print("The last correction counter is", cap_counter)


def _cv_analysis(in_file_h5: tb.File, bare_file_arg, bare_path_arg, base_group: tb.Group,
                 apply_correction_arg, first_boundaries: Optional[Union[tuple, Iterable[tuple]]],
                 second_boundaries: Optional[Union[tuple, Iterable[tuple]]], is_advanced: bool, is_inter_pixel: bool,
                 get_distribution, parasitic: float, parasitic_error: float,
                 **kwargs) -> Optional[tb.Group]:
    """
    _cv_analysis

    @author: Dominik Fischer
    @date: 2026-08-12

    Utility function performing analysis steps for determination of the C-V characteristics of a Silicon Pixelsensor
    using the data measured by PixCap65.
    In a first step for every (bias) voltage and every (measured) pixel on the sensor the capacitance is determined.
    Then, the capacitances are collected and inserted into a higher-dimensional matrix for the dependency on voltage and pixel (row, col).

    If requested, also these data is computed for an 'averaged' sensor.
    But in anyways all masked pixels are determined as having NaN as capacitance.

    The depletion voltage and further analysis of the doping profile of the sensor under test is only performed if
    boundaries for the two asymptotic regions of the c-v-curve are provided.
    The depletion voltage is estimated for every pixel by fitting two straight lines, to the low and to the high voltage asymptotic region.
    Then, the depletion voltage is determined by the intersection of these two lines.
    To investigate the doping profile, this must be requested explicitly by the corresponding keyword and the 'chi_group',
    which contains information about the sensors physical properties, e.g. the pixels dimensions, must be provided.


    :param in_file_h5: hdf file containing the measurement data
    :param bare_file_arg: hdf file containing the measurements and investigation of a bare pix cap sample to obtain
        information about intrinsic and parasitic capacitance. (Only required for the correction procedure, but in
        this case it must be present)
    :param bare_path_arg: hdf files hierarchy path to the group containing the bare pix cap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present)
    :param base_group: hdf file's hierarchy group containing the measurement data of c-v-curve! Must have a
        subgroup biasing.
    :param apply_correction_arg: boolean, indicates whether the measured capacitance should be
        corrected immediately; Will require the presence of further arguments as information about
        the parasitic capacitance needs to be submitted. (data corrected for parasitic capacitances of PixCap65,
        default: False)
    :param first_boundaries: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will only be estimated if this argument is provided.
    :param second_boundaries: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will be estimated if this argument is provided.
    :param is_advanced: boolean indicating if the advanced analysis strategy should be used
        or not (may require additional keyword arguments)
    :param is_inter_pixel: indicating whether these are inter-pixel-capacitance measurements
    :type is_inter_pixel: bool
    :param get_distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor. (default: False)
    :param parasitic: parasitic capcitance of the measurement circuit to be assumed
    :param parasitic_error: uncertainty/deviation of the parasitic capacitance of the circuit to be assumed.

    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :keyword full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used. (Only used for the advanced procedure)
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :keyword plot: boolean, indicates whether to plot the data. An output PDF object could be submitted here instead of an explicitly created one. (Default: False) (Only used for the advanced procedure)
    :keyword apply_contour: boolean, indicates whether to determine the contours and try to plot them. (default: False)
        (Only used for the advanced procedure, perhaps also 'apply_contours')
    :keyword fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure)
    :keyword apply_correction: boolean, False, indicates whether the measured capacitance should be
        corrected immediately; Will require the presence of further arguments as information about
        the parasitic capacitance needs to be submitted.
    :keyword bare_file: hdf file containing the measurements and investigation of a bare pix cap sample to obtain
        information about intrinsic and parasitic capacitance. (Only required for the correction procedure, but in
        this case it must be present)
    :keyword bare_path: hdf files hierarchy path to the group containing the bare pix cap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present)
    :keyword output_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided)
    :keyword hist_res_key: name/identifiert of the array/table which contains the on-resistance estimators if present.
    :keyword test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :keyword mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation. [array-like]
    :keyword mask_lower: threshold to mask all pixels below this value.
    :type mask_lower: float
    :keyword mask_upper: threshold to mask all pixels above this value.
    :type mask_upper: float
    :keyword hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :keyword chip_group_name: path hdf files' group containing the (physical) properties of the sensor pixels. This must include a matrix of the Physical dimensions of the individual pixels named ''.
    :keyword chip_group:
    :keyword fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :type fit_description_text: str
    :return: hdf files' hierarchy group, used as the reference (containing now the measurements as well as the analysis results)
    """
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

            # this would require detailed information about the sensor geometry!
            # we would need to know which pixels contribute to give an estimate
            # could we fetch the correct geometry from the corresponding data set?
            from scipy.constants import epsilon_0

            try:
                raw_pixel_areas = np.prod(chip_spec_group.PhysicalDimensions[:], axis=2)
                sensor_pixel_mask = np.isfinite(cv_data[:, :, 0])
                distribution_pixel_area = np.mean(raw_pixel_areas[sensor_pixel_mask])
            except (NameError, tb.NoSuchNodeError, np.exceptions.AxisError):
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


def analysis_data_handle(file: tb.File, data_group: GroupType, result_group: GroupType,
                         is_advanced=False, is_inter_pixel=False, **kwargs):
    """
    analysis_data_handle

    @author: Dominik Fischer
    last update: 2026-08-12

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
    :keyword full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used. (Only used for the advanced procedure)
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :keyword plot: boolean, indicates whether to plot the data. An output PDF object could be submitted here instead of an explicitly created one. (Default: False) (Only used for the advanced procedure)
    :keyword apply_contour: boolean, indicates whether to determine the contours and try to plot them. (default: False)
        (Only used for the advanced procedure)
    :keyword fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure)
    :keyword apply_correction: boolean, False, indicates whether the measured capacitance should be
        corrected immediately; Will require the presence of further arguments as information about
        the parasitic capacitance needs to be submitted.
    :keyword bare_file: hdf file containing the measurements and investigation of a bare pix cap sample to obtain
        information about intrinsic and parasitic capacitance. (Only required for the correction procedure, but in
        this case it must be present)
    :keyword bare_path: hdf files hierarchy path to the group containing the bare pix cap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present)
    :keyword output_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided)
    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
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
    # but then the question is whether it could leak the lock outside!
    _handle_cap_correction(result_group, **kwargs)
