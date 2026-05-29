"""
Analysis of Pixcap65 data. Fits freq vs current to extract the capacitance. A 2D histogram containing the capacitance
for each pixel is stored.
"""
import logging
import time
import warnings
from matplotlib.backends.backend_pdf import PdfPages
from tables import File, Group

from pixcap65.analysis_util.data_store import DepletionDataStore, DepletionTableStore, DepletionArrayStore, \
    DopingArrayStore
from pixcap65.utility import synchronized_process_open_file

# TODO: update the documentation of these implementations

try:
    # noinspection PyCompatibility
    from collections.abc import Sized, Iterable
except ImportError:
    # python 2.7
    import collections.Sized as Sized
    import collections.Iterable as Iterable

import numpy as np
import tables as tb
from typing import Optional, Tuple, Union, Callable, Any, Iterable
from warnings import deprecated

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
from pixcap65.plotting import CAPACITANCE_CONVERSION_FACTOR, evaluate_pixel_mask, X1_SCAN_2_FILE, X2_SCAN_2_FILE
from pixcap65.utility.tables_util import get_groups, get_leaves, copy_node, list_attributes, group_get_file, \
    set_group_attribute, get_group_attribute, get_group_attributes, get_parent_group
from pixcap65.utility.utils_2 import walk_to_node, GroupType, create_carray, prevent_group_mix_up

UNITS_ATTRIBUTE_KEY = "Units"
BOUNDARY_TYPE = Union[Tuple, Iterable[Tuple]]
ADVANCED_PARAMETER_TYPE = Union[bool, Iterable[bool]]

logger = logging.getLogger(__name__)


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
    """
    # handle deprecated keyword arguments.
    correction_key_value = kwargs.pop("use_corrected", None)
    if correction_key_value is not None:
        msg = "keyword argument `use_corrected` is deprecated, use `apply_correction` instead. If `apply_correction` is also present this value will take precedence, otherwise the provided value will be used. This keyword argument will be removed in the future."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        kwargs.setdefault("apply_correction", correction_key_value)

    # handle the additional PDF file in case of plotting enabled.
    fit_plot_pdf_name = kwargs.pop('fit_plot_pdf_name', None)
    if "plot" in kwargs and kwargs["plot"] and fit_plot_pdf_name is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        assert "fit_plot_pdf_name" not in kwargs
        with PdfPages(fit_plot_pdf_name) as pdf:
            kwargs['fit_plot_pdf'] = pdf
            analyze_data(raw_data, base_path, is_advanced, is_cv, first_boundaries,
                         second_boundaries, is_inter_pixel, **kwargs)
            return
    kargs = kwargs.copy()
    kargs.pop("plot", None)
    kargs.pop("fit_plot_pdf", None)
    kargs.pop("use_kafe2", None)
    kargs.pop("output_pdf", None)
    kargs['no_plot'] = True

    get_distribution = kwargs.get("distribution", False)
    get_total_cap_file = kwargs.get("total_cap_file", None)
    get_total_cap_group = kwargs.get("total_cap_group", None)

    propagate_key_args = kwargs.copy()
    apply_correction_arg = kwargs.pop("apply_correction", False)
    bare_file_arg = kwargs.pop("bare_file", None)
    bare_path_arg = kwargs.pop("bare_hdf_path", None)
    # handle the real analysis.
    with tb.open_file(raw_data, mode='a') as in_file_h5:
        # extract the hdf file groups to perform the analysis on.
        base_group = get_base_group(base_path, in_file_h5)

        # get capacitance to correct for.
        if apply_correction_arg:
            # extract the parasitic capacitance right here!
            with tb.open_file(bare_file_arg, mode='r') as correction_h5:
                assert isinstance(bare_path_arg, (str, None))
                parasitic, parasitic_error = _handle_parasitic_cap(bare_path_arg, correction_h5)
                if parasitic < 1.0:
                    print("Unfortunatley the parasitic capacitance vanishs.")
        else:
            parasitic = 0
            parasitic_error = 0

        # special handling for C-V characterization.
        if is_cv:
            # need to perform the analysis for every bias voltage
            cv_data = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]),
                              fill_value=np.nan)
            cv_err_data = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]),
                                  fill_value=np.nan)
            cv_data_corrected = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]),
                                        fill_value=np.nan)
            cv_err_data_corrected = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]),
                                            fill_value=np.nan)

            # make sure to not mix-up with previous analysis results
            # Why is this C-V specific? Due to the operation on a reference group otherwise.
            handle_analysis_mix_up(base_group.biasing)

            # Why check this only for cv data and total cap?
            # popping to make sure it is not propagated to the analysis handler to perform the parasitic correction
            # not for each bias voltage individually but for altogether.

            if get_distribution:
                in_file_h5.create_group(base_group.biasing, "analysis")
                dist_table = in_file_h5.create_table(where=base_group.biasing.analysis, name="CVDistribution",
                                                     description=CVDistributionData, filters=GLOBAL_FILTERS)
                dist_entry = dist_table.row
            else:
                dist_entry = {}

            # no progressbar as the overhead for this is much too large in most cases.
            bias_voltage_data = base_group.biasing.measurements.BiasVoltageHist[:]
            if len(bias_voltage_data.shape) > 1:
                # this wont use available measured voltage data but the settings instead. This may lower accuracy.
                bias_voltage_data = bias_voltage_data[:, 0]
            for k, bias_voltage in enumerate(bias_voltage_data):
                bias_name = "bias_{volt}_V".format(volt=bias_voltage).replace('-', "M_").replace(".", "__")
                data_group = base_group.biasing.measurements[bias_name]
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
                    # Will continue using tuples for performance!
                    # first get a histogram with all the data
                    temp_tuple = analyze_capacitance_distribution_delegate(None,
                                                                           None,
                                                                           capacitance=cap_data,
                                                                           **kargs)
                    n_pix, cap, cap_err, cap_std, cap_std_err = temp_tuple

                    # we need an independent fit for the corrected distributions
                    # interestingly only the parasitic negatives are present here.
                    cap_data_para = np.where(np.isfinite(cap_data), cap_data - (parasitic / CAPACITANCE_CONVERSION_FACTOR), np.nan)
                    cap_data_para_upper = np.where(np.isfinite(cap_data), cap_data - (parasitic + parasitic_error) / CAPACITANCE_CONVERSION_FACTOR, np.nan)
                    cap_data_para_lower = np.where(np.isfinite(cap_data), cap_data - (parasitic - parasitic_error) / CAPACITANCE_CONVERSION_FACTOR, np.nan)
                    try:
                        temp_tuple = analyze_capacitance_distribution_delegate(None,None,
                                                                               capacitance=cap_data_para,
                                                                               **kargs)
                        _, cap_corr, cap_err_corr, cap_std_corr, cap_std_err_corr = temp_tuple

                        # TODO: Evaluate the systematic uncertainties for the different capacitances
                        temp_tuple = analyze_capacitance_distribution_delegate(None, None,
                                                                               capacitance=cap_data_para_upper,
                                                                               **kargs)
                        _, cap_corr_1, cap_err_corr_1, cap_std_corr_1, cap_std_err_corr_1 = temp_tuple

                        temp_tuple = analyze_capacitance_distribution_delegate(None, None,
                                                                               capacitance=cap_data_para_lower,
                                                                               **kargs)
                        _, cap_corr_2, cap_err_corr_2, cap_std_corr_2, cap_std_err_corr_2 = temp_tuple
                    except:
                        print(np.count_nonzero(np.isfinite(cap_data_para)))
                        print(cap_data[np.isfinite(cap_data)])
                        print(cap_data_para[np.isfinite(cap_data)])
                        raise

                    # Fehler von Fehlern werden i.d.R. unterdrückt.
                    cap_corr_sys_abs = np.abs(np.array((cap_corr_1, cap_corr_2)) - cap_corr)
                    cap_corr_sys_errors = np.array([np.sqrt(cap_err_corr_1 ** 2 + cap_err_corr ** 2),
                                                    np.sqrt(cap_err_corr_2 ** 2 + cap_err_corr ** 2)])
                    cap_corr_systematic = np.average(cap_corr_sys_abs, weights=np.reciprocal(cap_corr_sys_errors ** 2))


                    # try to get to the on-resistance datasets to perform the same distribution handler!
                    # only issue with this attempt the capacitance will be saved as a parasitic one!
                    # FIXME: For interpix runs this will break as there are different names for the different channels.
                    if "HistRes" in ana_group and np.any(np.isfinite(ana_group.HistRes)):
                        resistance_data = ana_group.HistRes[:]
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

                    dist_entry['bias'] = bias_voltage
                    dist_entry['n_pixel'] = n_pix
                    dist_entry['capacitance'] = cap / CAPACITANCE_CONVERSION_FACTOR
                    dist_entry['cap_err'] = cap_err / CAPACITANCE_CONVERSION_FACTOR
                    dist_entry['cap_std'] = cap_std / CAPACITANCE_CONVERSION_FACTOR
                    dist_entry['cap_std_err'] = cap_std_err / CAPACITANCE_CONVERSION_FACTOR
                    dist_entry['r_on'] = res
                    dist_entry['r_on_err'] = res_err
                    dist_entry['r_on_std'] = res_std
                    dist_entry['r_on_std_err'] = res_std_err
                    if apply_correction_arg:
                        # TODO: Rework this part to save all the results for all the value
                        dist_entry['cap_corrected'] = cap_corr / CAPACITANCE_CONVERSION_FACTOR
                        dist_entry['cap_corrected_err'] = cap_std_corr / CAPACITANCE_CONVERSION_FACTOR
                        dist_entry['cap_parasitic'] = parasitic
                        dist_entry['cap_systematic_error'] = np.sqrt(
                            cap_err ** 2 + parasitic_error ** 2) / CAPACITANCE_CONVERSION_FACTOR
                    else:
                        dist_entry['cap_corrected'] = cap / CAPACITANCE_CONVERSION_FACTOR
                        dist_entry['cap_corrected_err'] = cap_std / CAPACITANCE_CONVERSION_FACTOR
                        dist_entry['cap_parasitic'] = 0
                        dist_entry['cap_systematic_error'] = cap_corr_systematic / CAPACITANCE_CONVERSION_FACTOR

                    if isinstance(dist_entry, tb.tableextension.Row):
                        dist_entry.append()
                    else:
                        print("No append of the distribution!")

            if get_distribution:
                dist_table.flush()

            in_file_h5.flush()

            create_carray(in_file_h5, base_group.biasing.analysis, name="UCHist", title="Histogram of the U-C-curve",
                          filters=GLOBAL_FILTERS, obj=cv_data, unit=HIST_CAP_UNIT)
            create_carray(in_file_h5, base_group.biasing.analysis, name="UCErrHist",
                          title="Error Histogram of the U-C-curve",
                          filters=GLOBAL_FILTERS,
                          obj=cv_err_data,
                          unit=HIST_CAP_UNIT)

            # in case of active correction, also the summary capacitance tables needs to be corrected, but it should
            # also an uncorrected table present.
            if apply_correction_arg:
                # perform the transfer
                apply_correction_simple(bare_file_arg, bare_path_arg, base_group.biasing.analysis, )

                for k, bias_voltage in enumerate(bias_voltage_data):
                    ana_group_correction, _ = walk_to_node(base_group.biasing,
                                                           str_join("/", ANALYSIS_CORRECTED_GROUP_NAME, bias_name),
                                                           create=True, verify_create=True)
                    cap_data = ana_group_correction.HistCap[:]
                    cap_error_data = ana_group_correction.HistCapErr[:]
                    cv_data_corrected[:, :, k] = cap_data[:, :]
                    cv_err_data_corrected[:, :, k] = cap_error_data[:, :]

                # save also the corrected capacitance data
                create_carray(in_file_h5, base_group.biasing.analysis_correction, name="UCHist",
                              title="Histogram of the U-C-curve",
                              filters=GLOBAL_FILTERS,
                              obj=cv_data, unit=HIST_CAP_UNIT)
                create_carray(in_file_h5, base_group.biasing.analysis_correction, name="UCErrHist",
                              title="Error Histogram of the U-C-curve",
                              filters=GLOBAL_FILTERS,
                              obj=cv_err_data, unit=HIST_CAP_UNIT)

            if first_boundaries is not None and second_boundaries is not None:
                # We will need all the usages as here might be a inconsitency with the data systems.
                dep_ana_group = get_analysis_group(base_group.biasing, use_corrected=apply_correction_arg)
                assert isinstance(dep_ana_group, tb.Group)
                chip_name = kwargs.pop("chip_group_name", None)
                # we need to put the plotting arguments back in
                if chip_name is not None:
                    chip_spec_group, _ = walk_to_node(in_file_h5.root, chip_name, create=False, verify_create=True)
                    kwargs['chip_group'] = chip_spec_group

                # Anyway it is necessary to perform this analysis also on a sensor-average basis!
                if apply_correction_arg:
                    analyze_depletion_delegate(base_group.biasing.measurements,
                                               base_group.biasing.analysis,
                                               first_boundaries, second_boundaries, **kwargs)
                analyze_depletion_delegate(base_group.biasing.measurements, dep_ana_group,
                                           first_boundaries, second_boundaries, **kwargs)

                if get_distribution:
                    print("The analysis group is: ", dep_ana_group)
                    # the appropriate options are in this case: create one combination table for each of the
                    # fit boundaries or save numpy arrays within the table?
                    dist_table.flush()
                    dist_table = base_group.biasing.analysis.CVDistribution[:]
                    depletion_data_table_raw = in_file_h5.create_table(where=base_group.biasing.analysis,
                                                                       name="SensorDepletionRaw1",
                                                                       description=DepletionData,
                                                                       filters=GLOBAL_FILTERS)
                    corrected_depletion_data_table = in_file_h5.create_table(where=base_group.biasing.analysis,
                                                                             name="SensorDepletionRaw2",
                                                                             description=DepletionData,
                                                                             filters=GLOBAL_FILTERS)

                    # will need to perform the operation with two different tables which only differ by their entries
                    # corrected_depletion_data_table = depletion_data_table_raw.copy(base_group.biasing.analysis,
                    #                                                                "SensorDepletionRaw2")
                    __distribution_depletion_estimation(dist_table, first_boundaries,
                                                        second_boundaries, depletion_data_table_raw, False, **kwargs)
                    depletion_data_table_raw.flush()
                    depletion_data_table = depletion_data_table_raw.copy(base_group.biasing.analysis,
                                                                         "SensorDepletionRaw")
                    depletion_data_table.flush()

                    if apply_correction_arg:
                        print("Will try to apply the distribution data.")
                        __distribution_depletion_estimation(dist_table, first_boundaries, second_boundaries,
                                                            corrected_depletion_data_table, True, **kwargs)
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
                        in_file_h5.copy_node(where=base_group.biasing.analysis, newparent=dep_ana_group, name="SensorDepletionRaw", newname="SensorDepletionRaw")

                    # remove the additonal tables
                    depletion_data_table_raw.remove()
                    corrected_depletion_data_table.remove()

        else:
            if is_inter_pixel:
                reference_group = base_group.inter_cap
            else:
                reference_group = base_group.total_cap

            handle_analysis_mix_up(reference_group)
            ana_group, _ = walk_to_node(reference_group, "analysis", create=True, verify_create=True)
            assert isinstance(ana_group, tb.Group)

            analysis_data_handle(in_file_h5, reference_group.measurements, ana_group,
                                 is_advanced=is_advanced,
                                 is_inter_pixel=is_inter_pixel, **propagate_key_args)

        # TODO: implement further processing to estimate the inter-pixel and in-pix capacitance. Also for distribution.
        if is_inter_pixel and get_total_cap_file is not None:
            # without the additional data from the total capactiance measurement
            assert get_total_cap_group is not None
            with synchronized_process_open_file(get_total_cap_file, mode='a') as total_h5_file:
                total_cap_data, _ = walk_to_node(total_h5_file.root, get_total_cap_group, create=False, verify_create=True)
                total_cap = total_cap_data.analysis.HistCap[:]
                total_cap_error = total_cap_data.analysis.HistCapErr[:]

                inter_pix_cap = total_cap - reference_group.analysis.TotalHistCap[:]
                inter_pix_cap_err = total_cap_error - reference_group.analysis.TotalHistCap_err[:]

                # account for systematic effects?
                in_pix_cap = reference_group.analysis.TotalHistCap[:]
                in_pix_cap_err = reference_group.analysis.TotalHistCap_err[:]
                create_carray(in_file_h5, where=reference_group.analysis, name="InPixHistCap", obj=in_pix_cap)
                create_carray(in_file_h5, where=reference_group.analysis, name="InPixHistCapErr", obj=in_pix_cap_err)
                create_carray(in_file_h5, where=reference_group.analysis, name="InterPixHistCap", obj=inter_pix_cap)
                create_carray(in_file_h5, where=reference_group.analysis, name="InterPixHistCapErr", obj=inter_pix_cap_err)
                if apply_correction_arg:
                    # must retrieve the parasitics first.
                    in_pix_cap -= parasitic
                    in_pix_cap_err = np.where(np.isfinite(in_pix_cap), np.sqrt(in_pix_cap_err ** 2 + parasitic_error ** 2), np.nan)
                    create_carray(in_file_h5, where=reference_group.analysis_correction, name="InPixHistCap", obj=in_pix_cap)
                    create_carray(in_file_h5, where=reference_group.analysis_correction, name="InPixHistCapErr",
                                  obj=in_pix_cap_err)
                    in_file_h5.copy_node(where=reference_group.analysis, newparent=reference_group.analysis_correction,
                                         newname="InterPixHistCap", name="InterPixHistCap")
                    in_file_h5.copy_node(where=reference_group.analysis, newparent=reference_group.analysis_correction,
                                         newname="InterPixHistCapErr", name="InterPixHistCapErr")
                    in_file_h5.flush()

                # TODO: handle the sensor distribution of this.
                # perform individual distribution fits for the corrected data, also accounting for the uncertaintiy
                # of the parasitic capacitance.





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


# perhaps extract the result table as an parameter in order to put the correct depletion values also in the table!
def __distribution_depletion_estimation(dist_table,
                                        first_boundaries: Union[tuple, Iterable[tuple]],
                                        second_boundaries: Union[tuple, Iterable[tuple]], depletion_data_table,
                                        is_corrected, **kwargs):
    """
    distribution_depletion_estimation

    Internal handler function to estimate the depletion voltage / reversed bias for full depletion of the sensor.
    This handler function uses the capacitance distribution over the full sensor (or at least the measured part)
    assuming a gaussian pdf.
    It takes the gaussian centered capacitances (more accuratley: the distributions c-v-curve), fits two straight lines
    to low voltage limit for the capacitance behaviour of the undepleted sensor and the high voltage limit for the
    capacitance behaviour for a depletion zone extending over the full physical dimensions of the sensor.
    The depletion voltage is estimated as the intersection of these two straight lines.
    Both the c-v-data is taken from a table and the results are written back to another table.

    These estimation attempt is redone with slightly adjust fit ranges to estimate the systematic uncertainty.
    This estimator for the systematic uncertainty ignores cross-correlation between different factors influencing
    the fit. In particular this approach only accounts for systematic effects by the fit procedure itself.
    In particular no systematic effects from the measurement setup itself or the parasitic capacitances are evaluated
    at this point.

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
    :key plot: boolean, False, indicates whether to plot the data. AN output PDF object could be submitted here
         instead of an explicitly created one. (No effect for the fit to estimate systematic effects)
    :key fit_plot_pdf: PDF object to save the fit figures to. (No effect for
        the fit to estimate systematic effects.)
    :key verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    :key cv_fit_plot_pdf: analog to `fit_plot_pdf` to activate the plotting for c-v- and depletion fits independent from
        the plotting for capacitance estimation fits. If this keyword argument is present also the `plot` arguments will
        be set automatically. (This keyword argument will be ignored for the fits to estimate systematic effects)
    """
    pixel_cap_data = _extract_table_data('capacitance', is_corrected, dist_table)
    pixel_cap_error_data = _extract_table_data('cap_std', is_corrected, dist_table)
    voltage_data = _extract_table_data('bias', is_corrected, dist_table)
    systematic_depletion_data = depletion_data_table.copy(depletion_data_table._v_parent, "SystematicDepletionTable")
    systematic_offset = kwargs.pop("systematic_offset", 2)
    systematic_key_args = kwargs.copy()
    systematic_key_args.pop("plot", False)
    systematic_key_args.pop("fit_plot_pdf", None)
    systematic_key_args.pop("output_pdf", None)
    systematic_key_args.pop("cv_fit_plot_pdf", None)
    if isinstance(first_boundaries, Iterable) and not isinstance(first_boundaries, Tuple):
        assert first_boundaries is not None
        assert second_boundaries is not None
        assert isinstance(first_boundaries, Sized)
        n_depletion_regions = len(first_boundaries)
        fit_result_storage = DepletionTableStore(depletion_data_table, n_depletions=n_depletion_regions, )
        systematic_fit_result_storage = DepletionTableStore(systematic_depletion_data,
                                                            n_depletions=n_depletion_regions, )
        for k, (first_bound, second_bound) in enumerate(zip(first_boundaries, second_boundaries)):
            first_lower, first_upper = first_bound
            second_lower, second_upper = second_bound
            fit_result_storage.set_depletion_region(k)

            # Now the real implementation starts.
            analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, pixel_cap_error_data,
                                    second_lower,
                                    second_upper, voltage_data, fit_result_storage,
                                    fit_description_text=" Sensor Distribution", **kwargs)
            analyze_pixel_depletion(first_lower - systematic_offset, first_upper + systematic_offset, pixel_cap_data,
                                    pixel_cap_error_data,
                                    second_lower - systematic_offset,
                                    second_upper + systematic_offset, voltage_data, systematic_fit_result_storage,
                                    fit_description_text=" Sensor Distribution Systematic", **systematic_key_args)
    else:
        first_lower, first_upper = first_boundaries
        second_lower, second_upper = second_boundaries
        fit_result_storage = DepletionTableStore(depletion_data_table)
        systematic_fit_result_storage = DepletionTableStore(systematic_depletion_data, )
        analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, pixel_cap_error_data,
                                second_lower,
                                second_upper, voltage_data, fit_result_storage,
                                fit_description_text=" Sensor Distribution", **kwargs)
        analyze_pixel_depletion(first_lower - systematic_offset, first_upper + systematic_offset, pixel_cap_data,
                                pixel_cap_error_data,
                                second_lower - systematic_offset,
                                second_upper + systematic_offset, voltage_data, systematic_fit_result_storage,
                                fit_description_text=" Sensor Distribution Systematic", **systematic_key_args)

    depletion_data_table.flush()
    systematic_depletion_data.flush()
    depletion_data_table.cols.Ubi_systematic[:] = np.abs(
        depletion_data_table.cols.Ubi[:] - systematic_depletion_data.cols.Ubi[:])
    depletion_data_table.flush()
    systematic_depletion_data.flush()
    group_get_file(depletion_data_table).flush()
    systematic_depletion_data.remove()
    group_get_file(depletion_data_table).flush()


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


@deprecated("Please use analsis_data_handle instead.")
def analysis_data_handle_temporary_replacement(file: tb.File, data_group: GroupType, result_group: GroupType,
                                               is_advanced=False, is_inter_pixel=False, **kwargs):
    """
    advanced_analysis_data_handle

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
    analysis_data_handle(file, data_group, result_group, is_advanced, is_inter_pixel, **kwargs)


def analysis_data_handle(file: tb.File, data_group: GroupType, result_group: GroupType,
                         is_advanced=False, is_inter_pixel=False, **kwargs):
    """
    advanced_analysis_data_handle

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
        # Read scan parameters
        scan_parameters = data_group.scan_params[:]
        assert isinstance(scan_parameters, tb.Table) or isinstance(scan_parameters, np.ndarray)
        assert isinstance(current_hist, tb.CArray) or isinstance(current_hist, np.ndarray)
        kwargs["current_error_hist"] = current_error_hist
        perform_analysis(file, result_group, current_hist, scan_parameters, **kwargs)

    # if necessary: directly apply the correction of the capacitance values
    _handle_cap_correction(result_group, **kwargs)


def _handle_inter_pix_capacitance(file: File, data_group: tb.Group, is_advanced: ADVANCED_PARAMETER_TYPE,
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
    current_hist = check_leaf_unit(data_group.InterHistCurrA, HIST_CURRENT_MEAS_UNIT)
    if is_advanced and "InterHistCurrErrA" in data_group:
        current_error_hist = check_leaf_unit(data_group.InterHistCurrErrA, HIST_CURRENT_MEAS_UNIT)
    else:
        current_error_hist = np.full_like(current_hist, fill_value=np.nan)

    kwargs["current_error_hist"] = current_error_hist
    assert isinstance(current_hist, tb.CArray) or isinstance(current_hist, np.ndarray)
    perform_analysis(file, result_group, current_hist, scan_parameters, **kwargs)

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
    perform_analysis(file, result_group, current_hist, scan_parameters, **kwargs)


def _handle_cap_correction(result_group: Group, **kwargs):
    if "apply_correction" in kwargs and kwargs["apply_correction"]:
        assert 'bare_file' in kwargs
        assert 'bare_hdf_path' in kwargs
        apply_correction_simple(kwargs['bare_file'], kwargs['bare_hdf_path'], result_group)


def _get_analyze(is_advanced: bool) -> Callable[..., None]:
    if is_advanced:
        from pixcap65.advanced_analysis import advanced_analysis_delegate
        perform_analysis = advanced_analysis_delegate
    else:
        from pixcap65.analysis_util import analyze_data_delegate
        perform_analysis = analyze_data_delegate
    return perform_analysis


def analyze_depletion_delegate(data_group: tb.Group, analysis_group: tb.Group,
                               first_boundaries: Optional[BOUNDARY_TYPE],
                               second_boundaries: Optional[BOUNDARY_TYPE], chip_group: Optional[tb.Group] = None,
                               apply_doping=False,
                               **kwargs):
    """
    analyse_depletion_delegate

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
    # extract the additional parameters for advanced fitting procedures
    propagate_kwargs = kwargs.copy()
    propagate_kargs = kwargs.copy()
    propagate_kargs.setdefault('output_pdf', kwargs.get('fit_plot_pdf', None))

    # verify and extract the raw data for further analysis
    cap_data = check_leaf_unit(analysis_group.UCHist, HIST_CAP_UNIT)
    cap_error_data = check_leaf_unit(analysis_group.UCErrHist, HIST_CAP_UNIT)
    voltage_data = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
    if len(voltage_data.shape) > 1:
        # TODO: adjust for usage of the correct voltages. Also find all the other places were the BiasVoltageHist is used.
        voltage_data = voltage_data[:, 0]

    if isinstance(first_boundaries, Iterable) and not isinstance(first_boundaries, Tuple):
        assert first_boundaries is not None
        assert second_boundaries is not None
        assert isinstance(first_boundaries, Sized)
        fit_result_storage = DepletionArrayStore(n_depletions=len(first_boundaries))
        for k, (first_bound, second_bound) in enumerate(zip(first_boundaries, second_boundaries)):
            first_lower, first_upper = first_bound
            second_lower, second_upper = second_bound
            fit_result_storage.set_depletion_region(k)
            depletion_delegation_impl(cap_data, cap_error_data, first_lower, first_upper, second_lower, second_upper,
                                      fit_result_storage, voltage_data, **propagate_kargs)

    else:
        # extract the required data and create arrays for temporary storage.
        first_lower, first_upper = first_boundaries
        second_lower, second_upper = second_boundaries
        fit_result_storage = DepletionArrayStore()
        depletion_delegation_impl(cap_data, cap_error_data, first_lower, first_upper, second_lower, second_upper,
                                  fit_result_storage, voltage_data, **propagate_kargs)

    # save the depletion voltage data.
    file_h5 = group_get_file(analysis_group)
    create_carray(file_h5, where=analysis_group, name="DepletionHist",
                  title="Histogram of the depletion voltages", obj=fit_result_storage.depletion_voltage,
                  filters=GLOBAL_FILTERS, unit=HIST_BIAS_MEAS_UNIT)
    create_carray(file_h5, where=analysis_group, name="DepletionErrHist",
                  title="Histogram of the depletion voltage errors", obj=fit_result_storage.depletion_error,
                  filters=GLOBAL_FILTERS, unit=HIST_BIAS_MEAS_UNIT)
    create_carray(file_h5, where=analysis_group, name="DepFitParamHist",
                  title="Histogram of the depletion voltages fit parameters",
                  obj=fit_result_storage.fit_parameter_estimators,
                  filters=GLOBAL_FILTERS, unit="NONE")
    create_carray(file_h5, where=analysis_group, name="DepFitParamErrHist",
                  title="Histogram of the depletion voltages fit parameter errors",
                  obj=fit_result_storage.fit_parameter_errors,
                  filters=GLOBAL_FILTERS, unit="NONE")

    if (not apply_doping or chip_group is None or "PhysicalDimensions" not in chip_group or
            chip_group.PhysicalDimensions.shape != (40, 40, 2) or
            np.any(~np.isfinite(chip_group.PhysicalDimensions[:]))):
        return
    physical_dimensions_data = chip_group.PhysicalDimensions[:]
    pixel_areas = np.prod(physical_dimensions_data, axis=2)
    doping_shape = (40, 40, voltage_data.shape[0])
    doping_result_storage = DopingArrayStore(doping_shape)

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
    bias_voltages = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
    bias_voltage_errors = np.full_like(bias_voltages, fill_value=np.nan)

    for col, row in np.ndindex(GENERAL_PIXCAP_SHAPE):
        propagate_kwargs['fit_description_text'] = ' for Pixel ({col},{row})'.format(col=col, row=row)
        # make sure the provided data is useful for further investigation.
        if not np.all(np.isfinite(physical_dimensions_data[col, row])):
            continue
        pixel_cap_data = cap_data[col, row]
        pixel_cap_error_data = cap_error_data[col, row]
        pixel_area = pixel_areas[col, row]
        if not np.all(np.isfinite(pixel_cap_data)):
            continue
        doping_result_storage.set_pixel(row, col)
        entry["row"] = row
        entry["col"] = col

        n_eff, depletion_width_data, pos_min = analyze_doping_profile(bias_voltages, bias_voltage_errors,
                                                                      doping_result_storage, entry, NA / 2, v_bi,
                                                                      pixel_area, pixel_cap_data,
                                                                      pixel_cap_error_data, **propagate_kwargs)

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
    file_h5.flush()


def depletion_delegation_impl(cap_data, cap_error_data, first_lower, first_upper, second_lower, second_upper,
                              fit_result_storage: DepletionArrayStore, voltage_data: Union[np.ndarray, tb.CArray],
                              **kwargs):
    """
    depletion_delegate_impl

    @author Dominik Fischer
    @date 2026-05-07

    Implementation of the pixel-wise depletion voltage estimation from a provided C-V characterization.

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
    """
    kwargs.setdefault("verbose", False)
    for ii, jj in np.ndindex(GENERAL_PIXCAP_SHAPE):
        pixel_cap_data = cap_data[ii, jj, :]
        pixel_cap_error_data = cap_error_data[ii, jj, :]
        # could only perform the analysis for pixels with trustable measurements.
        if np.any(~np.isfinite(pixel_cap_data)):
            continue
        fit_result_storage.set_pixel(jj, ii)
        analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, pixel_cap_error_data, second_lower,
                                second_upper, voltage_data, fit_result_storage,
                                fit_description_text=" for Pixel ({col},{row})".format(col=ii, row=jj),
                                **kwargs)


DOPING_RESULT_TYPE = Tuple[np.ndarray, np.ndarray, int]


def analyze_doping_profile(bias_voltages: np.ndarray,
                           bias_voltage_errors: np.ndarray,
                           doping_result_storage: DopingArrayStore, entry, n_a: float, v_bi: float,
                           pixel_area, pixel_cap_data: np.ndarray, pixel_cap_error_data: np.ndarray,
                           **kwargs) -> DOPING_RESULT_TYPE:
    """
    analyze:doping_profile

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
    depletion_width_data = np.asarray((constants.epsilon_0 * pixel_area) / pixel_cap_data * 1.e-6)
    depletion_width_error_data = (constants.epsilon_0 * pixel_area * effective_cap_errors) / (np.array(
        pixel_cap_data) ** 2) * 1.e-6

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
            handle_kafe2_advanced_options(fitter, apply_contours, "$U_\\text{{bi}}$ in \\unit{{\\volt}}",
                                          "$d$ in \\unit{{\\micro\\meter}}",
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
            handle_minuit_advanced_options(fitter, apply_contours, "$U_\\text{{bi}}$ in \\unit{{\\volt}}",
                                           "$d$ in \\unit{{\\micro\\meter}}",
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
                             diode_area=pixel_area)  # -> 10^(12) 1/m
    doping_result_storage.store_data("doping", n_eff)
    pos_min = int(np.argmin(n_eff))
    for key, value in depletion_fit_propagate_parameters.items():
        entry[key] = value
    entry.append()
    return n_eff, depletion_width_data, pos_min


def analyze_pixel_depletion(first_lower, first_upper,
                            pixel_cap_data: np.ndarray,
                            pixel_cap_error_data: np.ndarray, second_lower, second_upper,
                            voltage_data: TABLES_LEAF_COMPAT_TYPE,
                            result: DepletionDataStore, **kwargs):
    """
    analyze_pixel_depletion

    @author Dominik Fischer
    @date 2026-05-07

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
    if "cv_fit_plot_pdf" in kwargs:
        kwargs["fit_plot_pdf"] = kwargs.pop("cv_fit_plot_pdf")
        kwargs["plot"] = True
    # extract the information about the depletion voltage
    assert not isinstance(voltage_data, tb.Leaf)
    first_section_upper_mask = voltage_data <= first_upper
    first_section_lower_mask = voltage_data >= first_lower
    first_section_mask = np.logical_and(first_section_upper_mask, first_section_lower_mask)

    # print(second_upper, second_lower)
    second_section_upper_mask = voltage_data <= second_upper
    second_section_lower_mask = voltage_data >= second_lower
    # print(np.count_nonzero(second_section_upper_mask))
    # print(np.count_nonzero(second_section_lower_mask))
    # print(voltage_data)
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
    first_dep_cov, first_dep_errors, first_dep_parameters = get_depletion_fit(first_cap_data, first_cap_error_data,
                                                                              first_voltage_data, "First",
                                                                              **kwargs)

    # print(second_cap_data)
    # print(second_cap_data.shape)
    second_dep_cov, second_dep_errors, second_dep_parameters = get_depletion_fit(second_cap_data, second_cap_error_data,
                                                                                 second_voltage_data, "Second",
                                                                                 **kwargs)

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
    result.store_data("depletion", dep_voltage_2)
    result.store_data("depletion_error", dep_voltage_error_2)
    if verbose_output:
        print("The depletion voltage is {voltage}+-{error}".format(voltage=dep_voltage_2,
                                                                   error=dep_voltage_error_2))
    result.store_data("fit_result_first", first_dep_parameters)
    result.store_data("fit_result_second", second_dep_parameters)
    result.store_data("fit_error_first", first_dep_errors)
    result.store_data("fit_error_second", second_dep_errors)
    result.flush_data()


def get_depletion_fit(cap_data: np.ndarray, cap_error_data: np.ndarray, voltage_data: np.ndarray,
                      fit_reference, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    get_depletion_fit

    @author Dominik Fischer
    @date 2026-05-07

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
                handle_minuit_advanced_options(m, apply_contours, "$U$ in \\unit{{\\volt}}", "$\\frac{{1}}{{C^2}}$",
                                               "{} Fit{}".format(fit_reference, fit_description_text),
                                               output_pdf,
                                               "{} Contour{}".format(fit_reference, fit_description_text))
            try:
                assert first_dep_cov is not None
            except AssertionError:
                print(m.valid)
                print(m.fmin)
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
    :return: effective doping concentration in 10^(12) 1/m (maybe still an issue with the units)
    """
    from scipy import constants
    from findiff import Diff

    # make sure the data is provided as numpy arrays
    capacitance = np.asarray(capacitance)
    bias_voltages = np.asarray(bias_voltages)

    if diode_area is None:
        diode_area = 50 * 50  # measured in um^2
    temp_capacitance = np.reciprocal((capacitance * CAPACITANCE_CONVERSION_FACTOR) ** 2)

    # noinspection PyTypeChecker
    try:
        # handle duplicates
        unique_voltage, cap_mask, voltage_counts = np.unique(bias_voltages, return_index=True, return_counts=True)
        duplicate_mask = voltage_counts > 1
        unique_temp_capacitance = temp_capacitance[cap_mask]
        derivative = np.full_like(bias_voltages, np.nan)
        d_du = Diff(0, unique_voltage, acc=2)
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


    n_eff = 2 / (constants.elementary_charge * constants.epsilon_0 * EPS_SILICON * (diode_area ** 2) * np.asarray(
        derivative)) * 1e6
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
    :key use_kafe2: TODO: missing description.
    :key fit_plot_pdf: TODO: missing description.
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key mask_lower: float, threshold to mask all pixels below this value.
    :key mask_upper: float, threshold to mask all pixels above this value.
    :key exclude_cap_hist: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    :key capacitance: histogram of the capacitance to use instead of those extracted from the provided hdf files group.
    :key set_parasitic: boolean, whether to set the parasitic capacitance for this data set.
    :key no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
    """
    from pixcap65.plotting import CAPACITANCE_CONVERSION_FACTOR
    from pixcap65.plotting import DEFAULT_BIN_NUMBER
    from pixcap65.plotting import COUNTS_HIST_LABEL
    from pixcap65.plotting import HIST_PIX_CAP_LABEL
    from matplotlib import pyplot as plt
    from matplotlib import rcParams

    # extract further arguments for the performance of the fitting
    use_kafe2 = kwargs.pop("use_kafe2", False)
    if output_pdf is None:
        output_pdf = kwargs.pop("fit_plot_pdf", None)
    set_parasitic = kwargs.pop("set_parasitic", analysis_group is not None)
    assert not set_parasitic or analysis_group is not None

    # we want to fit a binned distribution; but some bins might be empty
    cap_hist = kwargs.pop("capacitance", None)
    kargs = kwargs.copy()
    temp_exclusion_flag = kargs.pop("exclude_cap_hist", None)
    if temp_exclusion_flag is not None:
        kargs["test_cap_exclusion"] = temp_exclusion_flag

    if cap_hist is None or not isinstance(cap_hist, np.ndarray):
        assert analysis_group is not None
        cap_hist = check_leaf_unit(analysis_group.HistCap, HIST_CAP_UNIT)
    cv_height, cv_width = rcParams['figure.figsize']
    fig, ax = plt.subplots(figsize=(cv_width, cv_height))
    hist_cap_hist = evaluate_pixel_mask(cap_hist, **kargs)
    temp_hist_back_data = hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR
    hist_data, bins, _ = ax.hist(temp_hist_back_data,
                                 bins=kwargs.get("hist_bins", DEFAULT_BIN_NUMBER), density=False,)
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
            handle_fitter_advanced_options(extended_fitter, False, "$C$ in \\unit{{\\femto\\farad}}", COUNTS_HIST_LABEL,
                                           "Capacitance distribution (EXTENDED)", output_pdf,
                                           "Contours for the capacitance distribution (EXTENDED)")
        # FIXME: issue when handling the minuit.
        handle_fitter_advanced_options(fitter, True, "$C$ in \\unit{{\\femto\\farad}}", COUNTS_HIST_LABEL,
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
            label="Model for C =  \\qty{{{c:.2f}\\pm{error:.2f}}}{{\\femto\\farad}}".format(c=mean_value,
                                                                                            error=std_value))

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
            title=f"GoF = {hypo_test['x']:.4f}\nndf = {hypo_test['ndf']: .4f}\np = {hypo_test['p']: .4f}\nu = "
                  f"{mean_value:.3f}+-{mean_error:.3f}\ns = {std_value:.3f}+-{std_error:.3f}")
    if kwargs.get('no_plot', False):
        plt.close(fig)
    elif output_pdf is None:
        plt.show()
    else:
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    return temp_hist_back_data.reshape(-1).shape[0], mean_value, mean_error, std_value, std_error


def apply_correction(raw_data, base_path=None, bare_data_path=None, bare_group=None):
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
    with tb.open_file(raw_data, mode='a') as in_file_h5_inner:
        base_group = get_base_group(base_path, in_file_h5_inner)
        apply_correction_simple(bare_data_path, bare_group, base_group.analysis)


def apply_correction_simple(bare_data_path: str, bare_path: str, analysis_group: tb.Group):
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
    """
    with tb.open_file(bare_data_path, mode='r') as in_file_h5:
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

def get_test_capacitance_data(group: tb.Group):
    test_cap = group.HistCap[:][:, 0]
    test_cap_error = group.HistCapErr[:][:, 0]
    for k, (cap, err) in enumerate(zip(test_cap, test_cap_error)):
        eff_cap = cap * 1e15
        eff_err = err * 1e15
        print(k, f"{eff_cap:.3f}+-{eff_err:.3f}")

    test_cap[16] = np.nan
    test_cap_error[16] = np.nan
    return test_cap[np.isfinite(test_cap)], test_cap_error[np.isfinite(test_cap_error)]

def generate_test_summary(files, groups, sensors, summary_file):
    with synchronized_process_open_file(summary_file, mode='a') as summary_file:
        field_names = [name.replace(" ", "_") for name in [
            "0 w_o bump", "1 w_o bump", "2 w_o bump", "3 w_o bump", "4 w_o bump", "5",
            "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "17",
            "18", "19", "20", "21", "22", "23", "24", "25", "26", "27 w_ bump",
            "28 w_ bump", "29 w_ bump", "30 w_ bump", "31 w_ bump", "32 w_ bump", "33 w_ bump",
            "34 w_ bump", "35 w_o bump", "36 w_o bump", "37 w_o bump", "38 w_o bump", "39 w_o bump"
        ]]
        table_description = [("Sensor", np.uint64),]
        for name in field_names:
            table_description.extend([(name, np.float64),])
            table_description.extend([("{}_error".format(name), np.float64),])

        if "TestCap" in summary_file.root:
            summary_file.root.TestCap.remove()
            time.sleep(10)

        print(table_description)
        table_type = np.dtype(table_description)
        print(table_type)
        table = summary_file.create_table(where=summary_file.root, name="TestCap", title="Test Capacitances from the different sensors", description=np.dtype(table_description))
        for file, group_path, sensor in zip(files, groups, sensors):
            with synchronized_process_open_file(file, mode='r') as h5_file:
                group, _ = walk_to_node(h5_file.root, group_path, create=False, verify_create=True)
                test_cap, test_cap_err = get_test_capacitance_data(group)
                assert isinstance(sensor, str)
                # result_data = [sensor]
                result_data = [int(sensor.encode().hex())]
                for cap, cap_err in zip(test_cap, test_cap_err):
                    result_data.append(cap)
                    result_data.append(cap_err)

                rec_array = np.rec.array(result_data, dtype=table_type)
                table.append([result_data, ])

        table.flush()




if __name__ == '__main__':
    # analyse_data(raw_data='/home/silab/git/pixcap65/pixcap_full_data_image1.h5')

    # some usage examples
    from pixcap65.utility.homogenize_plots import set_params

    analyze_data(raw_data="data/3D_Sensor_221_W13_X_Scan.h5", base_path="Thesis/ATLAS_ITk/X3/C_V_Characteristic", is_cv=True)
    analyze_data(raw_data="data/3D_Sensor_221_W6_j_Scan.h5", base_path="Thesis/ATLAS_ITk/X5/C_V_Characteristic", is_cv=True)
    analyze_data(raw_data="data/3D_Sensor_I14_S24_Scan.h5", base_path="Thesis/ATLAS_ITk/X6/C_V_Characteristic", is_cv=True)
    analyze_data(raw_data="data/3D_Sensor_H23_S24_Scan.h5", base_path="Thesis/ATLAS_ITk/X7/C_V_Characteristic", is_cv=True)
    analyze_data(raw_data="data/argparser.h5", base_path="Reference/R11/C_V_Characteristic", is_cv=True)
    analyze_data(raw_data="data/3D_Sensor_221_W5_S_Scan.h5", base_path="Thesis/ATLAS_ITk/X4/C_V_Characteristic", is_cv=True)
    # set_params(latex=True,
    #            latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors="
    #                        r"{stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}"
    #                        r"\sisetup{retain-zero-uncertainty}")
    #
    # bare_correction_args = {
    #     "apply_correction": True,
    #     "bare_file": "Bare_Repeat_2_Scan.h5",
    #     "bare_hdf_path": "Reference/bare/unbiased_8/total_cap",
    # }
    #
    #
    #
    # # noqa: S125
    # # analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
    # #                                  corrected_distribution=False,
    # #                                  exclude_test_cap=True, use_kafe2=False,
    # #                                  fit_plot_pdf_name="Bare_analysis_parasitic.pdf")
    # # analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', is_advanced=True,
    # #              **bare_correction_args)
    # # analyze_data(raw_data='R13-Interpixel_Scan.h5',
    # #              base_path="Reference/R13/demo_measurement_65_unbiased_1_discharge",
    # #              is_inter_pixel=True, is_advanced=True)
    # # analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic",
    # #              is_advanced=False, is_cv=True, use_corrected=True, apply_doping=True,
    # #              chip_group_name="ATLAS ITk/sensor",
    # #              first_boundaries=[(-60, -40), (-80, -75)], second_boundaries=[(-5, 0), (-70, -65)],
    # #              **bare_correction_args)
    #
    # # Second Try Bare
    # analyze_data(raw_data="packaged/Reference_Bare_renewed.h5", base_path="Reference/Bare/unbiased_31_renew", is_advanced=True, full_model=False)
    #
    # # Second Try R13
    # print("Analyze R13")
    # analyze_data(raw_data="packaged/R13_2_Scan.h5", base_path="ATLAS_ITk/X2/unbiased_1_full", is_advanced=True, **bare_correction_args)
    # analyze_data(raw_data="packaged/R13_2_Scan.h5", base_path="ATLAS_ITk/X2/biased_80_V_full", is_advanced=True,
    #              **bare_correction_args)
    # with PdfPages("R13_3_Scan_Combined_reference_fits.pdf") as pdf:
    #     analyze_data(raw_data="packaged/R13_3_Scan.h5", base_path="Reference/R13/C_V_Characteristic_refined", is_advanced=True, full_model=False, is_cv=True, first_boundaries=(-85,-33), second_boundaries=(-4,0), distribution=True, cv_fit_plot_pdf=pdf,
    #                  **bare_correction_args)
    #
    # # Second Try X1
    # print("Analyze X1")
    # # analyze_data(raw_data=X1_SCAN_2_FILE, base_path="ATLAS_ITk/X1/unbiased_61_full", is_advanced=True,
    # #              **bare_correction_args)
    # # analyze_data(raw_data=X1_SCAN_2_FILE, base_path="ATLAS_ITk/X1/biased_80_V_full", is_advanced=True,
    # #              **bare_correction_args)
    # with PdfPages("X1_Scan_Combined_reference_fits.pdf") as pdf:
    #     analyze_data(raw_data=X1_SCAN_2_FILE, base_path="ATLAS_ITk/X1/C_V_Characteristic_refined",
    #                  is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
    #                  first_boundaries=[(-60, -40), (-82, -77)], second_boundaries=[(-2, 0), (-73, -65)], distribution=True, cv_fit_plot_pdf=pdf,
    #                  **bare_correction_args)
    #
    # # Second Try X2
    # print("Analyze X2")
    # # analyze_data(raw_data=X2_SCAN_2_FILE, base_path="ATLAS_ITk/X2/unbiased_1_full", is_advanced=True,
    # #              **bare_correction_args)
    # # analyze_data(raw_data=X2_SCAN_2_FILE, base_path="ATLAS_ITk/X2/biased_80_V_full", is_advanced=True,
    # #              **bare_correction_args)
    # # with PdfPages("X2_SCAN_Combined_reference_fits.pdf") as pdf:
    # #     analyze_data(raw_data=X2_SCAN_2_FILE, base_path="ATLAS_ITk/X2/C_V_Characteristic_refined",
    # #                  is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
    # #                  first_boundaries=[(-55, -20), (-78, -73)], second_boundaries=[(-5, 0), (-68, -60)], distribution=True, cv_fit_plot_pdf=pdf,
    # #                  **bare_correction_args)
    #
    # # remaining analysis of the E1 sample
    # print("Analyze E1")
    # # analyze_data(raw_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/unbiased_4_full", is_advanced=True,
    # #              **bare_correction_args)
    # # e1_depletion_args = {
    # #     "first_boundaries": (-100, -80),
    # #     "second_boundaries": (-10, 0),
    # # }
    # # analyze_data(raw_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/C_V_Characteristic",
    # #              is_advanced=True, full_model=False, is_cv=True, use_corrected=True, distribution=True, cv_fit_plot_pdf_name="E1_C_V_Verify.pdf", **e1_depletion_args,
    # #              **bare_correction_args)
    #
    # print("Analyze Tests")
    # # analyze_data(raw_data=REFERENCE_TEST_FILE, base_path="Reference/TESTS/cv_only_simple",
    # #              is_advanced=True, is_cv=True, use_corrected=True,
    # #              **bare_correction_args)
    # # analyze_data(raw_data=REFERENCE_TEST_FILE, base_path="Reference/TESTS/cv_only_advanced",
    # #              is_advanced=True, is_cv=True, use_corrected=True,
    # #              **bare_correction_args)
    # # analyze_data(raw_data=REFERENCE_TEST_FILE, base_path="Reference/TESTS/cv_combined_simple",
    # #              is_advanced=True, is_cv=True, use_corrected=True,
    # #              **bare_correction_args)
    # # analyze_data(raw_data=REFERENCE_TEST_FILE, base_path="Reference/TESTS/cv_combined_advanced",
    # #              is_advanced=True, is_cv=True, use_corrected=True,
    # #              **bare_correction_args)
    # #
    # # analyze_data(raw_data=REFERENCE_TEST_FILE, base_path="Reference/TESTS/unbiased_30",
    # #              is_advanced=True, full_model=False, is_cv=False, use_corrected=False)
    # # analyze_data(raw_data=REFERENCE_TEST_FILE, base_path="Reference/TESTS/unbiased_31",
    # #              is_advanced=True, full_model=False, is_cv=False, use_corrected=False)

    print("generate summary")
    generate_test_summary(summary_files, summary_groups, summary_sensors, "conclude_result.h5")
