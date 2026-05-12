"""
Analysis of Pixcap65 data. Fits freq vs current to extract the capacitance. A 2D histogram containing the capacitance
for each pixel is stored.
"""
import logging

from tables import File, Group

from pixcap65.analysis_util.data_store import DepletionDataStore, DepletionTableStore, DepletionArrayStore, \
    DopingArrayStore

try:
    # noinspection PyCompatibility
    from collections.abc import Sized, Iterable
except ImportError:
    # python 2.7
    import collections.Sized as Sized
    import collections.Iterable as Iterable

import numpy as np
import tables as tb
from typing import Optional, Tuple, Union, Callable, Any
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
    CVDistributionData, CVDepletionCapacitanceData, handle_fitter_advanced_options
from pixcap65.plotting import CAPACITANCE_CONVERSION_FACTOR, evaluate_pixel_mask
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
        analysis.
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
    # handle the additional PDF file in case of plotting enabled
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

    # handle the real analysis
    with tb.open_file(raw_data, mode='a') as in_file_h5:
        # extract the hdf file groups to perform the analysis on.
        base_group = get_base_group(base_path, in_file_h5)

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

            apply_correction_arg = kwargs.pop("apply_correction", False)
            bare_file_arg = kwargs.pop("bare_file", None)
            bare_path_arg = kwargs.pop("bare_hdf_path", None)

            if kwargs.get("distribution", False):
                in_file_h5.create_group(base_group.biasing, "analysis")
                dist_table = in_file_h5.create_table(where=base_group.biasing.analysis, name="CVDistribution",
                                                     description=CVDistributionData, filters=GLOBAL_FILTERS)
                dist_entry = dist_table.row
            else:
                dist_entry = {}

            if kwargs.get("apply_correction", False):
                # extract the parasitic capacitance right here!
                with tb.open_file(kwargs["bare_file"], mode='r') as correction_h5:
                    bare_path = kwargs.get("bare_hdf_file", None)
                    assert isinstance(bare_path, (str, None))
                    parasitic, parasitic_error = _handle_parasitic_cap(bare_path, correction_h5)
            else:
                parasitic = 0
                parasitic_error = 0

            # no progressbar as the overhead for this is much too large in must cases.
            for k, bias_voltage in enumerate(base_group.biasing.measurements.BiasVoltageHist):
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

                if kwargs.get("distribution", False):
                    # first get a histogram with all the data
                    temp_tuple = analyze_capacitance_distribution_delegate(None,
                                                                           None,
                                                                           capacitance=cap_data,
                                                                           **kargs)
                    n_pix, cap, cap_err, cap_std, cap_std_err = temp_tuple

                    # try to get to the on-resistance datasets to perform the same distribution handler!
                    # only issue with this attempt the capacitance will be saved as a parasitic one!
                    if "HistRes" in ana_group:
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
                    dist_entry['capacitance'] = cap
                    dist_entry['cap_err'] = cap_err
                    dist_entry['cap_std'] = cap_std
                    dist_entry['cap_std_err'] = cap_std_err
                    dist_entry['r_on'] = res
                    dist_entry['r_on_err'] = res_err
                    dist_entry['r_on_std'] = res_std
                    dist_entry['r_on_std_err'] = res_std_err
                    if kwargs.get("apply_correction", False):
                        dist_entry['cap_corrected'] = cap - parasitic
                        dist_entry['cap_corrected_err'] = cap_std
                        dist_entry['cap_parasitic'] = parasitic
                        dist_entry['cap_systematic_error'] = np.sqrt(cap_err ** 2 + parasitic_error ** 2)
                    else:
                        dist_entry['cap_corrected'] = cap
                        dist_entry['cap_corrected_err'] = cap_std
                        dist_entry['cap_parasitic'] = 0
                        dist_entry['cap_systematic_error'] = cap_err

                    if isinstance(dist_entry, tb.tableextension.Row):
                        dist_entry.append()
                    else:
                        print("No append of the distribution!")

            # in case of active correction, also the summary capacitance tables needs to be corrected, but it should
            # also an uncorrected table present.
            if apply_correction_arg:
                # perform the transfer
                apply_correction_simple(bare_file_arg, bare_path_arg, base_group.biasing.analysis, )

                for k, bias_voltage in enumerate(base_group.biasing.measurements.BiasVoltageHist):
                    ana_group_correction, _ = walk_to_node(base_group.biasing,
                                                           str_join("/", ANALYSIS_CORRECTED_GROUP_NAME, bias_name),
                                                           create=True, verify_create=True)
                    cap_data = ana_group_correction.HistCap[:]
                    cap_error_data = ana_group_correction.HistCapErr[:]
                    cv_data_corrected[:, :, k] = cap_data[:, :]
                    cv_err_data_corrected[:, :, k] = cap_error_data[:, :]

            create_carray(in_file_h5, base_group.biasing.analysis, name="UCHist", title="Histogram of the U-C-curve",
                          filters=GLOBAL_FILTERS, obj=cv_data, unit=HIST_CAP_UNIT)
            create_carray(in_file_h5, base_group.biasing.analysis, name="UCErrHist",
                          title="Error Histogram of the U-C-curve",
                          filters=GLOBAL_FILTERS,
                          obj=cv_err_data,
                          unit=HIST_CAP_UNIT)

            # save also the corrected capacitance data if necessary
            if apply_correction_arg:
                create_carray(in_file_h5, base_group.biasing.analysis_correction, name="UCHist",
                              title="Histogram of the U-C-curve",
                              filters=GLOBAL_FILTERS,
                              obj=cv_data, unit=HIST_CAP_UNIT)
                create_carray(in_file_h5, base_group.biasing.analysis_correction, name="UCErrHist",
                              title="Error Histogram of the U-C-curve",
                              filters=GLOBAL_FILTERS,
                              obj=cv_err_data, unit=HIST_CAP_UNIT)

            if first_boundaries is not None and second_boundaries is not None:
                dep_ana_group = get_analysis_group(base_group.biasing, **kwargs)
                assert isinstance(dep_ana_group, tb.Group)
                chip_name = kwargs.pop("chip_group_name", None)
                # we need to put the plotting arguments back in
                if chip_name is not None:
                    chip_spec_group, _ = walk_to_node(in_file_h5.root, chip_name, create=False, verify_create=True)
                    kwargs['chip_group'] = chip_spec_group

                # Anyway it is necessary to perform this analysis also on a sensor-average basis!
                if kwargs.pop("use_corrected", False):
                    analyze_depletion_delegate(base_group.biasing.measurements,
                                               base_group.biasing.analysis,
                                               first_boundaries, second_boundaries, **kwargs)
                analyze_depletion_delegate(base_group.biasing.measurements, dep_ana_group,
                                           first_boundaries, second_boundaries, **kwargs)

                if kwargs.get("distribution", False):
                    dist_table.flush()
                    dist_table = base_group.biasing.analysis.CVDistribution[:]
                    first_lower, first_upper = first_boundaries
                    second_lower, second_upper = second_boundaries
                    pixel_cap_data = dist_table['capacitance']
                    pixel_cap_error_data = dist_table['cap_std']
                    voltage_data = dist_table['bias']
                    depletion_data_table = in_file_h5.create_table(where=base_group.biasing.analysis,
                                                                   name="SensorDepletionRaw", description=DepletionData,
                                                                   filters=GLOBAL_FILTERS)
                    depletion_distribution_table = in_file_h5.create_table(where=base_group.biasing.analysis,
                                                                           name="SensorDepletion",
                                                                           description=CVDepletionCapacitanceData,
                                                                           filters=GLOBAL_FILTERS)
                    fit_result_storage = DepletionTableStore(depletion_data_table)
                    analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, pixel_cap_error_data,
                                            second_lower,
                                            second_upper, voltage_data, fit_result_storage,
                                            fit_description_text=" Sensor Distribution", **kwargs)

                    depletion_data_table.flush()
                    # now combine the two tables at last!
                    for first_row, second_row in zip(base_group.biasing.analysis.CVDistribution.iterrows(),
                                                     base_group.biasing.analysis.SensorDepletionRaw.iterrows()):
                        distribution_data = first_row[:]
                        depletion_data = second_row[:]
                        new_data = np.concat((distribution_data, depletion_data), )
                        assert distribution_data != depletion_data
                        depletion_distribution_table.append(new_data)
                    depletion_distribution_table.flush()

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
                                 is_inter_pixel=is_inter_pixel, **kwargs)


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
    # TODO: is the unit correct?


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
    """
    # extract the additional parameters for advanced fitting procedures
    verbose_output = kwargs.pop("verbose", False)
    # extract the information about the depletion voltage
    assert not isinstance(voltage_data, tb.Leaf)
    first_section_upper_mask = voltage_data <= first_upper
    first_section_lower_mask = voltage_data >= first_lower
    first_section_mask = np.logical_and(first_section_upper_mask, first_section_lower_mask)

    second_section_upper_mask = voltage_data <= second_upper
    second_section_lower_mask = voltage_data >= second_lower
    second_section_mask = np.logical_and(second_section_upper_mask, second_section_lower_mask)

    # extract and select the usable voltage and capacitance data for the two fit ranges.
    effective_capacitance_data = np.reciprocal(pixel_cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 2
    effective_capacitance_error_data = np.reciprocal(
        pixel_cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 3 * pixel_cap_error_data \
        if np.all(np.isfinite(pixel_cap_error_data)) else pixel_cap_error_data
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
    temp_capacitance = np.reciprocal(capacitance ** 2)

    # noinspection PyTypeChecker
    d_du = Diff(0, bias_voltages)
    derivative = d_du(temp_capacitance)
    n_eff = 2 / (constants.elementary_charge * constants.epsilon_0 * EPS_SILICON * (diode_area ** 2) * np.array(
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
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key exclude_cap_hist: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    :key capacitance: histogram of the capacitance
    """
    from pixcap65.plotting import CAPACITANCE_CONVERSION_FACTOR
    from pixcap65.plotting import DEFAULT_BIN_NUMBER
    from pixcap65.plotting import COUNTS_HIST_LABEL
    from pixcap65.plotting import HIST_PIX_CAP_LABEL
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
    from matplotlib import rcParams
    cv_height, cv_width = rcParams['figure.figsize']
    fig, ax = plt.subplots(figsize=(cv_width, cv_height))
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
        from scipy.stats.distributions import norm
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
    # add some information about the model
    hypo_test = investigate_fit_convergence(fitter)
    ax.set_ylabel(COUNTS_HIST_LABEL)
    ax.set_xlabel(HIST_PIX_CAP_LABEL)
    ax.grid()
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


if __name__ == '__main__':
    # analyse_data(raw_data='/home/silab/git/pixcap65/pixcap_full_data_image1.h5')

    # some usage examples
    from pixcap65.utility.homogenize_plots import set_params

    set_params(latex=True,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors="
                           r"{stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}"
                           r"\sisetup{retain-zero-uncertainty}")

    bare_correction_args = {
        "apply_correction": True,
        "bare_file": "Bare_Repeat_2_Scan.h5",
        "bare_hdf_path": "Reference/bare/unbiased_8/total_cap",
    }

    # analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
    #                                  corrected_distribution=False,
    #                                  exclude_test_cap=True, use_kafe2=False,
    #                                  fit_plot_pdf_name="Bare_analysis_parasitic.pdf")
    # analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', is_advanced=True,
    #              **bare_correction_args)
    # analyze_data(raw_data='R13-Interpixel_Scan.h5',
    #              base_path="Reference/R13/demo_measurement_65_unbiased_1_discharge",
    #              is_inter_pixel=True, is_advanced=True)
    # analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic",
    #              is_advanced=False, is_cv=True, use_corrected=True, apply_doping=True,
    #              chip_group_name="ATLAS ITk/sensor",
    #              first_boundaries=[(-60, -40), (-80, -75)], second_boundaries=[(-5, 0), (-70, -65)],
    #              **bare_correction_args)
    analyze_data(raw_data="packaged/X2_2_Scan.h5", base_path="ATLAS_ITk/X2/unbiased_1_full", is_advanced=True,
                 **bare_correction_args)
    analyze_data(raw_data="packaged/X2_2_Scan.h5", base_path="ATLAS_ITk/X2/biased_80_V_full", is_advanced=True,
                 **bare_correction_args)
    analyze_data(raw_data="packaged/X2_2_Scan.h5", base_path="ATLAS_ITk/X2/C_V_Characteristic_refined",
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                 first_boundaries=[(-60, -40), (-80, -75)], second_boundaries=[(-5, 0), (-70, -65)],
                 **bare_correction_args)

    # remaining analysis of the E1 sample
    analyze_data(raw_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/unbiased_4_full", is_advanced=True,
                 **bare_correction_args)
    analyze_data(raw_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/C_V_Characteristic",
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                 **bare_correction_args)
