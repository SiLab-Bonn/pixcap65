"""
Analysis of Pixcap65 data. Fits freq vs current to extract the capacitance. A 2D histogram containing the capacitance
for each pixel is stored.
"""
import logging
from typing import Optional, Tuple
from warnings import deprecated

import numpy as np
import tables as tb

from analysis_util.physics_modelling import SILICON_V_BIAS, depletion_model, model_depletion, EPS_SILICON, gauss_model, \
    extended_gauss_integral
from analysis_util.utility import check_leaf_unit, str_join, ANALYSIS_CORRECTED_GROUP_NAME, ANALYSIS_GROUP_NAME, \
    GENERAL_PIXCAP_SHAPE, GLOBAL_FILTERS, DepletionData, handle_kafe2_advanced_options, handle_minuit_advanced_options, \
    PARASITIC_SUBTRACTION, get_base_group, handle_analysis_mix_up, HIST_CAP_UNIT, HIST_CURRENT_MEAS_UNIT, \
    HIST_BIAS_MEAS_UNIT, get_analysis_group, investigate_fit_convergence, TABLES_LEAF_COMPAT_TYPE
from pixcap65.utility.tables_util import get_groups, get_leaves, copy_node, list_attributes, group_get_file, \
    set_group_attribute, get_group_attribute, get_group_attributes, get_parent_group
from pixcap65.utility.utils_2 import walk_to_node, GroupType, create_carray, prevent_group_mix_up
from plotting import CAPACITANCE_CONVERSION_FACTOR, evaluate_pixel_mask

UNITS_ATTRIBUTE_KEY = "Units"

logger = logging.getLogger(__name__)


@deprecated("Please use analyze_data instead.")
def analyze_data_temporary_replacement(raw_data, base_path=None, is_advanced=False, is_cv=False,
                                       first_boundaries: Optional[Tuple] = None,
                                       second_boundaries: Optional[Tuple] = None,
                                       is_inter_pixel=False, **kwargs):
    """
    analyze_data

    Implementation of the analysis strategy for the capacitance measurement of a pixel sensor.
    But keep in mind that this function serves as a wrapper to handle file access and modification around
    the actual analysis implementation.
    The capacitances of the pixels are measured and investigated individually. For determination of the
    capacitance values either a linear fit or non-linear least square fit algorithms are used.
    Depending on the choice of the ´is_advanced` parameter non-linear techniques are used.
    In this case either 'kafe2' or 'iminuit' are used for the least-squares minimiztation depending on the choice
    of parameters.
    When using the advanced least-squares procedure the fit results will be plotted to verify the convergence of the
    fit.
    Thus, it is possible to use this wrapper to handle the pdf file to save fit-plot figures to instead of doing this
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
    Currently the C-V characterization for the inter-pixel measurements is not implemented yet.


    :param raw_data: path to the hdf file containing the raw data.
    :param base_path: path to the base group to look for the data.
    :param is_advanced: boolean indicating if the advanced analysis strategy should be used
        or not (may require additional keyword arguments)
    :param is_cv: boolean, indicates whether this is a C-V characterization.
    :param first_boundaries: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will only be estimated if this argument is provided.
    :param second_boundaries: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will be estimated if this argument is provided.
    :param is_inter_pixel: boolean, False, indicating whether the measurement to be analysed is a inter-pixel
        capacitance measurement. In this case more fits will be applied, adjusted to the specific structure of this
        problem
    :key use_kafe2: boolean, indicates whether kafe2 is used for the fit. (Only used for the advanced procedure)
    :key plot: boolean, indicates whether to plot the data. A output pdf object could be submitted here
         instead of an explicitly created one. (Only used for the advanced procedure)
    :key apply_contour: boolean, indicates whether to determine the contours and try to plot them.
        (Only used for the advanced procedure)
    :key full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used. (Only used for the advanced procedure)
    :key apply_correction: boolean, False, indicates whether the measured capacitances should be
        corrected immediately;
        Will require the presence of further arguments as information about the parasitics needs to be submitted.
    :key use_corrected: boolean, False, indicates whether to use the corrected capacitances for the depletion
        analysis.
    :key fit_plot_pdf_name:  Name of the pdf file to save fitting figures from the advanced procedures to.
        (Only used for the advanced procedure)
    :key bare_file: hdf file containing the measurements and investigation of a bare pixcap sample to obtain
        information about intrinisc and parasitic capacitances. (Only required for the correction procedure, but in
        this case it must be present)
    :key bare_path: hdf files hierarchy path to the group containing the bare pixcap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present)
    :key chip_group: HDF files hierarchy group with the data/specifications of the pixels on the current sensor.
        (Will only be usd if the depletion behaviour is investigated)
    """
    analyze_data(raw_data, base_path, is_advanced, is_cv, first_boundaries, second_boundaries, is_inter_pixel, **kwargs)


def analyze_data(raw_data, base_path=None, is_advanced=False, is_cv=False,
                 first_boundaries: Optional[Tuple] = None,
                 second_boundaries: Optional[Tuple] = None,
                 is_inter_pixel=False, **kwargs):
    """
    analyze_data

    Implementation of the analysis strategy for the capacitance measurement of a pixel sensor.
    But keep in mind that this function serves as a wrapper to handle file access and modification around
    the actual analysis implementation.
    The capacitances of the pixels are measured and investigated individually. For determination of the
    capacitance values either a linear fit or non-linear least square fit algorithms are used.
    Depending on the choice of the ´is_advanced` parameter non-linear techniques are used.
    In this case either 'kafe2' or 'iminuit' are used for the least-squares minimiztation depending on the choice
    of parameters.
    When using the advanced least-squares procedure the fit results will be plotted to verify the convergence of the
    fit.
    Thus, it is possible to use this wrapper to handle the pdf file to save fit-plot figures to instead of doing this
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
    Currently the C-V characterization for the inter-pixel measurements is not implemented yet.


    :param raw_data: path to the hdf file containing the raw data.
    :param base_path: path to the base group to look for the data.
    :param is_advanced: boolean indicating if the advanced analysis strategy should be used
        or not (may require additional keyword arguments)
    :param is_cv: boolean, indicates whether this is a C-V characterization.
    :param first_boundaries: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will only be estimated if this argument is provided.
    :param second_boundaries: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will be estimated if this argument is provided.
    :param is_inter_pixel: boolean, False, indicating whether the measurement to be analysed is a inter-pixel
        capacitance measurement. In this case more fits will be applied, adjusted to the specific structure of this
        problem
    :key use_kafe2: boolean, indicates whether kafe2 is used for the fit. (Only used for the advanced procedure)
    :key plot: boolean, indicates whether to plot the data. A output pdf object could be submitted here
         instead of an explicitly created one. (Only used for the advanced procedure)
    :key apply_contour: boolean, indicates whether to determine the contours and try to plot them.
        (Only used for the advanced procedure)
    :key full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used. (Only used for the advanced procedure)
    :key apply_correction: boolean, False, indicates whether the measured capacitances should be
        corrected immediately;
        Will require the presence of further arguments as information about the parasitics needs to be submitted.
    :key use_corrected: boolean, False, indicates whether to use the corrected capacitances for the depletion
        analysis.
    :key fit_plot_pdf_name:  Name of the pdf file to save fitting figures from the advanced procedures to.
        (Only used for the advanced procedure)
    :key bare_file: hdf file containing the measurements and investigation of a bare pixcap sample to obtain
        information about intrinisc and parasitic capacitances. (Only required for the correction procedure, but in
        this case it must be present)
    :key bare_hdf_path: hdf files hierarchy path to the group containing the bare pixcap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present)
    :key chip_group_name: HDF files hierarchy group with the data/specifications of the pixels on the current sensor.
        (Will only be usd if the depletion behaviour is investigated)
    """
    # handle the additonal pdf file in case of plotting enabled
    fit_plot_pdf_name = kwargs.pop('fit_plot_pdf_name', None)
    if "plot" in kwargs and kwargs["plot"] and fit_plot_pdf_name is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        assert "fit_plot_pdf_name" not in kwargs
        with PdfPages(fit_plot_pdf_name) as pdf:
            kwargs['fit_plot_pdf'] = pdf
            return analyze_data(raw_data, base_path, is_advanced, is_cv, first_boundaries,
                                second_boundaries, is_inter_pixel, **kwargs)

    # handle the real analysis
    with tb.open_file(raw_data, mode='a') as in_file_h5:
        # extract the hdf file groups to perform the analysis on.
        base_group = get_base_group(base_path, in_file_h5)

        # special handling for C-V characterization.
        # /biasing/analysis_correction/DepletionWidth
        # /biasing/analysis_correction/DepletionWidth
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

            # no progressbar as the overhead for this is much too large in must cases.
            for k, bias_voltage in enumerate(base_group.biasing.measurements.BiasVoltageHist):
                bias_name = "bias_{volt}_V".format(volt=bias_voltage).replace('-', "M_").replace(".", "__")
                data_group = base_group.biasing.measurements[bias_name]
                ana_group = walk_to_node(base_group.biasing,
                                         str_join("/", ANALYSIS_GROUP_NAME, bias_name), create=True)
                assert isinstance(ana_group, tb.Group)
                analysis_data_handle(in_file_h5, data_group, ana_group, is_advanced=is_advanced,
                                     is_inter_pixel=is_inter_pixel, **kwargs)

                # extract the capacitance data for tabular value; will also need coreected data.
                cap_data = ana_group.HistCap[:]
                cap_error_data = ana_group.HistCapErr[:]
                cv_data[:, :, k] = cap_data[:, :]
                cv_err_data[:, :, k] = cap_error_data[:, :]

            # in case of active correction, also the summary capacitance tables needs to be corrected, but it should
            # also a uncorrected table present.
            if apply_correction_arg:
                # perform the transfer
                apply_correction_simple(bare_file_arg, bare_path_arg, base_group.biasing.analysis, )

                for k, bias_voltage in enumerate(base_group.biasing.measurements.BiasVoltageHist):
                    ana_group_correction = walk_to_node(base_group.biasing,
                                                        str_join("/", ANALYSIS_CORRECTED_GROUP_NAME, bias_name),
                                                        create=True)
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
                    chip_spec_group = walk_to_node(in_file_h5.root, chip_name, create=False)
                    kwargs['chip_group'] = chip_spec_group

                # Anyway it is necessary to perform this analysis also on a sensor-average basis!
                if kwargs.pop("use_corrected", False):
                    analyze_depletion_delegate(base_group.biasing.measurements,
                                               base_group.biasing.analysis,
                                               first_boundaries, second_boundaries, **kwargs)
                analyze_depletion_delegate(base_group.biasing.measurements, dep_ana_group,
                                           first_boundaries, second_boundaries, **kwargs)
        else:
            if is_inter_pixel:
                reference_group = base_group.inter_cap
            else:
                reference_group = base_group.total_cap

            handle_analysis_mix_up(reference_group)
            ana_group = walk_to_node(reference_group, "analysis", create=True)
            assert isinstance(ana_group, tb.Group)

            analysis_data_handle(in_file_h5, reference_group.measurements, ana_group,
                                 is_advanced=is_advanced,
                                 is_inter_pixel=is_inter_pixel, **kwargs)


@deprecated("Please use analsis_data_handle instead.")
def analysis_data_handle_temporary_replacement(file: tb.File, data_group: GroupType, result_group: GroupType,
                                               is_advanced=False, is_inter_pixel=False, **kwargs):
    """
    advanced_analysis_data_handle

    Implementation of the analysis strategy for the capacitance measurement of a pixel sensor.
    The capacitances of the pixels are measured and investigated individually. For determination of the
    capacitance values either a linear fit or non-linear least square fit algorithms are used.
    Depending on the choice of the ´is_advanced` parameter non-linear techniques are used.
    In this case either 'kafe2' or 'iminuit' are used for the least-squares minimiztation depending on the choice
    of parameters.
    When using the advanced least-squares procedure the fit results will be plotted to verify the convergence of the
    fit.

    Afterwards, it is possible to directly correct the results for the capacitance by the connection and the
    measurement circuit.

    In Addition, there is the special case of an inter-pixel capacitance measurement.
    If the data to be analyzed comes from such a measurement this needs to be specified.
    Thus, in this case it will be checked which of the total current or the two inter-pix current data sets exist.
    The analysis will be done for each existing data set.
    Combinations with a C-V-Characterization might still be an issue.

    :param file: h5 file object containing the data to be analysed.
    :param data_group: hierachy group of the opend hdf file containing the raw data (measurements).
    :param result_group: hierachy group of the opened hdf file to write the analysis results to.
    :param is_advanced: boolean indicating if the advanced analysis strategy should be used or not (may require additional keyword arguments)
    :param is_inter_pixel: boolean, False, indicating whether the measurement to be analysed is a inter-pixel
        capacitance measurement. In this case more fits will be applied, adjusted to the specific structure of this
        problem
    :key full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used. (Only used for the advanced procedure)
    :key use_kafe2: boolean, indicates whether kafe2 is used for the fit. (Only used for the advanced procedure)
    :key plot: boolean, indicates whether to plot the data. A output pdf object could be submitted here
         instead of an explicitly created one. (Only used for the advanced procedure)
    :key apply_contour: boolean, indicates whether to determine the contours and try to plot them.
        (Only used for the advanced procedure)
    :key fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure)
    :key apply_correction: boolean, False, indicates whether the measured capacitances should be
        corrected immediately;
        Will require the presence of further arguments as information about the parasitics needs to be submitted.
    :key bare_file: hdf file containing the measurements and investigation of a bare pixcap sample to obtain
        information about intrinisc and parasitic capacitances. (Only required for the correction procedure, but in
        this case it must be present)
    :key bare_path: hdf files hierarchy path to the group containing the bare pixcap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present)
    """
    analysis_data_handle(file, data_group, result_group, is_advanced, is_inter_pixel, **kwargs)


def analysis_data_handle(file: tb.File, data_group: GroupType, result_group: GroupType,
                         is_advanced=False, is_inter_pixel=False, **kwargs):
    """
    advanced_analysis_data_handle

    Implementation of the analysis strategy for the capacitance measurement of a pixel sensor.
    The capacitances of the pixels are measured and investigated individually. For determination of the
    capacitance values either a linear fit or non-linear least square fit algorithms are used.
    Depending on the choice of the ´is_advanced` parameter non-linear techniques are used.
    In this case either 'kafe2' or 'iminuit' are used for the least-squares minimiztation depending on the choice
    of parameters.
    When using the advanced least-squares procedure the fit results will be plotted to verify the convergence of the
    fit.

    Afterwards, it is possible to directly correct the results for the capacitance by the connection and the
    measurement circuit.

    In Addition, there is the special case of an inter-pixel capacitance measurement.
    If the data to be analyzed comes from such a measurement this needs to be specified.
    Thus, in this case it will be checked which of the total current or the two inter-pix current data sets exist.
    The analysis will be done for each existing data set.
    Combinations with a C-V-Characterization might still be an issue.

    :param file: h5 file object containing the data to be analysed.
    :param data_group: hierachy group of the opend hdf file containing the raw data (measurements).
    :param result_group: hierachy group of the opened hdf file to write the analysis results to.
    :param is_advanced: boolean indicating if the advanced analysis strategy should be used or not (may require additional keyword arguments)
    :param is_inter_pixel: boolean, False, indicating whether the measurement to be analysed is a inter-pixel
        capacitance measurement. In this case more fits will be applied, adjusted to the specific structure of this
        problem
    :key full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used. (Only used for the advanced procedure)
    :key use_kafe2: boolean, indicates whether kafe2 is used for the fit. (Only used for the advanced procedure)
    :key plot: boolean, indicates whether to plot the data. A output pdf object could be submitted here
         instead of an explicitly created one. (Only used for the advanced procedure)
    :key apply_contour: boolean, indicates whether to determine the contours and try to plot them.
        (Only used for the advanced procedure)
    :key fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure)
    :key apply_correction: boolean, False, indicates whether the measured capacitances should be
        corrected immediately;
        Will require the presence of further arguments as information about the parasitics needs to be submitted.
    :key bare_file: hdf file containing the measurements and investigation of a bare pixcap sample to obtain
        information about intrinisc and parasitic capacitances. (Only required for the correction procedure, but in
        this case it must be present)
    :key bare_path: hdf files hierarchy path to the group containing the bare pixcap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present)
    """
    # get the correct analysis function
    if is_advanced:
        from advanced_analysis import advanced_analysis_delegate
        perform_analysis = advanced_analysis_delegate
    else:
        from analysis_util import analyze_data_delegate
        perform_analysis = analyze_data_delegate

    # Read pixel map
    assert isinstance(data_group, tb.Group)
    assert isinstance(result_group, tb.Group)
    if is_inter_pixel:
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
    if "apply_correction" in kwargs and kwargs["apply_correction"]:
        assert 'bare_file' in kwargs
        assert 'bare_hdf_path' in kwargs
        apply_correction_simple(kwargs['bare_file'], kwargs['bare_hdf_path'], result_group)


class DepletionDataStore:
    def store_data(self, key, data):
        pass

    def flush_data(self):
        pass


class DepletionTableStore(DepletionDataStore):
    def __init__(self, table: tb.Table):
        self.table = table
        self.entry = self.table.row

    def store_data(self, key, data):
        match key:
            case "depletion":
                self.entry["Ubi"] = data
            case "depletion_error":
                self.entry["Ubi_error"] = data
            case "fit_result_first":
                self.entry["a"] = data[0]
                self.entry["b"] = data[1]
            case "fit_error_first":
                self.entry["a_error"] = data[1]
                self.entry["b_error"] = data[2]
            case "fit_result_second":
                self.entry["c"] = data[0]
                self.entry["d"] = data[1]
            case "fit_error_second":
                self.entry["c_error"] = data[0]
                self.entry["d_error"] = data[1]
            case _:
                raise ValueError(f"The provided storage key is unknown: {key}")

    def flush_data(self):
        self.entry.append()


class DepletionArrayStore(DepletionDataStore):
    def __init__(self):
        self.depletion_voltage = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
        self.depletion_error = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
        self.fit_parameter_estimators = np.full(shape=(40, 40, 4), fill_value=np.nan)
        self.fit_parameter_errors = np.full(shape=(40, 40, 4), fill_value=np.nan)
        self.pixel_row = 1
        self.pixel_col = 1

    def set_pixel(self, i_row, i_col):
        self.pixel_row = i_row
        self.pixel_col = i_col

    def store_data(self, key, data):
        match key:
            case "depletion":
                self.depletion_voltage[self.pixel_col, self.pixel_row] = data
            case "depletion_error":
                self.depletion_error[self.pixel_col, self.pixel_row] = data
            case "fit_result_first":
                self.fit_parameter_estimators[self.pixel_col, self.pixel_row, :2] = data
            case "fit_error_first":
                self.fit_parameter_errors[self.pixel_col, self.pixel_row, :2] = data
            case "fit_result_second":
                self.fit_parameter_estimators[self.pixel_col, self.pixel_row, 2:] = data
            case "fit_error_second":
                self.fit_parameter_errors[self.pixel_col, self.pixel_row, 2:] = data
            case _:
                raise ValueError(f"The provided storage key is unknown: {key}")


class DopingArrayStore(DepletionDataStore):
    def __init__(self, doping_shape):
        self.pixel_row = 1
        self.pixel_col = 1
        self.depletion_width_plate = np.full(shape=doping_shape, fill_value=np.nan)
        self.depletion_width_plate_error = np.full(shape=doping_shape, fill_value=np.nan)
        self.depletion_fit_parameter_table = np.full((40, 40, 4), fill_value=np.nan)
        self.depletion_fit_parameter_error_table = np.full((40, 40, 4), fill_value=np.nan)
        self.depletion_fit_covariance_table = np.full((40, 40, 4, 4), fill_value=np.nan)
        self.effective_doping_table = np.full(shape=doping_shape, fill_value=np.nan)

    def set_pixel(self, i_row, i_col):
        self.pixel_row = i_row
        self.pixel_col = i_col

    def store_data(self, key, data):
        match key:
            case "width":
                self.depletion_width_plate[self.pixel_col, self.pixel_row] = data
            case "width_error":
                self.depletion_width_plate_error[self.pixel_col, self.pixel_row] = data
            case "fit_parameters":
                self.depletion_fit_parameter_table[self.pixel_col, self.pixel_row] = data
            case "fit_parameters_error":
                self.depletion_fit_parameter_error_table[self.pixel_col, self.pixel_row] = data
            case "fit_covariance":
                self.depletion_fit_covariance_table[self.pixel_col, self.pixel_row] = data
            case "doping":
                self.effective_doping_table[self.pixel_col, self.pixel_row] = data
            case _:
                raise ValueError(f"Unknown data storage key {key}")


def analyze_depletion_delegate(data_group: GroupType, analysis_group: GroupType, first_boundaries: Optional[Tuple],
                               second_boundaries: Optional[Tuple], chip_group: GroupType = None, apply_doping=False,
                               **kwargs):
    """
    analyze_depletion_delegate

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

    Second, the depletion width and it's consequences are analyzed if requested.
    This part will only be performed if it is requested by the 'apply_doping' parameter.
    This second analysis also requires the presence of the 'chip_group' parameter and within it the table/array
    'PhysicalDimensions' with he physical dimensions of the pixels on the module. Otherwise, it is not possible
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
    :key plot: boolean, False, indicates whether to plot the data. A output pdf object could be submitted here
         instead of an explicitly created one.
    :key apply_contour: boolean, indicates whether to determine the contours and try to plot them.
    :key fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure)
    :key verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    """
    # extract the additional parameters for advanced fitting procedures
    use_kafe2 = kwargs.pop("use_kafe2", False)
    apply_contours = kwargs.pop("apply_contours", False)
    plot = kwargs.pop("plot", False)
    output_pdf = kwargs.pop("fit_plot_pdf", None)

    # extract the required data and create arrays for temporary storage.
    voltage_data = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
    first_lower, first_upper = first_boundaries
    second_lower, second_upper = second_boundaries
    fit_result_storage = DepletionArrayStore()

    # verify and extract the raw data for further analysis
    check_leaf_unit(analysis_group.UCHist, HIST_CAP_UNIT)
    check_leaf_unit(analysis_group.UCErrHist, HIST_CAP_UNIT)
    cap_data = analysis_group.UCHist[:, :, :]
    cap_error_data = analysis_group.UCErrHist[:, :, :]
    for ii, jj in np.ndindex(GENERAL_PIXCAP_SHAPE):
        pixel_cap_data = cap_data[ii, jj, :]
        pixel_cap_error_data = cap_error_data[ii, jj, :]
        # could only perform the analysis for pixels with trustable measurements.
        if np.any(~np.isfinite(pixel_cap_data)):
            continue
        fit_result_storage.set_pixel(jj, ii)
        analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, pixel_cap_error_data, second_lower,
                                second_upper, voltage_data, fit_result_storage, use_kafe2=use_kafe2,
                                apply_contours=apply_contours, plot=plot, fit_plot_pdf=output_pdf,
                                fit_description_text=" for Pixel ({col},{row})".format(col=ii, row=jj),
                                verbose=kwargs.get("verbose", False))

    # save the depletion voltage data.
    assert isinstance(analysis_group, tb.Group)
    file_h5 = group_get_file(analysis_group)
    assert isinstance(file_h5, tb.File)
    assert isinstance(analysis_group, tb.Group)  # necessary as only groups could contain subelements.
    create_carray(file_h5, where=analysis_group, name="DepletionHist",
                  title="Histogram of the depletion voltages", obj=fit_result_storage.depletion_voltage,
                  filters=GLOBAL_FILTERS, unit=HIST_BIAS_MEAS_UNIT)
    create_carray(file_h5, where=analysis_group, name="DepletionErrHist",
                  title="Histogram of the depletion voltage erors", obj=fit_result_storage.depletion_error,
                  filters=GLOBAL_FILTERS, unit=HIST_BIAS_MEAS_UNIT)
    create_carray(file_h5, where=analysis_group, name="DepFitParamHist",
                  title="Histogram of the depletion voltages fit parameters",
                  obj=fit_result_storage.fit_parameter_estimators,
                  filters=GLOBAL_FILTERS, unit="NONE")
    create_carray(file_h5, where=analysis_group, name="DepFitParamErrHist",
                  title="Histogram of the depletion voltages fit parameter errors",
                  obj=fit_result_storage.fit_parameter_errors,
                  filters=GLOBAL_FILTERS, unit="NONE")

    if apply_doping and chip_group is not None and "PhysicalDimensions" in chip_group and chip_group.PhysicalDimensions.shape == (
            40, 40, 2) and np.all(np.isfinite(chip_group.PhysicalDimensions[:])):
        physical_dimensions_data = chip_group.PhysicalDimensions[:]
        pixel_areas = np.prod(physical_dimensions_data, axis=2)
        doping_shape = (40, 40, voltage_data.shape[0])
        doping_result_storage = DopingArrayStore(doping_shape)

        # some further definitions for the loop
        from findiff import Diff
        import scipy.constants as constants
        # this values will not be correct as I don't known the doping concentrationor the intrinisc bias voltage;
        # the intrinisc bias voltage could be estimated from a fit to the forward bias I-V characteristic.
        NA, ND = 1e16, 1e16
        V_bi = SILICON_V_BIAS
        assert isinstance(analysis_group, tb.Group)
        dep_table = file_h5.create_table(where=analysis_group, name="DepletionParamTable", description=DepletionData,
                                         filters=GLOBAL_FILTERS)
        entry = dep_table.row
        bias_voltages = check_leaf_unit(data_group.BiasVoltageHist, HIST_BIAS_MEAS_UNIT)
        bias_voltage_errors = np.full_like(bias_voltages, fill_value=np.nan)

        for col, row in np.ndindex(GENERAL_PIXCAP_SHAPE):
            kwargs['fit_description_text'] = ' for Pixel ({col},{row})'.format(col=col, row=row)
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

            Neff, depletion_width_data, pos_min = analyze_doping_profile(bias_voltages, bias_voltage_errors,
                                                                         doping_result_storage, entry, NA / 2, V_bi,
                                                                         pixel_area, pixel_cap_data,
                                                                         pixel_cap_error_data,
                                                                         use_kafe2=use_kafe2,
                                                                         plot=plot, apply_contours=apply_contours,
                                                                         fit_plot_pdf=output_pdf, **kwargs)

            if kwargs.get("verbose", False):
                print(
                    f"The minimum concentration is {Neff[pos_min]} and depth {depletion_width_data[pos_min]} for pixel ({col}, {row})")

        # save the computed information about the depletion behaviour
        assert isinstance(analysis_group, tb.Group)
        create_carray(file_h5, where=analysis_group, name="DepletionWidth",
                      title="Depletion width from the pixel capacitances",
                      filters=GLOBAL_FILTERS, obj=doping_result_storage.depletion_width_plate, unit="um")
        create_carray(file_h5, where=analysis_group, name="DepletionWidthErr",
                      title="Uncertainty of the depletion width from the pixel capacitances",
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


def analyze_doping_profile(bias_voltages: TABLES_LEAF_COMPAT_TYPE,
                           bias_voltage_errors: np.ndarray,
                           doping_result_storage: DopingArrayStore, entry, NA: float, V_bi: float,
                           pixel_area, pixel_cap_data, pixel_cap_error_data, **kwargs) -> tuple[
    np.ndarray, np.ndarray, int]:
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
    depletion_width_data = (constants.epsilon_0 * pixel_area) / (
        np.array(pixel_cap_data)) * 1.e-6  # provides the width in um
    depletion_width_error_data = (constants.epsilon_0 * pixel_area * effective_cap_errors) / (np.array(
        pixel_cap_data) ** 2) * 1.e-6  # provides the width in um

    # fit the theoretical expected depletion width to determine some of the properties of the pixel diode
    parameter_guess = {
        "NAD": NA,
        "V": V_bi,
        "dep": -10,
        "sat": np.max(depletion_width_data),
    }
    # this model fits may fail due to two indistinguishable parameters.
    if use_kafe2:
        from kafe2 import XYFit, XYContainer
        assert isinstance(bias_voltages, np.ndarray) or isinstance(bias_voltages, tb.CArray)
        xy_data = XYContainer(x_data=bias_voltages, y_data=depletion_width_data)
        xy_data.add_error(axis='y', err_val=depletion_width_error_data)
        if np.all(bias_voltage_errors[np.isfinite(bias_voltages)]):
            xy_data.add_error(axis='x', err_val=bias_voltage_errors)

        fitter = XYFit(xy_data, model_function=model_depletion, minimizer="iminuit")
        fitter.assign_parameter_latex_names(voltages="U_\\text{{bi}}", NA="N_\\text{{A}}", ND="N_\\text{{D}}",
                                            V="U_\\text{{th}}", )
        fitter.assign_model_function_latex_name("d_\\text{{depletion}}")
        fitter.assign_model_function_latex_expression(
            "\\sqrt{{\\frac{{2\\epsilon_0\\epsilon}}{{e}}\\cdot\\frac{{{NA}+{ND}}}{{{NA}\\cdot{ND}}}\\cdot ({V}+{voltages})}}")
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
    Neff = effective_doping(pixel_cap_data, -bias_voltages,
                            diode_area=pixel_area)  # -> 10^(12) 1/m
    doping_result_storage.store_data("doping", Neff)
    pos_min = np.argmin(Neff)
    for key, value in depletion_fit_propagate_parameters.items():
        entry[key] = value
    entry.append()
    return Neff, depletion_width_data, pos_min


def analyze_pixel_depletion(first_lower, first_upper,
                            pixel_cap_data: np.ndarray,
                            pixel_cap_error_data: np.ndarray, second_lower, second_upper,
                            voltage_data: TABLES_LEAF_COMPAT_TYPE,
                            result: DepletionDataStore, **kwargs):
    """

    :param first_upper: upper limit of the first fit range for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel.
    :param first_lower: lower limit of the first fit range for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel.
    :param second_upper: upper limit of the second fit range for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel.
    :param second_lower: lower limit of the second fit range for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel.
    :param pixel_cap_data: capacitance data for one particular pixel on the sensor
    :param pixel_cap_error_data: uncertainties of the capacitance data for one particular pixel on the sensor.
    :param voltage_data: array of the biasing HV voltages (with the correct sign)
    :param result: data store container to write the results back
    :param kwargs: further keyword arguments for fitting and output.
    :key verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    """
    # extract the additional parameters for advanced fitting procedures
    use_kafe2 = kwargs.pop("use_kafe2", False)
    apply_contours = kwargs.pop("apply_contours", False)
    plot = kwargs.pop("plot", False)
    output_pdf = kwargs.pop("fit_plot_pdf", None)
    fit_description_text = kwargs.pop("fit_description_text", "")
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
        pixel_cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 3 * pixel_cap_error_data if np.all(np.isfinite(
        pixel_cap_error_data)) else pixel_cap_error_data
    first_voltage_data = voltage_data[first_section_mask]
    second_voltage_data = voltage_data[second_section_mask]
    first_cap_data = effective_capacitance_data[first_section_mask]
    second_cap_data = effective_capacitance_data[second_section_mask]
    first_cap_error_data = effective_capacitance_error_data[first_section_mask]
    second_cap_error_data = effective_capacitance_error_data[second_section_mask]

    # perform fits to the two boundary regions specified to estimate the two distinc behaviours.
    if np.all(np.isfinite(first_cap_error_data)):
        if use_kafe2:
            # perform the fit
            from kafe2 import XYContainer, XYFit
            data_container = XYContainer(first_voltage_data, first_cap_data)
            data_container.add_error(axis='y', err_val=first_cap_error_data)
            m = XYFit(data_container, model_function=depletion_model)
            m.do_fit()

            # extract the fit parameters
            assert m.did_fit
            first_dep_parameters = np.array([m.parameter_values[0], m.parameter_values[1]])
            first_dep_errors = np.array([m.parameter_errors[0], m.parameter_errors[1]])
            first_dep_cov = m.parameter_cov_mat
            if plot:
                handle_kafe2_advanced_options(m, apply_contours, "U in V", "\\frac{{1}}{{C^2}}",
                                              "First Fit{}".format(fit_description_text),
                                              output_pdf,
                                              "First Contour{}".format(fit_description_text))

        else:
            # perform the fit
            from iminuit import Minuit
            # noinspection PyProtectedMember
            from iminuit.cost import LeastSquares, Model
            assert isinstance(depletion_model, Model)
            cost = LeastSquares(first_voltage_data, first_cap_data, first_cap_error_data, depletion_model)
            m = Minuit(cost, a=1, b=0)
            m.migrad()
            m.hesse()

            # extract the parameters
            first_dep_parameters = np.array([m.values['a'], m.values['b']])
            first_dep_errors = np.array([m.errors['a'], m.errors['b']])
            first_dep_cov = m.covariance
            if plot:
                handle_minuit_advanced_options(m, apply_contours, "$U$ in \\unit{{\\volt}}", "$\\frac{{1}}{{C^2}}$",
                                               "First Fit{}".format(fit_description_text),
                                               output_pdf,
                                               "First Contour{}".format(fit_description_text))
    else:
        first_result = np.polyfit(first_voltage_data, first_cap_data, deg=1, cov=True)

        # extract the parameters and fit results.
        first_dep_parameters = np.asarray(first_result[0])
        first_dep_cov = np.asarray(first_result[1])
        first_dep_errors = np.sqrt(np.diag(first_dep_cov))

    if np.all(np.isfinite(second_cap_error_data)):
        if use_kafe2:
            # perform the fit
            from kafe2 import XYContainer, XYFit
            data_container = XYContainer(second_voltage_data, second_cap_data)
            data_container.add_error(axis='y', err_val=second_cap_error_data)
            m = XYFit(data_container, model_function=depletion_model)
            m.do_fit()

            # extract the fit parameters
            assert m.did_fit
            second_dep_parameters = np.array([m.parameter_values[0], m.parameter_values[1]])
            second_dep_errors = np.array([m.parameter_errors[0], m.parameter_errors[1]])
            second_dep_cov = m.parameter_cov_mat
            if plot:
                handle_kafe2_advanced_options(m, apply_contours, "U in V", "\\frac{{1}}{{C^2}}",
                                              "Second Fit{}".format(fit_description_text),
                                              output_pdf,
                                              "Second Contour{}".format(fit_description_text))

        else:
            # perform the fit
            from iminuit import Minuit
            # noinspection PyProtectedMember
            from iminuit.cost import LeastSquares, Model
            assert isinstance(depletion_model, Model)
            cost = LeastSquares(second_voltage_data, second_cap_data, second_cap_error_data, depletion_model)
            m = Minuit(cost, a=1, b=0)
            m.migrad()
            m.hesse()

            # extract the parameters
            second_dep_parameters = np.array([m.values['a'], m.values['b']])
            second_dep_errors = np.array([m.errors['a'], m.errors['b']])
            second_dep_cov = m.covariance
            if plot:
                handle_minuit_advanced_options(m, apply_contours, "$U$ in \\unit{{\\volt}}", "$\\frac{{1}}{{C^2}}$",
                                               "Second Fit{}".format(fit_description_text),
                                               output_pdf,
                                               "Second Contour{}".format(fit_description_text))
        # MARK: What about an analysis of the fits convergence properties.
    else:
        second_result = np.polyfit(second_voltage_data, second_cap_data, deg=1, cov=True)

        # extract the parameters and fit results.
        second_dep_parameters = np.asarray(second_result[0])
        second_dep_cov = np.asarray(second_result[1])
        second_dep_errors = np.sqrt(np.diag(second_dep_cov))

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
    full_cov = np.full((4, 4), fill_value=0)
    full_cov_2 = np.zeros((4, 4))  # cross correlations between the two fits are not known
    full_cov[:2, :2] = first_dep_cov
    full_cov[2:, 2:] = second_dep_cov
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


def effective_doping(capacitances, bias_voltages, diode_area=None) -> np.ndarray:
    """
    effective doping

    Helper function to calculate the effective doping for every bias voltage/depletion depth.

    :param capacitances: measured (and corrected) capacitances of the CV characterization
    :param bias_voltages: bias voltages corresponding to the provided capacitances
    :param diode_area: area of the individual pixel.
    :return: effective doping concentration in 10^(12) 1/m (maybe still an issue with the units)
    """
    from scipy import constants
    from findiff import Diff

    # make sure the data is provided as numpy arrays
    capacitances = np.asarray(capacitances)
    bias_voltages = np.asarray(bias_voltages)

    if diode_area is None:
        diode_area = 50 * 50  # measured in um^2
    temp_capacitances = np.reciprocal(capacitances ** 2)

    # noinspection PyTypeChecker
    d_du = Diff(0, bias_voltages)
    derivative = d_du(temp_capacitances)
    Neff = 2 / (constants.elementary_charge * constants.epsilon_0 * EPS_SILICON * (diode_area ** 2) * np.array(
        derivative)) * 1e6
    return Neff


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
        should be analysed.
    :param base_path: hdf files group witht the measurement data.
    :key fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure)
    :key output_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure)
    :key fit_plot_pdf_name:  Name of the pdf file to save fitting figures from the advanced procedures to.
        (Only used for the advanced procedure)
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key exclude_cap_hist: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    """
    temp_plot_name = kwargs.pop("fit_plot_pdf_name", None)
    if temp_plot_name is not None:
        assert "fit_plot_pdf_name" not in kwargs
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(temp_plot_name) as pdf:
            return analyze_capacitance_distribution(raw_data, base_path, corrected_distribution,
                                                    fit_plot_pdf=pdf, **kwargs)
    else:
        output_pdf = kwargs.pop("output_pdf", None)
        if output_pdf is None:
            output_pdf = kwargs.pop("fit_plot_pdf", None)
    with tb.open_file(raw_data, mode='a') as in_file_h5:
        base_group = get_base_group(base_path, in_file_h5)
        if corrected_distribution:
            analyze_capacitance_distribution_delegate(base_group.total_cap.analysis_correction, output_pdf, **kwargs)
        else:
            analyze_capacitance_distribution_delegate(base_group.total_cap.analysis, output_pdf, **kwargs)


def analyze_capacitance_distribution_delegate(analysis_group, output_pdf, **kwargs):
    """
    analyze_capacitance_distribution_delegate

    Implementation of the investigation in the distribution of the capacitances on the chip.
    It should predominantly be used to determine the intrinsic and parasitic capacitances of a bare pixcap chip.
    To achieve the distribution the data is first binned and presented into a histogram.
    Next, a gaussian shape is fitted to the histogram to match it's shape.

    Last the main results are written back to the analysis group as an attribute.

    :param analysis_group: hdf file's group where to find the capacitances to be analyzed.
    :param output_pdf: pdf object to write the created figures to for long-term saving.
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key exclude_cap_hist: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    :key capacitance: histogram of the capacitances
    """
    from plotting import CAPACITANCE_CONVERSION_FACTOR
    from plotting import DEFAULT_BIN_NUMBER
    from plotting import COUNTS_HIST_LABEL
    from plotting import HIST_PIX_CAP_LABEL
    from matplotlib import pyplot as plt

    # extract further arguments for the performance of the fitting
    use_kafe2 = kwargs.pop("use_kafe2", False)
    if output_pdf is None:
        output_pdf = kwargs.pop("fit_plot_pdf", None)

    # we want to fit a binned distribution; but some bins might be empty
    cap_hist = kwargs.pop("capacitance", None)
    if cap_hist is None or not isinstance(cap_hist, np.ndarray):
        cap_hist = check_leaf_unit(analysis_group.HistCap, HIST_CAP_UNIT)
    fig, ax = plt.subplots()
    hist_cap_hist = evaluate_pixel_mask(cap_hist, **kwargs)
    temp_hist_back_data = hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR
    hist_data, bins, _ = ax.hist(temp_hist_back_data,
                                 bins=kwargs.get("hist_bins", DEFAULT_BIN_NUMBER), density=False, )
    bin_positions = bins[:-1] + np.diff(bins)

    # now fit this to a gauss function
    initial_estimator = dict(u=np.mean(temp_hist_back_data), s=np.std(temp_hist_back_data))
    if use_kafe2:
        from kafe2 import Plot, HistContainer, HistFit
        data_container = HistContainer(bin_edges=bins, fill_data=temp_hist_back_data)
        fitter = HistFit(data=data_container, density=False, model_function=gauss_model)
        fitter.set_parameter_values(**initial_estimator)
        fitter.do_fit()
        assert fitter.did_fit
        mean_value = fitter.parameter_values[0]
        std_value = fitter.parameter_values[1]
        norm = fitter.parameter_values[2]
        if output_pdf is not None:
            handle_kafe2_advanced_options(fitter, True, "$C$ in \\unit{{\\femto\\farad}}", COUNTS_HIST_LABEL,
                                          "Capacitance distribution", output_pdf,
                                          "Contours for the capacitance distribution")
        fit_plot = Plot(fitter)
        fit_plot.plot()
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
        initial_estimator_2["b"] = 1
        cost = BinnedNLL(hist_data, bins, cdf=norm.cdf, name=('u', 's'))
        cost_2 = ExtendedBinnedNLL(hist_data, bins, scaled_cdf=extended_gauss_integral, name=("b", 'u', 's'))
        fitter = Minuit(cost_2, **initial_estimator_2)
        fitter.migrad()
        fitter.hesse()
        print("There are results for the extended fit")
        print(fitter.fmin)
        print(fitter.values)
        print(fitter.errors)
        if output_pdf is not None:
            handle_minuit_advanced_options(fitter, False, "$C$ in \\unit{{\\femto\\farad}}", COUNTS_HIST_LABEL,
                                           "Capacitance distribution (EXTENDED)", output_pdf,
                                           "Contours for the capacitance distribution (EXTENDED)")
        fitter = Minuit(cost, **initial_estimator)
        fitter.migrad()
        fitter.hesse()
        print("There are results for the standard fit")
        print(fitter.fmin)
        print(fitter.values)
        print(fitter.errors)
        fit_results = fitter.values
        fit_cov = fitter.covariance
        norm = np.sum(hist_data)
        mean_value = fitter.values["u"]
        std_value = fitter.values["s"]
        if output_pdf is not None:
            handle_minuit_advanced_options(fitter, True, "$C$ in \\unit{{\\femto\\farad}}", COUNTS_HIST_LABEL,
                                           "Capacitance distribution", output_pdf,
                                           "Contours for the capacitance distribution")

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
        y, ycov = propagate(lambda p: gauss_model(binning_span, p[0], p[1], normalisation), fit_results,
                            fit_cov)
        yerr_prop = np.diag(ycov) ** 0.5
        ax.fill_between(binning_span, y - yerr_prop, y + yerr_prop, facecolor="C1", alpha=0.5)

    except ImportError:
        pass
    # add some information about the model
    hypo_test = investigate_fit_convergence(fitter)
    #     ax.text(0.5, 0.5, f"""GoF = {hypo_test['x']:.2f}
    # ndf = {hypo_test['ndf']:.2f}
    # p = {hypo_test['p']:.2f}""", transform=ax.transAxes)
    ax.set_ylabel(COUNTS_HIST_LABEL)
    ax.set_xlabel(HIST_PIX_CAP_LABEL)
    ax.grid()
    ax.legend(title=f"GoF = {hypo_test['x']:.4f}\nndf = {hypo_test['ndf']: .4f}\np = {hypo_test['p']: .4f}")
    if output_pdf is None:
        plt.show()
    else:
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def apply_correction(raw_data, base_path=None, bare_data_path=None, bare_group=None):
    """
    apply_correction

    Wrapper function to handle the file access, when correcting the capacitance data tables.

    :param raw_data: hdf file containing the measurements and investigation of a pixcap sample which needs correction
    :param base_path: hdf files group with the analysis data/capacitance data to be corrected.
    :param bare_data_path: file containing the analysis of the bare pixcap sample to be used for capacitance correction.
    :param bare_group: hdf files group for the bare analysis (holding the parasitic capacitance information)
    """
    # first extract the parasitic capacitances
    assert bare_data_path is not None
    with tb.open_file(raw_data, mode='a') as in_file_h5_inner:
        base_group = get_base_group(base_path, in_file_h5_inner)
        apply_correction_simple(bare_data_path, bare_group, base_group.analysis)


def apply_correction_simple(bare_data_path: str, bare_path: str, analysis_group: tb.Group):
    """
    apply_correction_simple

    Minimal wrapper for the correction of the measured capacitance for the capacitance of the bump-bond and the
    switching circuit itself.
    It will open the data file and analysis group of the bare pixcap measurement, extract the parasitic capacitances
    (including their uncertainties which will be applied as systematic uncertainties to the measurement), correct for
    these effect and write all the results of the current analysis with the correction applied back.
    As the bump-bond capacitance and the intrinsic capacitance of the switching circuit can be considered as beeing
    parallel no further effects needs to be accounted for. Also these capacitances could be considered beeing parallel
    to the pixel sensors capacitances. So the correction could be done by a simple subtraction.


    :param bare_data_path: path to the hdf file of the bare pixcap measurement to obtain information about parasitic
        capacitances
    :param bare_path: hdf files hierarchy path to the group containing the bare pixcap analysis with the information
        about the parasitic after investigating the capacitance distribution.
    :param analysis_group: hdf files hierarchy group with the analysis results which needs to be corrected for the
        intrinsic and parasitic effectcs.
    """
    with tb.open_file(bare_data_path, mode='r') as in_file_h5:
        if bare_path is None:
            bare_group = in_file_h5.root
        else:
            bare_group = walk_to_node(in_file_h5.root, bare_path)

        assert "parasitic" in get_group_attributes(bare_group.analysis)
        assert "parasitic_error" in get_group_attributes(bare_group.analysis)
        # assert "parasitic" in bare_group.analysis._v_attrs
        # assert "parasitic_error" in bare_group.analysis._v_attrs
        parasitic = get_group_attribute(bare_group.analysis, "parasitic")
        parasitic_error = get_group_attribute(bare_group.analysis, "parasitic_error")
        # parasitic = bare_group.analysis._f_getattr("parasitic")
        # parasitic_error = bare_group.analysis._f_getattr("parasitic_error")

        # second perform correction of the capacitance data
        apply_correction_delegate(parasitic, parasitic_error, analysis_group)


def apply_correction_delegate(parasitic, parasitic_error, group: GroupType):
    """
    apply_correction_delegate

    Implementation of the capacitance correction for the parasitic capacitance of the bump-bond and the intrinsic
    capacitance of the switching circuit.
    As the bump-bond capacitance and the intrinsic capacitance of the switching circuit can be considered as beeing
    parallel no further effects needs to be accounted for. Also these capacitances could be considered beeing parallel
    to the pixel sensors capacitances. So the correction could be done by a simple subtraction.

    To make it possible to still plot all the data later on, the new tables get also the parasitic capacitance
    subtracted as an attribute.

    :param parasitic: parasitic capacitance to correct for.
    :param parasitic_error: (systematic) error of the parasitic capacitance to correct for.
    :param group: hdf files group containing the capacitance data to be corrected.
    """
    assert isinstance(group, tb.Group)
    # define the corrected analysis group
    prevent_group_mix_up(get_parent_group(group), "analysis_correction")
    # if "analysis_correction" in group._v_parent:
    #     group._v_parent.analysis_correction._f_remove(recursive=True)
    #     time.sleep(2)

    file_h5 = group_get_file(group)
    correction_group = file_h5.create_group(where=get_parent_group(group), name="analysis_correction")
    # correction_group = group._v_file.create_group(where=group._v_parent, name="analysis_correction")

    # first extract the capacitance histograms (errors shall be copied to the new group
    # for key, value in group._v_leaves.items():
    for key, value in get_leaves(group):
        if "Cap" in key:
            logger.debug("Found the key %s for transfering capacitance data.", key)
            if "Err" in key:
                copy_node(value, newparent=correction_group)
                # value._f_copy(newparent=correction_group)
            else:
                cap_data_hist = value[:]
                assert isinstance(cap_data_hist, np.ndarray)
                corrected_cap_data = np.where(np.isfinite(cap_data_hist), cap_data_hist - parasitic * 1.e-15, np.nan)
                corrected_cap_systemtatics = np.full_like(corrected_cap_data, fill_value=parasitic_error)
                temp_data_array = file_h5.create_carray(where=correction_group, name=key, title=value.title,
                                                        filters=value.filters, obj=corrected_cap_data)
                # careful the parasitics are provided in fF were all other capacitacnes are saved in F
                temp_data_array.attrs[PARASITIC_SUBTRACTION] = parasitic
                temp_error_array = file_h5.create_carray(where=correction_group, name="{}Systematics".format(key),
                                                         title=value.title,
                                                         filters=value.filters, obj=corrected_cap_systemtatics)
                # for name in value.attrs._f_list():
                for name in list_attributes(value):
                    temp_data_array.attrs[name] = value.attrs[name]
                    temp_error_array.attrs[name] = value.attrs[name]
        else:
            copy_node(value, newparent=correction_group)
            # value._f_copy(newparent=correction_group)

    # will need to handle also the subgroups (only one level down)!
    for group_key, next_group in get_groups(group):
        next_correction_group = file_h5.create_group(where=correction_group, name=group_key)
        for key, value in get_leaves(next_group):
            # And waht about tables here?
            assert isinstance(value, tb.Node)
            if "Cap" in key:
                if "Err" in key:
                    copy_node(value, newparent=next_correction_group)
                    # value._f_copy(newparent=next_correction_group)
                else:
                    cap_data_hist = value[:]
                    corrected_cap_data = cap_data_hist - parasitic
                    corrected_cap_systemtatics = np.full_like(corrected_cap_data, fill_value=parasitic_error)
                    temp_data_array = file_h5.create_carray(where=next_correction_group, name=key,
                                                            title=value.title,
                                                            filters=value.filters, obj=corrected_cap_data)
                    temp_error_array = file_h5.create_carray(where=next_correction_group,
                                                             name="{}Systematics".format(key), title=value.title,
                                                             filters=value.filters,
                                                             obj=corrected_cap_systemtatics)
                    # for name in value.attrs._f_list():
                    assert isinstance(value, tb.Leaf)
                    for name in list_attributes(value):
                        temp_data_array.attrs[name] = value.attrs[name]
                        temp_error_array.attrs[name] = value.attrs[name]
            else:
                copy_node(value, newparent=next_correction_group)
                # value._f_copy(newparent=next_correction_group)


if __name__ == '__main__':
    # analyze_data(raw_data='/home/silab/git/pixcap65/pixcap_full_data_image1.h5')
    # advanced_analysis(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1")
    # advanced_analysis(raw_data='Data/r13-measurement/R13_Initial_3_Scan.h5',base_path="ATLAS ITk/unbiased_1")
    # analyze_data(raw_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', is_cv=True, first_boundaries=(-100,-40), second_boundaries=(-10, 0),)
    from pixcap65.utility.homogenize_plots import set_params

    set_params(latex=True,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{retain-zero-uncertainty}")

    bare_correction_args = {
        "apply_correction": True,
        "bare_file": "Bare_Repeat_2_Scan.h5",
        "bare_hdf_path": "Reference/bare/unbiased_8/total_cap",
    }

    # analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
    #                                  corrected_distribution=False,
    #                                  exclude_test_cap=True, use_kafe2=False, fit_plot_pdf_name="Bare_analysis_parasitic.pdf")
    analyze_data(raw_data='R13-Interpixel_Scan.h5', base_path="Reference/R13/demo_measurement_52_biased_80_V_1_charge",
                 is_inter_pixel=True, is_advanced=True)
    # analyze_data(raw_data='Data/r13-measurement/R13_Full_Scan_80V.h5', is_advanced=False, **bare_correction_args)

    # plot_data(interpreted_data='Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data_run_corrected", use_group=False, use_corrected=True)
    # analyze_data(raw_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', is_advanced=False, is_cv=True,
    #              first_boundaries=(-100, -40),
    #              second_boundaries=(-10, 0), apply_doping=True, chip_group_name="sensor", use_corrected=True, plot=True, apply_contours=False,
    #              fit_plot_pdf_name="r13-CV_COMBI_6_C_V_Fits.pdf",
    #              **bare_correction_args)
    # plot_combined_data(interpreted_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', first_lower=-100,
    #                    first_upper=-40, second_lower=-8, second_upper=0, use_corrected=True, apply_doping=True)
    analyze_data(raw_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/unbiased_1_test", is_advanced=True)
