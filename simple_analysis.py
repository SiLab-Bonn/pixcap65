from warnings import deprecated

import numpy as np
import tables as tb
import time

from analysis import analyze_depletion_delegate, apply_correction_simple
from analysis_util import analyze_data_delegate
from analysis_util.utility import str_join, ANALYSIS_CORRECTED_GROUP_NAME, check_leaf_unit, ANALYSIS_GROUP_NAME, \
    GLOBAL_FILTERS, get_analysis_group
from utility.utils_2 import walk_to_node, GroupType


@deprecated("Use the general implementation of analysis.analyze_data_temporary_replacement instead.")
def analyze_data(raw_data, base_path=None, is_cv=False, first_boundaries=None, second_boundaries=None,
                 is_inter_pixel=False, **kwargs):
    """
    analyze data


    # FIXME: update the docstring
    Analze the provided raw data to determine the (total) capacitance of each pixel in the measurement.
    Due to its simplified analysis strategy this method is only valid for sufficiently small frequencies.
    What sufficiently smalls is, dependes on the measured capacitance and the on-resistance of the measurement circuit.
    Please note, this function is only a wrapper around the actual analysis to handle files and output strategy.

    The capacitances are determined by a simple linear fit without paying attention to measurement uncertainties.

    This function could also be used for the investigation of a C-V curve.
    In this case the fits are performed for every bias voltage used for the characterization.

    :param raw_data: path to the hdf file containing the raw data.
    :param is_cv: boolean, indicating whether this a C-V- characterization instead of a simple pixel scan.
    :param base_path: path to the base group withing the hdf files hierachry.
    """
    with tb.open_file(raw_data, mode='a') as in_file_h5:
        if base_path is None:
            base_group = in_file_h5.root
        else:
            try:
                base_group = walk_to_node(in_file_h5.root, base_path)
            except:
                print(in_file_h5)
                raise
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
            if "analysis" in base_group.biasing:
                base_group.biasing.analysis._f_remove(recursive=True)
                time.sleep(1)

            for k, bias_voltage in enumerate(base_group.biasing.measurements.BiasVoltageHist):
                bias_name = f"bias_{bias_voltage}_V".replace('-', "M_").replace(".", "__")
                data_group = base_group.biasing.measurements[bias_name]
                ana_group = walk_to_node(base_group.biasing, str_join("/", ANALYSIS_GROUP_NAME, bias_name), create=True)
                assert isinstance(ana_group, tb.Group)


                analyze_data_handle_data(in_file_h5, data_group, ana_group, is_inter_pixel=is_inter_pixel, **kwargs)

                # extract the capacitance data for tabular value; will also need coreected data.
                cap_data = ana_group.HistCap[:]
                cap_error_data = ana_group.HistCapErr[:]
                cv_data[:, :, k] = cap_data[:, :]
                cv_err_data[:, :, k] = cap_error_data[:, :]
                if kwargs.get("apply_correction", False):
                    ana_group_correction = walk_to_node(base_group.biasing,
                                                str_join("/", ANALYSIS_CORRECTED_GROUP_NAME, bias_name),
                                                create=True)
                    cap_data = ana_group_correction.HistCap[:]
                    cap_error_data = ana_group_correction.HistCapErr[:]
                    cv_data_corrected[:, :, k] = cap_data[:, :]
                    cv_err_data_corrected[:, :, k] = cap_error_data[:, :]

            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCHist",
                                                  title="Histogram of the U-C-curve",
                                                  filters=GLOBAL_FILTERS,
                                                  obj=cv_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()
            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCErrHist",
                                                  title="Error Histogram of the U-C-curve",
                                                  filters=GLOBAL_FILTERS,
                                                  obj=cv_err_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()

            # save also the corrected capacitance data if necessary
            if kwargs.get("apply_correction", False):
                temp_array = in_file_h5.create_carray(base_group.biasing.analysis_correction, name="UCHist",
                                                      title="Histogram of the U-C-curve",
                                                      filters=GLOBAL_FILTERS,
                                                      obj=cv_data)
                temp_array.attrs["Units"] = "F"
                temp_array.flush()
                temp_array = in_file_h5.create_carray(base_group.biasing.analysis_correction, name="UCErrHist",
                                                      title="Error Histogram of the U-C-curve",
                                                      filters=GLOBAL_FILTERS,
                                                      obj=cv_err_data)
                temp_array.attrs["Units"] = "F"
                temp_array.flush()

            if first_boundaries is not None and second_boundaries is not None:
                dep_ana_group = get_analysis_group(base_group.biasing, **kwargs)
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
            if "analysis" in reference_group:
                reference_group.analysis._f_remove(recursive=True)
                time.sleep(1)
            ana_group = walk_to_node(reference_group, "analysis", create=True)
            assert isinstance(ana_group, tb.Group)


            analyze_data_handle_data(in_file_h5, reference_group.measurements, ana_group, is_inter_pixel=is_inter_pixel, **kwargs)


@deprecated("Use the general implementation of analysis.analysis_data_handle_temporary_replacement instead.")
def analyze_data_handle_data(file: tb.File, data_group: GroupType, result_group: GroupType,
                             is_inter_pixel=False,  **kwargs):
    """
        analyze data_delegate

        # FIXME: implement the correct doc string here.
        Analze the provided raw data to determine the (total) capacitance of each pixel in the measurement.
        Due to its simplified analysis strategy this method is only valid for sufficiently small frequencies.
        What sufficiently smalls is, dependes on the measured capacitance and the on-resistance of the measurement circuit.

        The capacitances are determined by a simple linear fit without paying attention to measurement uncertainties.

        This function could also be used for the investigation of a C-V curve.
        In this case the fits are performed for every bias voltage used for the characterization.

        :param file: open hdf file to write the analysis results to.
        :param data_group: hdf file's hierachy group containing the measured data.
        :param result_group: hdf file's hierachy group to write the analysis results to.
        """
    # Read pixel map
    assert isinstance(data_group, tb.Group)
    assert isinstance(result_group, tb.Group)
    if is_inter_pixel:
        current_hist = check_leaf_unit(data_group.TotalHistCurr, "A")
        # Read scan parameters
        scan_parameters = data_group.scan_params[:]

        analyze_data_delegate(file, result_group, current_hist, scan_parameters, **kwargs)

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
        current_hist = check_leaf_unit(data_group.InterHistCurrA, "A")
        analyze_data_delegate(file, result_group, current_hist, scan_parameters, **kwargs)

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
        current_hist = check_leaf_unit(data_group.InterHistCurrB, "A")
        analyze_data_delegate(file, result_group, current_hist, scan_parameters, **kwargs)


    else:
        current_hist = check_leaf_unit(data_group.HistCurr, "A")
        # Read scan parameters
        scan_parameters = data_group.scan_params[:]
        analyze_data_delegate(file, result_group, current_hist, scan_parameters, **kwargs)

    # if necessary: directly apply the correction of the capacitance values
    if "apply_correction" in kwargs and kwargs["apply_correction"]:
        assert 'bare_file' in kwargs
        assert 'bare_hdf_path' in kwargs
        apply_correction_simple(kwargs['bare_file'], kwargs['bare_hdf_path'], result_group)


