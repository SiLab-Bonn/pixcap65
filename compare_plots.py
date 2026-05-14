import numpy as np
import tables as tb
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pixcap65.analysis_util.utility import check_leaf_unit, HIST_CURRENT_MEAS_UNIT
from pixcap65.plotting import BIAS_CURVE_X_LABEL, BIAS_CURVE_Y_LABEL, CURRENT_CONVERSION_FACTOR
from pixcap65.plotting import get_pdf_name, get_base_group, CURRENT_LABEL, FREQUENCY_LABEL


def plot_bias_data(interpreted_data, base_path=None, second_data=None, second_path=None, suffix="bias_curve",
                   use_group=False):
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    with PdfPages(pdf_name) as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            if second_data is None:
                base_group = get_base_group(base_path, in_file_h5)
                ref_group = get_base_group(second_path, in_file_h5)
                plot_bias_delegate(base_group.biasing.measurements, ref_group.biasing.measurements, output_pdf)
                return
            with tb.open_file(second_data, mode='r') as second_h5:
                base_group = get_base_group(base_path, in_file_h5)
                ref_group = get_base_group(second_path, second_h5)
                plot_bias_delegate(base_group.biasing.measurements, ref_group.biasing.measurements, output_pdf)


def plot_bias_delegate(data_group, second_group, output_pdf: PdfPages):
    tabular = data_group.BiasTable
    ref_tabular = second_group.BiasTable
    assert isinstance(tabular, tb.Table)
    assert isinstance(ref_tabular, tb.Table)
    voltage_data = np.abs(tabular.col("U"))
    ref_voltages = np.abs(ref_tabular.col("U"))
    current_data = np.abs(tabular.col("I"))
    ref_currents = np.abs(ref_tabular.col("I"))
    current_errors = tabular.col("DI")
    ref_current_errors = ref_tabular.col("DI")
    if not np.all(np.isfinite(voltage_data)):
        current_errors = None
    if np.any(~np.isfinite(ref_voltages)):
        ref_current_errors = None
    fig, ax = plt.subplots()
    ax.set(title="Bias data from the measurement", xlabel=BIAS_CURVE_X_LABEL, ylabel=BIAS_CURVE_Y_LABEL)
    ax.errorbar(voltage_data, current_data * CURRENT_CONVERSION_FACTOR,
                yerr=current_errors, xerr=None, fmt='o', label="Bias data")
    ax.errorbar(ref_voltages, ref_currents * CURRENT_CONVERSION_FACTOR,
                yerr=ref_current_errors, xerr=None, fmt='o', label="Reference data")
    output_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def plot_data(interpreted_data, base_path=None, reference_data=None, reference_path=None, suffix="general_data",
              use_group=False, **kwargs):
    # determine the pdf file
    pdf_name = get_pdf_name(base_path, interpreted_data, suffix, use_group)
    with PdfPages(pdf_name) as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            if reference_data is None:
                base_group = get_base_group(base_path, in_file_h5)
                reference_group = get_base_group(reference_path,in_file_h5)
                plot_data_delegate(base_group.total_cap.measurements, reference_group.total_cap.measurements,
                                   output_pdf, **kwargs)
                return
            with tb.open_file(reference_data, mode='r') as reference_h5:
                base_group = get_base_group(base_path, in_file_h5)
                reference_group = get_base_group(reference_path, reference_h5)
                plot_data_delegate(base_group.total_cap.measurements, reference_group.total_cap.measurements,
                                   output_pdf, **kwargs)


def linear_model(x, a, b):
    return a * x + b

def plot_data_delegate(data_group: tb.Group, reference_group: tb.Group, output_pdf: PdfPages, **kwargs):
    # Read pixel map
    current_hist = check_leaf_unit(data_group.HistCurr, HIST_CURRENT_MEAS_UNIT)
    current_err_hist = check_leaf_unit(data_group.HistCurrErr, HIST_CURRENT_MEAS_UNIT)
    ref_current_hist = check_leaf_unit(reference_group.HistCurr, HIST_CURRENT_MEAS_UNIT)
    ref_current_err_hist = check_leaf_unit(reference_group.HistCurrErr, HIST_CURRENT_MEAS_UNIT)

    # Read scan parameters
    scan_parameters = data_group.scan_params[:]
    reference_parameters = reference_group.scan_params[:]
    relative_err = []
    relative_err_uncert = []
    scale_err = []
    scale_err_uncert = []
    for col, row in np.ndindex(current_hist.shape[:2]):
        if np.all(np.isfinite(current_hist[col, row])) and np.all(np.isfinite(ref_current_hist[col, row])):
            fig, ax = plt.subplots()
            ax.set_title(f"Current data for ({col}, {row})")
            data_mask = current_hist[col, row] < 1e30
            ref_mask = ref_current_hist[col, row] < 1e30
            ax.errorbar(scan_parameters['frequency'][data_mask], current_hist[col, row, data_mask]*CURRENT_CONVERSION_FACTOR, yerr=current_err_hist[col, row, data_mask]*CURRENT_CONVERSION_FACTOR,
                        label="Current data", ls='', marker='.', markersize=5, capsize=6)
            ax.errorbar(reference_parameters['frequency'][ref_mask], ref_current_hist[col, row, ref_mask]*CURRENT_CONVERSION_FACTOR, yerr=ref_current_err_hist[col, row, ref_mask]*CURRENT_CONVERSION_FACTOR,
                        label="reference data", ls='', marker='x', capsize=6)
            ax.set_ylabel(CURRENT_LABEL)
            ax.set_xlabel(FREQUENCY_LABEL)
            ax.legend()
            ax.grid()
            output_pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            fig, ax = plt.subplots()
            ax.set_title(f"Current error data for ({col}, {row})")
            ax.errorbar(scan_parameters['frequency'][data_mask],
                        current_err_hist[col, row, data_mask] * CURRENT_CONVERSION_FACTOR,
                        label="Current data", ls='', marker='.', markersize=5, capsize=6)
            ax.errorbar(reference_parameters['frequency'][ref_mask],
                        ref_current_err_hist[col, row, ref_mask] * CURRENT_CONVERSION_FACTOR,
                        label="reference data", ls='', marker='x', capsize=6)
            ax.set_ylabel(CURRENT_LABEL)
            ax.set_xlabel(FREQUENCY_LABEL)
            ax.legend()
            ax.grid()
            output_pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            fig, ax = plt.subplots()
            ax.set_title(f"Current error analysis for ({col}, {row})")
            ax.errorbar(current_hist[col, row, data_mask] * CURRENT_CONVERSION_FACTOR,
                        current_err_hist[col, row, data_mask] * CURRENT_CONVERSION_FACTOR,
                        label="Current data", ls='', marker='.', markersize=5, capsize=6, color='C1')
            # ax.errorbar(ref_current_hist[col, row, ref_mask] * CURRENT_CONVERSION_FACTOR,
            #             ref_current_err_hist[col, row, ref_mask] * CURRENT_CONVERSION_FACTOR,
            #             label="reference data", ls='', marker='x', capsize=6, color='C2')
            sample_currents = np.linspace(0, np.max(current_hist[col, row, data_mask]) * 1.2 * CURRENT_CONVERSION_FACTOR, 1000)
            data_res = np.polyfit(current_hist[col, row, data_mask] * CURRENT_CONVERSION_FACTOR,
                                  current_err_hist[col, row, data_mask] * CURRENT_CONVERSION_FACTOR, deg=1,cov='unscaled')
            data_err = np.sqrt(np.diag(data_res[1]))
            ref_res = np.polyfit(ref_current_hist[col, row, ref_mask] * CURRENT_CONVERSION_FACTOR,
                                 ref_current_err_hist[col, row, ref_mask] * CURRENT_CONVERSION_FACTOR, deg=1, cov='unscaled')
            ref_err = np.sqrt(np.diag(ref_res[1]))
            from jacobi import propagate
            data_y, data_y_cov = propagate(lambda p: linear_model(sample_currents, *p), data_res[0], data_res[1])
            ref_y, ref_y_cov = propagate(lambda p: linear_model(sample_currents, *p), ref_res[0], ref_res[1])
            data_y_err_prop = np.diag(data_y_cov) ** 0.5
            ref_y_err_prop = np.diag(ref_y_cov) ** 0.5
            ax.plot(sample_currents, data_y, label="error analysis fit", ls='--', color='C1')
            # ax.plot(sample_currents, ref_y, label="error analysis fit (reference)", ls=':', color='C2')
            ax.fill_between(sample_currents, data_y - data_y_err_prop, data_y + data_y_err_prop, facecolor='C1', alpha=0.5)
            # ax.fill_between(sample_currents, ref_y - ref_y_err_prop, ref_y + ref_y_err_prop, facecolor='C2', alpha=0.5)
            print(f"Data: a=({data_res[0][0]}+-{data_err[0]}); b=({data_res[0][1]}+-{data_err[1]})")
            relative_err.append(data_res[0][0] * 100)
            relative_err_uncert.append(data_err[0] * 100)
            scale_err.append(data_res[0][1])
            scale_err_uncert.append(data_err[1])
            print(f"Reference: a=({ref_res[0][0]}+-{ref_err[0]}); b=({ref_res[0][1]}+-{ref_err[1]})")
            ax.set_ylabel(CURRENT_LABEL)
            ax.set_xlabel(CURRENT_LABEL)
            ax.legend()
            ax.grid()
            output_pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    # investigate the uncertainties for all the measurements
    relative_err = np.asarray(relative_err)
    relative_err_uncert = np.asarray(relative_err_uncert)
    scale_err = np.asarray(scale_err)
    scale_err_uncert = np.asarray(scale_err_uncert)
    weights_relative = np.reciprocal(relative_err_uncert)**2
    weights_scale = np.reciprocal(scale_err_uncert)**2
    final_relative_error = np.average(relative_err, weights=weights_relative)
    final_relative_error_std = np.std(relative_err)
    final_relative_error_sys = np.average(relative_err_uncert, weights=weights_relative)
    final_scale_error = np.average(scale_err, weights=weights_scale)
    final_scale_error_std = np.std(scale_err)
    final_scale_error_sys = np.average(scale_err_uncert, weights=weights_scale)
    print(f"Final relative error: {final_relative_error}+-{final_relative_error_std}+-{final_relative_error_sys}")
    print(f"Final scale error: {final_scale_error}+-{final_scale_error_std}+-{final_scale_error_sys}")


if __name__ == "__main__":
    from pixcap65.analysis import analyze_data
    from pixcap65.plotting import plot_cv_data

    plot_bias_data(interpreted_data="packaged/DEMO_HV_Sweep.h5", second_path="Reference/R13/simple_bias", base_path="Reference/R13/improved_bias")
    analyze_data(raw_data="packaged/DEMO_HV_Sweep.h5", base_path="Reference/R13/cv_bias", is_cv=True, is_advanced=False)
    plot_cv_data(interpreted_data="packaged/DEMO_HV_Sweep.h5", base_path="Reference/R13/cv_bias", is_cv=True, is_advanced=False, use_group=True)
    plot_data(interpreted_data="packaged/Reference_Demo.h5", base_path="Reference/TESTS/unbiased_5_full", reference_path="Reference/TESTS/unbiased_4_full", suffix="comparison_1")
    plot_data(interpreted_data="packaged/Reference_Demo.h5", base_path="Reference/TESTS/unbiased_6_full",
              reference_path="Reference/TESTS/unbiased_4_full", suffix="comparison_2")
    plot_data(interpreted_data="packaged/Reference_Demo.h5", base_path="Reference/TESTS/unbiased_7_full",
              reference_path="Reference/TESTS/unbiased_4_full", suffix="comparison_3")
    plot_data(interpreted_data="packaged/Reference_Demo.h5", base_path="Reference/TESTS/unbiased_9_full",
              reference_path="Reference/TESTS/unbiased_4_full", suffix="comparison_4")
    plot_data(interpreted_data="packaged/Reference_Demo.h5", base_path="Reference/TESTS/unbiased_10_full",
              reference_path="Reference/TESTS/unbiased_9_full", suffix="comparison_5")
    plot_data(interpreted_data="packaged/Reference_Demo.h5", base_path="Reference/TESTS/unbiased_11_full",
              reference_path="Reference/TESTS/unbiased_9_full", suffix="comparison_6")
