"""
Plotting of Pixcap65 data.
"""
import os.path

import numpy as np
import tables as tb
from matplotlib import cm, pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis import check_leaf_unit, GENERAL_PIXCAP_SHAPE
from utils_2 import walk_to_node, GroupType

cmap = cm.get_cmap('viridis')


def plot_data(interpreted_data, base_path=None, suffix="general_data", use_group=False):
    """
    plot_data

    Helper function to graphical present/plot the analysis results of a simple pixel capacitance scan.
    The plotting is only performed for the pixels which contribute a usable capacitance measurement.
    In Addition to the fits for estimating the capacitance also the capacitance distribution and frequency is plotted.
    The name of the resulting pdf is derived from the file name with the measurement data.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix: additonal suffix to use for naming the pdf containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the pdf name.
    """
    # determine the pdf file
    if use_group and base_path is not None:
        _, group_component = os.path.split(base_path)
        pdf_name = "{file}_{s}_{group}.pdf".format(s=suffix, file=interpreted_data[:-3], group=group_component)
    else:
        pdf_name = "{file}_{s}.pdf".format(s=suffix, file=interpreted_data[:-3])
    with PdfPages(pdf_name) as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            # actual plotting.
            if base_path is None:
                base_group = in_file_h5.root
            else:
                base_group = walk_to_node(in_file_h5.root, base_path)
            plot_data_delegate(base_group.total_cap.measurements, base_group.total_cap.analysis, output_pdf)

def plot_bias_data(interpreted_data, base_path=None, suffix="bias_curve", use_group=False):
    """
    plot_bias_data

    Plot the data acquired for the pixel-diodes I-V characterization.
    :param interpreted_data: path to the hdf file which holds the raw data.
    :param base_path: path within the files hierarchy for the base group.
    :param suffix: additonal suffix to use for naming the pdf containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the pdf name.
    """
    if use_group and base_path is not None:
        _, group_component = os.path.split(base_path)
        pdf_name = "{file}_{s}_{group}.pdf".format(s=suffix, file=interpreted_data[:-3], group=group_component)
    else:
        pdf_name = "{file}_{s}.pdf".format(s=suffix, file=interpreted_data[:-3])
    with PdfPages(pdf_name) as output_pdf:
        # with PdfPages(interpreted_data[:-3] + '.pdf') as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            if base_path is None:
                base_group = in_file_h5.root
            else:
                base_group = walk_to_node(in_file_h5.root, base_path)
            plot_bias_delegate(base_group.biasing.measurements, output_pdf)

def plot_cv_data(interpreted_data, base_path=None, first_upper=None, first_lower=None, second_upper=None,
                 second_lower=None, suffix="C_V_characteristic", use_group=False):
    """
    plot_cv_data

    Plot the results of the C-V characterization of the scanned pixels.
    To achieve this we need the different c-v-data.
    Then the C-V curve is plotted for every pixel.
    If requested also fits to the boundary regions of the c-v-curve are performed to determine the depletion voltage of the pixel.
    To do so, two fit ranges for the two boundaries with physically distinct behaviour needs to be supplied.


    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param first_upper: upper limit of the first fit range
    :param first_lower: lower limit of the first fit range
    :param second_upper: upper limit of the second fit range
    :param second_lower: lower limit of the second fit range
    :param suffix:  additonal suffix to use for naming the pdf containing the plots.
    :param use_group:   boolean, whether to append the group name of the measurements to the pdf name.
    """
    if use_group and base_path is not None:
        _, group_component = os.path.split(base_path)
        pdf_name = "{file}_{s}_{group}.pdf".format(s=suffix, file=interpreted_data[:-3], group=group_component)
    else:
        pdf_name = "{file}_{s}.pdf".format(s=suffix, file=interpreted_data[:-3])
    with PdfPages(pdf_name) as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            if base_path is None:
                base_group = in_file_h5.root
            else:
                base_group = walk_to_node(in_file_h5.root, base_path)
            plot_cv_data_delegate(base_group.biasing.measurements, base_group.biasing.analysis, output_pdf, first_upper, first_lower, second_upper, second_lower)

def plot_combined_data(interpreted_data, base_path=None, first_upper=None, first_lower=None, second_upper=None,
                       second_lower=None, suffix="combined_bias_cv_curve", use_group=False):
    """
    plot_combined_data

    Plot the data acquired for the pixel-diodes I-V characterization and the C-V characterization of the pixels.
    Plot the results of the C-V characterization of the scanned pixels.
    To achieve this we need the different c-v-data.
    Then the C-V curve is plotted for every pixel.
    If requested also fits to the boundary regions of the c-v-curve are performed to determine the depletion voltage of the pixel.
    To do so, two fit ranges for the two boundaries with physically distinct behaviour needs to be supplied.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param first_upper: upper limit of the first fit range
    :param first_lower: lower limit of the first fit range
    :param second_upper: upper limit of the second fit range
    :param second_lower: lower limit of the second fit range
    :param suffix: additonal suffix to use for naming the pdf containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the pdf name.
    """
    if use_group and base_path is not None:
        _, group_component = os.path.split(base_path)
        pdf_name = "{file}_{s}_{group}.pdf".format(s=suffix, file=interpreted_data[:-3], group=group_component)
    else:
        pdf_name = "{file}_{s}.pdf".format(s=suffix, file=interpreted_data[:-3])
    with PdfPages(pdf_name) as output_pdf:
        # with PdfPages(interpreted_data[:-3] + '.pdf') as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            if base_path is None:
                base_group = in_file_h5.root
            else:
                base_group = walk_to_node(in_file_h5.root, base_path)
            plot_bias_delegate(base_group.biasing.measurements, output_pdf)
            plot_cv_data_delegate(base_group.biasing.measurements, base_group.biasing.analysis, output_pdf, first_upper, first_lower, second_upper, second_lower)

def plot_bias_delegate(data_group, output_pdf: PdfPages):
    """
    plot_bias_delegate

    Actual implementation for presenting the results of the I-V characterization.
    :param data_group: hdf file's hierarchy group containing the raw data.
    :param output_pdf: pdf object to write the plots to.
    """
    tabular = data_group.BiasTable
    assert isinstance(tabular, tb.Table)
    voltage_data = np.abs(tabular.col("U"))
    current_data = np.abs(tabular.col("I"))
    current_errors = tabular.col("DI")
    if not np.all(np.isfinite(voltage_data)):
        current_errors = None
    fig, ax = plt.subplots()
    ax.set(title="Bias data from the measurement", xlabel="U in V", ylabel="I in A")
    ax.errorbar(voltage_data, current_data, yerr=current_errors, fmt='o', label="Bias data")
    output_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def plot_cv_data_delegate(data_group, analysis_group, output_pdf, first_upper=None, first_lower=None, second_upper=None, second_lower=None, **kwargs):
    """
    plot_cv_data_delegate

    Actual implementation for presenting the results of the C-V characterization and if necessary the determination of the full depletion voltage.
    For each with an successful capacitance measurement for each bias voltage in use the C-V-curve will be plotted.
    If necessary, meaning if the bounaries for the two fit ranges for the two physically distinc regions of the c-v-curve are provided the
    full depletion voltage will be calculated and the necessary fits be plotted.
    If the analysis results provided already contain the necessary data sets for the estimation of the full depletion voltage,
    these will be used and the fit will be plotted, as well.

    :param data_group: hdf files hierarchy group containing the raw measurement data.
    :param analysis_group: hdf files hierarchy group containing the analysis results.
    :param output_pdf: pdf object to write the created figures to for long-term saving.
    :param first_upper: upper limit of the first fit range
    :param first_lower: lower limit of the first fit range
    :param second_upper: upper limit of the second fit range
    :param second_lower: lower limit of the second fit range
    """
    # extract the bias data
    voltage_data = check_leaf_unit(data_group.BiasVoltageHist, "V")

    # investigate all the pixel for plotting
    for ii, jj in np.ndindex(GENERAL_PIXCAP_SHAPE):
        check_leaf_unit(analysis_group.UCHist, "F")
        check_leaf_unit(analysis_group.UCErrHist, "F")

        cap_data = analysis_group.UCHist[:, :, :]
        cap_errors = analysis_group.UCErrHist[:, :, :]
        if np.any(np.isnan(cap_data[ii, jj, :])):
            continue

        # extract the information about the depletion voltage
        approx_depletion = True
        if first_upper is None:
            approx_depletion = False
        if first_lower is None:
            approx_depletion = False
        if second_upper is None:
            approx_depletion = False
        if second_lower is None:
            approx_depletion = False
        if approx_depletion and "DepletionHist" not in analysis_group:
            first_section_upper_mask =voltage_data <= first_upper
            first_section_lower_mask =voltage_data >= first_lower
            first_section_mask = np.logical_and(first_section_upper_mask, first_section_lower_mask)

            second_section_upper_mask =voltage_data <= second_upper
            second_section_lower_mask =voltage_data >= second_lower
            second_section_mask = np.logical_and(second_section_upper_mask, second_section_lower_mask)

            effective_capacitance_data = np.reciprocal(cap_data[ii, jj, :]*1e15)**2
            first_voltage_data = voltage_data[first_section_mask]
            second_voltage_data = voltage_data[second_section_mask]
            first_cap_data = effective_capacitance_data[first_section_mask]
            second_cap_data = effective_capacitance_data[second_section_mask]
            first_result = np.polyfit(first_voltage_data, first_cap_data, deg=1, cov=True)
            first_dep_parameters = np.asarray(first_result[0])
            first_dep_cov = np.asarray(first_result[1])
            print("The result of the first fit is:")
            print(first_result)
            second_result = np.polyfit(second_voltage_data, second_cap_data, deg=1, cov=True)
            second_dep_parameters = np.asarray(second_result[0])
            second_dep_cov = np.asarray(second_result[1])
            print("The result of the second fit is:")
            print(second_result)
            dep_voltage = (first_dep_parameters[1] - second_dep_parameters[1]) / (first_dep_parameters[0] -
                                                                                  second_dep_parameters[0])
            first_norm_factor = first_dep_parameters[0] - second_dep_parameters[0]
            dxx = first_dep_cov[0][0]
            dxy = first_dep_cov[0][1]
            dyx = first_dep_cov[1][0]
            dyy = first_dep_cov[1][1]
            dvv = second_dep_cov[0][0]
            dvw = second_dep_cov[0][1]
            dwv = second_dep_cov[1][0]
            dww = second_dep_cov[1][1]
            first_error_term = (dww*first_norm_factor + dvw*(second_result[0][1]- first_dep_parameters[1])) / first_norm_factor**3
            second_error_term = (dyy*first_norm_factor + dxy*(second_result[0][1]- first_dep_parameters[1])) / first_norm_factor**3
            third_error_term = (dvv*(second_dep_parameters[1]- first_dep_parameters[1])-dwv*first_norm_factor) * (
                        second_dep_parameters[1]-
                                                                                                             first_dep_parameters[1]) / first_norm_factor**4
            fourth_error_term = (dyx*first_norm_factor + dxx*(second_dep_parameters[1]- first_dep_parameters[1])) * (
                    second_dep_parameters[1] - first_dep_parameters[1]) / first_norm_factor**4
            dep_voltage_error = np.sqrt(first_error_term + second_error_term + third_error_term + fourth_error_term)
            print("The depletion voltage is {voltage}+-{error}".format(voltage=dep_voltage, error=dep_voltage_error))
        elif "DepletionHist" in analysis_group:
            approx_depletion = True
            first_dep_parameters = analysis_group.DepFitParamHist[ii, jj, :2]
            second_dep_parameters = analysis_group.DepFitParamHist[ii, jj, 2:]
        else:
            approx_depletion = False
            first_dep_parameters = None
            second_dep_parameters = None


        fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
        if approx_depletion:
            first_voltage_x = np.linspace(first_lower, 0, 1000)
            second_voltage_x = np.linspace(second_lower, second_upper, 100)
            first_cap_calc = first_dep_parameters[0] * first_voltage_x + first_dep_parameters[1]
            second_cap_calc = second_dep_parameters[0] * second_voltage_x + second_dep_parameters[1]
            ax[1].plot(first_voltage_x, first_cap_calc, '-', label="First section fit")
            ax[1].plot(second_voltage_x, second_cap_calc, '-', label="Second section fit")
        effective_capacitance_error_data = np.reciprocal(cap_data[ii, jj, :] * 1e15) ** 3 * cap_errors[
            ii, jj, :] if np.all(np.isfinite(cap_errors[ii, jj, :])) else None
        eff_cap_errors = cap_errors[ii, jj, :] if np.all(np.isfinite(cap_errors[ii, jj, :])) else None
        ax[0].set(title=f"Bias data from the \nmeasurement for pixel ({ii},{jj})",xlabel="U in V", ylabel="C in fF")
        ax[0].errorbar(voltage_data, cap_data[ii, jj, :] * 1e15, yerr=eff_cap_errors, fmt='o', label="Bias data")
        ax[1].set(title=f"Suited Bias data from the \nmeasurement for pixel ({ii},{jj})", xlabel="U in V", ylabel="1/ C^2 in 1/(fF)^2")
        ax[1].errorbar(voltage_data, 1 / (cap_data[ii, jj, :] * 1e15)**2, yerr=effective_capacitance_error_data, fmt='o', label="Bias data")
        ax[0].legend()
        ax[1].legend()
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # for k, bias_voltage in enumerate(voltage_data):
    #     fig, ax = plt.subplots()
    #     ax.set(title=f"Capacitance distribution for bias voltage {bias_voltage}", xlabel="C in fF")
    #     ax.hist(analysis_group.UCHist[:, :, k] * 1e15, bins=50)
    #     output_pdf.savefig(fig, bbox_inches='tight')
    #     plt.close(fig)

def plot_data_delegate(data_group: tb.Group, analysis_group: tb.Group, output_pdf: PdfPages):
    """
    plot_data_delegate

    Actual implementation to plot the results of the analysis of simple pixel capacitance scan.
    Besides the naming it is not just plotting but also a bit of analysis as the distribution of the capacitances is investigated
    in this function.
    The current-frequency dependendy will plotted for each scanned pixel with finite currents.
    Also the capacitance distribution over the whole sensor and the frequency of capacitance values are plotted.

    :param data_group: hdf files hierarchy group containing the raw measurement data.
    :param analysis_group: hdf files hierarchy group containing the analysis results.
    :param output_pdf: pdf object to write the created figures to for long-term saving.
    """
    # Read pixel map
    current_hist = check_leaf_unit(data_group.HistCurr, "A")
    current_err_hist = check_leaf_unit(data_group.HistCurrErr, "A")
    cap_hist = check_leaf_unit(analysis_group.HistCap, "F")
    leak_hist = check_leaf_unit(analysis_group.HistLeak, "nA")

    # Read scan parameters
    scan_parameters = data_group.scan_params[:]

    # 2D Pixel Capacitance Hist
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    im = ax.imshow(cap_hist * 1e15)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label='Pixel Capacitance / fF')

    ax.set_ylabel('Column')
    ax.set_xlabel('Row')
    output_pdf.savefig(fig, bbox_inches='tight')

    # 1D Pixel Capacitance Hist
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.hist(cap_hist[~np.isnan(cap_hist)].reshape(-1) * 1e15, bins=50)
    ax.set_ylabel('Counts / #')
    ax.set_xlabel('Pixel Capacitance / fF')
    ax.grid()
    output_pdf.savefig(fig, bbox_inches='tight')

    # Current vs. frequency
    print("The histograms shape is {}".format(current_hist.shape))
    for col, row in np.ndindex(current_hist.shape[:2]):
        if np.isfinite(current_hist[col, row, 0]):
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            # res = np.polyfit(scan_parameters['frequency'], current_hist[col, row] * 1e9, deg=1, cov=True)
            f = np.arange(0, scan_parameters['frequency'].max() * 1.1, 0.1)
            actual_cap = cap_hist[col, row] * 1e15
            if "HistRes" in analysis_group and np.isfinite(analysis_group.HistRes[col, row]):
                from analysis import full_capacitance_model
                ax.plot(f, full_capacitance_model(f, c=cap_hist[col, row] * 1e6, r=analysis_group.HistRes[col, row],
                                                  i=leak_hist[col, row] * 1.e-9, u0=1) * 1e9, color=cmap(0.6), ls='--', marker='',
                        label='Fit to data:\n$C_d = %.1f\,$fF' % actual_cap)
            else:
                ax.plot(f, cap_hist[col, row] * 1e15 * f + leak_hist[col,row], color=cmap(0.6), ls='--', marker='',
                        label='Fit to data:\n$C_d = %.1f\,$fF' % actual_cap)
            if np.all(np.isfinite(current_err_hist[col, row, :])):
                ax.errorbar(scan_parameters['frequency'], current_hist[col, row] * 1e9, yerr=current_err_hist[col, row] * 1e9, fmt='o',
                            ls='',
                            label='Pixel({i_col},{i_row})'.format(i_col=col, i_row=row), color=cmap(0.2))
            else:
                ax.plot(scan_parameters['frequency'], current_hist[col, row] * 1e9, marker='o', ls='',
                        label='Pixel({i_col},{i_row})'.format(i_col=col, i_row=row), color=cmap(0.2))
            ax.set_ylabel('Current / nA')
            ax.set_xlabel('Frequency / MHz')
            ax.legend()
            ax.grid()
            output_pdf.savefig(fig, bbox_inches='tight')
            # ax.plot(freq_sweep_array, fit_fn, label = 'a={a:.3E}, b={b:.3E}'.format(a=a, b=b))
        else:
            continue

        # #apply linear fit to measured current values; also returns covariance matrix
        # matrix = np.polyfit(freq_sweep_array, current_array, 1, cov=True)

        # a, b = matrix[0][0], matrix[0][1]
        # #da = matrix[1][0][0] #squared fit error of a
        # #db = matrix[1][1][1] #squared fit error of b

        # #data structure in txt file: "slope, offset (y-intercept)"

        # # fit_fn = a*freq_sweep_array + b
        # # pl.plot(freq_sweep_array, current_array, 'o', label = 'COL({i_col})PIX(0)'.format(i_col=i_col))
        # # pl.plot(freq_sweep_array, fit_fn, label = 'a={a:.3E}, b={b:.3E}'.format(a=a, b=b))

def plot_compare_delegate(first_group: GroupType, second_group: GroupType, output_pdf: PdfPages):
    first_cap_hist = first_group.HistCurr[:]
    second_cap_hist = second_group.HistCurr[:]
    first_nan_mask = np.isfinite(first_cap_hist)
    second_nan_mask = np.isfinite(second_cap_hist)
    full_mask = np.logical_and(first_nan_mask, second_nan_mask)
    diff_cap_hist = np.where(full_mask, first_cap_hist - second_cap_hist, np.nan)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    im = ax.imshow(diff_cap_hist * 1e15)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label='Pixel Capacitance / fF')

    ax.set_ylabel('Column')
    ax.set_xlabel('Row')
    output_pdf.savefig(fig, bbox_inches='tight')


if __name__ == '__main__':
    # plot_data(interpreted_data=os.path.expanduser('~/git/pixcap65/pixcap_LF_50x50_DC_R3_80V_HV.h5'))
    plot_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1", suffix="general_data_80V", use_group=True)
    # plot_bias_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/I_V_Characteristic", use_group=True)
    # plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic",
    #                    use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0)