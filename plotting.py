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

from utils_2 import walk_to_node

cmap = cm.get_cmap('viridis')


def plot_data(interpreted_data, base_path=None):
    with PdfPages(interpreted_data[:-3] + '.pdf') as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            if base_path is None:
                base_group = in_file_h5.root
            else:
                base_group = walk_to_node(in_file_h5.root, base_path)
            plot_data_delegate(base_group.total_cap.measurements, base_group.total_cap.analysis, output_pdf)

def plot_bias_data(interpreted_data, base_path=None):
    with PdfPages(interpreted_data[:-3] + '.pdf') as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            if base_path is None:
                base_group = in_file_h5.root
            else:
                base_group = walk_to_node(in_file_h5.root, base_path)
            plot_bias_delegate(base_group.biasing.measurements, base_group.biasing.measurements, output_pdf)

def plot_cv_data(interpreted_data, base_path=None, first_upper=None, first_lower=None, second_upper=None,
                 second_lower=None):
    with PdfPages(interpreted_data[:-3] + '.pdf') as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            if base_path is None:
                base_group = in_file_h5.root
            else:
                base_group = walk_to_node(in_file_h5.root, base_path)
            plot_cv_data_delegate(base_group.biasing.measurements, base_group.biasing.analysis, output_pdf, first_upper, first_lower, second_upper, second_lower)


def plot_combined_data(interpreted_data, base_path=None, first_upper=None, first_lower=None, second_upper=None,
                       second_lower=None):
    with PdfPages(interpreted_data[:-3] + '.pdf') as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            if base_path is None:
                base_group = in_file_h5.root
            else:
                base_group = walk_to_node(in_file_h5.root, base_path)
            plot_bias_delegate(base_group.biasing.measurements, base_group.biasing.measurements, output_pdf)
            plot_cv_data_delegate(base_group.biasing.measurements, base_group.biasing.analysis, output_pdf, first_upper, first_lower, second_upper, second_lower)

def plot_bias_delegate(data_group, analysis_group, output_pdf):
    tabular = data_group.BiasTable
    assert isinstance(tabular, tb.Table)
    voltage_data = np.abs(tabular.col("U"))
    current_data = np.abs(tabular.col("I"))
    current_errors = tabular.col("DI")
    fig, ax = plt.subplots()
    ax.set(title="Bias data from the measurement", xlabel="U in V", ylabel="I in A")
    ax.errorbar(voltage_data, current_data, yerr=None, fmt='o', label="Bias data")
    output_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def plot_cv_data_delegate(data_group, analysis_group, output_pdf, first_upper=None, first_lower=None, second_upper=None, second_lower=None):
    voltage_data = data_group.BiasVoltageHist[:]
    for ii, jj in np.ndindex((40, 40)):
        cap_data = analysis_group.UCHist[:, :, :]
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
        if approx_depletion:
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
            print("The result of the first fit is:")
            print(first_result)
            second_result = np.polyfit(second_voltage_data, second_cap_data, deg=1, cov=True)
            print("The result of the second fit is:")
            print(second_result)
            dep_voltage = (first_result[0][1] - second_result[0][1]) / (first_result[0][0] - second_result[0][0])
            first_norm_factor = first_result[0][0] - second_result[0][0]
            dxx = first_result[1][0][0]
            dxy = first_result[1][0][1]
            dyx = first_result[1][1][0]
            dyy = first_result[1][1][1]
            dvv = second_result[1][0][0]
            dvw = second_result[1][0][1]
            dwv = second_result[1][1][0]
            dww = second_result[1][1][1]
            first_error_term = (dww*first_norm_factor + dvw*(second_result[0][1]-first_result[0][1])) / first_norm_factor**3
            second_error_term = (dyy*first_norm_factor + dxy*(second_result[0][1]-first_result[0][1])) / first_norm_factor**3
            third_error_term = (dvv*(second_result[0][1]-first_result[0][1])-dwv*first_norm_factor) * (second_result[0][1]-first_result[0][1]) / first_norm_factor**4
            fourth_error_term = (dyx*first_norm_factor + dxx*(second_result[0][1]-first_result[0][1])) * (
                        second_result[0][1] - first_result[0][1]) / first_norm_factor**4
            dep_voltage_error = np.sqrt(first_error_term + second_error_term + third_error_term + fourth_error_term)
            print("The depletion voltage is {voltage}+-{error}".format(voltage=dep_voltage, error=dep_voltage_error))


        fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
        if approx_depletion:
            first_voltage_x = np.linspace(first_lower, 0, 1000)
            second_voltage_x = np.linspace(second_lower, second_upper, 100)
            first_cap_calc = first_result[0][0] * first_voltage_x + first_result[0][1]
            second_cap_calc = second_result[0][0] * second_voltage_x + second_result[0][1]
            ax[1].plot(first_voltage_x, first_cap_calc, '-', label="First section fit")
            ax[1].plot(second_voltage_x, second_cap_calc, '-', label="Second section fit")
        ax[0].set(title=f"Bias data from the \nmeasurement for pixel ({ii},{jj})",xlabel="U in V", ylabel="C in fF")
        ax[0].errorbar(voltage_data, cap_data[ii, jj, :] * 1e15, yerr=None, fmt='o', label="Bias data")
        ax[1].set(title=f"Suited Bias data from the \nmeasurement for pixel ({ii},{jj})", xlabel="U in V", ylabel="1/ C^2 in 1/(fF)^2")
        ax[1].errorbar(voltage_data, 1 / (cap_data[ii, jj, :] * 1e15)**2, yerr=None, fmt='o', label="Bias data")

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
    # Read pixel map
    current_hist = data_group.HistCurr[:]
    cap_hist = analysis_group.HistCap[:]
    leak_hist = analysis_group.HistLeak[:]
    # cap_hist[:, 0] = np.nan
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
    for col in range(0, current_hist.shape[0]):
        for row in range(0, current_hist.shape[1]):
            if np.isfinite(current_hist[col, row, 0]):
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                res = np.polyfit(scan_parameters['frequency'], current_hist[col, row] * 1e9, deg=1, cov=True)
                f = np.arange(0, scan_parameters['frequency'].max() * 1.1, 0.1)
                actual_cap = cap_hist[col, row] * 1e9
                ax.plot(f, cap_hist[col, row] * 1e15 * f + leak_hist[col,row] * 1e9, color=cmap(0.6), ls='--', marker='',
                        label='Fit to data:\n$C_d = %.1f\,$fF' % actual_cap)
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


if __name__ == '__main__':
    plot_data(interpreted_data=os.path.expanduser('~/git/pixcap65/pixcap_LF_50x50_DC_R3_80V_HV.h5'))
