"""
Plotting of Pixcap65 data.
"""
import os.path

import numpy as np
import tables as tb
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

cmap = cm.get_cmap('viridis')


def plot_data(interpreted_data):
    with PdfPages(interpreted_data[:-3] + '.pdf') as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            plot_data_delegate(in_file_h5.root, in_file_h5.root, output_pdf)


def plot_data_delegate(data_group: tb.Group, analysis_group: tb.Group, output_pdf: PdfPages):
    # Read pixel map
    current_hist = data_group.HistCurr[:]
    cap_hist = analysis_group.HistCap[:]
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
                ax.plot(f, res[0][0] * f + res[0][1], color=cmap(0.6), ls='--', marker='',
                        label='Fit to data:\n$C_d = %.1f\,$fF' % res[0][0])
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
