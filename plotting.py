"""
The latest version of the Pixcap65 test script for measuring the total pixel capacitance.

Changes compared to original script:
- Remote control of depletion voltage source
- Reading some current values before actual measurement to avoid incorrect currents due to initial oscillation effects of SMU
- Vary the order of column/row routing and switching frequency using the reversed arrays (uncomment corresponding lines in code)
- Fit also returns covariance matrix in order to extract the errors of the fit parameters if needed
- Output in txt file also includes offset (y-intercept) next to the slope
"""

import tables as tb
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

cmap = cm.get_cmap('viridis')


def plot_data(interpreted_data):
    with PdfPages(interpreted_data[:-3] + '.pdf') as output_pdf:
        with tb.open_file(interpreted_data, mode='r') as in_file_h5:
            # Read pixel map
            current_hist = in_file_h5.root.HistCurr[:]
            cap_hist = in_file_h5.root.HistCap[:]
            # cap_hist[:, 0] = np.nan
            # Read scan parameters
            scan_parameters = in_file_h5.root.scan_params[:]

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
            for col in range(0, current_hist.shape[0]):
                for row in range(0, current_hist.shape[1]):
                    if np.isfinite(current_hist[col, row, 0]):
                        fig = Figure()
                        _ = FigureCanvas(fig)
                        ax = fig.add_subplot(111)
                        res = np.polyfit(scan_parameters['frequency'], current_hist[col, row] * 1e9, deg=1, cov=True)
                        f = np.arange(0, scan_parameters['frequency'].max() * 1.1, 0.1)
                        ax.plot(f, res[0][0] * f + res[0][1], color=cmap(0.6), ls='--', marker='', label='Fit to data:\n$C_d = %.1f\,$fF' % res[0][0])
                        ax.plot(scan_parameters['frequency'], current_hist[col, row] * 1e9, marker='o', ls='', label='Pixel({i_col},{i_row})'.format(i_col=col, i_row=row), color=cmap(0.2))
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
    plot_data(interpreted_data='/home/silab/git/pixcap65/pixcap_LF_50x50_DC_R3_80V_HV.h5')
