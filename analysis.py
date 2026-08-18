"""
Analysis of Pixcap65 data. Fits freq vs current to extract the capacitance. A 2D histogram containing the capcitance for each pixel is stored.
"""

import tables as tb
import matplotlib.pyplot as plt
import numpy as np


def analyze_data(raw_data):
    cap_hist = np.full(shape=(40, 40), fill_value=np.nan)  # capacitance for each pixel
    with tb.open_file(raw_data, mode='a') as in_file_h5:
        # Read pixel map
        current_hist = in_file_h5.root.HistCurr[:]
        # Read scan parameters
        scan_parameters = in_file_h5.root.scan_params[:]

        # Fit pixel data in order to extract capacitance for each pixel
        for col in range(current_hist.shape[0]):
            for row in range(current_hist.shape[1]):
                if np.isfinite(current_hist[col, row, 0]):
                    res = np.polyfit(scan_parameters['frequency'], current_hist[col, row], deg=1, cov=True)
                    cap = res[0][0] * 1e-6  # convert to F
                else:
                    cap = np.nan

                cap_hist[col, row] = cap


            # #apply linear fit to measured current values; also returns covariance matrix
            # matrix = np.polyfit(freq_sweep_array, current_array, 1, cov=True)

            # a, b = matrix[0][0], matrix[0][1]
            # #da = matrix[1][0][0] #squared fit error of a
            # #db = matrix[1][1][1] #squared fit error of b

            # #data structure in txt file: "slope, offset (y-intercept)"

            # # fit_fn = a*freq_sweep_array + b
            # # pl.plot(freq_sweep_array, current_array, 'o', label = 'COL({i_col})PIX(0)'.format(i_col=i_col))
            # # pl.plot(freq_sweep_array, fit_fn, label = 'a={a:.3E}, b={b:.3E}'.format(a=a, b=b))

        # Store capacitance values
        in_file_h5.create_carray(in_file_h5.root,
                                 name='HistCap',
                                 title='Capacitance Histogram',
                                 obj=cap_hist,
                                 filters=tb.Filters(complib='blosc',
                                                    complevel=5,
                                                    fletcher32=False))


if __name__ == '__main__':
    analyze_data(raw_data='/home/silab/git/pixcap65/pixcap_full_data_image1.h5')
