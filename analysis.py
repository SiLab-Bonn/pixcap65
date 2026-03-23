"""
Analysis of Pixcap65 data. Fits freq vs current to extract the capacitance. A 2D histogram containing the capcitance for each pixel is stored.
"""

import tables as tb
import matplotlib.pyplot as plt
import numpy as np


def full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1):
    # ignores the reference voltage for now
    return (u0 * c * freq + i) / (1 + r * c * freq)


def advanced_analysis(raw_data):
    cap_hist = np.full(shape=(40, 40), fill_value=np.nan)
    cap_error_hist = np.full(shape=(40, 40), fill_value=np.nan)
    leak_hist = np.full(shape=(40, 40), fill_value=np.nan)
    leak_error_hist = np.full(shape=(40, 40), fill_value=np.nan)
    resistor_hist = np.full(shape=(40, 40), fill_value=np.nan)
    resistor_error_hist = np.full(shape=(40, 40), fill_value=np.nan)
    fit_cov = np.full(shape=(40, 40, 4, 4), fill_value=np.nan)

    from iminuit import Minuit
    from iminuit.cost import LeastSquares
    from kafe2 import XYContainer
    from kafe2 import XYFit
    with tb.open_file(raw_data, mode='a') as in_file_h5:
        # select the correct group to write the analysis results to
        group = in_file_h5.root

        # select the correct group to read the data from
        select_group = in_file_h5.root

        # Read pixel map
        current_hist = select_group.HistCurr[:]
        # Read scan parameters
        scan_parameters = select_group.scan_params[:]

        # prepare the fit model
        # Fit pixel data in order to extract capacitance for each pixel
        print(current_hist.shape)
        for ii, jj in np.indices(current_hist.shape):
            if np.isfinite(current_hist[ii, jj, 0]):
                xy_data = XYContainer(scan_parameters['frequency'], current_hist[ii, jj])
                # errors?
                xy_fit = XYFit(xy_data, model_function=full_capacitance_model)
                xy_fit.fix_parameter('u0', 1)
                fit_results = xy_fit.do_fit()

                # unused minuit
                # cost = LeastSquares(scan_parameters['frequency'], current_hist[ii, jj],
                #                     np.zeros_like(scan_parameters['frequency']), model=full_capcitance_model)
                # m = Minuit(cost, c=1e-6, r=1e6, i=0)
                # m.fixto('u0', 1)
                # m.migrad()
                # m.hesse()

                # extract the fit parameters
                cap = fit_results['parameter_values']['c'] * 1e-6  # convert to F
                cap_error = fit_results['parameter_errors']['c'] * 1e-6
                leakage = fit_results['parameter_values']['i']
                leakage_error = fit_results['parameter_errors']['i']
                resistor = fit_results['parameter_values']['r']
                resistor_error = fit_results['parameter_errors']['r']
                fit_cov[ii, jj] = fit_results['covariance_matrix']

                # iMinuit is currently not in use
                # cap = m.values['c'] * 1e-6  # convert to F
                # cap_error = m.errors['c'] * 1e-6
                # leakage = m.values['i']
                # leakage_error = m.errors['i']
                # resistor = m.values['r']
                # resistor_error = m.errors['r']
                # fit_cov[ii, jj] = m.covariance
            else:
                cap = np.nan
                cap_error = np.nan
                leakage = np.nan
                leakage_error = np.nan
                resistor = np.nan
                resistor_error = np.nan
                fit_cov[ii, jj] = np.full(shape=(4, 4), fill_value=np.nan)

            cap_hist[ii, jj] = cap
            cap_error_hist[ii, jj] = cap_error
            leak_hist[ii, jj] = leakage
            leak_error_hist[ii, jj] = leakage_error
            resistor_hist[ii, jj] = resistor
            resistor_error_hist[ii, jj] = resistor_error

        # Store capacitance values
        in_file_h5.create_carray(group,
                                 name='HistCap',
                                 title='Capacitance Histogram',
                                 obj=cap_hist,
                                 filters=tb.Filters(complib='blosc',
                                                    complevel=5,
                                                    fletcher32=False))
        in_file_h5.create_carray(group,
                                 name='HistCapErr',
                                 title='Capacitance Error Histogram',
                                 obj=cap_error_hist,
                                 filters=tb.Filters(complib='blosc',
                                                    complevel=5,
                                                    fletcher32=False))

        in_file_h5.create_carray(group,
                                 name='HistLeak',
                                 title='Leakage Current Histogram',
                                 obj=leak_hist,
                                 filters=tb.Filters(complib='blosc',
                                                    complevel=5,
                                                    fletcher32=False))
        in_file_h5.create_carray(group,
                                 name='HistLeakErr',
                                 title='Leakage Current Error Histogram',
                                 obj=leak_error_hist,
                                 filters=tb.Filters(complib='blosc',
                                                    complevel=5,
                                                    fletcher32=False))

        in_file_h5.create_carray(group,
                                 name='HistRes',
                                 title='On-Resistance Histogram',
                                 obj=resistor_hist,
                                 filters=tb.Filters(complib='blosc',
                                                    complevel=5,
                                                    fletcher32=False))
        in_file_h5.create_carray(group,
                                 name='HistResErr',
                                 title='On-Resistance Error Histogram',
                                 obj=resistor_error_hist,
                                 filters=tb.Filters(complib='blosc',
                                                    complevel=5,
                                                    fletcher32=False))

        in_file_h5.create_carray(group,
                                 name='HistFitCov',
                                 title='Fit Covariance Matrix',
                                 obj=fit_cov,
                                 )


def analyze_data(raw_data):
    cap_hist = np.full(shape=(40, 40), fill_value=np.nan)  # capacitance for each pixel
    leak_hist = np.full(shape=(40, 40), fill_value=np.nan)
    fit_cov = np.full(shape=(40, 40, 2, 2), fill_value=np.nan)
    with tb.open_file(raw_data, mode='a') as in_file_h5:
        # select the correct group to save the analysis results to
        group = in_file_h5.root

        # select the correct group to read the data from
        select_group = in_file_h5.root

        # Read pixel map
        current_hist = select_group.HistCurr[:]
        # Read scan parameters
        scan_parameters = select_group.scan_params[:]

        # Fit pixel data in order to extract capacitance for each pixel
        for col in range(current_hist.shape[0]):
            for row in range(current_hist.shape[1]):
                if np.isfinite(current_hist[col, row, 0]):
                    res = np.polyfit(scan_parameters['frequency'], current_hist[col, row], deg=1, cov=True)
                    cap = res[0][0] * 1e-6  # convert to F
                    leak = res[0][1]
                    cov = res[1]
                else:
                    cap = np.nan
                    leak = np.nan
                    cov = np.full(shape=(2, 2), fill_value=np.nan)

                cap_hist[col, row] = cap
                leak_hist[col, row] = leak
                fit_cov[col, row] = cov

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
        in_file_h5.create_carray(group,
                                 name='HistCap',
                                 title='Capacitance Histogram',
                                 obj=cap_hist,
                                 filters=tb.Filters(complib='blosc',
                                                    complevel=5,
                                                    fletcher32=False))

        in_file_h5.create_carray(group,
                                 name='HistLeak',
                                 title='Leakage Current Histogram',
                                 obj=leak_hist,
                                 filters=tb.Filters(complib='blosc',
                                                    complevel=5,
                                                    fletcher32=False))

        in_file_h5.create_carray(group,
                                 name='HistFitCov',
                                 title='Fit Covariance Matrix',
                                 obj=fit_cov,
                                 )


if __name__ == '__main__':
    analyze_data(raw_data='/home/silab/git/pixcap65/pixcap_full_data_image1.h5')
