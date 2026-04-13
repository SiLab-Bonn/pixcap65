"""
Analysis of Pixcap65 data. Fits freq vs current to extract the capacitance. A 2D histogram containing the capacitance
for each pixel is stored.
"""

import numpy as np
import tables as tb
from iminuit import Minuit
from iminuit.cost import LeastSquares

from utils_2 import walk_to_node


def full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1):
    # ignores the reference voltage for now
    return (u0 * c * freq + i) / (1 + r * c * freq)

# TODO: REFACTOR the actual measurement paths in the file structure.
def advanced_analysis(raw_data, use_kafe2=False, is_cv=False, base_path=None):
    with tb.open_file(raw_data, mode='a') as in_file_h5:
        if base_path is None:
            base_group = in_file_h5.root
        else:
            base_group = walk_to_node(in_file_h5.root, base_path)
        if is_cv:
            # need to perform the analysis for every bias voltage
            cv_data = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]),
                              fill_value=np.nan)
            for k, bias_voltage in enumerate(base_group.biasing.measurements.BiasVoltageHist):
                bias_name = f"bias_{bias_voltage}_V".replace('-', "M_").replace(".", "__")
                data_group = base_group.biasing.measurements[bias_name]
                ana_group = walk_to_node(base_group.biasing, str_join("/", "analysis", bias_name), create=True)
                advanced_analysis_delegate(in_file_h5, data_group, ana_group, use_kafe2=use_kafe2)
                cap_data = ana_group.HistCap[:]
                cv_data[:, :, k] = cap_data[:, :]

            in_file_h5.create_carray(base_group.biasing.analysis, name="UCHist", title="Histogram of the U-C-curve",
                                     filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False), obj=cv_data)
        else:
            ana_group = walk_to_node(base_group.total_cap, "analysis", create=True)
            advanced_analysis_delegate(in_file_h5, base_group.total_cap.measurements, ana_group, use_kafe2=use_kafe2)

def advanced_analysis_delegate(file: tb.File, data_group: tb.Group, result_group: tb.Group, use_kafe2=True, is_cv=False):
    cap_hist = np.full(shape=(40, 40), fill_value=np.nan)
    cap_error_hist = np.full(shape=(40, 40), fill_value=np.nan)
    leak_hist = np.full(shape=(40, 40), fill_value=np.nan)
    leak_error_hist = np.full(shape=(40, 40), fill_value=np.nan)
    resistor_hist = np.full(shape=(40, 40), fill_value=np.nan)
    resistor_error_hist = np.full(shape=(40, 40), fill_value=np.nan)
    fit_cov = np.full(shape=(40, 40, 4, 4), fill_value=np.nan)

    from kafe2 import XYContainer
    from kafe2 import XYFit
    # select the correct group to write the analysis results to
    group = result_group

    # select the correct group to read the data from
    select_group = data_group

    # Read pixel map
    current_hist = select_group.HistCurr[:]
    # Read scan parameters
    scan_parameters = select_group.scan_params[:]

    # prepare the fit model
    # Fit pixel data in order to extract capacitance for each pixel
    print(current_hist.shape)
    for ii, jj in np.indices(current_hist.shape):
        if np.isfinite(current_hist[ii, jj, 0]):
            if use_kafe2:
                xy_data = XYContainer(scan_parameters['frequency'], current_hist[ii, jj])
                # errors?
                xy_fit = XYFit(xy_data, model_function=full_capacitance_model)
                xy_fit.fix_parameter('u0', 1)
                fit_results = xy_fit.do_fit()
            else:
                # unused minuit
                cost = LeastSquares(scan_parameters['frequency'], current_hist[ii, jj],
                                    np.zeros_like(scan_parameters['frequency']), model=full_capacitance_model)
                m = Minuit(cost, c=1e-6, r=1e6, i=0)
                m.fixto('u0', 1)
                m.migrad()
                m.hesse()

            # extract the fit parameters
            if use_kafe2:
                cap = fit_results['parameter_values']['c'] * 1e-6  # convert to F
                cap_error = fit_results['parameter_errors']['c'] * 1e-6
                leakage = fit_results['parameter_values']['i']
                leakage_error = fit_results['parameter_errors']['i']
                resistor = fit_results['parameter_values']['r']
                resistor_error = fit_results['parameter_errors']['r']
                fit_cov[ii, jj] = fit_results['covariance_matrix']
            else:
                # iMinuit is currently not in use
                cap = m.values['c'] * 1e-6  # convert to F
                cap_error = m.errors['c'] * 1e-6
                leakage = m.values['i']
                leakage_error = m.errors['i']
                resistor = m.values['r']
                resistor_error = m.errors['r']
                fit_cov[ii, jj] = m.covariance
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
    file.create_carray(group,
                       name='HistCap',
                       title='Capacitance Histogram',
                       obj=cap_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))
    file.create_carray(group,
                       name='HistCapErr',
                       title='Capacitance Error Histogram',
                       obj=cap_error_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))

    file.create_carray(group,
                       name='HistLeak',
                       title='Leakage Current Histogram',
                       obj=leak_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))
    file.create_carray(group,
                       name='HistLeakErr',
                       title='Leakage Current Error Histogram',
                       obj=leak_error_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))

    file.create_carray(group,
                       name='HistRes',
                       title='On-Resistance Histogram',
                       obj=resistor_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))
    file.create_carray(group,
                       name='HistResErr',
                       title='On-Resistance Error Histogram',
                       obj=resistor_error_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))

    file.create_carray(group,
                       name='HistFitCov',
                       title='Fit Covariance Matrix',
                       obj=fit_cov, filters=tb.Filters(complib='blosc',
                                                      complevel=5,
                                                      fletcher32=False)
                       )

def str_join(delimiter, *args):
    return delimiter.join(args)

def analyze_data(raw_data, is_cv=False, base_path=None):
    with tb.open_file(raw_data, mode='a') as in_file_h5:
        if base_path is None:
            base_group = in_file_h5.root
        else:
            base_group = walk_to_node(in_file_h5.root, base_path)
        if is_cv:
            # need to perform the analysis for every bias voltage
            cv_data = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]), fill_value=np.nan)
            for k, bias_voltage in enumerate(base_group.biasing.measurements.BiasVoltageHist):
                bias_name = f"bias_{bias_voltage}_V".replace('-', "M_").replace(".", "__")
                data_group = base_group.biasing.measurements[bias_name]
                ana_group = walk_to_node(base_group.biasing, str_join("/", "analysis", bias_name), create=True)
                analyze_data_delegate(in_file_h5, data_group, ana_group)
                cap_data = ana_group.HistCap[:]
                cv_data[:, :, k] = cap_data[:, :]

            in_file_h5.create_carray(base_group.biasing.analysis, name="UCHist", title="Histogram of the U-C-curve", filters=tb.Filters(complib='blosc',complevel=5,fletcher32=False), obj=cv_data)
        else:
            ana_group = walk_to_node(base_group.total_cap, "analysis", create=True)
            analyze_data_delegate(in_file_h5, base_group.total_cap.measurements, ana_group)

def analyze_data_delegate(file: tb.File, data_group: tb.Group, result_group: tb.Group):
    cap_hist = np.full(shape=(40, 40), fill_value=np.nan)  # capacitance for each pixel
    leak_hist = np.full(shape=(40, 40), fill_value=np.nan)
    fit_cov = np.full(shape=(40, 40, 2, 2), fill_value=np.nan)
    # select the correct group to save the analysis results to
    group = result_group

    # select the correct group to read the data from
    select_group = data_group

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
                # print(f"unexpected infinite value {current_hist[col, row, 0]} for {col} and {row};")
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
    file.create_carray(group,
                       name='HistCap',
                       title='Capacitance Histogram',
                       obj=cap_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))

    file.create_carray(group,
                       name='HistLeak',
                       title='Leakage Current Histogram',
                       obj=leak_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))

    file.create_carray(group,
                       name='HistFitCov',
                       title='Fit Covariance Matrix',
                       obj=fit_cov,filters=tb.Filters(complib='blosc',complevel=5,fletcher32=False)
                       )


if __name__ == '__main__':
    analyze_data(raw_data='/home/silab/git/pixcap65/pixcap_full_data_image1.h5')
