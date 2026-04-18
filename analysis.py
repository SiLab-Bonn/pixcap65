"""
Analysis of Pixcap65 data. Fits freq vs current to extract the capacitance. A 2D histogram containing the capacitance
for each pixel is stored.
"""
import time

import numpy as np
import tables as tb
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib.backends.backend_pdf import PdfPages

from utils_2 import walk_to_node, GroupType

GENERAL_PIXCAP_SHAPE = (40, 40)
COVARIANCE_PIXCAP_SHAPE = (40, 40, 4, 4)

FULL_MODEL_LABEL = "I_\\text{{full}}"
SIMPLE_MODEL_LABEL = "I_\\text{{approximation}}"
FULL_MODEL_EXPRESSION = "\\frac{{{u0}\\cdot{c}\\cdot{freq}+{i}}}{{1+{r}\\cdot{c}\\cdot{freq}}}"
SIMPLE_MODEL_EXPRESSION = "{u0}\\cdot{c\\cdot{freq}+{i}}"
FULL_MODEL_PARAMETER_DICT = {"c": "C", "r": "R", "i": "I_\\text{{Leakage}}", "u0": "U_{{0}}", "freq": "\\nu"}
SIMPLE_MODEL_PARAMETER_DICT = {"c": "C", "i": "I", "u0": "U_{{0}}", "freq": "\\nu"}
GENERAL_TRANSFORMATION_MATRIX = np.array([[1.e-12, 1.e-6, 1.e3, 1.e-6], [1.e-6, 1, 1.e9, 1], [1.e3, 1.e9, 1.e18, 1.e9], [1.e-6, 1, 1.e9, 1]])

def transform_covariance(cov):
    """
    transform_covariance

    Transforms the returned covariance such that the units match the specified ones.
    For the more advanced fits also the contributions of the fixed reference voltage parameter are removed from the covariance matrix.

    :param cov: covariance matrix to be transformed.
    :return: transformed covariance matrix.
    """
    cov = np.asarray(cov)
    match (cov.shape):
        case (2, 2):
            assert cov.shape == (2, 2)
            mask = np.array([[True, False, True, False], [False, False, False, False], [True, False, True, False], [False, False, False, False]])
            return cov * GENERAL_TRANSFORMATION_MATRIX[mask].reshape((2, 2))
        case (3, 3):
            assert cov.shape == (3, 3)
            # here it is necessary to reduce the parts from the covariance of u0 to
            mask = np.array([[True, False, True, False], [False, False, False, False], [True, False, True, False],
                             [False, False, False, False]])
            mask_reduction = np.array([[True, True, False], [True, True, False],[False, False, False]])
            return cov[mask_reduction].reshape((2,2)) * GENERAL_TRANSFORMATION_MATRIX[mask].reshape((2, 2))
        case (4, 4):
            assert cov.shape == (4, 4)
            # here it is necessary to reduce the parts from the covariance of u0 to
            mask = np.array([[True, True, True, False], [True, True, True, False], [True, True, True, False],
                                       [False, False, False, False]])
            return cov[mask].reshape((3,3)) * GENERAL_TRANSFORMATION_MATRIX[mask].reshape((3, 3))
        case _:
            raise ValueError("The dimension of the covariance matrix does not fit to any of the fitting functions and their parameters.")

def full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1):
    # ignores the reference voltage for now
    return (u0 * c * freq + i) / (1 + r * c * freq)

def simple_capacitance_model(freq, c=1e-6, i=0, u0=1):
    return u0*c*freq + i

def depletion_model(x, a=1, b=0):
    return a * x + b

def advanced_analysis(raw_data, use_kafe2=False, is_cv=False, base_path=None, plot=False, apply_contour=False, full_model=True,
                      first_boundaries=None, second_boundaries=None,):
    """
    advanced_analysis

    Implementation of the advanced analysis strategy for the capacitance measurement of a pixel sensor.
    For determination of the capacitance values non-linear fit algorithms are used.
    Depeneding on the choice of parameters either kafe2 or iminuit is used for least squares minimization.
    But keep in mind that this function serves as a wrapper for file access and modification around the actual analysis implementation.

    For measurements of the C-V-Characteristic of a sensor, the fits will be applied for every bias voltage measured.
    :param raw_data: path to the hdf file containing the raw data.
    :param use_kafe2: boolean, indicates whether kafe2 is used.
    :param is_cv: boolean, indicates whether this is a C-V characterization.
    :param base_path: path to the base group to look for the data.
    :param plot: boolean, indicates whether to plot the data. A output pdf object could be submitted here instead of an explicitly created one.
    :param apply_contour: boolean, indicates whether to determine the contours and try to plot them.
    :param full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
    """
    with tb.open_file(raw_data, mode='a') as in_file_h5:
        if base_path is None:
            base_group = in_file_h5.root
        else:
            base_group = walk_to_node(in_file_h5.root, base_path)
        if is_cv:
            # need to perform the analysis for every bias voltage
            cv_data = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]),
                              fill_value=np.nan)
            cv_err_data = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]),
                              fill_value=np.nan)
            if "analysis" in base_group.biasing:
                base_group.biasing.analysis._f_remove(recursive=True)
                time.sleep(1)
            for k, bias_voltage in enumerate(base_group.biasing.measurements.BiasVoltageHist):
                bias_name = f"bias_{bias_voltage}_V".replace('-', "M_").replace(".", "__")
                data_group = base_group.biasing.measurements[bias_name]
                ana_group = walk_to_node(base_group.biasing, str_join("/", "analysis", bias_name), create=True)
                advanced_analysis_delegate(in_file_h5, data_group, ana_group, use_kafe2=use_kafe2, plot=plot,
                                           apply_contour=apply_contour, full_model=full_model)
                cap_data = ana_group.HistCap[:]
                cap_error_data = ana_group.HistCapErr[:]
                cv_data[:, :, k] = cap_data[:, :]
                cv_err_data[:, :, k] = cap_error_data[:, :]

            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCHist", title="Histogram of the U-C-curve",
                                     filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False), obj=cv_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()
            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCErrHist", title="Error Histogram of the U-C-curve",
                                     filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False), obj=cv_err_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()
            if first_boundaries is not None and second_boundaries is not None:
                analyze_depletion_delegate(base_group.biasing.measurements, base_group.biasing.analysis,first_boundaries, second_boundaries,)
        else:
            if "analysis" in base_group.total_cap:
                base_group.total_cap.analysis._f_remove(recursive=True)
                time.sleep(1)
            ana_group = walk_to_node(base_group.total_cap, "analysis", create=True)
            advanced_analysis_delegate(in_file_h5, base_group.total_cap.measurements, ana_group, use_kafe2=use_kafe2, plot=plot, apply_contour=apply_contour, full_model=full_model)

def advanced_analysis_delegate(file: tb.File, data_group: tb.Group, result_group: tb.Group, use_kafe2=True, full_model=True, apply_contour=False, plot=False):
    """
        advanced_analysis_delegate

        Implementation of the advanced analysis strategy for the capacitance measurement of a pixel sensor.
        For determination of the capacitance values non-linear fit algorithms are used.
        Depeneding on the choice of parameters either kafe2 or iminuit is used for least squares minimization.
        If requested the fit results will also be plotted to verify the convergence of the fit.

        For measurements of the C-V-Characteristic of a sensor, the fits will be applied for every bias voltage measured.
        :param file: h5 file object containing the data to be analysed.
        :param data_group: hierachy group of the opend hdf file containing the raw data
        :param result_group: hierachy group of the opened hdf file to write the analysis results to.
        :param use_kafe2: boolean, indicates whether kafe2 is used.
        :param plot: boolean, indicates whether to plot the data. A output pdf object could be submitted here instead of an explicitly created one.
        :param apply_contour: boolean, indicates whether to determine the contours and try to plot them.
        :param full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        """

    # create array_like objects to save the analysis results temporarily.
    cap_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    cap_error_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    leak_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    leak_error_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    resistor_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    resistor_error_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    if full_model:
        fit_cov = np.full(shape=COVARIANCE_PIXCAP_SHAPE, fill_value=np.nan)
    else:
        fit_cov = np.full(shape=(40,40,3,3), fill_value=np.nan)

    # select the correct group to write the analysis results to
    group = result_group
    # select the correct group to read the data from
    select_group = data_group

    # Read pixel map
    check_leaf_unit(select_group.HistCurr, "A")
    current_hist = select_group.HistCurr[:]
    if "HistCurrErr" in select_group:
        check_leaf_unit(select_group.HistCurrErr, "A")
        current_error_hist = select_group.HistCurrErr[:]
    else:
        current_error_hist = np.full_like(current_hist, fill_value=np.nan)
    # Read scan parameters
    scan_parameters = select_group.scan_params[:]

    # prepare the fit model
    if full_model:
        effective_model = full_capacitance_model
        effective_label = FULL_MODEL_LABEL
        effective_expression = FULL_MODEL_EXPRESSION
        effective_parameter_dict = FULL_MODEL_PARAMETER_DICT
    else:
        effective_model = simple_capacitance_model
        effective_label = SIMPLE_MODEL_LABEL
        effective_expression = SIMPLE_MODEL_EXPRESSION
        effective_parameter_dict = SIMPLE_MODEL_PARAMETER_DICT

    if plot:
        output_pdf_name = f"{file.filename[:-3]}_{data_group._v_pathname.replace('/', '----')}_fit_results.pdf"
        output_pdf = plot if isinstance(plot, PdfPages) else PdfPages(output_pdf_name)
    # Fit pixel data in order to extract capacitance for each pixel

    for ii, jj in np.ndindex(current_hist.shape[:2]):
        if np.all(np.isfinite(current_hist[ii, jj, :])):
            if use_kafe2:
                from kafe2 import XYContainer
                from kafe2 import XYFit
                xy_data = XYContainer(scan_parameters['frequency'], current_hist[ii, jj])
                # errors?
                if np.all(np.isfinite(current_error_hist[ii, jj, :])):
                    xy_data.add_error('y', err_val=current_error_hist[ii, jj, :])
                fitter = XYFit(xy_data, model_function=effective_model)
                fitter.assign_model_function_latex_name(effective_label)
                fitter.assign_model_function_latex_expression(effective_expression)
                fitter.assign_parameter_latex_names(**effective_parameter_dict)
                fitter.fix_parameter('u0', 1)
                fitter.do_fit()
            else:
                # unused minuit
                if np.all(np.isfinite(current_error_hist[ii, jj, :])):
                    errors = current_error_hist[ii, jj, :]
                else:
                    errors = np.full_like(current_error_hist, fill_value=1)
                cost = LeastSquares(x=scan_parameters['frequency'], y=current_hist[ii, jj],
                                    yerror=errors, model=effective_model)
                if full_model:
                    fitter = Minuit(cost, c=1e-6, r=1e6, i=0, u0=1)
                else:
                    fitter = Minuit(cost, c=1e-6, i=0, u0=1)
                fitter.fixto('u0', 1)
                fitter.migrad()
                fitter.hesse()

            # extract the fit parameters
            if use_kafe2:
                assert fitter.did_fit
                cap = fitter.parameter_values['c'] * 1e-6  # convert to F
                cap_error = fitter.parameter_errors['c'] * 1e-6
                leakage = fitter.parameter_values['i'] * 1e9  # convert to nA
                leakage_error = fitter.parameter_errors['i'] * 1e9  # convert to nA
                if full_model:
                    # otherwise the requested information may not be present
                    resistor = fitter.parameter_values['r']
                    resistor_error = fitter.parameter_errors['r']
                else:
                    resistor = np.nan
                    resistor_error = np.nan
                fit_cov[ii, jj] = fitter.parameter_cov_mat

                if plot:
                    from kafe2 import Plot
                    fit_plot = Plot(fitter)
                    fit_plot.x_label = "$\\nu$ in MHz"
                    fit_plot.y_label = "$I$ in A"
                    fit_plot.plot(residual=True)
                    fit_plot.axes.set_title(f"Fit of the frequency dependence for pixel ({ii}, {jj})")
                    for (fig, axes) in zip(fit_plot.figures, fit_plot.axes):
                        for ax in axes.values():
                            ax.set_title(f"Fit of the frequency dependence for pixel ({ii}, {jj})")
                        output_pdf.savefig(fig, bbox_inches='tight')

                if apply_contour and plot:
                    from kafe2 import ContoursProfiler
                    cpf = ContoursProfiler(fitter)
                    cpf_figure = cpf.plot_profiles_contours_matrix()
                    from matplotlib.figure import Figure
                    assert isinstance(cpf_figure, Figure)
                    cpf_figure.axes[0][0].set_title(f"Contour profiles for pixel ({ii}, {jj})")
                    output_pdf.savefig(cpf_figure, bbox_inches='tight')

            else:
                # iMinuit is currently not in use
                cap = fitter.values['c'] * 1e-6  # convert to F
                cap_error = fitter.errors['c'] * 1e-6
                leakage = fitter.values['i'] * 1e9  # convert to nA
                leakage_error = fitter.errors['i'] * 1e9  # convert to nA
                if full_model:
                    # otherwise the requested information may not be present.
                    resistor = fitter.values['r']
                    resistor_error = fitter.errors['r']
                else:
                    resistor = np.nan
                    resistor_error = np.nan
                fit_cov[ii, jj] = fitter.covariance
                if apply_contour and plot:
                    assert isinstance(fitter, Minuit)
                    print(fitter.draw_mnmatrix())
                    from matplotlib import pyplot as plt
                    print(plt.get_fignums())
                    current_fig = plt.figure(plt.get_fignums()[0])
                    current_fig.axes[0, 0].set_title(f"Fit of the frequency dependence for pixel ({ii}, {jj})")
                    output_pdf.savefig(current_fig, bbox_inches='tight')

            if full_model:
                fit_cov[ii, jj, :3, :3] = transform_covariance(fit_cov[ii, jj])
            else:
                fit_cov[ii, jj, :2, :2] = transform_covariance(fit_cov[ii, jj])

        else:
            cap = np.nan
            cap_error = np.nan
            leakage = np.nan
            leakage_error = np.nan
            resistor = np.nan
            resistor_error = np.nan
            fit_cov[ii, jj] = np.full(shape=(4, 4), fill_value=np.nan)

        # temporarily save the results to memory
        cap_hist[ii, jj] = cap
        cap_error_hist[ii, jj] = cap_error
        leak_hist[ii, jj] = leakage
        leak_error_hist[ii, jj] = leakage_error
        resistor_hist[ii, jj] = resistor
        resistor_error_hist[ii, jj] = resistor_error

        # make sure that no plots/figures are opened anymore.
        from matplotlib import pyplot as plt
        plt.close('all')

    # Store capacitance values and specify the used units as an attribute.
    if plot and not isinstance(plot, PdfPages):
        output_pdf.close()
    temp_array = file.create_carray(group,
                       name='HistCap',
                       title='Capacitance Histogram',
                       obj=cap_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))
    temp_array.attrs["Units"] = "F"
    temp_array.flush()
    temp_array = file.create_carray(group,
                       name='HistCapErr',
                       title='Capacitance Error Histogram',
                       obj=cap_error_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))
    temp_array.attrs["Units"] = "F"
    temp_array.flush()

    temp_array = file.create_carray(group,
                       name='HistLeak',
                       title='Leakage Current Histogram',
                       obj=leak_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))
    temp_array.attrs["Units"] = "nA"
    temp_array.flush()
    temp_array = file.create_carray(group,
                       name='HistLeakErr',
                       title='Leakage Current Error Histogram',
                       obj=leak_error_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))
    temp_array.attrs["Units"] = "nA"
    temp_array.flush()

    temp_array = file.create_carray(group,
                       name='HistRes',
                       title='On-Resistance Histogram',
                       obj=resistor_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))
    temp_array.attrs["Units"] = "O"
    temp_array.flush()
    temp_array = file.create_carray(group,
                       name='HistResErr',
                       title='On-Resistance Error Histogram',
                       obj=resistor_error_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))
    temp_array.attrs["Units"] = "O"
    temp_array.flush()

    temp_array = file.create_carray(group,
                       name='HistFitCov',
                       title='Fit Covariance Matrix',
                       obj=fit_cov, filters=tb.Filters(complib='blosc',
                                                      complevel=5,
                                                      fletcher32=False)
                       )
    temp_array.attrs["Units"] = "{{F^2, F O, F nA, F V},{O F, O^2, O nA, O V},{nA F, nA O, nA^2, nA V}, {V F, V O, V nA, V^2}"
    temp_array.flush()

def str_join(delimiter, *args):
    return delimiter.join(args)

def analyze_data(raw_data, is_cv=False, base_path=None, first_boundaries=None, second_boundaries=None,):
    """
    analyze data

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
            cv_data = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]), fill_value=np.nan)
            cv_err_data = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]),
                                  fill_value=np.nan)
            if "analysis" in base_group.biasing:
                base_group.biasing.analysis._f_remove(recursive=True)
                time.sleep(1)
            for k, bias_voltage in enumerate(base_group.biasing.measurements.BiasVoltageHist):
                bias_name = f"bias_{bias_voltage}_V".replace('-', "M_").replace(".", "__")
                data_group = base_group.biasing.measurements[bias_name]
                ana_group = walk_to_node(base_group.biasing, str_join("/", "analysis", bias_name), create=True)
                analyze_data_delegate(in_file_h5, data_group, ana_group)
                cap_data = ana_group.HistCap[:]
                cap_error_data = ana_group.HistCapErr[:]
                cv_data[:, :, k] = cap_data[:, :]
                cv_err_data[:, :, k] = cap_error_data[:, :]

            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCHist", title="Histogram of the U-C-curve", filters=tb.Filters(complib='blosc',complevel=5,fletcher32=False), obj=cv_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()
            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCErrHist",
                                     title="Error Histogram of the U-C-curve",
                                     filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False),
                                     obj=cv_err_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()
            if first_boundaries is not None and second_boundaries is not None:
                analyze_depletion_delegate(base_group.biasing.measurements, base_group.biasing.analysis, first_boundaries, second_boundaries)
        else:
            if "analysis" in base_group.total_cap:
                base_group.total_cap.analysis._f_remove(recursive=True)
                time.sleep(1)
            ana_group = walk_to_node(base_group.total_cap, "analysis", create=True)
            analyze_data_delegate(in_file_h5, base_group.total_cap.measurements, ana_group)

def analyze_data_delegate(file: tb.File, data_group: tb.Group, result_group: tb.Group):
    """
        analyze data_delegate

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
    # create array like objects to temporarily save the analysis results
    cap_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)  # capacitance for each pixel
    cap_error_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    leak_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    leak_error_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    fit_cov = np.full(shape=(40, 40, 2, 2), fill_value=np.nan)

    # select the correct group to save the analysis results to
    group = result_group
    # select the correct group to read the data from
    select_group = data_group

    # Read pixel map
    check_leaf_unit(select_group.HistCurr, "A")
    current_hist = select_group.HistCurr[:]
    # Read scan parameters
    scan_parameters = select_group.scan_params[:]

    # Fit pixel data in order to extract capacitance for each pixel
    for col in range(current_hist.shape[0]):
        for row in range(current_hist.shape[1]):
            if np.isfinite(current_hist[col, row, 0]):
                res = np.polyfit(scan_parameters['frequency'], current_hist[col, row], deg=1, cov=True)
                cap = res[0][0] * 1e-6  # convert to F
                leak = res[0][1] * 1e9  # convert to nA
                cov = transform_covariance(res[1])
                cap_error = np.sqrt(cov[0,0])
                leak_error = np.sqrt(cov[1,1])
            else:
                cap = np.nan
                cap_error = np.nan
                leak = np.nan
                leak_error = np.nan
                cov = np.full(shape=(2, 2), fill_value=np.nan)

            cap_hist[col, row] = cap
            cap_error_hist[col, row] = cap_error
            leak_hist[col, row] = leak
            leak_error_hist[col, row] = leak_error
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
    temp_array = file.create_carray(group,
                       name='HistCap',
                       title='Capacitance Histogram',
                       obj=cap_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))
    temp_array.attrs["Units"] = "F"
    temp_array.flush()
    temp_array = file.create_carray(group,
                                    name='HistCapErr',
                                    title='Capacitance Error Histogram',
                                    obj=cap_error_hist,
                                    filters=tb.Filters(complib='blosc',
                                                       complevel=5,
                                                       fletcher32=False))
    temp_array.attrs["Units"] = "F"
    temp_array.flush()

    temp_array = file.create_carray(group,
                       name='HistLeak',
                       title='Leakage Current Histogram',
                       obj=leak_hist,
                       filters=tb.Filters(complib='blosc',
                                          complevel=5,
                                          fletcher32=False))
    temp_array.attrs["Units"] = "nA"
    temp_array.flush()
    temp_array = file.create_carray(group,
                                    name='HistLeakErr',
                                    title='Leakage Current Error Histogram',
                                    obj=leak_error_hist,
                                    filters=tb.Filters(complib='blosc',
                                                       complevel=5,
                                                       fletcher32=False))
    temp_array.attrs["Units"] = "nA"
    temp_array.flush()

    temp_array = file.create_carray(group,
                       name='HistFitCov',
                       title='Fit Covariance Matrix',
                       obj=fit_cov,filters=tb.Filters(complib='blosc',complevel=5,fletcher32=False)
                       )
    temp_array.attrs["Units"] = "{{F^2, F nA},{nA F, nA^2}}"
    temp_array.flush()

def check_leaf_unit(leaf, unit: str):
    """
    check_leaf_unit(leaf, unit)

    Verify that the table or array has a units attribute and the unit is the one expected.
    As last step return the data structure requested.

    This may not work for tables with multiple units (one per column)
    :param leaf: Data structure to be verified and extracted.
    :param unit: Expected unit for the data structure.
    :return: array_like of the data structures contents.
    """
    if "Units" not in leaf.attrs or leaf.attrs["Units"] != unit:
        raise AssertionError
    return leaf[:]

def analyze_depletion_delegate(data_group: GroupType, analysis_group: GroupType, first_boundaries, second_boundaries,):
    voltage_data = check_leaf_unit(data_group.BiasVoltageHist, "V")
    first_lower, first_upper = first_boundaries
    second_lower, second_upper = second_boundaries
    depletion_voltage = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    depletion_error = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    fit_parameter_estimators = np.full(shape=(40,40,4), fill_value=np.nan)
    fit_parameter_errors = np.full(shape=(40, 40, 4), fill_value=np.nan)
    for ii, jj in np.ndindex(GENERAL_PIXCAP_SHAPE):
        # verify and extract the raw data for further analysis
        check_leaf_unit(analysis_group.UCHist, "F")
        check_leaf_unit(analysis_group.UCErrHist, "F")
        cap_data = analysis_group.UCHist[:, :, :]
        cap_error_data = analysis_group.UCErrHist[:, :, :]
        if np.any(np.isnan(cap_data[ii, jj, :])):
            continue

        # extract the information about the depletion voltage
        first_section_upper_mask = voltage_data <= first_upper
        first_section_lower_mask = voltage_data >= first_lower
        first_section_mask = np.logical_and(first_section_upper_mask, first_section_lower_mask)

        second_section_upper_mask = voltage_data <= second_upper
        second_section_lower_mask = voltage_data >= second_lower
        second_section_mask = np.logical_and(second_section_upper_mask, second_section_lower_mask)

        # extract and select the usable voltage and capacitance data for the two fit ranges.
        effective_capacitance_data = np.reciprocal(cap_data[ii, jj, :] * 1e15) ** 2
        effective_capacitance_error_data = np.reciprocal(cap_data[ii, jj, :] * 1e15) ** 3 * cap_error_data[ii, jj, :] if np.all(np.isfinite(cap_error_data[ii, jj, :])) else cap_error_data[ii, jj, :]
        first_voltage_data = voltage_data[first_section_mask]
        second_voltage_data = voltage_data[second_section_mask]
        first_cap_data = effective_capacitance_data[first_section_mask]
        second_cap_data = effective_capacitance_data[second_section_mask]
        first_cap_error_data = effective_capacitance_error_data[first_section_mask]
        second_cap_error_data = effective_capacitance_error_data[second_section_mask]

        # perform fits to the two boundary regions specified to estimate the two distinc behaviours.
        if np.all(np.isfinite(first_cap_error_data)):
            from iminuit import Minuit
            from iminuit.cost import LeastSquares
            cost = LeastSquares(first_voltage_data, first_cap_data, first_cap_error_data, depletion_model)
            m = Minuit(cost, a=1, b=0)
            m.migrad()
            m.hesse()
            first_dep_parameters = np.array([m.values['a'], m.values['b']])
            print(first_dep_parameters)
            first_dep_errors = np.array([m.errors['a'], m.errors['b']])
            first_dep_cov = m.covariance
        else:
            first_result = np.polyfit(first_voltage_data, first_cap_data, deg=1, cov=True)
            first_dep_parameters = np.asarray(first_result[0])
            first_dep_cov = np.asarray(first_result[1])
            first_dep_errors = np.sqrt(np.diag(first_dep_cov))
        if np.all(np.isfinite(second_cap_error_data)):
            from iminuit import Minuit
            from iminuit.cost import LeastSquares
            cost = LeastSquares(second_voltage_data, second_cap_data, second_cap_error_data, depletion_model)
            m = Minuit(cost, a=1, b=0)
            m.migrad()
            m.hesse()
            second_dep_parameters = np.array([m.values['a'], m.values['b']])
            second_dep_errors = np.array([m.errors['a'], m.errors['b']])
            second_dep_cov = m.covariance
        else:
            second_result = np.polyfit(second_voltage_data, second_cap_data, deg=1, cov=True)
            second_dep_parameters = np.asarray(second_result[0])
            second_dep_cov = np.asarray(second_result[1])
            second_dep_errors = np.sqrt(np.diag(second_dep_cov))

        # estimate the depletion voltage
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
        first_error_term = (dww * first_norm_factor + dvw * (
                second_dep_parameters[1] - first_dep_parameters[1])) / first_norm_factor ** 3
        second_error_term = (dyy * first_norm_factor + dxy * (
                second_dep_parameters[1] - first_dep_parameters[1])) / first_norm_factor ** 3
        third_error_term = (dvv * (second_dep_parameters[1] - first_dep_parameters[1]) - dwv * first_norm_factor) * (
                second_dep_parameters[1] - first_dep_parameters[1]) / first_norm_factor ** 4
        fourth_error_term = (dyx * first_norm_factor + dxx * (second_dep_parameters[1] - first_dep_parameters[1])) * (
                second_dep_parameters[1] - first_dep_parameters[1]) / first_norm_factor ** 4
        dep_voltage_error = np.sqrt(first_error_term + second_error_term + third_error_term + fourth_error_term)
        depletion_voltage[ii, jj] = dep_voltage
        depletion_error[ii, jj] = dep_voltage_error
        print("The depletion voltage is {voltage}+-{error}".format(voltage=dep_voltage, error=dep_voltage_error))
        fit_parameter_estimators[ii, jj, :2] = first_dep_parameters
        fit_parameter_estimators[ii, jj, 2:] = second_dep_parameters
        fit_parameter_errors[ii, jj, :2] = first_dep_errors
        fit_parameter_errors[ii, jj, 2:] = second_dep_errors

    file_h5 = analysis_group._v_file
    assert isinstance(file_h5, tb.File)
    temp_array = file_h5.create_carray(where=analysis_group, name="DepletionHist", title="Histogram of the depletion voltages", obj=depletion_voltage, filters=tb.Filters(complevel=5, complib='blosc', fletcher32=False))
    temp_array.attrs["Units"] = "V"
    temp_array.flush()
    temp_array = file_h5.create_carray(where=analysis_group, name="DepletionErrHist",
                                       title="Histogram of the depletion voltage erors", obj=depletion_error,
                                       filters=tb.Filters(complevel=5, complib='blosc', fletcher32=False))
    temp_array.attrs["Units"] = "V"
    temp_array.flush()
    temp_array = file_h5.create_carray(where=analysis_group, name="DepFitParamHist",
                                       title="Histogram of the depletion voltages fit parameters", obj=fit_parameter_estimators,
                                       filters=tb.Filters(complevel=5, complib='blosc', fletcher32=False))
    temp_array.attrs["Units"] = "NONE"
    temp_array.flush()
    temp_array = file_h5.create_carray(where=analysis_group, name="DepFitParamErrHist",
                                       title="Histogram of the depletion voltages fit parameter errors",
                                       obj=fit_parameter_errors,
                                       filters=tb.Filters(complevel=5, complib='blosc', fletcher32=False))
    temp_array.attrs["Units"] = "NONE"
    temp_array.flush()


        # TODO: analyze the depletion depths and the charge carrier densities here; Write everything back into a table or something similar.

if __name__ == '__main__':
    # analyze_data(raw_data='/home/silab/git/pixcap65/pixcap_full_data_image1.h5')
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic", is_cv=True, first_boundaries=(-60,-40), second_boundaries=(-5,0),)
    # advanced_analysis(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3")
