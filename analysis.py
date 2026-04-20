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
SILICON_V_BIAS = 0.7
FULL_MODEL_LABEL = "I_\\text{{full}}"
SIMPLE_MODEL_LABEL = "I_\\text{{approximation}}"
FULL_MODEL_EXPRESSION = "\\frac{{{u0}\\cdot{c}\\cdot{freq}+{i}}}{{1+{r}\\cdot{c}\\cdot{freq}}}"
SIMPLE_MODEL_EXPRESSION = "{u0}\\cdot{c\\cdot{freq}+{i}}"
FULL_MODEL_PARAMETER_DICT = {"c": "C", "r": "R", "i": "I_\\text{{Leakage}}", "u0": "U_{{0}}", "freq": "\\nu"}
SIMPLE_MODEL_PARAMETER_DICT = {"c": "C", "i": "I", "u0": "U_{{0}}", "freq": "\\nu"}
GENERAL_TRANSFORMATION_MATRIX = np.array(
    [[1.e-12, 1.e-6, 1.e3, 1.e-6], [1.e-6, 1, 1.e9, 1], [1.e3, 1.e9, 1.e18, 1.e9], [1.e-6, 1, 1.e9, 1]])
EPS_SILICON = 11.7
GLOBAL_FILTERS = tb.Filters(complevel=5, complib='blosc', shuffle=False, fletcher32=False)


class DepletionData(tb.IsDescription):
    col = tb.Int64Col(pos=0)
    row = tb.Int64Col(pos=1)
    V = tb.Float32Col(pos=2)
    NA = tb.Float32Col(pos=3)
    ND = tb.Float32Col(pos=4)


def handle_kafe2_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf, contours_title=None):
    from kafe2 import Plot
    fit_plot = Plot(fit_object)
    fit_plot.x_label = x_label
    fit_plot.y_label = y_label
    fit_plot.plot(residual=True)
    for fig_dict in fit_plot.axes:
        for ax in fig_dict.values():
            ax.set_title(title)
    for (fig, axes) in zip(fit_plot.figures, fit_plot.axes):
        for ax in axes.values():
            ax.set_title(title)
        pdf.savefig(fig, bbox_inches='tight')

    if apply_contour:
        from kafe2 import ContoursProfiler
        from matplotlib.figure import Figure
        cpf = ContoursProfiler(fit_object)
        cpf_figure = cpf.plot_profiles_contours_matrix()
        assert isinstance(cpf_figure, Figure)
        cpf_figure.axes[0][0].set_title(contours_title)
        pdf.savefig(cpf_figure, bbox_inches='tight')


def handle_minuit_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf,
                                   contours_title=None):
    from matplotlib import pyplot as plt
    assert isinstance(fit_object, Minuit)
    fig, ax = plt.subplots()
    plt.figure(fig.number)
    print(plt.gca)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fit_object.visualize()
    pdf.savefig(fig, bbox_inches='tight')

    if apply_contour:
        fig, ax = fit_object.draw_mnmatrix()
        fig.suptitle(contours_title)
        print(fit_object.draw_mnmatrix())
        print(plt.get_fignums())
        current_fig = plt.figure(plt.get_fignums()[0])
        current_fig.axes[0, 0].set_title(contours_title)
        pdf.savefig(current_fig, bbox_inches='tight')
        pdf.savefig(fig, bbox_inches='tight')


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
            mask = np.array([[True, False, True, False], [False, False, False, False], [True, False, True, False],
                             [False, False, False, False]])
            return cov * GENERAL_TRANSFORMATION_MATRIX[mask].reshape((2, 2))
        case (3, 3):
            assert cov.shape == (3, 3)
            # here it is necessary to reduce the parts from the covariance of u0 to
            mask = np.array([[True, False, True, False], [False, False, False, False], [True, False, True, False],
                             [False, False, False, False]])
            mask_reduction = np.array([[True, True, False], [True, True, False], [False, False, False]])
            return cov[mask_reduction].reshape((2, 2)) * GENERAL_TRANSFORMATION_MATRIX[mask].reshape((2, 2))
        case (4, 4):
            assert cov.shape == (4, 4)
            # here it is necessary to reduce the parts from the covariance of u0 to
            mask = np.array([[True, True, True, False], [True, True, True, False], [True, True, True, False],
                             [False, False, False, False]])
            return cov[mask].reshape((3, 3)) * GENERAL_TRANSFORMATION_MATRIX[mask].reshape((3, 3))
        case _:
            raise ValueError(
                "The dimension of the covariance matrix does not fit to any of the fitting functions and their parameters.")


def full_capacitance_model(freq, c=1e-6, r=1e6, i=0, u0=1):
    # ignores the reference voltage for now
    return (u0 * c * freq + i) / (1 + r * c * freq)


def simple_capacitance_model(freq, c=1e-6, i=0, u0=1):
    return u0 * c * freq + i


def depletion_model(x, a=1, b=0):
    return a * x + b


def model_depletion(voltages, NA=1e16, ND=1e16, V=SILICON_V_BIAS):
    import scipy.constants as constants
    # modified the sign as the bias voltages are saved with correct sign assigned to them.
    return np.sqrt(2 * constants.epsilon_0 * EPS_SILICON / constants.e * (NA + ND) / (NA * ND) * (
            V - np.array(voltages)))


def semi_bias_model(voltages, bias=SILICON_V_BIAS, thermic=1, i=1):
    return np.where(voltages >= 0, i * (np.exp((voltages - bias) / thermic) - 1), i)


def advanced_analysis(raw_data, is_cv=False, base_path=None, full_model=True,
                      first_boundaries=None, second_boundaries=None, **kwargs):
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
                advanced_analysis_delegate(in_file_h5, data_group, ana_group, full_model=full_model, **kwargs)
                cap_data = ana_group.HistCap[:]
                cap_error_data = ana_group.HistCapErr[:]
                cv_data[:, :, k] = cap_data[:, :]
                cv_err_data[:, :, k] = cap_error_data[:, :]

            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCHist",
                                                  title="Histogram of the U-C-curve",
                                                  filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False),
                                                  obj=cv_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()
            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCErrHist",
                                                  title="Error Histogram of the U-C-curve",
                                                  filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False),
                                                  obj=cv_err_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()
            if first_boundaries is not None and second_boundaries is not None:
                analyze_depletion_delegate(base_group.biasing.measurements, base_group.biasing.analysis,
                                           first_boundaries, second_boundaries, **kwargs)
        else:
            if "analysis" in base_group.total_cap:
                base_group.total_cap.analysis._f_remove(recursive=True)
                time.sleep(1)
            ana_group = walk_to_node(base_group.total_cap, "analysis", create=True)
            advanced_analysis_delegate(in_file_h5, base_group.total_cap.measurements, ana_group, full_model=full_model,
                                       **kwargs)


def advanced_analysis_delegate(file: tb.File, data_group: tb.Group, result_group: tb.Group, full_model=True, **kwargs):
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
    # extract the additional keyword arguments
    use_kafe2 = kwargs.get("use_kafe2", False)
    apply_contour = kwargs.get("apply_contour", False)
    plot = kwargs.get("plot", False)

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
        fit_cov = np.full(shape=(40, 40, 3, 3), fill_value=np.nan)

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
                try:
                    cap = fitter.parameter_values[0] * 1e-6  # convert to F
                    cap_error = fitter.parameter_errors[0] * 1e-6
                    leakage = fitter.parameter_values[2] * 1e9  # convert to nA
                    leakage_error = fitter.parameter_errors[2] * 1e9  # convert to nA
                except:
                    print(fitter.parameter_values)
                    print(fitter.get_result_dict()["parameter_values"])
                    raise
                if full_model:
                    # otherwise the requested information may not be present
                    resistor = fitter.parameter_values[1]
                    resistor_error = fitter.parameter_errors[1]
                else:
                    resistor = np.nan
                    resistor_error = np.nan
                fit_cov[ii, jj] = fitter.parameter_cov_mat

                if plot:
                    assert isinstance(fitter, XYFit)
                    handle_kafe2_advanced_options(fitter, apply_contour, "$\\nu$ in MHz", "$I$ in A",
                                                  f"Fit of the frequency dependence for pixel ({ii}, {jj})",
                                                  output_pdf,
                                                  f"Contour profiles for pixel ({ii}, {jj})")
                # if plot:
                #     from kafe2 import Plot
                #     fit_plot = Plot(fitter)
                #     fit_plot.x_label = "$\\nu$ in MHz"
                #     fit_plot.y_label = "$I$ in A"
                #     fit_plot.plot(residual=True)
                #     fit_plot.axes.set_title(f"Fit of the frequency dependence for pixel ({ii}, {jj})")
                #     for (fig, axes) in zip(fit_plot.figures, fit_plot.axes):
                #         for ax in axes.values():
                #             ax.set_title(f"Fit of the frequency dependence for pixel ({ii}, {jj})")
                #         output_pdf.savefig(fig, bbox_inches='tight')
                #
                # if apply_contour and plot:
                #     from kafe2 import ContoursProfiler
                #     cpf = ContoursProfiler(fitter)
                #     cpf_figure = cpf.plot_profiles_contours_matrix()
                #     from matplotlib.figure import Figure
                #     assert isinstance(cpf_figure, Figure)
                #     cpf_figure.axes[0][0].set_title(f"Contour profiles for pixel ({ii}, {jj})")
                #     output_pdf.savefig(cpf_figure, bbox_inches='tight')

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

                if plot:
                    assert isinstance(fitter, Minuit)
                    handle_minuit_advanced_options(fitter, apply_contour, None, None, None, output_pdf,
                                                   f"Fit of the frequency dependence for pixel ({ii}, {jj})")

                # if apply_contour and plot:
                #     assert isinstance(fitter, Minuit)
                #     print(fitter.draw_mnmatrix())
                #     from matplotlib import pyplot as plt
                #     print(plt.get_fignums())
                #     current_fig = plt.figure(plt.get_fignums()[0])
                #     current_fig.axes[0, 0].set_title(f"Fit of the frequency dependence for pixel ({ii}, {jj})")
                #     output_pdf.savefig(current_fig, bbox_inches='tight')

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
            if full_model:
                fit_cov[ii, jj, :3, :3] = np.full(shape=(3, 3), fill_value=np.nan)
            else:
                fit_cov[ii, jj, :2, :2] = np.full(shape=(2, 2), fill_value=np.nan)

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
    temp_array.attrs[
        "Units"] = "{{F^2, F O, F nA, F V},{O F, O^2, O nA, O V},{nA F, nA O, nA^2, nA V}, {V F, V O, V nA, V^2}"
    temp_array.flush()


def str_join(delimiter, *args):
    return delimiter.join(args)


def analyze_data(raw_data, is_cv=False, base_path=None, first_boundaries=None, second_boundaries=None, ):
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
                analyze_data_delegate(in_file_h5, data_group, ana_group)
                cap_data = ana_group.HistCap[:]
                cap_error_data = ana_group.HistCapErr[:]
                cv_data[:, :, k] = cap_data[:, :]
                cv_err_data[:, :, k] = cap_error_data[:, :]

            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCHist",
                                                  title="Histogram of the U-C-curve",
                                                  filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False),
                                                  obj=cv_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()
            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCErrHist",
                                                  title="Error Histogram of the U-C-curve",
                                                  filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False),
                                                  obj=cv_err_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()
            if first_boundaries is not None and second_boundaries is not None:
                analyze_depletion_delegate(base_group.biasing.measurements, base_group.biasing.analysis,
                                           first_boundaries, second_boundaries)
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
                cap_error = np.sqrt(cov[0, 0])
                leak_error = np.sqrt(cov[1, 1])
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
                                    obj=fit_cov, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False)
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


def analyze_depletion_delegate(data_group: GroupType, analysis_group: GroupType, first_boundaries, second_boundaries,
                               chip_group: GroupType = None, apply_doping=False, **kwargs):
    # extract the additional parameters for advanced fitting procedures
    use_kafe2 = kwargs.pop("use_kafe2", False)
    apply_contours = kwargs.pop("apply_contours", False)
    plot = kwargs.pop("plot", False)

    voltage_data = check_leaf_unit(data_group.BiasVoltageHist, "V")
    first_lower, first_upper = first_boundaries
    second_lower, second_upper = second_boundaries
    depletion_voltage = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    depletion_error = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    fit_parameter_estimators = np.full(shape=(40, 40, 4), fill_value=np.nan)
    fit_parameter_errors = np.full(shape=(40, 40, 4), fill_value=np.nan)

    # verify and extract the raw data for further analysis
    check_leaf_unit(analysis_group.UCHist, "F")
    check_leaf_unit(analysis_group.UCErrHist, "F")
    cap_data = analysis_group.UCHist[:, :, :]
    cap_error_data = analysis_group.UCErrHist[:, :, :]
    for ii, jj in np.ndindex(GENERAL_PIXCAP_SHAPE):
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
        effective_capacitance_error_data = np.reciprocal(cap_data[ii, jj, :] * 1e15) ** 3 * cap_error_data[
            ii, jj, :] if np.all(np.isfinite(cap_error_data[ii, jj, :])) else cap_error_data[ii, jj, :]
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

        # estimate the depletion voltage TODO: improve the error estimation for directly using the covariance matrices and make use of numpy mat-multiplication
        dep_voltage = (first_dep_parameters[1] - second_dep_parameters[1]) / (first_dep_parameters[0] -
                                                                              second_dep_parameters[0])
        # combine both cov matrices into a single one:
        full_cov = np.full((4,4), fill_value=np.nan)
        full_cov[:2, :2] = first_dep_cov
        full_cov[2:, 2:] = second_dep_cov

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
    temp_array = file_h5.create_carray(where=analysis_group, name="DepletionHist",
                                       title="Histogram of the depletion voltages", obj=depletion_voltage,
                                       filters=tb.Filters(complevel=5, complib='blosc', fletcher32=False))
    temp_array.attrs["Units"] = "V"
    temp_array.flush()
    temp_array = file_h5.create_carray(where=analysis_group, name="DepletionErrHist",
                                       title="Histogram of the depletion voltage erors", obj=depletion_error,
                                       filters=tb.Filters(complevel=5, complib='blosc', fletcher32=False))
    temp_array.attrs["Units"] = "V"
    temp_array.flush()
    temp_array = file_h5.create_carray(where=analysis_group, name="DepFitParamHist",
                                       title="Histogram of the depletion voltages fit parameters",
                                       obj=fit_parameter_estimators,
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
    if apply_doping and chip_group is not None and "PhysicalDimensions" in chip_group and chip_group.PhysicalDimensions.shape == (
            40, 40, 2) and np.all(np.isfinite(
            chip_group.PhysicalDimensions[:])):

        physical_dimensions_data = chip_group.PhysicalDimensions[:]
        pixel_areas = np.prod(physical_dimensions_data, axis=2)
        depletion_width_plate = np.full_like(pixel_areas, fill_value=np.nan)
        depletion_width_plate_error = np.full_like(pixel_areas, fill_value=np.nan)
        depletion_fit_parameter_table = np.full((40, 40, 3), fill_value=np.nan)
        depletion_fit_parameter_error_table = np.full((40, 40, 3), fill_value=np.nan)
        depletion_fit_covariance_table = np.full((40, 40, 3, 3), fill_value=np.nan)
        effective_doping_table = np.full((40, 40, voltage_data.shape[0]), fill_value=np.nan)

        # some further definitions for the loop
        from findiff import Diff
        import scipy.constants as constants
        from iminuit import Minuit
        from iminuit.cost import LeastSquares
        if use_kafe2:
            from kafe2 import XYContainer, XYFit, Plot, ContoursProfiler

        # this values will not be correct as I don't known the doping concentrationor the intrinisc bias voltage;
        # the intrinisc bias voltage could be estimated from a fit to the forward bias I-V characteristic.
        NA, ND = 1e16, 1e16
        V_bi = SILICON_V_BIAS
        dep_table = file_h5.create_table(where=analysis_group, name="DepletionParamTable", description=DepletionData,
                                         filters=GLOBAL_FILTERS)
        entry = dep_table.row
        for col, row in np.ndindex(GENERAL_PIXCAP_SHAPE):
            if not np.all(np.isfinite(physical_dimensions_data[col, row])):
                continue
            if not np.isfinite(cap_data[col, row]):
                continue

            # calculate the depletion width

            bias_voltages = check_leaf_unit(data_group.BiasVoltageHist, "V")
            if np.isfinite(cap_error_data[col, row]):
                effective_cap_errors = cap_error_data[col, row]
            else:
                effective_cap_errors = np.full_like(bias_voltages, 1)

            # What is the unit of this result
            # use this theoretical prescription as a model to fit
            # W = np.sqrt(2 * constants.epsilon_0 / constants.e * (NA + ND) / (NA * ND) * (V_bi + np.array(bias_voltages))) * 1e4
            depletion_width_plate[col, row] = (constants.epsilon_0 * pixel_areas[col, row]) / (
                np.array(cap_data[col, row])) * 1e9  # provides the width in um
            depletion_width_plate_error[col, row] = (constants.epsilon_0 * pixel_areas[col, row] * effective_cap_errors[
                col, row]) / (
                                                            np.array(cap_data[
                                                                         col, row]) ** 2) * 1e9  # provides the width in um

            # fit the theoretical expected depletion width to determine some of the properties of the pixel diode
            parameter_guess = {
                "NA": NA,
                "ND": ND,
                "V": V_bi
            }
            if use_kafe2:
                xy_data = XYContainer(x_data=bias_voltages, y_data=depletion_width_plate[col, row], )
                xy_data.add_error(axis='y', err_val=depletion_width_plate_error[col, row])
                # TODO: apply the error for the x-coordinates here, too

                fitter = XYFit(xy_data, model_function=model_depletion, minimizer="iminuit")
                fitter.assign_parameter_latex_names(voltages="U_\\text{{bi}}", NA="N_\\text{{A}}", ND="N_\\text{{D}}",
                                                    V="U_\\text{{th}}", )
                fitter.assign_model_function_latex_name("d_\\text{{depletion}}")
                fitter.assign_model_function_latex_expression(
                    "\\sqrt{{\\frac{{2\\epsilon_0\\epsilon}}{{e}}\\cdot\\frac{{{NA}+{ND}}}{{{NA}\\cdot{ND}}}\\cdot ({V}+{voltages})}}")
                fitter.set_parameter_values(**parameter_guess)
                fitter.limit_parameter(name="V", lower=0.0)
                fitter.limit_parameter(name="NA", lower=0.0)
                fitter.limit_parameter(name="ND", lower=0.0)

                fitter.do_fit()
                assert fitter.did_fit
                depletion_fit_propagate_parameters = fitter.parameter_name_value_dict
                depletion_fit_params = fitter.parameter_values
                depletion_fit_errors = fitter.parameter_errors
                depletion_fit_cov = fitter.parameter_cov_mat

                if plot:
                    assert isinstance(fitter, XYFit)
                    # what about the pdf pages object?
                    handle_kafe2_advanced_options(fitter, apply_contours, "$U_\\text{{bi}}$ in \\unit{{\\volt}}",
                                                  "$d$ in \\unit{{\\micro\\meter}}", "???", None, None)
            else:
                cost = LeastSquares(x=bias_voltages, y=depletion_width_plate[col, row],
                                    yerror=depletion_width_plate_error[col, row], model=model_depletion, )
                fitter = Minuit(cost, **parameter_guess)
                fitter.limits["V"] = (0.0, None)
                fitter.limits["NA"] = (0.0, None)
                fitter.limits["ND"] = (0.0, None)
                fitter.migrad()
                fitter.hesse()
                depletion_fit_propagate_parameters = fitter.values.to_dict()
                depletion_fit_params = np.array(fitter.values)
                depletion_fit_errors = np.array(fitter.errors)
                depletion_fit_cov = np.asarray(fitter.covariance)
                if plot:
                    assert isinstance(fitter, Minuit)
                    # what about the pdf pages object?
                    handle_minuit_advanced_options(fitter, apply_contours, "$U_\\text{{bi}}$ in \\unit{{\\volt}}",
                                                   "$d$ in \\unit{{\\micro\\meter}}", "???", None, None)
            depletion_fit_parameter_table[col, row] = depletion_fit_params
            depletion_fit_parameter_error_table[col, row] = depletion_fit_errors
            depletion_fit_covariance_table[col, row] = depletion_fit_cov

            Neff = effective_doping(cap_data[col, row], -bias_voltages, diode_area=pixel_areas[col, row])
            effective_doping_table[col, row] = Neff
            pos_min = np.argmin(Neff)
            print(f"The minimum concentration is {Neff[pos_min]} and depth {depletion_width_plate[pos_min]}")

            entry["row"] = row
            entry["col"] = col
            for key, value in depletion_fit_propagate_parameters.items():
                entry[key] = value

            entry.append()

            # TODO: Move these plotting parts to the plotting file.
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(2)
            if np.all(np.isfinite(depletion_width_plate_error[col, row])):
                ax[0].errorbar(bias_voltages, depletion_width_plate[col, row],
                               yerr=depletion_width_plate_error[col, row], label='d-measurement')
            else:
                ax[0].plot(bias_voltages, depletion_width_plate, label='d-measurement')

            sample_voltage = -1 * np.linspace(np.min(-bias_voltages), np.max(-bias_voltages) * 1.1, 1000)
            ax[0].plot(-sample_voltage, model_depletion(sample_voltage, **depletion_fit_propagate_parameters),
                       label=f'd-theory,NA={NA},ND={ND}')
            ax[0].set_xlabel('Bias Voltage [V]')
            ax[0].set_ylabel('Depletion Width [µm]')
            ax[0].set_title(f"Analysis of the depletion width for pixel ({col}, {row}).")
            ax[0].grid(True)
            ax[0].legend()

            # TODO: again extrac the plotting functionality.
            ax[1].plot(-bias_voltages, Neff)
            ax[1].set_xlabel('Bias Voltage [V]')
            ax[1].set_ylabel('Effective doping concentration [cm-3]')
            ax[1].set_title("Analysis of the effective doping for pixel ({col}, {row}).")
            ax[1].grid(True)
            ax[1].yscale('log')

        # save the computed information about the depletion behaviour
        temp_array = file_h5.create_carray(where=analysis_group, name="DepletionParameters",
                                           title="Depletion Parameters from fitting the depletion width",
                                           filters=GLOBAL_FILTERS, obj=depletion_fit_parameter_table)
        temp_array.attrs["Units"] = "cm^-3; cm^-3; V"
        temp_array.flush()
        temp_array = file_h5.create_carray(where=analysis_group, name="DepletionErrors",
                                           title="Depletion Parameters from fitting the depletion width",
                                           filters=GLOBAL_FILTERS, obj=depletion_fit_parameter_error_table)
        temp_array.attrs["Units"] = "cm^-3; cm^-3; V"
        temp_array.flush()
        temp_array = file_h5.create_carray(where=analysis_group, name="DepletionCovariance",
                                           title="Covariance matrices for Depletion Parameters from fitting the depletion width",
                                           filters=GLOBAL_FILTERS, obj=depletion_fit_covariance_table)
        temp_array.attrs["Units"] = "{{cm^-6, cm^-6, cm^-3 V},{cm^-6, cm^-6, cm^-3 V},{V cm^-3, V cm^-3, V^2}}"
        temp_array.flush()
        temp_array = file_h5.create_carray(where=analysis_group, name="DepletionEffDoping",
                                           title="Data for the effective doping from the cv-analysis",
                                           filters=GLOBAL_FILTERS, obj=effective_doping_table)
        temp_array.attrs["Units"] = "cm^-3"
        temp_array.flush()  # TODO: is the unit correct?


def effective_doping(capacitances, bias_voltages, diode_area=None):
    import scipy.constants
    from findiff import Diff
    capacitances = np.asarray(capacitances)
    bias_voltages = np.asarray(bias_voltages)
    if diode_area is None:
        diode_area = 50 * 50  # But what is the unit for this.
    temp_capacitances = np.reciprocal(capacitances ** 2)
    d_du = Diff(0, bias_voltages)
    derivative = d_du(temp_capacitances)
    Neff = 2 / (scipy.constants.elementary_charge * scipy.constants.epsilon_0 * (diode_area ** 2) * np.array(
        derivative))
    return Neff


if __name__ == '__main__':
    # analyze_data(raw_data='/home/silab/git/pixcap65/pixcap_full_data_image1.h5')
    # advanced_analysis(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1")
    advanced_analysis(raw_data='Data/r13-measurement/R13_Initial_3_Scan.h5',base_path="ATLAS ITk/unbiased_1")
    # analyze_data(raw_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', is_cv=True, first_boundaries=(-100,-40), second_boundaries=(-10, 0),)

