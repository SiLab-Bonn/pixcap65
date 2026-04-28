from warnings import deprecated

import numpy as np
import tables as tb
import time

# from iminuit import Minuit
# from iminuit.cost import LeastSquares
# from matplotlib.backends.backend_pdf import PdfPages
# from tables import Group
from analysis import analyze_depletion_delegate, apply_correction_simple
from analysis_util.physics_modelling import full_capacitance_model, simple_capacitance_model
from analysis_util.utility import str_join, check_leaf_unit, \
    transform_covariance, TABLES_ARRAY_TYPE, GENERAL_PIXCAP_SHAPE, COVARIANCE_PIXCAP_SHAPE, FULL_MODEL_LABEL, \
    SIMPLE_MODEL_LABEL, FULL_MODEL_EXPRESSION, SIMPLE_MODEL_EXPRESSION, FULL_MODEL_PARAMETER_DICT, \
    SIMPLE_MODEL_PARAMETER_DICT, GLOBAL_FILTERS, handle_kafe2_advanced_options, handle_minuit_advanced_options, \
    FARAD_CONVERSION_FACTOR, CURRENT_CONVERSION_FACTOR, TABLES_TABLE_TYPE, get_analysis_group
from utility.utils_2 import walk_to_node, GroupType

ANALYSIS_FIT_Y_LABEL = "$I$ in A"
ANALYSIS_FIT_X_LABEL = "$\\nu$ in MHz"
ANALYSIS_FIT_CONTOUR_LEGEND = "Contour profiles for pixel ({col}, {row})"
ANALYSIS_FIT_PLOT_LEGEND = "Fit of the frequency dependence for pixel ({col}, {row})"
ANALYSIS_GROUP_NAME = "analysis"
ANALYSIS_CORRECTED_GROUP_NAME = "analysis_correction"

@deprecated("Use the general implementation of analysis.analyze_data_temporary_replacement instead.")
def advanced_analysis(raw_data, base_path=None, is_cv=False, first_boundaries=None, second_boundaries=None,
                      is_inter_pixel=False, **kwargs):
    """
    advanced_analysis

    Implementation of the advanced analysis strategy for the capacitance measurement of a pixel sensor.
    For determination of the capacitance values non-linear fit algorithms are used.
    Depeneding on the choice of parameters either kafe2 or iminuit is used for least-squares minimization.
    But keep in mind that this function serves as a wrapper for file access and modification around
    the actual analysis implementation.

    For measurements of the C-V-Characteristic of a sensor, the fits will be applied for every bias voltage measured.
    :param raw_data: path to the hdf file containing the raw data.
    :param use_kafe2: boolean, indicates whether kafe2 is used.
    :param is_cv: boolean, indicates whether this is a C-V characterization.
    :param base_path: path to the base group to look for the data.
    :param plot: boolean, indicates whether to plot the data. A output pdf object could be submitted here instead of an explicitly created one.
    :param apply_contour: boolean, indicates whether to determine the contours and try to plot them.
    :param full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
    :param apply_correction: boolean, False, indicates whether the measured capacitances should be
        corrected immediately;
        Will require the presence of further arguments as information about the parasitics needs to be submitted.
    :param use_corrected: boolean, False, indicates whether to use the corrected capacitances for the depletion analysis.
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
            cv_data_corrected = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]),
                              fill_value=np.nan)
            cv_err_data_corrected = np.full(shape=(40, 40, base_group.biasing.measurements.BiasVoltageHist.shape[0]),
                                  fill_value=np.nan)

            # make sure to not mix-up with previous analysis results
            if "analysis" in base_group.biasing:
                base_group.biasing.analysis._f_remove(recursive=True)
                time.sleep(1)

            for k, bias_voltage in enumerate(base_group.biasing.measurements.BiasVoltageHist):
                bias_name = f"bias_{bias_voltage}_V".replace('-', "M_").replace(".", "__")
                data_group = base_group.biasing.measurements[bias_name]
                ana_group = walk_to_node(base_group.biasing, str_join("/", ANALYSIS_GROUP_NAME, bias_name), create=True)
                assert isinstance(ana_group, tb.Group)


                advanced_analysis_data_handle(in_file_h5, data_group, ana_group,
                                              is_inter_pixel=is_inter_pixel, **kwargs)

                # extract the capacitance data for tabular value; will also need coreected data.
                cap_data = ana_group.HistCap[:]
                cap_error_data = ana_group.HistCapErr[:]
                cv_data[:, :, k] = cap_data[:, :]
                cv_err_data[:, :, k] = cap_error_data[:, :]
                if kwargs.get("apply_correction", False):
                    ana_group_correction = walk_to_node(base_group.biasing,
                                                str_join("/", ANALYSIS_CORRECTED_GROUP_NAME, bias_name),
                                                create=True)
                    cap_data = ana_group_correction.HistCap[:]
                    cap_error_data = ana_group_correction.HistCapErr[:]
                    cv_data_corrected[:, :, k] = cap_data[:, :]
                    cv_err_data_corrected[:, :, k] = cap_error_data[:, :]

            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCHist",
                                                  title="Histogram of the U-C-curve",
                                                  filters=GLOBAL_FILTERS,
                                                  obj=cv_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()
            temp_array = in_file_h5.create_carray(base_group.biasing.analysis, name="UCErrHist",
                                                  title="Error Histogram of the U-C-curve",
                                                  filters=GLOBAL_FILTERS,
                                                  obj=cv_err_data)
            temp_array.attrs["Units"] = "F"
            temp_array.flush()

            # save also the corrected capacitance data if necessary
            if kwargs.get("apply_correction", False):
                temp_array = in_file_h5.create_carray(base_group.biasing.analysis_correction, name="UCHist",
                                                      title="Histogram of the U-C-curve",
                                                      filters=GLOBAL_FILTERS,
                                                      obj=cv_data)
                temp_array.attrs["Units"] = "F"
                temp_array.flush()
                temp_array = in_file_h5.create_carray(base_group.biasing.analysis_correction, name="UCErrHist",
                                                      title="Error Histogram of the U-C-curve",
                                                      filters=GLOBAL_FILTERS,
                                                      obj=cv_err_data)
                temp_array.attrs["Units"] = "F"
                temp_array.flush()

            if first_boundaries is not None and second_boundaries is not None:
                dep_ana_group = get_analysis_group(base_group.biasing, **kwargs)
                if kwargs.pop("use_corrected", False):
                    analyze_depletion_delegate(base_group.biasing.measurements,
                                               base_group.biasing.analysis,
                                               first_boundaries, second_boundaries, **kwargs)
                analyze_depletion_delegate(base_group.biasing.measurements, dep_ana_group,
                                           first_boundaries, second_boundaries, **kwargs)
        else:
            if is_inter_pixel:
                reference_group = base_group.inter_cap
            else:
                reference_group = base_group.total_cap
            if "analysis" in reference_group:
                reference_group.analysis._f_remove(recursive=True)
                time.sleep(1)
            ana_group = walk_to_node(reference_group, "analysis", create=True)
            assert isinstance(ana_group, tb.Group)


            advanced_analysis_data_handle(in_file_h5, reference_group.measurements, ana_group,
                                          is_inter_pixel=is_inter_pixel, **kwargs)
            # should already be done by the data handle delegation
            # if kwargs.get("apply_correction", False):
            #     apply_correction_simple(kwargs['bare_file'], kwargs['bare_hdf_path'], ana_group)


@deprecated("Use the general implementation of analysis.analysis_data_handle_temporary_replacement instead.")
def advanced_analysis_data_handle(file: tb.File, data_group: GroupType, result_group: GroupType,
                                  is_inter_pixel=False, **kwargs):
    """
    advanced_analysis_data_handle

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
    :param apply_correction: boolean, False, indicates whether the measured capacitances should be
        corrected immediately;
        Will require the presence of further arguments as information about the parasitics needs to be submitted.
    """
    # Read pixel map
    assert isinstance(data_group, tb.Group)
    assert isinstance(result_group, tb.Group)
    if is_inter_pixel:
        current_hist = check_leaf_unit(data_group.TotalHistCurr, "A")
        if "TotalHistCurr" in data_group:
            current_error_hist = check_leaf_unit(data_group.TotalHistCurr, "A")
        else:
            current_error_hist = np.full_like(current_hist, fill_value=np.nan)
        # Read scan parameters
        scan_parameters = data_group.scan_params[:]

        advanced_analysis_delegate(file, result_group, current_hist, scan_parameters, current_error_hist=current_error_hist,
                                   **kwargs)

        inter_a_output_dict = {
            "cap_name": "HistCapInterA",
            "cap_title": "Capacitance Histogram of inter pixel A",
            "cap_err_name": "HistCapErrInterA",
            "cap_err_title": "Capacitance Error Histogram of inter pixel A",
            "leak_name": "HistLeakInterA",
            "leak_title": "Leakage Current Histogram of inter pixel A",
            "leak_error_name": "HistLeakErrInterA",
            "leak_error_title": "Leakage Current Error Histogram of inter pixel A",
            "resistor_name": "HistResInterA",
            "resistor_title": "On-Resistance Histogram of inter pixel A",
            "resistor_error_name": "HistResErrInterA",
            "resistor_error_title": "On-Resistance Error Histogram of inter pixel A",
            "cov_name": "HistFitCovInterA",
            "cov_title": 'Fit Covariance Matrix of inter pixel A'
        }
        kwargs.update(inter_a_output_dict)
        current_hist = check_leaf_unit(data_group.InterHistCurrA, "A")
        if "InterHistCurrErrA" in data_group:
            current_error_hist = check_leaf_unit(data_group.InterHistCurrErrA, "A")
        else:
            current_error_hist = np.full_like(current_hist, fill_value=np.nan)

        advanced_analysis_delegate(file, result_group, current_hist, scan_parameters, current_error_hist=current_error_hist,
                                   **kwargs)

        inter_b_output_dict = {
            "cap_name": "HistCapInterB",
            "cap_title": "Capacitance Histogram of inter pixel B",
            "cap_err_name": "HistCapErrInterB",
            "cap_err_title": "Capacitance Error Histogram of inter pixel B",
            "leak_name": "HistLeakInterB",
            "leak_title": "Leakage Current Histogram of inter pixel B",
            "leak_error_name": "HistLeakErrInterB",
            "leak_error_title": "Leakage Current Error Histogram of inter pixel B",
            "resistor_name": "HistResInterB",
            "resistor_title": "On-Resistance Histogram of inter pixel B",
            "resistor_error_name": "HistResErrInterB",
            "resistor_error_title": "On-Resistance Error Histogram of inter pixel B",
            "cov_name": "HistFitCovInterB",
            "cov_title": 'Fit Covariance Matrix of inter pixel B'
        }
        kwargs.update(inter_b_output_dict)
        current_hist = check_leaf_unit(data_group.InterHistCurrB, "A")
        if "InterHistCurrErrB" in data_group:
            current_error_hist = check_leaf_unit(data_group.InterHistCurrErrB, "A")
        else:
            current_error_hist = np.full_like(current_hist, fill_value=np.nan)

        advanced_analysis_delegate(file, result_group, current_hist, scan_parameters, current_error_hist=current_error_hist,
                                   **kwargs)


    else:
        current_hist = check_leaf_unit(data_group.HistCurr, "A")
        if "HistCurrErr" in data_group:
            check_leaf_unit(data_group.HistCurrErr, "A")
            current_error_hist = data_group.HistCurrErr[:]
        else:
            current_error_hist = np.full_like(current_hist, fill_value=np.nan)
        # Read scan parameters
        scan_parameters = data_group.scan_params[:]

        advanced_analysis_delegate(file, result_group, current_hist, scan_parameters, current_error_hist=current_error_hist,
                                   **kwargs)

    # if necessary: directly apply the correction of the capacitance values
    if "apply_correction" in kwargs and kwargs["apply_correction"]:
        assert 'bare_file' in kwargs
        assert 'bare_hdf_path' in kwargs
        apply_correction_simple(kwargs['bare_file'], kwargs['bare_hdf_path'], result_group)


def advanced_analysis_delegate(file: tb.File, group: tb.Group, current_hist: TABLES_ARRAY_TYPE,
                               scan_parameters: TABLES_TABLE_TYPE, **kwargs):
    """
        advanced_analysis_delegate

        Implementation of the advanced analysis strategy for the capacitance measurement of a pixel sensor.
        For determination of the capacitance values non-linear fit algorithms are used.
        Depeneding on the choice of parameters either kafe2 or iminuit is used for least squares minimization.
        If requested the fit results will also be plotted to verify the convergence of the fit.
        If this mode is activated, additional keyword arguments must be present to specify the output of the plots.

        :param file: h5 file object containing the data to be analysed.
        :param group: hierachy group of the opened hdf file to write the analysis results to.
        :param current_hist: 2D-Array for the current data to fit the model to.
        :param scan_parameters: table of the scan parameters used for each measurement point within the frequency and/or
            voltage scan.
        :param full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
            Otherwise, the linear model is used.
        :param current_error_hist: 2D-Array for the errors of the current data. This keyword argument must be present
            for the advanced analysis strategy.
        :param use_kafe2: boolean, indicates whether kafe2 is used for the fit.
        :param plot: boolean, indicates whether to plot the data. A output pdf object could be submitted here
             instead of an explicitly created one.
        :param apply_contour: boolean, indicates whether to determine the contours and try to plot them.
        :param fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided)
        """

    # extract the additional keyword arguments
    full_model = kwargs.get("full_model", True)
    assert "current_error_hist" in kwargs
    current_error_hist = kwargs['current_error_hist']
    assert isinstance(current_error_hist, TABLES_ARRAY_TYPE)
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
        from matplotlib.backends.backend_pdf import PdfPages
        output_pdf_name = f"{file.filename[:-3]}_{group._v_pathname.replace('/', '----')}_fit_results.pdf"
        fit_plot_pdf = kwargs.get("fit_plot_pdf", None)
        if fit_plot_pdf is not None and isinstance(fit_plot_pdf, PdfPages):
            output_pdf = fit_plot_pdf
        elif isinstance(plot, PdfPages):
            output_pdf = plot
        else:
            output_pdf = PdfPages(output_pdf_name)
    else:
        output_pdf = None

    # Fit pixel data in order to extract capacitance for each pixel
    for ii, jj in np.ndindex(current_hist.shape[:2]):
        if np.all(np.isfinite(current_hist[ii, jj, :])):
            if use_kafe2:
                from kafe2 import XYContainer, XYFit
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
                from iminuit import Minuit
                from iminuit.cost import LeastSquares
                if np.all(np.isfinite(current_error_hist[ii, jj, :])):
                    errors = current_error_hist[ii, jj, :]
                else:
                    errors = np.full_like(current_error_hist, fill_value=1)
                # noinspection PyTypeChecker
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
                from kafe2 import XYFit
                assert isinstance(fitter, XYFit)
                assert fitter.did_fit
                try:
                    cap = fitter.parameter_values[0] * FARAD_CONVERSION_FACTOR  # convert to F
                    cap_error = fitter.parameter_errors[0] * FARAD_CONVERSION_FACTOR
                    leakage = fitter.parameter_values[2] * CURRENT_CONVERSION_FACTOR  # convert to nA
                    leakage_error = fitter.parameter_errors[2] * CURRENT_CONVERSION_FACTOR  # convert to nA
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
                    handle_kafe2_advanced_options(fitter, apply_contour, ANALYSIS_FIT_X_LABEL, ANALYSIS_FIT_Y_LABEL,
                                                  ANALYSIS_FIT_PLOT_LEGEND.format(col=ii, row=jj),
                                                  output_pdf,
                                                  ANALYSIS_FIT_CONTOUR_LEGEND.format(col=ii, row=jj))
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
                from iminuit import Minuit
                assert isinstance(fitter, Minuit)
                cap = fitter.values['c'] * FARAD_CONVERSION_FACTOR  # convert to F
                cap_error = fitter.errors['c'] * FARAD_CONVERSION_FACTOR
                leakage = fitter.values['i'] * CURRENT_CONVERSION_FACTOR  # convert to nA
                leakage_error = fitter.errors['i'] * CURRENT_CONVERSION_FACTOR  # convert to nA
                if full_model:
                    # otherwise the requested information may not be present.
                    resistor = fitter.values['r']
                    resistor_error = fitter.errors['r']
                else:
                    resistor = np.nan
                    resistor_error = np.nan
                fit_cov[ii, jj] = fitter.covariance

                if plot:
                    handle_minuit_advanced_options(fitter, apply_contour, ANALYSIS_FIT_X_LABEL,
                                                   ANALYSIS_FIT_Y_LABEL,
                                                   ANALYSIS_FIT_PLOT_LEGEND.format(col=ii, row=jj), output_pdf,
                                                   ANALYSIS_FIT_CONTOUR_LEGEND.format(col=ii, row=jj))

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

    if plot and not (isinstance(plot, PdfPages) or isinstance(kwargs.get("fit_plot_pdf", None), PdfPages)):
        output_pdf.close()

    # Store capacitance values and specify the used units as an attribute.
    temp_array = file.create_carray(group,
                                    name=kwargs.get("cap_name", "HistCap"),
                                    title=kwargs.get("cap_title", "Capacitance Histogram"),
                                    obj=cap_hist,
                                    filters=tb.Filters(complib='blosc',
                                                       complevel=5,
                                                       fletcher32=False))
    temp_array.attrs["Units"] = "F"
    temp_array.flush()
    temp_array = file.create_carray(group,
                                    name=kwargs.get("cap_err_name", "HistCapErr"),
                                    title=kwargs.get("cap_err_title", "Capacitance Error Histogram"),
                                    obj=cap_error_hist,
                                    filters=tb.Filters(complib='blosc',
                                                       complevel=5,
                                                       fletcher32=False))
    temp_array.attrs["Units"] = "F"
    temp_array.flush()

    temp_array = file.create_carray(group,
                                    name=kwargs.get("leak_name", "HistLeak"),
                                    title=kwargs.get("leak_title", "Leakage Current Histogram"),
                                    obj=leak_hist,
                                    filters=tb.Filters(complib='blosc',
                                                       complevel=5,
                                                       fletcher32=False))
    temp_array.attrs["Units"] = "nA"
    temp_array.flush()
    temp_array = file.create_carray(group,
                                    name=kwargs.get("leak_error_name", "HistLeakErr"),
                                    title=kwargs.get("leak_error_title", "Leakage Current Error Histogram"),
                                    obj=leak_error_hist,
                                    filters=tb.Filters(complib='blosc',
                                                       complevel=5,
                                                       fletcher32=False))
    temp_array.attrs["Units"] = "nA"
    temp_array.flush()

    temp_array = file.create_carray(group,
                                    name=kwargs.get("resistor_name", "HistRes"),
                                    title=kwargs.get("resistor_title", "On-Resistance Histogram"),
                                    obj=resistor_hist,
                                    filters=tb.Filters(complib='blosc',
                                                       complevel=5,
                                                       fletcher32=False))
    temp_array.attrs["Units"] = "O"
    temp_array.flush()
    temp_array = file.create_carray(group,
                                    name=kwargs.get("resistor_error_name", "HistResErr"),
                                    title=kwargs.get("resistor_error_title", "On-Resistance Error Histogram"),
                                    obj=resistor_error_hist,
                                    filters=tb.Filters(complib='blosc',
                                                       complevel=5,
                                                       fletcher32=False))
    temp_array.attrs["Units"] = "O"
    temp_array.flush()

    temp_array = file.create_carray(group,
                                    name=kwargs.get("cov_name", "HistFitCov"),
                                    title=kwargs.get("cov_title", 'Fit Covariance Matrix'),
                                    obj=fit_cov, filters=tb.Filters(complib='blosc',
                                                                    complevel=5,
                                                                    fletcher32=False)
                                    )
    temp_array.attrs[
        "Units"] = "{{F^2, F O, F nA, F V},{O F, O^2, O nA, O V},{nA F, nA O, nA^2, nA V}, {V F, V O, V nA, V^2}"
    temp_array.flush()
