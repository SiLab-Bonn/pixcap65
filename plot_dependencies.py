import numpy as np
import tables as tb
from collections.abc import Iterable, Mapping
from iminuit import Minuit
from iminuit.cost import LeastSquares
from inspect import Parameter
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats.distributions import chi2 as sample_chi2

from pixcap65.analysis_util.summary import field_names, test_design_values, spatial_identifier
from pixcap65.plotting import CAPACITANCE_CONVERSION_FACTOR
from pixcap65.utility.homogenize_plots import set_params


def chi2(x: float, dof: int):
    return sample_chi2.cdf(x, dof)


interesting_keys = [
    "implantation_area",
    "implantation_depth",
    "pixel_separation_x",
    "pixel_separation_y",
    "Perimeter",
]


def linear_model(x, a, b):
    return a + x * b


def reciprocal_model(x, a, b):
    return a + b / x


def combined_model(x, a, b, c):
    return a + b * x + c / x


def exponential_model(x, a, b):
    return a * np.exp(b * x)


# format the table output
GENERAL_SIUNITX_FORMAT = "\\qty{{{:#0.6g}({:d})({:d})({:d})}}{{{}}}"
GENERAL_SIUNITX_FORMAT_2 = "\\qty{{{:.6f}({:.6f})({:.6f})({:.6f})}}{{{}}}"

GENERAL_PART = "\\qty{{{{{{:#0.{}g}}({{:d}})({{:d}})({{:d}})}}}}{{{{{{}}}}}}"
REDUCED_GENERAL_PART = "{{:#0.{}g}}({{:d}})({{:d}})({{:d}})"
SECOND_GENERAL_PART = "{{:#0.{}f}}({{:d}})({{:d}})({{:d}})"

GENERAL_PREC = 6
GENERAL_TEST_FORMAT = "{:-6g}"
GENERAL_UNCERT_MULTIPLIER = 1e6


def generate_siunitx(data_set: np.recarray, depletion=False) -> str:
    if depletion:
        test_format = "{:.3g}".format(data_set.dep_voltage_err)
        _, text_n_digits = test_format.removeprefix('-').split('.', 1)
        n_digits = len(text_n_digits)
        return SECOND_GENERAL_PART.format(n_digits).format(data_set.dep_voltage,
                                                           int(data_set.dep_voltage_err * 10 ** n_digits),
                                                           int(data_set.dep_voltage_systematic_general * 10 ** n_digits),
                                                           int(data_set.dep_voltage_systematic_dispersion * 10 ** n_digits), )

        # old approach
        # test_format = GENERAL_TEST_FORMAT.format(data_set.dep_voltage)
        # int_part, _ = test_format.removeprefix('-').split('.', 1)
        # sig = GENERAL_PREC if int(int_part) == 0 else GENERAL_PREC + len(int_part)
        # return GENERAL_PART.format(sig).format(data_set.dep_voltage,
        #                                     int(data_set.dep_voltage_err * GENERAL_UNCERT_MULTIPLIER),
        #                                     int(data_set.dep_voltage_systematic_general * GENERAL_UNCERT_MULTIPLIER),
        #                                     int(data_set.dep_voltage_systematic_dispersion * GENERAL_UNCERT_MULTIPLIER), "\\volt")
    else:
        if np.isnan(data_set.capacitance):
            return "\\infty"

        test_format = "{:.3g}".format(data_set.capacitance_err)
        _, text_n_digits = test_format.removeprefix('-').split('.', 1)
        n_digits = len(text_n_digits)
        return SECOND_GENERAL_PART.format(n_digits).format(data_set.capacitance,
                                                           int(data_set.capacitance_err * 10 ** n_digits),
                                                           int(data_set.capacitance_systematic_general * 10 ** n_digits),
                                                           int(data_set.capacitance_systematic_dispersion * 10 ** n_digits), )

        # old approach
        # test_format = GENERAL_TEST_FORMAT.format(data_set.capacitance)
        # int_part, _ = test_format.removeprefix('-').split('.', 1)
        # sig = GENERAL_PREC if int(int_part) == 0 else GENERAL_PREC + len(int_part)
        # return GENERAL_PART.format(sig).format(data_set.capacitance,
        #                                        int(data_set.capacitance_err * GENERAL_UNCERT_MULTIPLIER),
        #                                        int(data_set.capacitance_systematic_general * GENERAL_UNCERT_MULTIPLIER),
        #                                        int(data_set.capacitance_systematic_dispersion * GENERAL_UNCERT_MULTIPLIER), "\\femto\\farad")


def generate_siunitx_2(data_set: np.recarray, depletion=False) -> str:
    if depletion:
        return GENERAL_SIUNITX_FORMAT_2.format(data_set.dep_voltage,
                                               data_set.dep_voltage_err,
                                               data_set.dep_voltage_systematic_general,
                                               data_set.dep_voltage_systematic_dispersion, "\\volt")
    else:
        if np.isnan(data_set.capacitance):
            return "\\infty"
        return GENERAL_SIUNITX_FORMAT_2.format(data_set.capacitance,
                                               data_set.capacitance_err,
                                               data_set.capacitance_systematic_general,
                                               data_set.capacitance_systematic_dispersion, "\\femto\\farad")


def read_rec_array(table: tb.Table, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read(*args, **kwargs), dtype=table.dtype)


def read_rec_array_sorted(table: tb.Table, primary_key, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read_sorted(primary_key, *args, **kwargs), dtype=table.dtype)


def read_rec_array_where(table: tb.Table, condition, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read_where(condition, *args, **kwargs), dtype=table.dtype)


def cap_model(xy, a0, a1, a2, a3, a4):
    A, W, p = xy
    return a0 + a1 * A + a2 * W + a3 * p + a4 * A * W


def simplified_cap_model(xy, a0, a1, a2):
    A, W = xy
    return a0 + a1 * A + a2 * W


def extended_cap_model(xy, a0, a1, a2, a3, a4, a5, a6, a7):
    A, W, p, separation_x, separation_y = xy
    return a0 + a1 * A + a2 * W + a3 / W + a4 * p + a5 * A * W + a6 * separation_x + a7 * separation_y


def extended_cap_model_2(xy, a0, a1, a2, a4, a3):
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(a4 * p) + a1 * A + a2 * W + a3 * A * W


def extended_cap_model_3(xy, a0, a1, a2, a4, a5):
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(a4 * p) + a1 * A + a2 * W + a5 / separation_x


def extended_cap_model_4(xy, a0, a1, a2, a3, a4, a5, a6, a7):
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(a4 * p) + a1 * A + a2 * W + a3 / W + a5 * A * W + np.where(separation_x > 0, a6 / separation_x,
                                                                                  0) + np.where(separation_y > 0,
                                                                                                a7 / separation_y, 0)


def inter_cap_model(xy, a0, a1, a2, a3, a4):
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(- a4 * separation_x) + a1 * A + a2 * W + a3 * p


def inter_cap_model_2(xy, a0, a1, a2, a3, a4):
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(a3 * p - a4 * separation_x) + a1 * A + a2 * W


def inter_cap_model_3(xy, a0, a1, a2, a3):
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(a3 * p / separation_x) + a1 * A + a2 * W


def investigate_dependences_graphical(summary_data: np.recarray, property_data: np.recarray,
                                      interesting_data: Iterable):
    spatial_mask = np.array([sensor.decode() in spatial_identifier for sensor in summary_data.sensor], dtype=bool)

    plotter("Dependencies.pdf", spatial_mask, interesting_data, summary_data, property_data)

    plotter("Dependencies-log.pdf", spatial_mask, interesting_data, summary_data, property_data, scaley='log')

    plotter("Dependencies-log-log.pdf", spatial_mask, interesting_data, summary_data, property_data,
            scalex='log', scaley='log')


def investiagte_dependences_fitting(summary_data: np.recarray, property_data: np.recarray, interesting_data: Mapping):
    spatial_mask = np.array([sensor.decode() in spatial_identifier for sensor in summary_data.sensor], dtype=bool)
    for key, models in interesting_data.items():
        total_model, inter_model = models
        print("Will handle the dependent:", key)
        if total_model is not None:
            mask = np.logical_not(spatial_mask)
            handle_model_simple_fit(property_data[key][mask], summary_data.biased_capacitance.capacitance[mask],
                                    summary_data.biased_capacitance.capacitance_systematic_dispersion[mask],
                                    total_model, 'total')

        if inter_model is not None:
            mask = np.logical_and(np.logical_not(spatial_mask), np.isfinite(
                summary_data.biased_inter_capacitance.capacitance))
            handle_model_simple_fit(property_data[key][mask], summary_data.biased_inter_capacitance.capacitance[mask],
                                    summary_data.biased_inter_capacitance.capacitance_systematic_dispersion[mask],
                                    inter_model, 'inter')


def plotter(file, spatial_mask, keys: Iterable, data: np.recarray, properties: np.recarray, **keywords):
    def __plot_dependence(pdf: PdfPages, spatial_mask, physical_property, name, quantity, unit, x, y, **kwargs):
        scalex = kwargs.pop("scalex", None)
        scaley = kwargs.pop("scaley", None)
        fig, ax = plt.subplots()
        ax.set_title("Dependence of the {} on {}".format(name, physical_property))
        ax.set_ylabel("${}$ in \\unit{{{}}}".format(quantity, unit))
        if scalex is not None:
            ax.set_xscale(scalex)
        if scaley is not None:
            ax.set_yscale(scaley)
        condition = kwargs.pop("condition", None)
        mask = np.isfinite(y)
        if condition is not None:
            mask = np.logical_and(mask, condition(y))
        first_mask = np.logical_and(mask, spatial_mask)
        second_mask = np.logical_and(mask, np.logical_not(spatial_mask))
        ax.errorbar(x[physical_property][first_mask],
                    y[first_mask], fmt='x', label='3D')
        ax.errorbar(x[physical_property][second_mask],
                    y[second_mask], fmt='x', label='planar')
        ax.legend()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    with PdfPages(file) as pdf:
        physical_property = None
        for physical_property in keys:
            __plot_dependence(pdf, spatial_mask, physical_property, "capacitance", 'C',
                              '\\femto\\farad', properties, data.biased_capacitance.capacitance,
                              **keywords)

            __plot_dependence(pdf, spatial_mask, physical_property, 'inter-pixel capacitance',
                              'C_\\text{inter}', '\\femto\\farad', properties,
                              data.biased_inter_capacitance.capacitance, condition=lambda x: x >= 0,
                              **keywords)

            scalex = keywords.get("scalex", None)
            scaley = keywords.get("scaley", None)
            fig, ax = plt.subplots()
            ax.set_title("Dependence of the depletion voltage on {}".format(physical_property))
            ax.set_ylabel("$U$ in \\unit{{\\volt}}")
            if scalex is not None:
                ax.set_xscale(scalex)
            if scaley is not None:
                ax.set_yscale(scaley)
            mask = np.isfinite(final_summary.depletion_voltage.dep_voltage)
            first_mask = np.logical_and(mask, spatial_mask)
            second_mask = np.logical_and(mask, np.logical_not(spatial_mask))
            ax.errorbar(final_properties[physical_property][first_mask], final_summary.depletion_voltage.dep_voltage[first_mask], fmt='x', label='3D')
            ax.errorbar(final_properties[physical_property][second_mask],
                        final_summary.depletion_voltage.dep_voltage[second_mask], fmt='x', label='planar')
            ax.legend()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        if physical_property is None:
            print("No physical property was investigated at all.")


def handle_model_simple_fit(x_data, y_data, y_errors, model_function, name, *initial_args, **limits):
    from inspect import signature
    from warnings import warn
    print("Handle", name)
    model = LeastSquares(x_data, y_data, y_errors, model_function)
    # get the number of parameters from the actual function!
    sig = signature(model_function)
    position_argument = []
    starting_values = []
    positional_counter = 0
    for param in sig.parameters.values():
        if param.kind == param.POSITIONAL_OR_KEYWORD:
            position_argument.append(param.name)
            positional_counter += 1
            start_value = param.default
            if start_value == Parameter.empty:
                start_value = 0
            starting_values.append(start_value)
        elif param.kind == param.KEYWORD_ONLY:
            warn("Keyword-only arguments for the model function are currently not supported.", RuntimeWarning)

    del starting_values[0]

    for k, estimator in enumerate(initial_args):
        starting_values[k] = estimator

    m = Minuit(model, *starting_values)
    for parameter, limit in limits.items():
        m.limits[parameter] = limit
    m.migrad()
    m.hesse()
    try:
        m.minos()
    except RuntimeError as e:
        from warnings import warn
        warn(str(e), RuntimeWarning)
    print(m.fmin)
    print(m.params)
    print("p-value", 1 - chi2(m.fmin.fval, m.ndof))
    m.visualize()
    plt.show()


def handle_model_fit(x_data, y_data, y_errors, model_function, name, *initial_args, **limits):
    from inspect import signature
    from warnings import warn
    print("Handle the", name, "model for capacitance's.")
    model = LeastSquares(x_data, y_data, y_errors, model_function)
    # get the number of parameters from the actual function!
    sig = signature(model_function)
    position_argument = []
    starting_values = []
    positional_counter = 0
    for param in sig.parameters.values():
        if param.kind == param.POSITIONAL_OR_KEYWORD:
            position_argument.append(param.name)
            positional_counter += 1
            start_value = param.default
            if start_value == Parameter.empty:
                start_value = 0
            starting_values.append(start_value)
        elif param.kind == param.KEYWORD_ONLY:
            warn("Keyword-only arguments for the model function are currently not supported.", RuntimeWarning)

    del starting_values[0]

    for k, estimator in enumerate(initial_args):
        starting_values[k] = estimator

    m = Minuit(model, *starting_values)
    if 'a0' in position_argument:
        m.limits['a0'] = (0, None)
    for parameter, limit in limits.items():
        m.limits[parameter] = limit
    m.migrad()
    m.hesse()
    try:
        m.minos()
    except RuntimeError as e:
        from warnings import warn
        warn(str(e), RuntimeWarning)
    print(m.fmin)
    print(m.params)
    print("p-value", 1 - chi2(m.fmin.fval, m.ndof))
    print("Investigate the residuals")
    prediction = np.array([model_function(pair, **m.values.to_dict()) for pair in zip(*x_data)])
    print(np.abs(y_data - prediction))
    print(final_summary.sensor)
    print(prediction)
    print(y_data)
    return m.fmin.fval, m.ndof


if __name__ == "__main__":
    # we need to acquire all the data from the
    set_params(latex=True,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                           r"stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                           r"retain-zero-uncertainty}", fig_height=8.26772, fig_width=11.69291,
               minor=True, fontsize=16, dpi=1200)
    with tb.open_file('conclude_summary.h5', mode='a') as h5_conclusion:
        summary_data = h5_conclusion.root.GeneralSummaryTable
        sensor_properties = h5_conclusion.root.SensorTypes
        assert isinstance(sensor_properties, tb.Table)
        final_properties = read_rec_array_sorted(sensor_properties, 'sensor')
        final_summary = read_rec_array_sorted(summary_data, 'sensor')
        assert isinstance(final_properties, np.recarray)

        investigate_dependences_graphical(final_summary, final_properties, interesting_keys)

        model_mapper = {
            "implantation_area": (linear_model, linear_model),
            "implantation_depth": (linear_model, linear_model),
            "pixel_separation_x": (exponential_model, exponential_model),
            "pixel_separation_y": (exponential_model, exponential_model),
            "Perimeter": (exponential_model, exponential_model),
        }
        print(np.vstack([[sensor.decode() for sensor in final_properties.sensor], final_summary.sensor,
                         final_properties.implantation_depth, final_summary.biased_capacitance.capacitance]).T)

        investiagte_dependences_fitting(final_summary, final_properties, model_mapper)

        # it is also necessary to get a ND-Fit of our model for the capacitance distribution!
        upper_limit = 5
        suited_implant_area = final_properties.implantation_area[:-upper_limit]
        suited_depth = final_properties.implantation_depth[:-upper_limit]
        suited_perimeter = final_properties.Perimeter[:-upper_limit]
        suited_x_separation = final_properties.pixel_separation_x[:-upper_limit]
        suited_y_separation = final_properties.pixel_separation_y[:-upper_limit]

        full_implant_area = final_properties.implantation_area
        full_depth = final_properties.implantation_depth
        full_perimeter = final_properties.Perimeter
        full_x_separation = final_properties.pixel_separation_x
        full_y_separation = final_properties.pixel_separation_y

        capacitance_data = final_summary.biased_capacitance.capacitance[:-upper_limit]
        capacitance_errors = final_summary.biased_capacitance.capacitance_systematic_dispersion[:-upper_limit]
        # capacitance_errors = final_summary.biased_capacitance.capacitance_err[:-upper_limit]
        inter_capacitance_data = final_summary.biased_inter_capacitance.capacitance[:-upper_limit]
        inter_capacitance_errors = final_summary.biased_inter_capacitance.capacitance_systematic_dispersion[
            :-upper_limit]

        full_capacitance_data = final_summary.biased_capacitance.capacitance
        full_capacitance_errors = final_summary.biased_capacitance.capacitance_systematic_dispersion
        full_inter_capacitance_data = final_summary.biased_inter_capacitance.capacitance
        full_inter_capacitance_errors = final_summary.biased_inter_capacitance.capacitance_systematic_dispersion

        inter_cap_mask = np.isfinite(inter_capacitance_data)
        inter_capacitance_data = inter_capacitance_data[inter_cap_mask]
        inter_capacitance_errors = inter_capacitance_errors[inter_cap_mask]

        full_inter_cap_mask = np.isfinite(full_inter_capacitance_data)
        full_inter_capacitance_data = full_inter_capacitance_data[full_inter_cap_mask]
        full_inter_capacitance_errors = full_inter_capacitance_errors[full_inter_cap_mask]

        standard_model_dependences = (suited_implant_area, suited_depth, suited_perimeter)
        simple_model_dependences = (suited_implant_area, suited_depth)
        extended_model_dependences = (suited_implant_area, suited_depth, suited_perimeter, suited_x_separation,
                                      suited_y_separation)
        full_extended_model_dependences = (full_implant_area, full_depth, full_perimeter, full_x_separation,
                                  full_y_separation)
        inter_pix_model_dependences = (suited_implant_area[inter_cap_mask], suited_depth[inter_cap_mask],
                                       suited_perimeter[inter_cap_mask], suited_x_separation[inter_cap_mask],
                                       suited_y_separation[inter_cap_mask])
        full_inter_pix_model_dependences = (full_implant_area[full_inter_cap_mask], full_depth[full_inter_cap_mask],
                                            full_perimeter[full_inter_cap_mask], full_x_separation[full_inter_cap_mask],
                                            full_y_separation[full_inter_cap_mask])

        cost_0, dof_0 = handle_model_fit(simple_model_dependences, capacitance_data, capacitance_errors,
                                         simplified_cap_model, 'simplified', 2, 0.005, 0.16)

        # cost_1, dof_1 = handle_model_fit(standard_model_dependences, capacitance_data, capacitance_errors, cap_model, 'standard', 2, 0.005, 0.16)
        # print("Hypothesis test!")
        # print(cost_0 - cost_1, dof_0 - dof_1)
        # print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))

        # cost_1, dof_1 = handle_model_fit(extended_model_dependences, capacitance_data, capacitance_errors, extended_cap_model, "first_extension", 2, 0.005, 0.16)
        # print("Hypothesis test!")
        # print(cost_0 - cost_1, dof_0 - dof_1)
        # print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))

        cost_1, dof_1 = handle_model_fit(extended_model_dependences, capacitance_data, capacitance_errors,
                                         extended_cap_model_2,
                                         "second_extension", 2, 0.005, 0.16, 12.64e-3)
        print("Hypothesis test!")
        print(cost_0 - cost_1, dof_0 - dof_1)
        print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))

        cost_1, dof_1 = handle_model_fit(full_extended_model_dependences, full_capacitance_data,
                                         full_capacitance_errors, extended_cap_model_2,
                                         "third_extension", 3.4, 4.7e-3, 3.25, 13.2e-3, 0.15e-3)
        print("Hypothesis test!")
        print(cost_0 - cost_1, dof_0 - dof_1)
        print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))

        # cost_1, dof_1 = handle_model_fit(extended_model_dependences, capacitance_data, capacitance_errors,
        #                                  extended_cap_model_4,
        #                                  "fourth_extension", 0.2, 0.065, 8.9, 18.4, 0.1, -6.8e-3, 7.4e3, -7.7e3)
        # print("Hypothesis test!")
        # print(cost_0 - cost_1, dof_0 - dof_1)
        # print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))

        cost_inter_1, inter_dof_1 = handle_model_fit(inter_pix_model_dependences, inter_capacitance_data,
                                                     inter_capacitance_errors, inter_cap_model, "inter-pix", 0.1)

        cost_inter_2, inter_dof_2 = handle_model_fit(inter_pix_model_dependences, inter_capacitance_data,
                                                     inter_capacitance_errors, inter_cap_model_2,
                                                     "inter-pix-second", 0.1, 0.01, 0.01)

        print("Perform the inter-pixel hypothesis test!")
        print(cost_inter_1, inter_dof_1)
        print(cost_inter_2, inter_dof_2)
        effective_inter_cost = cost_inter_1 - cost_inter_2
        print(1 - chi2(effective_inter_cost, 1))

        handle_model_fit(full_inter_pix_model_dependences, full_inter_capacitance_data, full_inter_capacitance_errors,
                         inter_cap_model_2,
                         "inter-pix-full", 1, 3.32e-3, 2.16, 0.0594, 0.132)

        # generate a latex table of the test capacitances:
        test_table = h5_conclusion.root.TestCap
        test_properties = read_rec_array_sorted(test_table, 'Sensor')

        with open("tab_test_capacitance.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c", **print_args)
            header_output = "{\\(\\text{sensor}\\)}"
            actual_b_output = "Design"
            for key in field_names:
                print("S[table-format=3.3(3), separate-uncertainty, round-mode = uncertainty, round-precision = 3]",
                      **print_args)
                header_output += " & {{\\(C_\\text{{{}}}\\)}}".format(key.removeprefix('test_').split('_')[0])
                actual_b_output += " & {:.2f}".format(test_design_values[key])

            header_output += "\\\\"
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print(header_output, **print_args)
            print("\\midrule", **print_args)
            sensor_dict = {}
            print(actual_b_output, "\\\\", **print_args)

            for record in test_properties:
                # ignore any filtering for now and it get arbitraryly large
                actual_a_output = record.Sensor.decode().replace('_', "\\_")
                for key in field_names:
                    if key not in sensor_dict:
                        sensor_dict[key] = key.removeprefix('test_').split('_')[0]
                        sensor_dict[key] += " & {:.2f}".format(test_design_values[key])
                    actual_a_output += " & \\num{{{:.6g}\\pm{:.6g}}}".format(
                        record["test_" + key] * CAPACITANCE_CONVERSION_FACTOR,
                        record["test_" + key + "_error"] * CAPACITANCE_CONVERSION_FACTOR)
                    sensor_dict[key] += " & \\num{{{:.6g}\\pm{:.6g}}}".format(
                        record["test_" + key] * CAPACITANCE_CONVERSION_FACTOR,
                        record["test_" + key + "_error"] * CAPACITANCE_CONVERSION_FACTOR)

                print(actual_a_output, "\\\\", **print_args)

            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        with open("tab_test_capacitance_trans.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c", **print_args)
            print("S[table-format=3.2, round-precision = 2]", **print_args)
            header_output = "{\\(\\text{sensor}\\)} & {\\(C_\\text{design}\\)}"
            for key in test_properties.Sensor:
                print("S[table-format=3.3(3), separate-uncertainty, round-mode = uncertainty, round-precision = 3]",
                      **print_args)
                header_output += " & {{\\(C_\\text{{{}}}\\)}}".format(key.decode().replace('_', "\\_"))

            header_output += "\\\\"
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print(header_output, **print_args)
            print("\\midrule", **print_args)
            for item in sensor_dict.values():
                print(item, "\\\\", **print_args)
            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        # generate the biased capacitance latex table
        print("Create the tables to use.")
        with open("tab_capacitance_data.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c", **print_args)
            print("S[table-format=3.1, round-precision = 2]", **print_args)
            print("S[table-format=2.4(4)e20, separate-uncertainty]",
                  **print_args)
            print("S[table-format=2.4(4)e20, separate-uncertainty, retain-zero-uncertainty]",
                  **print_args)
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print("{\\(\\text{Sensor}\\)}", "&", "{\\(U\\text{ in }\\unit{\\volt}\\)}", "&", "{\\(C_\\text{biased}\\)}",
                  "&", "{\\(C_\\text{inter}\\)}", "\\\\", **print_args)
            print("\\midrule", **print_args)
            for record in final_summary:
                print(record.sensor.decode().replace('_', "\\_"), "&",
                      record.bias_voltage, "&",
                      generate_siunitx(record.biased_capacitance, False), "&",
                      generate_siunitx(record.biased_inter_capacitance, False),
                      "\\\\", **print_args)

            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        # generate the planar depletion voltages latex table
        with open("tab_depletion_planar_data.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c", **print_args)
            print("S[table-format=2.4(4)e20, separate-uncertainty]",
                  **print_args)
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print("{\\(\\text{Sensor}\\)}", "&", "{\\(U_\\text{depletion}\\)}", "\\\\", **print_args)
            print("\\midrule", **print_args)
            for sensor, dep_set in zip(final_summary.sensor, final_summary.depletion_voltage):
                if sensor.decode() in ("X5", "X6", "X7", "X8", "X3", "X4"):
                    continue
                print(sensor.decode().replace('_', '\\_'), "&", generate_siunitx(dep_set, True), "\\\\", **print_args)

            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        # generate the 3d depletion voltages latex table
        with open("tab_depletion_3d_data.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c", **print_args)
            print("S[table-format=2.4(4)e20, separate-uncertainty]",
                  **print_args)
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print("{\\(\\text{{Sensor}}\\)}", "&", "{\\(U_\\text{depletion}\\)}", "\\\\", **print_args)
            print("\\midrule", **print_args)
            for sensor, dep_set in zip(final_summary.sensor, final_summary.depletion_voltage):
                if sensor.decode() not in ("X5", "X6", "X7", "X8", "X3", "X4"):
                    continue
                print(sensor.decode().replace('_', '\\_'), "&", generate_siunitx(dep_set, True), "\\\\", **print_args)

            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        sample_data = np.arange(0.1, 10, 0.05)
        plt.plot(sample_data, 1 / sample_data)
        plt.plot(sample_data, 1 / sample_data ** 2)
        plt.plot(sample_data, 1 / np.sqrt(sample_data))
        plt.plot(sample_data, 20 * np.exp(-sample_data))
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        plt.plot(sample_data, 1 / sample_data)
        plt.plot(sample_data, 1 / sample_data ** 2)
        plt.plot(sample_data, 1 / np.sqrt(sample_data))
        plt.plot(sample_data, 20 * np.exp(-sample_data))
        plt.yscale('log')
        plt.show()
        plt.plot(sample_data, 1 / sample_data)
        plt.plot(sample_data, 1 / sample_data ** 2)
        plt.plot(sample_data, 1 / np.sqrt(sample_data))
        plt.plot(sample_data, 20 * np.exp(-sample_data))
        plt.show()
