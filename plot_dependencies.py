import locale
import numpy as np
import tables as tb
from collections.abc import Iterable, Mapping, Callable
from iminuit import Minuit
from iminuit.cost import LeastSquares
from inspect import Parameter
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats.distributions import chi2 as sample_chi2
from typing import Optional

from capacitance_models import linear_model, simplified_cap_model, extended_cap_model, extended_cap_model_2, \
    extended_cap_model_3, extended_cap_model_4, extended_cap_model_5, extended_cap_model_6, extended_cap_model_7, \
    extended_cap_model_8
from general_model import exponential_model
from inter_capacitance_models import inter_cap_model, inter_cap_model_2, inter_cap_model_3, inter_cap_model_4, \
    inter_cap_model_5, inter_cap_model_6, inter_cap_model_7
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
    "pitch_x",
    "implantation_size_x",
]


def read_sorted_where(table: tb.Table,
                      sortby: tb.Column | str,
                      condition: str,
                      condvars: dict[str, tb.Column | np.ndarray] | None = None,
                      checkCSI: bool = False,
                      field=None,
                      start: str | None = None,
                      stop: str | None = None,
                      step: str | None = None,) -> np.ndarray:
    table._g_check_open()
    index = table._check_sortby_csi(sortby, checkCSI)

    filtered_coords = [
        p.nrow for p in table._where(condition, condvars, start, stop, step)
    ]
    coords = [q for q in index[None:None:None] if q in filtered_coords]
    table._where_condition = None  # reset the conditions
    if len(coords) > 1:
        cstart, cstop = coords[0], coords[-1] + 1
        if cstop - cstart == len(coords):
            # Chances for monotonically increasing row values. Refine.
            inc_seq = np.all(np.arange(cstart, cstop) == np.array(coords))
            if inc_seq:
                return table.read(cstart, cstop, field=field)
    return table.read_coordinates(coords, field)

# format the table output
GENERAL_SIUNITX_FORMAT = "\\qty{{{:#0.6g}({:d})({:d})({:d})}}{{{}}}"
GENERAL_SIUNITX_FORMAT_2 = "\\qty{{{:.6f}({:.6f})({:.6f})({:.6f})}}{{{}}}"

GENERAL_PART = "\\qty{{{{{{:#0.{}g}}({{:d}})({{:d}})({{:d}})}}}}{{{{{{}}}}}}"
REDUCED_GENERAL_PART = "{{:#0.{}g}}({{:d}})({{:d}})({{:d}})"
SECOND_GENERAL_PART = "{{:#0.{}f}}({{:d}})({{:d}})({{:d}})"
THIRD_GENERAL_PART = "{{:#0.{}f}}({{:d}})"

GENERAL_PREC = 6
GENERAL_TEST_FORMAT = "{:-6g}"
GENERAL_UNCERT_MULTIPLIER = 1e6

axis_mapping = {
    "pitch_x": "$p_\\text{{x}}$ / \\unit{{\\micro\\meter}}",
    "pitch_y": "$p_\\text{{y}}$ / \\unit{{\\micro\\meter}}",
    "implantation_size_x": "$w_\\text{{x}}$ / \\unit{{\\micro\\meter}}",
    "implantation_size_y": "$w_\\text{{y}}$ / \\unit{{\\micro\\meter}}",
    "pixel_area": "$A$ / \\unit{{\\micro\\meter\\squared}}",
    "implantation_area": "$A$ / \\unit{{\\micro\\meter\\squared}}",
    "pixel_separation_x": "$\\Delta x$ / \\unit{{\\micro\\meter}}",
    "pixel_separation_y": "$\\Delta y$ / \\unit{{\\micro\\meter}}",
    "pixel_separation_area": "$A_\\text{{separation}}$ / \\unit{{\\micro\\meter\\squared}}",
    "implantation_depth": "$d$ / \\unit{{\\micro\\meter}}",
    "sensor_depth": "$D$ / \\unit{{\\micro\\meter}}",
    "Perimeter": "$U$ / \\unit{{\\micro\\meter}}",
}

def enhanced_logical_or(*args):
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return np.logical_or(*args)
    args = list(args)
    current_result = np.logical_or(args.pop(0), args.pop(0))
    for further_mask in args:
        current_result |= further_mask

    return current_result

def generate_siunitx(data_set: np.recarray, depletion=False) -> str:
    if depletion:
        test_format = "{:.3g}".format(data_set.stat_error)
        _, text_n_digits = test_format.removeprefix('-').split('.', 1)
        n_digits = len(text_n_digits)
        return SECOND_GENERAL_PART.format(n_digits).format(data_set.magnitude,
                                                           int(data_set.stat_error * 10 ** n_digits),
                                                           int(data_set.systematic_general * 10 ** n_digits),
                                                           int(data_set.systematic_dispersion * 10 ** n_digits), )

        # old approach
        # test_format = GENERAL_TEST_FORMAT.format(data_set.magnitude)
        # int_part, _ = test_format.removeprefix('-').split('.', 1)
        # sig = GENERAL_PREC if int(int_part) == 0 else GENERAL_PREC + len(int_part)
        # return GENERAL_PART.format(sig).format(data_set.magnitude,
        #                                     int(data_set.stat_error * GENERAL_UNCERT_MULTIPLIER),
        #                                     int(data_set.systematic_general * GENERAL_UNCERT_MULTIPLIER),
        #                                     int(data_set.systematic_dispersion * GENERAL_UNCERT_MULTIPLIER), "\\volt")
    else:
        if np.isnan(data_set.magnitude):
            return "\\text{k.A.}"

        test_format = "{:.3g}".format(data_set.stat_error)
        _, text_n_digits = test_format.removeprefix('-').split('.', 1)
        n_digits = len(text_n_digits)
        return SECOND_GENERAL_PART.format(n_digits).format(data_set.magnitude,
                                                           int(data_set.stat_error * 10 ** n_digits),
                                                           int(data_set.systematic_general * 10 ** n_digits),
                                                           int(data_set.systematic_dispersion * 10 ** n_digits), )

        # old approach
        # test_format = GENERAL_TEST_FORMAT.format(data_set.magnitude)
        # int_part, _ = test_format.removeprefix('-').split('.', 1)
        # sig = GENERAL_PREC if int(int_part) == 0 else GENERAL_PREC + len(int_part)
        # return GENERAL_PART.format(sig).format(data_set.magnitude,
        #                                        int(data_set.stat_error * GENERAL_UNCERT_MULTIPLIER),
        #                                        int(data_set.systematic_general * GENERAL_UNCERT_MULTIPLIER),
        #                                        int(data_set.systematic_dispersion * GENERAL_UNCERT_MULTIPLIER), "\\femto\\farad")

def generate_siunitx_2(data_set: np.recarray, depletion=False) -> str:
    if depletion:
        return GENERAL_SIUNITX_FORMAT_2.format(data_set.magnitude,
                                               data_set.stat_error,
                                               data_set.systematic_general,
                                               data_set.systematic_dispersion, "\\volt")
    else:
        if np.isnan(data_set.magnitude):
            return "\\text{k.A.}"
        return GENERAL_SIUNITX_FORMAT_2.format(data_set.magnitude,
                                               data_set.stat_error,
                                               data_set.systematic_general,
                                               data_set.systematic_dispersion, "\\femto\\farad")

def generate_siunitx_3(data_set: np.recarray, depletion=False) -> str:
    if depletion:
        test_format = "{:.3g}".format(data_set.stat_error)
        _, text_n_digits = test_format.removeprefix('-').split('.', 1)
        n_digits = len(text_n_digits)
        return THIRD_GENERAL_PART.format(n_digits).format(data_set.magnitude,
                                                           int(data_set.stat_error * 10 ** n_digits),
                                                           int(data_set.systematic_general * 10 ** n_digits),
                                                           int(data_set.systematic_dispersion * 10 ** n_digits), )

        # old approach
        # test_format = GENERAL_TEST_FORMAT.format(data_set.magnitude)
        # int_part, _ = test_format.removeprefix('-').split('.', 1)
        # sig = GENERAL_PREC if int(int_part) == 0 else GENERAL_PREC + len(int_part)
        # return GENERAL_PART.format(sig).format(data_set.magnitude,
        #                                     int(data_set.stat_error * GENERAL_UNCERT_MULTIPLIER),
        #                                     int(data_set.systematic_general * GENERAL_UNCERT_MULTIPLIER),
        #                                     int(data_set.systematic_dispersion * GENERAL_UNCERT_MULTIPLIER), "\\volt")
    else:
        if np.isnan(data_set.magnitude):
            # TODO: könnte man hier auch direkt 'k.A.' raus machen?
            return "\\text{k.A.}"

        test_format = "{:.3g}".format(data_set.stat_error)
        _, text_n_digits = test_format.removeprefix('-').split('.', 1)
        n_digits = len(text_n_digits)
        return THIRD_GENERAL_PART.format(n_digits).format(data_set.magnitude,
                                                           int(data_set.stat_error * 10 ** n_digits),
                                                           int(data_set.systematic_general * 10 ** n_digits),
                                                           int(data_set.systematic_dispersion * 10 ** n_digits), )

        # old approach
        # test_format = GENERAL_TEST_FORMAT.format(data_set.magnitude)
        # int_part, _ = test_format.removeprefix('-').split('.', 1)
        # sig = GENERAL_PREC if int(int_part) == 0 else GENERAL_PREC + len(int_part)
        # return GENERAL_PART.format(sig).format(data_set.magnitude,
        #                                        int(data_set.stat_error * GENERAL_UNCERT_MULTIPLIER),
        #                                        int(data_set.systematic_general * GENERAL_UNCERT_MULTIPLIER),
        #                                        int(data_set.systematic_dispersion * GENERAL_UNCERT_MULTIPLIER), "\\femto\\farad")

def read_rec_array(table: tb.Table, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read(*args, **kwargs), dtype=table.dtype)

def read_rec_array_sorted(table: tb.Table, primary_key, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read_sorted(primary_key, *args, **kwargs), dtype=table.dtype)

def read_rec_array_where(table: tb.Table, condition, *args, **kwargs) -> np.recarray:
    return np.rec.array(table.read_where(condition, *args, **kwargs), dtype=table.dtype)

def read_rec_array_sorted_where(table: tb.Table, condition, primary_key, *args, **kwargs) -> np.recarray:
    return np.rec.array(read_sorted_where(table, primary_key, condition, *args, **kwargs), dtype=table.dtype)

def investigate_dependences_graphical(summary_data: np.recarray, property_data: np.recarray,
                                      interesting_data: Iterable, **keys):
    spatial_mask = np.array([sensor.decode() in spatial_identifier for sensor in summary_data.sensor], dtype=bool)

    plotter("Dependencies.pdf", spatial_mask, interesting_data, summary_data, property_data, **keys)

    plotter("Dependencies-log.pdf", spatial_mask, interesting_data, summary_data, property_data, scaley='log', **keys)

    plotter("Dependencies-log-log.pdf", spatial_mask, interesting_data, summary_data, property_data,
            scalex='log', scaley='log', **keys)


def get_test_capacitance_data_correction(group: tb.Group, **kwargs):
    locale.setlocale(locale.LC_NUMERIC, "DE")
    table = read_rec_array(group.TestCapCorrected)
    capacitances = [item for item in table.dtype.names if item != "sensor" and not item.endswith('_error')]
    exclusion_list = kwargs.get("exlusion_list", [])
    if kwargs.get("print_result", True):
        for sensor in np.strings.decode(table.sensor):
            if sensor in exclusion_list:
                continue
            record = table[sensor]
            print("handling now the test capacitances for board", sensor)
            for k, cap_name in enumerate(capacitances):
                cap = record[cap_name]
                err = record[cap_name + "_error"]
                eff_cap = cap * 1e15
                eff_err = err * 1e15
                print(k, f"{eff_cap:.3n}+-{eff_err:.3n}")



def __dependence_fit(spatial_mask, properties: np.recarray, y_data_set: np.recarray, model_name, model: Callable, condition: Optional[Callable]=None, systematic=True, x_log=False, y_log=False):
    if condition is None:
        mask = np.logical_not(spatial_mask)
    else:
        mask = np.logical_and(np.logical_not(spatial_mask), condition(y_data_set.magnitude))

    x_data = properties[key][mask]
    y_data = y_data_set.magnitude[mask]
    if systematic:
        y_error = y_data_set.systematic_dispersion[mask]
    else:
        y_error = y_data_set.stat_error[mask]

    if x_log:
        x_data = np.log(x_data)
    if y_log:
        y_error = y_error / y_data
        y_data = np.log(y_data)

    handle_model_simple_fit(x_data, y_data, y_error, model, model_name)

def investigate_dependencies_fitting(summary_data: np.recarray, property_data: np.recarray, interesting_data: Mapping, x_log=False, y_log=False):
    spatial_mask = np.array([sensor.decode() in spatial_identifier for sensor in summary_data.sensor], dtype=bool)
    for key, models in interesting_data.items():
        total_model, inter_model = models
        print("Will handle the dependent:", key)
        if total_model is not None:
            mask = np.logical_not(spatial_mask)
            x_data = property_data[key][mask]
            y_data = summary_data.biased_capacitance.magnitude[mask]
            y_error = summary_data.biased_capacitance.systematic_dispersion[mask]
            if x_log:
                x_data = np.where(x_data > 0, np.log(x_data), 0)
            if y_log:
                y_error = y_error / y_data
                y_data = np.where(y_data > 0, np.log(y_data), 0)
            handle_model_simple_fit(x_data, y_data, y_error, total_model, 'total' + key)

        if inter_model is not None:
            mask = np.logical_and(np.logical_not(spatial_mask), np.isfinite(
                summary_data.biased_inter_capacitance.magnitude))
            x_data = property_data[key][mask]
            y_data = summary_data.biased_inter_capacitance.magnitude[mask]
            y_error = summary_data.biased_inter_capacitance.systematic_dispersion[mask]
            if x_log:
                x_data = np.where(x_data > 0, np.log(x_data), 0)
            if y_log:
                y_error = y_error / y_data
                y_data = np.where(y_data > 0, np.log(y_data), 0)
            handle_model_simple_fit(x_data, y_data, y_error, inter_model, 'inter' + key)

marker_list = ['X1', 'X2', 'R1']

def plotter(file, spatial_mask, keys: Iterable, data: np.recarray, properties: np.recarray, **keywords):
    def __plot_dependence(pdf: PdfPages, spatial_mask, physical_property, name, quantity, unit, x, y, **kwargs):
        scalex = kwargs.pop("scalex", None)
        scaley = kwargs.pop("scaley", None)
        fig, ax = plt.subplots()
        if not kwargs.get('no_plot', False):
            ax.set_title("Dependence of the {} on {}".format(name, physical_property))
        ax.set_ylabel("${}$ / \\unit{{{}}}".format(quantity, unit))
        ax.set_xlabel(axis_mapping[physical_property])
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

        # in addition there are some sensors which needs to be marked properly
        first_properties = x[first_mask]
        first_points = y[first_mask]
        second_properties = x[second_mask]
        second_points = y[second_mask]
        sensor_names_first = np.strings.decode(first_properties.sensor)
        sensor_names_second = np.strings.decode(second_properties.sensor)
        first_sensor_mask_components = [np.strings.endswith(sensor_names_first, ref) for ref in marker_list]
        second_sensor_mask_components = [np.strings.endswith(sensor_names_second, ref) for ref in marker_list]
        first_sensor_mask = enhanced_logical_or(*first_sensor_mask_components)
        second_sensor_mask = enhanced_logical_or(*second_sensor_mask_components)
        for x_c, y_c, name in zip(first_properties[first_sensor_mask][physical_property], first_points[first_sensor_mask],sensor_names_first[first_sensor_mask]):
            ax.annotate(name, xy=(x_c, y_c), color='b')

        for x_c, y_c, name in zip(second_properties[second_sensor_mask][physical_property],
                                  second_points[second_sensor_mask], sensor_names_second[second_sensor_mask]):
            ax.annotate(name, xy=(x_c, y_c), color='orange')


        ax.legend()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    with PdfPages(file) as pdf:
        physical_property = None
        for physical_property in keys:
            __plot_dependence(pdf, spatial_mask, physical_property, "capacitance", 'C',
                              '\\femto\\farad', properties, data.biased_capacitance.magnitude,
                              **keywords)

            __plot_dependence(pdf, spatial_mask, physical_property, 'inter-pixel capacitance',
                              'C_\\text{inter}', '\\femto\\farad', properties,
                              data.biased_inter_capacitance.magnitude, condition=lambda x: x >= 0,
                              **keywords)

            __plot_dependence(pdf, spatial_mask, physical_property, 'backplane capacitance',
                              'C_\\text{back}', '\\femto\\farad', properties,
                              data.biased_back_capacitance.magnitude, condition=lambda x: x >= 0,
                              **keywords)

            scalex = keywords.get("scalex", None)
            scaley = keywords.get("scaley", None)
            fig, ax = plt.subplots()
            if not keywords.get('no_plot', False):
                ax.set_title("Dependence of the depletion voltage on {}".format(physical_property))
            ax.set_ylabel("$U$ in \\unit{{\\volt}}")
            if scalex is not None:
                ax.set_xscale(scalex)
            if scaley is not None:
                ax.set_yscale(scaley)
            mask = np.isfinite(final_summary.depletion_voltage.magnitude)
            first_mask = np.logical_and(mask, spatial_mask)
            second_mask = np.logical_and(mask, np.logical_not(spatial_mask))
            ax.errorbar(final_properties[physical_property][first_mask], final_summary.depletion_voltage.magnitude[first_mask], fmt='x', label='3D')
            ax.errorbar(final_properties[physical_property][second_mask],
                        final_summary.depletion_voltage.magnitude[second_mask], fmt='x', label='planar')
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
    plt.title(name)
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
    # if 'a0' in position_argument:
    #     m.limits['a0'] = (0, None)
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
               minor=True, fontsize=24, dpi=1200)
    with tb.open_file('conclude_summary.h5', mode='r') as h5_conclusion:
        summary_data = h5_conclusion.root.GeneralSummaryTable
        # CHECK: whether 'test_data' is still used!
        test_data = read_rec_array(h5_conclusion.root.TestCapCorrected)
        sensor_properties = h5_conclusion.root.SensorTypes
        assert isinstance(sensor_properties, tb.Table)
        final_properties = read_rec_array_sorted(sensor_properties, 'sensor')
        final_summary = read_rec_array_sorted(summary_data, 'sensor')
        assert isinstance(final_properties, np.recarray)

        enhanced_properties = final_properties.copy()
        enhanced_summary = final_summary.copy()
        for cap_key in [
            'biased_back_capacitance',
            'biased_inter_capacitance',
            'biased_capacitance',
        ]:
            enhanced_summary[cap_key].magnitude /= final_properties.pixel_area

        investigate_dependences_graphical(final_summary, final_properties, interesting_keys, no_plot=True)
        # investigate_dependences_graphical(enhanced_summary, final_properties, interesting_keys, no_plot=True)

        model_mapper = {
            "implantation_area": (linear_model, linear_model),
            "implantation_depth": (linear_model, linear_model),
            "pixel_separation_x": (exponential_model, exponential_model),
            "pixel_separation_y": (exponential_model, exponential_model),
            "Perimeter": (exponential_model, exponential_model),
        }
        second_model_mapper = {
            "pixel_separation_x": (linear_model, linear_model),
            "pixel_separation_y": (linear_model, linear_model),
            "Perimeter": (linear_model, linear_model),
        }
        third_model_mapper = {
            "implantation_area": (linear_model, linear_model),
            "implantation_depth": (linear_model, linear_model),
            "pixel_separation_x": (linear_model, linear_model),
            "pixel_separation_y": (linear_model, linear_model),
            "Perimeter": (linear_model, linear_model),
        }
        # investigate_dependencies_fitting(final_summary, final_properties, model_mapper)
        # print("Try the log plots")
        # investigate_dependencies_fitting(final_summary, final_properties, second_model_mapper, y_log=True)
        # print("Try the log-log plots")
        # investigate_dependencies_fitting(final_summary, final_properties, third_model_mapper, x_log=True, y_log=True)

        # it is also necessary to get a ND-Fit of our model for the capacitance distribution!
        # TODO: refine the selection procedure!
        upper_limit = 6
        general_upper_selection = slice(0, -6)
        assert np.shape(final_properties.implantation_area[:-upper_limit]) == np.shape(final_properties.implantation_area[general_upper_selection])
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

        capacitance_data = final_summary.biased_capacitance.magnitude[:-upper_limit]
        capacitance_errors = final_summary.biased_capacitance.systematic_dispersion[:-upper_limit]
        # capacitance_errors = final_summary.biased_capacitance.stat_error[:-upper_limit]
        inter_capacitance_data = final_summary.biased_inter_capacitance.magnitude[:-upper_limit]
        inter_capacitance_errors = final_summary.biased_inter_capacitance.stat_error[:-upper_limit]

        full_capacitance_data = final_summary.biased_capacitance.magnitude
        full_capacitance_errors = final_summary.biased_capacitance.systematic_dispersion
        full_inter_capacitance_data = final_summary.biased_inter_capacitance.magnitude
        full_inter_capacitance_errors = final_summary.biased_inter_capacitance.stat_error

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

        total_cap_costs = []
        total_cap_dof = []

        cost_0, dof_0 = handle_model_fit(simple_model_dependences, capacitance_data, capacitance_errors,
                                         simplified_cap_model, 'simplified', 2, 0.005, 0.16)

        cost_1, dof_1 = handle_model_fit(extended_model_dependences, capacitance_data, capacitance_errors,
                                         extended_cap_model, "first_extension",
                                         0.06, 0.11, -0.0017, -0.00029, 1.57e-5, 3.9e-6)
        print("Hypothesis test!")
        print(cost_0 - cost_1, dof_0 - dof_1)
        print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))
        total_cap_costs.append(cost_1)
        total_cap_dof.append(dof_1)

        cost_1, dof_1 = handle_model_fit(extended_model_dependences, capacitance_data, capacitance_errors,
                                         extended_cap_model_2,
                                         "second_extension", 6.45, 0, 10.87e-3, 0)
        print("Hypothesis test!")
        print(cost_0 - cost_1, dof_0 - dof_1)
        print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))
        total_cap_costs.append(cost_1)
        total_cap_dof.append(dof_1)

        cost_1, dof_1 = handle_model_fit(extended_model_dependences, capacitance_data,
                                         capacitance_errors, extended_cap_model_3,
                                         "third_extension", 0.2, 0.065)

        print("Hypothesis test!")
        print(cost_0 - cost_1, dof_0 - dof_1)
        print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))
        total_cap_costs.append(cost_1)
        total_cap_dof.append(dof_1)

        cost_1, dof_1 = handle_model_fit(extended_model_dependences, capacitance_data,
                                         capacitance_errors, extended_cap_model_4,
                                         "fourth_extension", 0.06, 0.11, -0.0017, -0.00029, 1.57e-5, 3.9e-6)

        print("Hypothesis test!")
        print(cost_0 - cost_1, dof_0 - dof_1)
        print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))
        total_cap_costs.append(cost_1)
        total_cap_dof.append(dof_1)

        cost_1, dof_1 = handle_model_fit(extended_model_dependences, capacitance_data,
                                         capacitance_errors, extended_cap_model_5,
                                         "fifth_extension", 0.06, 0.11, -0.0017, -0.00029, 1.57e-5, 3.9e-6, 0, 0)

        print("Hypothesis test!")
        print(cost_0 - cost_1, dof_0 - dof_1)
        print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))
        total_cap_costs.append(cost_1)
        total_cap_dof.append(dof_1)

        cost_1, dof_1 = handle_model_fit(extended_model_dependences, capacitance_data,
                                         capacitance_errors, extended_cap_model_6,
                                         "sixth_extension", 0.06, 0.11, -0.0017, -0.00029, 1.57e-5, 3.9e-6)

        print("Hypothesis test!")
        print(cost_0 - cost_1, dof_0 - dof_1)
        print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))
        total_cap_costs.append(cost_1)
        total_cap_dof.append(dof_1)

        cost_1, dof_1 = handle_model_fit(extended_model_dependences, capacitance_data,
                                         capacitance_errors, extended_cap_model_7,
                                         "seventh_extension", 0.06, 0.11, -0.0017, -0.00029)

        print("Hypothesis test!")
        print(cost_0 - cost_1, dof_0 - dof_1)
        print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))
        total_cap_costs.append(cost_1)
        total_cap_dof.append(dof_1)

        cost_1, dof_1 = handle_model_fit(extended_model_dependences, capacitance_data,
                                         capacitance_errors, extended_cap_model_8,
                                         "eigth_extension", 0.06, 0.11, -0.0017, -0.00029, 1.57e-5, 3.9e-6, 0, 0)

        print("Hypothesis test!")
        print(cost_0 - cost_1, dof_0 - dof_1)
        print(1 - chi2(cost_0 - cost_1, dof_0 - dof_1))
        total_cap_costs.append(cost_1)
        total_cap_dof.append(dof_1)

        print("INVESTIGATE THE HYPOTHESIS OF THE DIFFERENT MODELS")
        total_p_values = np.full((len(total_cap_costs), len(total_cap_dof)), np.nan)
        total_p_values_compare = np.full((len(total_cap_costs), len(total_cap_dof)), np.nan)

        for ii, jj in np.ndindex(len(total_cap_costs), len(total_cap_dof)):
            if ii == jj:
                total_p_values[ii, ii] = 1 - chi2(total_cap_costs[ii], total_cap_dof[ii])
                continue
            effective_dof = np.abs(total_cap_dof[ii] - total_cap_dof[jj])
            if effective_dof == 0:
                total_p_values_compare[ii, jj] = np.inf * np.where(total_cap_costs[ii] > total_cap_costs[jj], 1, -1)
            elif total_cap_costs[ii] > total_cap_costs[jj]:
                p = 1 - chi2(total_cap_costs[ii] - total_cap_costs[jj], effective_dof)
                total_p_values_compare[ii, jj] = p
                if p < 0.01:
                   print( "Drop model {} in favour of model {}".format(ii + 1, jj + 1))
            else:
                p = (1 - chi2(total_cap_costs[jj] - total_cap_costs[ii], effective_dof))
                total_p_values_compare[ii, jj] = p * -1
                if p < 0.01:
                    print("Drop model {} in favour of model {}".format(jj + 1, ii + 1))


        print(total_p_values)
        print("And for the comparison")
        print(total_p_values_compare)
        print("Diagonals")
        print(np.diag(total_p_values_compare))

        print("INVESTIGATION OF THE INTER-PIXEL-CAPACITANCE")
        inter_cap_costs = []
        inter_cap_dof = []
        cost_inter_1, inter_dof_1 = handle_model_fit(inter_pix_model_dependences, inter_capacitance_data,
                                                     inter_capacitance_errors, inter_cap_model, "inter-pix", 0.1)
        inter_cap_costs.append(cost_inter_1)
        inter_cap_dof.append(inter_dof_1)

        cost_inter_2, inter_dof_2 = handle_model_fit(inter_pix_model_dependences, inter_capacitance_data,
                                                     inter_capacitance_errors, inter_cap_model_2,
                                                     "inter-pix-second", 0.1, 0.01)
        inter_cap_costs.append(cost_inter_2)
        inter_cap_dof.append(inter_dof_2)

        print("Perform the inter-pixel hypothesis test!")
        print(cost_inter_1, inter_dof_1)
        print(cost_inter_2, inter_dof_2)
        effective_inter_cost = cost_inter_1 - cost_inter_2
        print(1 - chi2(effective_inter_cost, 1))

        cost_inter_1, inter_dof_1 = handle_model_fit(inter_pix_model_dependences, inter_capacitance_data,
                                                     inter_capacitance_errors, inter_cap_model_3, "inter-pix third", 0.1)
        inter_cap_costs.append(cost_inter_1)
        inter_cap_dof.append(inter_dof_1)

        cost_inter_1, inter_dof_1 = handle_model_fit(inter_pix_model_dependences, inter_capacitance_data,
                                                     inter_capacitance_errors, inter_cap_model_4, "inter-pix fourth",
                                                     0.1, 0.06, 0. -0.0018, 0, 0.018e-3)
        inter_cap_costs.append(cost_inter_1)
        inter_cap_dof.append(inter_dof_1)

        cost_inter_1, inter_dof_1 = handle_model_fit(inter_pix_model_dependences, inter_capacitance_data,
                                                     inter_capacitance_errors, inter_cap_model_5, "inter-pix fifth",
                                                     0.1, 0.06, 0. - 0.0018, 0, 0.018e-3)
        inter_cap_costs.append(cost_inter_1)
        inter_cap_dof.append(inter_dof_1)

        cost_inter_1, inter_dof_1 = handle_model_fit(inter_pix_model_dependences, inter_capacitance_data,
                                                     inter_capacitance_errors, inter_cap_model_6, "inter-pix sixth",
                                                     0.1, 0.06, 0. - 0.0018)
        inter_cap_costs.append(cost_inter_1)
        inter_cap_dof.append(inter_dof_1)

        cost_inter_1, inter_dof_1 = handle_model_fit(inter_pix_model_dependences, inter_capacitance_data,
                                                     inter_capacitance_errors, inter_cap_model_7, "inter-pix seventh",
                                                     0.1, 0.06, 0. - 0.0018)
        inter_cap_costs.append(cost_inter_1)
        inter_cap_dof.append(inter_dof_1)

        # generate a latex table of the test capacitances:
        test_table = h5_conclusion.root.TestCapCorrected
        test_properties = read_rec_array_sorted(test_table, 'Sensor')

        standard_cap_list = [
            "6",
            "7",
            "8",
            "9",
            "32",
        ]

        sensor_trans_list = [
        ]

        with open("tab_test_capacitance.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c|", **print_args)
            header_output = "{\\(\\text{Sensor}\\)}"
            actual_b_output = "Design"
            for key in field_names:
                formatted_key = key.removeprefix('test_').split('_')[0]
                if formatted_key in standard_cap_list:
                    print("S[table-format=3.4(3), separate-uncertainty, round-mode = uncertainty, round-precision = 3]|",
                          **print_args)
                    header_output += " & {{\\(C_\\text{{{}}}\\)}}".format(formatted_key)
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
                    if key.removeprefix('test_').split('_')[0] in standard_cap_list:
                        actual_a_output += " & {:.6g}\\pm{:.6g}".format(
                            record["test_" + key] * CAPACITANCE_CONVERSION_FACTOR,
                            record["test_" + key + "_error"] * CAPACITANCE_CONVERSION_FACTOR)
                    sensor_dict[key] += " & {:.6g}\\pm{:.6g}".format(
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
            print("c|", **print_args)
            print("S[table-format=3.2, round-precision = 2]|", **print_args)
            header_output = "{\\(\\text{Sensor}\\)} & {\\(C_\\text{design}\\)}"
            for key in test_properties.Sensor:
                print("S[table-format=3.4(3), separate-uncertainty, round-mode = uncertainty, round-precision = 3]|",
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
            print("c|", **print_args)
            print("S[table-format=3.1, round-precision = 2]|", **print_args)
            print("S[table-format=2.3,tight-spacing]", **print_args)
            print("S[table-format=1.4,table-align-text-after=false]", **print_args)
            print("S[table-format=1.4,table-align-text-after=false]", **print_args)
            print("S[table-format=1.7,table-align-text-after=false]|", **print_args)
            print("S[table-format=2.4(4), separate-uncertainty, retain-zero-uncertainty]",
                  **print_args)
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print("{\\(\\text{Sensor}\\)}", "&", "{\\(U\\text{ / }\\unit{\\volt}\\)}", "&", "\\multicolumn{4}{c}{\\(C_\\text{biased}\\text{ / }\\unit{\\femto\\farad}\\)}",
                  "&", "{\\(C_\\text{inter}\\text{ / }\\unit{\\femto\\farad}\\)}", "\\\\", **print_args)
            print("\\midrule", **print_args)
            for record in final_summary:
                test_format = "{:.3g}".format(record.biased_capacitance.stat_error)
                _, text_n_digits = test_format.removeprefix('-').split('.', 1)
                n_digits = len(text_n_digits)
                number_format = "{{:.{}f}}".format(n_digits)

                print(record.sensor.decode().replace('_', '\\_'), "&", record.bias_voltage, "&", number_format.format(record.biased_capacitance.magnitude),
                      "&", "+-" + number_format.format(record.biased_capacitance.stat_error) + '\\textstatistic', "&",
                      "+-" + number_format.format(record.biased_capacitance.systematic_general) + '\\textsystematic',
                      "&", "+-" + number_format.format(record.biased_capacitance.systematic_dispersion) + '\\textdispersion',
                      "&", generate_siunitx_3(record.biased_inter_capacitance), "\\\\", **print_args)


                # print(record.sensor.decode().replace('_', "\\_"), "&",
                #       record.bias_voltage, "&",
                #       generate_siunitx(record.biased_capacitance, False), "&",
                #       generate_siunitx(record.biased_inter_capacitance, False),
                #       "\\\\", **print_args)

            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        # generate the planar depletion voltages latex table
        with open("tab_depletion_planar_data.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c|", **print_args)
            print("S[table-format=-2.2]", **print_args)
            print("S[table-format=2.4,table-align-text-after=false]", **print_args)
            print("S[table-format=2.4,table-align-text-after=false]", **print_args)
            print("S[table-format=2.7,table-align-text-after=false]", **print_args)
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print("{\\(\\text{Sensor}\\)}", "&", "\\multicolumn{4}{c}{\\(U_\\text{depletion}\\) / \\unit{\\volt}}", "\\\\", **print_args)
            print("\\midrule", **print_args)
            for sensor, dep_set in zip(final_summary.sensor, final_summary.depletion_voltage):
                if sensor.decode() in ("X5", "X6", "X7", "X8", "X3", "X4"):
                    continue

                test_format = "{:.3g}".format(dep_set.stat_error)
                try:
                    _, text_n_digits = test_format.removeprefix('-').split('.', 1)
                except:
                    print(test_format)
                    raise
                n_digits = len(text_n_digits)
                number_format = "{{:.{}f}}".format(n_digits)
                print(sensor.decode().replace('_', '\\_'), "&", number_format.format(dep_set.magnitude),
                      "&", "+-" + number_format.format(dep_set.stat_error) + '\\textstatistic', "&", "+-" + number_format.format(dep_set.systematic_general) + '\\textsystematic',
                      "&", "+-" + number_format.format(dep_set.systematic_dispersion) + '\\textdispersion', "\\\\", **print_args)

            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        # generate the 3d depletion voltages latex table
        with open("tab_depletion_3d_data.tex", "w") as f:
            print_args = {
                "file": f
            }
            print("\\begin{tabular}{@{}", **print_args)
            print("c|", **print_args)
            print("S[table-format=-2.2]", **print_args)
            print("S[table-format=2.4,table-align-text-after=false]", **print_args)
            print("S[table-format=2.4,table-align-text-after=false]", **print_args)
            print("S[table-format=2.7,table-align-text-after=false]", **print_args)
            print("@{}}", **print_args)
            print("\\toprule", **print_args)
            print("{\\(\\text{{Sensor}}\\)}", "&", "\\multicolumn{4}{c}{\\(U_\\text{depletion}\\) / \\unit{\\volt}}", "\\\\", **print_args)
            print("\\midrule", **print_args)
            for sensor, dep_set in zip(final_summary.sensor, final_summary.depletion_voltage):
                if sensor.decode() not in ("X5", "X6", "X7", "X8", "X3", "X4"):
                    continue
                test_format = "{:.3g}".format(dep_set.stat_error)
                _, text_n_digits = test_format.removeprefix('-').split('.', 1)
                n_digits = len(text_n_digits)
                number_format = "{{:.{}f}}".format(n_digits)
                print(sensor.decode().replace('_', '\\_'), "&", number_format.format(dep_set.magnitude),
                      "&", "+-" + number_format.format(dep_set.stat_error) + '\\textstatistic', "&",
                      "+-" + number_format.format(dep_set.systematic_general) + '\\textsystematic',
                      "&", "+-" + number_format.format(dep_set.systematic_dispersion) + '\\textdispersion', "\\\\", **print_args)

            print("\\bottomrule", **print_args)
            print("\\end{tabular}", **print_args)

        # sample_data = np.arange(0.1, 10, 0.05)
        # plt.plot(sample_data, 1 / sample_data)
        # plt.plot(sample_data, 1 / sample_data ** 2)
        # plt.plot(sample_data, 1 / np.sqrt(sample_data))
        # plt.plot(sample_data, 20 * np.exp(-sample_data))
        # plt.plot(sample_data, 20 * np.exp(sample_data))
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        # plt.plot(sample_data, 1 / sample_data)
        # plt.plot(sample_data, 1 / sample_data ** 2)
        # plt.plot(sample_data, 1 / np.sqrt(sample_data))
        # plt.plot(sample_data, 20 * np.exp(-sample_data))
        # plt.plot(sample_data, 20 * np.exp(sample_data))
        # plt.yscale('log')
        # plt.show()
        # plt.plot(sample_data, 1 / sample_data)
        # plt.plot(sample_data, 1 / sample_data ** 2)
        # plt.plot(sample_data, 1 / np.sqrt(sample_data))
        # plt.plot(sample_data, 20 * np.exp(-sample_data))
        # plt.plot(sample_data, 20 * np.exp(sample_data))
        # plt.show()


        # some additional conclusions:
        # * the quadratic LF pixels have for each depth a perfect proportionality to the implantation area but indeed both implantation depths have different slopes => TODO: estimate the slopes for the 'A' dependence independently for the two implantation depths and determine it's dependence on the implantation depth
        # * the rectangular LF pixel does not match the slope behaviour of the other two. => capacitance increase by dependendence on the implantation area exposed to the p-stop implantations? (This would need a parameter: 'perimeter * depth')
        # * the two HPK sensors are significant outliers as they have a much higher pixel capacitance with the same implantation depth (their pixel separation is much smaller)
        # * TODO: ask whether there are different resistivities used for the foundrys? -> answer: should not be the case!
        # * Behaviour of the implantation depth is difficult to say, as there are only to different implanation depths for same area sensors!
        # * all the planar sensors could be matched perfectly well by their pixel separation (the capacitance seems to decay strongly with the pixel separation, it would assume a exponential decay but polynomial/reciprocal one could not be excluded but in the later case the hpk pixel separation would be an significant issue; pixel separation seems to be different for the different implantation depths??; would also need to exclude R1 for a refined fit here as it strongly deviates from the behaviour of all the others)
        # * for the perimeter it is the same as for the pixel separation fits.
