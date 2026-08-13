# ----------------------------------------------------------
#  Copyright (c) 2026. SiLab, Institute of Physics, University of Bonn.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ----------------------------------------------------------
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ----------------------------------------------------------
import numpy as np
import tables as tb
from collections.abc import Iterable, Callable
from iminuit import Minuit
from iminuit.cost import LeastSquares
from iminuit.util import _detect_log_spacing, _smart_sampling
from matplotlib import pyplot as plt, rc_context
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from numpy._typing import NDArray, ArrayLike
from typing import Tuple, Union, Sequence, Optional, List

from capacitance_models import linear_model, reciprocal_model, extended_cap_model_5, inverted_reciprocal_model
from examples.plot_dependencies import read_rec_array_sorted, read_rec_array_sorted_where, chi2
from general_model import exponential_model
from pixcap65.plotting import GENERATE_THESIS_PLOTS
from pixcap65.utility.homogenize_plots import set_params, get_error_cycler


def quadratic_model(x, a, b, c):
    return a + b * x + c * x ** 2


def quadratic_model_grad(x, a, b, c):
    if isinstance(x, Iterable):
        return np.vstack([np.array([1, val, val ** 2]) for val in x]).T
    else:
        return np.array([1, x, x ** 2])


def get_polynomial_model(dof):
    def method(x, *args):
        return np.polyval(args, x)

    return method


def linear_model_grad(x, a, b):
    if isinstance(x, Iterable):
        return np.vstack([np.array([1, val]) for val in x]).T
    else:
        return np.array([1, x])


def exponential_model_grad(x, a, b):
    if isinstance(x, Iterable):
        return np.vstack([np.array([np.exp(b * val), a * b * np.exp(b * val)]) for val in x]).T
    else:
        return np.array([np.exp(b * x), a * b * np.exp(b * x)])

def get_new_axes():
    plt.close('all')
    fig, ax = plt.subplots()
    fig_full, ax_full = plt.subplots(3)
    assert isinstance(ax_full, np.ndarray)
    ax_std, ax_log, ax_loglog = ax_full.flatten()
    return fig, fig_full, ax, ax_std, ax_log, ax_loglog


def visualize(figures, axes, pdf1, pdf2, fitter: Minuit, title, parameter, *args, **kwargs):
    fig_full, fig_combi = figures
    _, _, ax_log, ax_loglog = axes
    for ax in axes:
        plt.sca(ax)
        fitter.visualize()
        ax.figure.suptitle(title)
        if len(args) > 0:
            ax.errorbar(*args, **kwargs)
        ax.set_ylabel("C / \\unit{{\\femto\\farad}}")
        ax.set_xlabel(parameter)

    ax_log.set_yscale('log')
    ax_loglog.set_yscale('log')
    ax_loglog.set_xscale('log')
    pdf1.savefig(fig_full, bbox_inches='tight')
    pdf2.savefig(fig_combi, bbox_inches='tight')


def get_advanced_visualizer(obj):
    def visualize_advanced_visualizer(
            args: ArrayLike, model_points: Union[int, Sequence[float]] = 0, **kwargs
    ) -> Tuple[Tuple[NDArray, NDArray, NDArray], Tuple[NDArray, NDArray]]:
        """
        Visualize data and model agreement (requires matplotlib).

        The visualization is drawn with matplotlib.pyplot into the current axes.

        Parameters
        ----------
        args : array-like
            Parameter values.

        model_points : int or array-like, optional
            How many points to use to draw the model. Default is 0, in this case
            an smart sampling algorithm selects the number of points. If array-like,
            it is interpreted as the point locations.
        """
        kwargs.setdefault('fmt', 'ok')

        if obj._ndim > 1:
            raise ValueError("visualize is not implemented for multi-dimensional data")

        x, y, ye = obj._masked.T
        plt.errorbar(x, y, ye, fmt=kwargs.setdefault('fmt', 'ok'))

        xmin = np.min(x)
        xmax = np.max(x)
        if isinstance(model_points, Iterable):
            xm = np.array(model_points)
            ym = obj.model(xm, *args)
        elif model_points > 0:
            # beware, x may not be sorted
            if _detect_log_spacing(x):
                xm = np.geomspace(xmin, xmax, model_points)
            else:
                xm = np.linspace(xmin, xmax, model_points)
            ym = obj.model(xm, *args)
        else:
            xm, ym = _smart_sampling(lambda x: obj.model(x, *args), xmin, xmax)
        plt.plot(xm, ym)
        return (x, y, ye), (xm, ym)

    return visualize_advanced_visualizer


def visualize_full(figures, axes, pdf1, pdf2, fitter: Iterable[Minuit], title, parameter, cost=None):
    fig_full, fig_combi = figures
    _, _, ax_log, ax_loglog = axes
    for ax in axes:
        plt.sca(ax)
        labels = []
        if cost is None:
            for k, fit_m in enumerate(fitter):
                assert isinstance(fit_m, Minuit)
                fit_m.visualize()
                labels.append("{} with p = {}%".format(k, (1 - chi2(fit_m.fmin.fval, fit_m.ndof)) * 100))
        else:
            for k, (fit_m, cost_m) in enumerate(zip(fitter, cost)):
                visu = get_advanced_visualizer(cost_m)
                fit_m.visualize(plot=visu, fmt="o")
                labels.append("{} with p = {}%".format(k, (1 - chi2(fit_m.fmin.fval, fit_m.ndof)) * 100))

        ax.figure.suptitle(title)
        ax.set_ylabel("C / \\unit{{\\femto\\farad}}")
        ax.set_xlabel(parameter)
        ax.legend(labels)

    ax_log.set_yscale('log')
    ax_loglog.set_yscale('log')
    ax_loglog.set_xscale('log')
    pdf1.savefig(fig_full, bbox_inches='tight')
    pdf2.savefig(fig_combi, bbox_inches='tight')


def detailed_fit(x, y, error, model, title, parameter, *args, model_gradient=None, **kwargs):
    if model_gradient is not None and model_gradient == "numeric":
        # use numdifftools for accurate numeric derivatives
        import numdifftools as nd
        def __gradient_method(x, *args):
            def __method(*args, **keys):
                return model_gradient(x, *args, **keys)

            return nd.Gradient(__method)(args, **kwargs)

        model_gradient = __gradient_method

    fig, fig_full, ax, ax_std, ax_log, ax_loglog = get_new_axes()
    cost = LeastSquares(x, y, error, model, grad=model_gradient)
    m = Minuit(cost, *args)
    m.migrad()
    m.hesse()
    try:
        m.minos()
    except:
        pass
    print(m.fmin)
    print(m.params)
    visualize((fig, fig_full), (ax, ax_std, ax_log, ax_loglog), pdf, pdf_log, m, title, parameter)

    return cost, m

if __name__ == "__main__":
    plt.close('all')

    set_params(latex=True,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                           r"stat,sys,sys-disp.}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                           r"retain-zero-uncertainty}", fig_height=8.26772, fig_width=11.69291,
               minor=True, fontsize=24, dpi=300)

    # need a fit for the areas with the same implantation depth
    # => thus could only use effectively R13 and E1 sensors
    # afterwards estimate the dependence of the A parameter on the depletion width.
    with tb.open_file('../conclude_summary.h5', mode='r') as h5_conclusion,\
        PdfPages("../Dependencies_fitted.pdf") as pdf,\
        PdfPages("../Dependencies_fitted_full.pdf") as pdf_log,\
            PdfPages("../Dependencies_fitted_full_full.pdf") as pdf_2:

        print("HANDLE THE AREA REDUCTION")
        full_properties = read_rec_array_sorted(h5_conclusion.root.SensorTypes, 'sensor')
        dnw_properties_raw = read_rec_array_sorted_where(h5_conclusion.root.SensorTypes,
                                                     """(implantation_depth >= {})""".format(4.),
                                                     'sensor')
        nw_properties = read_rec_array_sorted_where(h5_conclusion.root.SensorTypes,
                                                     """(implantation_depth < {})""".format(4.),
                                                     'sensor')

        full_properties = read_rec_array_sorted(h5_conclusion.root.SensorTypes, "sensor")

        # now create a selection tree
        dnw_selection = None
        for sensor in dnw_properties_raw.sensor:
            if dnw_selection is None:
                dnw_selection = """(sensor == {})""".format(sensor)
            else:
                dnw_selection += """ | (sensor == {})""".format(sensor)

        nw_selection = None
        full_nw_selection = None
        full_full_nw_selection = None
        hpk_nw_selection = None
        for sensor in nw_properties.sensor:
            if full_full_nw_selection is None:
                full_full_nw_selection = """(sensor == {})""".format(sensor)
            else:
                full_full_nw_selection += """ | (sensor == {})""".format(sensor)
            if sensor.decode().startswith("X"):
                if hpk_nw_selection is None:
                    hpk_nw_selection = """(sensor == {})""".format(sensor)
                else:
                    hpk_nw_selection += """ | (sensor == {})""".format(sensor)
                continue
            if sensor.decode() == 'R1' or sensor.decode().startswith('X'):
                pass
            else:
                if nw_selection is None:
                    nw_selection = """(sensor == {})""".format(sensor)
                else:
                    nw_selection += """ | (sensor == {})""".format(sensor)
                if hpk_nw_selection is None:
                    hpk_nw_selection = """(sensor == {})""".format(sensor)
                else:
                    hpk_nw_selection += """ | (sensor == {})""".format(sensor)
            if full_nw_selection is None:
                full_nw_selection = """(sensor == {})""".format(sensor)
            else:
                full_nw_selection += """ | (sensor == {})""".format(sensor)

        # we will also need to remove these from our list
        def exclude_properties(a):
            return a.decode() == "R1" or a.decode().startswith("X")

        mask = np.logical_not(np.array([exclude_properties(a) for a in nw_properties.sensor], dtype=bool))
        second_mask = np.logical_not(np.array([a.decode().startswith("X") for a in nw_properties.sensor], dtype=bool))
        hpk_mask = np.logical_not(np.array([a.decode() == 'R1' for a in nw_properties.sensor], dtype=bool))

        dnw_planar_slice = slice(0, -4)
        dnw_3d_slice = slice(len(dnw_properties_raw)-4, len(dnw_properties_raw))
        dnw_data_raw = read_rec_array_sorted_where(h5_conclusion.root.GeneralSummaryTable, dnw_selection, 'sensor')

        lf_selection_slice = slice(0, -6)

        dnw_properties = dnw_properties_raw[dnw_planar_slice]
        dnw_data = dnw_data_raw[dnw_planar_slice]
        d3_properties = dnw_properties_raw[dnw_3d_slice]
        d3_data = dnw_data_raw[dnw_3d_slice]
        nw_data = read_rec_array_sorted_where(h5_conclusion.root.GeneralSummaryTable, nw_selection, 'sensor')
        full_nw_data = read_rec_array_sorted_where(h5_conclusion.root.GeneralSummaryTable, full_nw_selection, 'sensor')
        hpk_nw_data = read_rec_array_sorted_where(h5_conclusion.root.GeneralSummaryTable, hpk_nw_selection, 'sensor')
        full_data = read_rec_array_sorted(h5_conclusion.root.GeneralSummaryTable, 'sensor')
        full_full_nw_data = read_rec_array_sorted_where(h5_conclusion.root.GeneralSummaryTable, full_full_nw_selection, 'sensor')

        def enhanced_full_visualisation(pdf, fitter, title, parameter, callback: Optional[Callable], cost=None, labels=None, extension_factor=1.05, **kwargs):
            fig, ax = plt.subplots()
            ax.set(xlabel=parameter, ylabel="$C$ / \\unit{{\\femto\\farad}}")
            from pixcap65.plotting import GENERATE_THESIS_PLOTS
            if not GENERATE_THESIS_PLOTS:
                ax.set_title(title)

            result = [] if callback is None else callback(ax)

            # need to asses the fit ranges!
            minimum_value = np.inf
            maximum_value = -np.inf
            maximum_y_value = -np.inf
            for line in result:
                assert isinstance(line, Line2D)
                original_data = np.asarray(line.get_xdata(orig=True), dtype=np.float64)
                original_y_data = np.asarray(line.get_ydata(orig=True), dtype=np.float64)
                orig_lower = np.min(original_data)
                orig_upper = np.max(original_data)
                orig_y_upper = np.max(original_y_data)
                if orig_lower < minimum_value:
                    minimum_value = orig_lower
                if orig_upper > maximum_value:
                    maximum_value = orig_upper
                if orig_y_upper > maximum_y_value:
                    maximum_y_value = orig_y_upper

            if cost is None:
                for k, fit_m in enumerate(fitter):
                    assert isinstance(fit_m, Minuit)
                    fit_m.visualize()
                    labels.append("{} with p = {}%".format(k, (1 - chi2(fit_m.fmin.fval, fit_m.ndof)) * 100))
            else:
                assert labels is not None
                for k, (fit_m, cost_m, label) in enumerate(zip(fitter, cost, labels)):
                    assert isinstance(cost_m, LeastSquares)
                    assert isinstance(fit_m, Minuit)
                    cost_x = np.asarray(cost_m.x, dtype=np.float64)
                    actual_minimum = np.min(cost_x)
                    actual_maximum = np.max(cost_x)
                    if actual_minimum > minimum_value:
                        actual_minimum = minimum_value

                    if actual_maximum < maximum_value:
                        actual_maximum = maximum_value

                    actual_minimum = actual_minimum / extension_factor if actual_minimum >= 0 else actual_minimum * extension_factor
                    actual_maximum = actual_maximum * extension_factor if actual_maximum >= 0 else actual_maximum / extension_factor

                    x_data = np.linspace(actual_minimum, actual_maximum * 1.2, 10000)
                    y_data = cost_m.model(x_data, *fit_m.values.to_dict().values())
                    ax.plot(x_data, y_data, label=label)
                    # ax.plot(x_data, y_data, label="{}\nwith p = {:.4g}\\%".format(label, (
                    #         1 - chi2(fit_m.fmin.fval, fit_m.ndof)) * 100))

            assert isinstance(pdf, PdfPages)
            ax.legend()
            ax.set_ylim(None, maximum_y_value * 1.05)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # could not be extracted to top-level as it has a dependence on local-variables
        def get_data_plotter(target, property, error=2):
            def method(axes) -> List[Line2D]:
                from matplotlib import rc_context
                with rc_context(rc={'axes.prop_cycle': get_error_cycler()}):
                    object_list = []
                    # dnw_x_data = dnw_properties[property][:-4]
                    # dnw_y_data = dnw_data[target].magnitude[:-4]
                    dnw_x_data = dnw_properties[property]
                    dnw_y_data = dnw_data[target].magnitude
                    match error:
                        case 0: dnw_y_error= dnw_data[target].stat_error
                        case 1:
                            dnw_y_error = dnw_data[target].systematic_general
                        case 2:
                            dnw_y_error = dnw_data[target].systematic_dispersion
                        case _:
                            raise RuntimeError

                    dnw_mask = np.isfinite(dnw_y_data)
                    container, _, _ = axes.errorbar(dnw_x_data[dnw_mask], dnw_y_data[dnw_mask], yerr=dnw_y_error[dnw_mask], fmt="x", label="DNW, planar", capsize=15, markersize=20)
                    object_list.append(container)

                    nw_x_data = nw_properties[property]
                    nw_y_data = full_full_nw_data[target].magnitude
                    match error:
                        case 0:
                            nw_y_error = full_full_nw_data[target].stat_error
                        case 1:
                            nw_y_error = full_full_nw_data[target].systematic_general
                        case 2:
                            nw_y_error = full_full_nw_data[target].systematic_dispersion
                        case _:
                            raise RuntimeError
                    nw_mask = np.isfinite(nw_y_data)
                    container, _, _ = axes.errorbar(nw_x_data[nw_mask], nw_y_data[nw_mask], yerr=nw_y_error[nw_mask], fmt="x", label="NW, planar", capsize=15, markersize=20)
                    assert isinstance(container, Line2D)
                    object_list.append(container)

                    d3_x_data = d3_properties[property]
                    d3_y_data = d3_data[target].magnitude
                    match error:
                        case 0:
                            d3_y_error = d3_data[target].stat_error
                        case 1:
                            d3_y_error = d3_data[target].systematic_general
                        case 2:
                            d3_y_error = d3_data[target].systematic_dispersion
                        case _:
                            raise RuntimeError

                    d3_mask = np.isfinite(d3_y_data)
                    container, _, _ = axes.errorbar(d3_x_data[d3_mask], d3_y_data[d3_mask],
                                                    yerr=d3_y_error[d3_mask], fmt="x", label="3D-Sensoren", capsize=15,
                                                    markersize=20)
                    object_list.append(container)
                    return object_list
            return method

        model_mapper = {
            "implantation_area": (linear_model, linear_model),
            "implantation_depth": (linear_model, linear_model),
            "pixel_separation_x": (exponential_model, exponential_model),
            "pixel_separation_y": (exponential_model, exponential_model),
            "Perimeter": (exponential_model, exponential_model),
        }
        # first run a test for the dnw with the new implementation
        dnw_cost, dnw_m = detailed_fit(dnw_properties.implantation_area,
                                       dnw_data.biased_capacitance.magnitude,
                                       dnw_data.biased_capacitance.systematic_dispersion,
                                       linear_model, "Area - DNW",
                                       "A / \\unit{{\\micro\\meter\\squared}}", 0, 1,
                                       model_gradient=linear_model_grad)

        nw_cost, nw_m = detailed_fit(nw_properties.implantation_area[mask],
                                       nw_data.biased_capacitance.magnitude,
                                       nw_data.biased_capacitance.systematic_dispersion,
                                       linear_model, "Area - NW",
                                       "A / \\unit{{\\micro\\meter\\squared}}", 0, 1,
                                       model_gradient=linear_model_grad)

        fig, fig_full, ax, ax_std, ax_log, ax_loglog = get_new_axes()
        visualize_full((fig, fig_full), (ax, ax_std, ax_log, ax_loglog), pdf, pdf_log, (dnw_m, nw_m), "Area",
                       "A / \\unit{{\\micro\\meter\\squared}}", cost=(dnw_cost, nw_cost))
        enhanced_full_visualisation(pdf_2, (dnw_m, nw_m), "Area", "A / \\unit{{\\micro\\meter\\squared}}", get_data_plotter("biased_capacitance", "implantation_area", 2), cost=(dnw_cost, nw_cost), labels=("DNW, planar", "NW, planar"))

        area_depth_poly_offset = np.polyfit([3, 5], [nw_m.values['a'], dnw_m.values['a']], 1)
        area_depth_poly_slope = np.polyfit([3, 5], [nw_m.values['b'], dnw_m.values['b']], 1)
        print(area_depth_poly_offset)
        print(area_depth_poly_slope)

        print("SKIP THE IMPLANTATION DEPTH REDUCTION")
        fig, ax = plt.subplots()
        with rc_context(rc={'axes.prop_cycle': get_error_cycler()}):
            ax.errorbar(full_properties.implantation_depth[dnw_planar_slice], full_data.biased_capacitance.magnitude[dnw_planar_slice], yerr=full_data.biased_capacitance.systematic_dispersion[dnw_planar_slice], fmt='x', capsize=3.0)
            ax.set_xlabel("$d$ / \\unit{{\\micro\\meter}}")
            ax.set_ylabel("$C$ / \\unit{{\\femto\\farad}}")
            if not GENERATE_THESIS_PLOTS:
                ax.set_title("Dependence on the implantation depth.")
            pdf_2.savefig(fig, bbox_inches="tight")
        plt.close(fig)


        print("INVESTIGATE THE PERIMETER DEPENDENCE!")
        print("For R1 we would expect", nw_m.values['a'] + nw_m.values['b'] * 8 * 81, "But measured", 42.568,
              (nw_m.values['a'] + nw_m.values['b'] * 8 * 81) / 42.568)
        print("The equivalent quadratic pixel length is", np.sqrt(8*81), "which leads to a perimeter of", 4 * np.sqrt(
            8 * 81), 4 * np.sqrt(8 * 81) / (8 + 8 + 81 + 81))
        r1_p_a = 2 * (8 + 81)
        r1_p_b = 4 * np.sqrt(8 * 81)

        # this leads to the conclusion that the specific shape leads to a more than doubled perimneter and an almost twice as large capacitance => this could be exponential increase.
        # we should also compare the parameters of X1 and X2 sensors in the same way!
        # but if it is exponentially scaling: does it apply only to our exponential offset (which seemingly depends on the implanation depth) or must be rescale the whole model used so far?)
        # on the other hand the capacitance's could be estimated quite well by a exponential without distinguishing any thing.
        # So we should perform this one again, distinguishing by the implantation depth.

        dnw_cost_p, dnw_m_p = detailed_fit(dnw_properties.Perimeter, dnw_data.biased_capacitance.magnitude,
                                           dnw_data.biased_capacitance.systematic_dispersion,
                                           exponential_model, "Perimeter - DNW",
                                        "U / \\unit{{\\micro\\meter}}", 15, 0,
                                           model_gradient=exponential_model_grad)
        plt.errorbar(dnw_properties.Perimeter, dnw_data.biased_capacitance.magnitude, label='all', fmt='+')

        dnw_cost_p_2, dnw_m_p_2 = detailed_fit(dnw_properties.Perimeter, dnw_data.biased_capacitance.magnitude,
                                           dnw_data.biased_capacitance.systematic_dispersion,
                                           quadratic_model, "Perimeter - DNW - Quad",
                                           "U / \\unit{{\\micro\\meter}}", 15, 0, 0,
                                           model_gradient=quadratic_model_grad)

        fig, fig_full, ax, ax_std, ax_log, ax_loglog = get_new_axes()
        dnw_cost_p_2 = LeastSquares(dnw_properties.Perimeter, dnw_data.biased_capacitance.magnitude,
                                  dnw_data.biased_capacitance.systematic_dispersion,
                                  quadratic_model)

        nw_cost_p, nw_m_p = detailed_fit(nw_properties.Perimeter[second_mask], full_nw_data.biased_capacitance.magnitude,
                                           full_nw_data.biased_capacitance.systematic_dispersion,
                                           exponential_model, "Perimeter - NW",
                                           "U / \\unit{{\\micro\\meter}}", 10, 0,
                                           model_gradient=exponential_model_grad)

        nw_cost_p_2, nw_m_p_2 = detailed_fit(nw_properties.Perimeter[second_mask],
                                               full_nw_data.biased_capacitance.magnitude,
                                               full_nw_data.biased_capacitance.systematic_dispersion,
                                               quadratic_model, "Perimeter - NW - Quad",
                                               "U / \\unit{{\\micro\\meter}}", 10, 0, 0,
                                               model_gradient=quadratic_model_grad)

        nw_cost_p_3, nw_m_p_3 = detailed_fit(nw_properties.Perimeter[second_mask],
                                             full_nw_data.biased_capacitance.magnitude,
                                             full_nw_data.biased_capacitance.systematic_dispersion,
                                             get_polynomial_model(4), "Perimeter - NW - 4",
                                             "U / \\unit{{\\micro\\meter}}", 0, 0, 2.93e-3, -0.27, 20.4)

        fig, fig_full, ax, ax_std, ax_log, ax_loglog = get_new_axes()
        visualize_full((fig, fig_full), (ax, ax_std, ax_log, ax_loglog), pdf, pdf_log,
                       (dnw_m_p, dnw_m_p_2, nw_m_p, nw_m_p_2, nw_m_p_3), "Perimeter",
                       "U / \\unit{{\\micro\\meter}}",
                       cost=(dnw_cost_p, dnw_cost_p_2, nw_cost_p, nw_cost_p_2, nw_cost_p_3),)

        enhanced_full_visualisation(pdf_2,
                                    (dnw_m_p, dnw_m_p_2, nw_m_p, nw_m_p_2, nw_m_p_3), "Perimeter",
                                    "U / \\unit{{\\micro\\meter}}",
                                    get_data_plotter("biased_capacitance", "Perimeter", 2),
                                    cost=(dnw_cost_p, dnw_cost_p_2, nw_cost_p, nw_cost_p_2,),
                                    labels=["DNW, planar, exponentiell", "DNW, planar, quadratisch", "NW, planar, exponentiell", "NW, planar, quadratisch",])

        print("parameter depth dependence for the exponential model!")
        print(np.polyfit([3, 5], [nw_m_p.values['a'], dnw_m_p.values['a']], 1))
        print(np.polyfit([3, 5], [nw_m_p.values['b'], dnw_m_p.values['b']], 1))
        print(np.exp(nw_m_p.values['b'] * (r1_p_b - r1_p_a)))
        print("parameter depth dependence for the quadratic model model!")
        print(np.polyfit([3, 5], [nw_m_p_2.values['a'], dnw_m_p_2.values['a']], 1))
        print(np.polyfit([3, 5], [nw_m_p_2.values['b'], dnw_m_p_2.values['b']], 1))
        print(np.polyfit([3, 5], [nw_m_p_2.values['c'], dnw_m_p_2.values['c']], 1))

        # taking the HPK sensors for the perimeter directly into account leads to strong deviations => exclude them for now from the full data set.
        # the constant term has here the same slope as the constant offset for the area investigation (but they have quite different starting values)
        # the variation of the (decay) coefficient with the implantation depth is quite low!
        # maybe we could apply these here soley on the constant term of the linear model describing the dependence on the implantation area.
        # but the variation of the decay coefficient seems to be significant enough to be accounted for!
        # maybe we must directly account also for the difference in the pixel separation and implantation area

        # this model is able to fully explain the residual for the r1 with linear fit of the implantation area.


        print("INVESTIGATE THE DEPENDENCE ON THE PIXEL SEPARATION.")
        # before investigating the dependency on the pixel separation: What capacitances are to expect for the hpk sensor
        print(nw_m.values['a'] + nw_m.values['b'] * 50 * 50)
        print(nw_m.values['a'] * np.exp(nw_m_p.values['b'] * 4 * 50))
        print(nw_m_p.values['a'] * np.exp(nw_m_p.values['b'] * 4 * 50) + nw_m.values['b'] * 50 * 50)
        print((nw_m_p.values['a'] + nw_m.values['b'] * 50 * 50) * np.exp(nw_m_p.values['b'] * 4 * 50))
        # the expected value is in the range (72-75) fF.
        # but then the last two would require an influence reducing the capacity again (significant reduction required).
        # or the perimeter model needs to be corrected to account only for combination from perimeter and area
        # it seems to be quite inprobable that the pixel separation would lead to a decrease in capacitance
        # or use it such that the perimeter does only have a effect if there is a deviation from the quadratic structure?
        # p / 4*sqrt(A) for example.


        dnw_cost_s, dnw_m_s = detailed_fit(dnw_properties.pixel_separation_x,
                                           dnw_data.biased_capacitance.magnitude,
                                           dnw_data.biased_capacitance.systematic_dispersion,
                                           reciprocal_model, "Separation - DNW",
                                           "$\\Delta$ / \\unit{{\\micro\\meter}}", 1, 0)
        dnw_cost_s_q, dnw_m_s_q = detailed_fit(dnw_properties.pixel_separation_x,
                                           dnw_data.biased_capacitance.magnitude,
                                           dnw_data.biased_capacitance.systematic_dispersion,
                                           exponential_model, "Separation - DNW - EXP",
                                           "$\\Delta$ / \\unit{{\\micro\\meter}}", 1, 0,
                                               model_gradient=exponential_model_grad)
        if True:
            nw_cost_s, nw_m_s = detailed_fit(nw_properties.pixel_separation_x[mask],
                                               nw_data.biased_capacitance.magnitude,
                                               nw_data.biased_capacitance.systematic_dispersion,
                                               reciprocal_model, "Separation - NW",
                                               "$\\Delta$ / \\unit{{\\micro\\meter}}", 1, 0)
            nw_cost_s_q, nw_m_s_q = detailed_fit(nw_properties.pixel_separation_x[mask],
                                                   nw_data.biased_capacitance.magnitude,
                                                   nw_data.biased_capacitance.systematic_dispersion,
                                                   exponential_model, "Separation - NW - EXP",
                                                   "$\\Delta$ / \\unit{{\\micro\\meter}}", 1, 0,
                                                   model_gradient=exponential_model_grad)
        else:
            nw_cost_s, nw_m_s = detailed_fit(nw_properties.pixel_separation_x[hpk_mask],
                                             hpk_nw_data.biased_capacitance.magnitude,
                                             hpk_nw_data.biased_capacitance.systematic_dispersion,
                                             reciprocal_model, "Separation - NW",
                                             "$\\Delta$ / \\unit{{\\micro\\meter}}", 1, 0)
            nw_cost_s_q, nw_m_s_q = detailed_fit(nw_properties.pixel_separation_x[hpk_mask],
                                                 hpk_nw_data.biased_capacitance.magnitude,
                                                 hpk_nw_data.biased_capacitance.systematic_dispersion,
                                                 exponential_model, "Separation - NW - EXP",
                                                 "$\\Delta$ / \\unit{{\\micro\\meter}}", 1, 0,
                                                 model_gradient=exponential_model_grad)
        fig, fig_full, ax, ax_std, ax_log, ax_loglog = get_new_axes()
        visualize_full((fig, fig_full), (ax, ax_std, ax_log, ax_loglog), pdf, pdf_log, (dnw_m_s, nw_m_s, dnw_m_s_q, nw_m_s_q), "Separation",
                       "$\\Delta$ / \\unit{{\\micro\\meter}}", cost=(dnw_cost_s, nw_cost_s, dnw_cost_s_q, nw_cost_s_q))
        enhanced_full_visualisation(pdf_2,
                                    (dnw_m_s, nw_m_s, dnw_m_s_q, nw_m_s_q), "Separation",
                                    "$\\Delta$ / \\unit{{\\micro\\meter}}",
                                    get_data_plotter("biased_capacitance", "pixel_separation_x", 2),
                                    cost=(dnw_cost_s, nw_cost_s, dnw_cost_s_q, nw_cost_s_q),
                                    labels=("DNW, planar, reziprok", "DNW, planar, exponentiell", "NW, planar, reziprok", "NW, planar, exponentiell"))

        # from point of the cost function it seems like reciprocal models are better suited
        exp_dnw_s = 19.87
        exp_nw_s = 29.09

        print(1- chi2(exp_dnw_s - dnw_m_s.fmin.fval, 1))
        print(1 - chi2(exp_nw_s - nw_m_s.fmin.fval, 1))
        # by p-value test reject the exponential model.
        # but this will introduce significant divergences for the HPK sensors.
        # and for the hpk sensors be need to lower the pixel capacitance after all.
        # the actual coefficients would make this contribution quite large for the HPK sensors.
        #
        print("parameter depth dependence for the reciprocal model on the separation")
        print(np.polyfit([3, 5], [nw_m_s.values['a'], dnw_m_s.values['a']], 1))
        print(np.polyfit([3, 5], [nw_m_s.values['b'], dnw_m_s.values['b']], 1))

        # also what happends if we also account for R1 in this investigation!
        def active_model(x):
            A, W, p, sep_x, sep_y = x
            # it may be the case that we need to estimate the primary parameters (within the brackets) by a different
            # formula. This will hold in particular if the first 10 residuals are much to large.
            return (area_depth_poly_offset[1] + area_depth_poly_offset[0] * W + area_depth_poly_slope[1] * A + area_depth_poly_slope[0] * A * W) * np.exp(nw_m_p.values['b'] * (p - 4 * np.sqrt(A)))

        # calculate some residue for our model
        x_ref_data = (full_properties.implantation_area, full_properties.implantation_depth, full_properties.Perimeter, full_properties.pixel_separation_x, full_properties.pixel_separation_y)
        y_ref_data = full_data.biased_capacitance.magnitude
        predictions = active_model(x_ref_data)
        residuals = predictions - y_ref_data
        absolute_residuals = np.abs(residuals)
        print(absolute_residuals[lf_selection_slice])
        normed_residuals = residuals / full_data.biased_capacitance.stat_error
        general_cost = np.sum(np.asarray(normed_residuals[lf_selection_slice]) ** 2)
        print("Cost information to share.")
        print(general_cost)
        # should not be used anymore!
        print(1 - chi2(general_cost, 4))

        # investigate the inter-pix capacitance by the same means.
        # first we need to fetch a sample
        print("INVESTIGATE PERIMETER DEPENDENCE FOR INTER-PIX")
        inter_pix_mask = np.isfinite(full_data.biased_inter_capacitance.magnitude)
        nw_inter_pix_mask = np.isfinite(full_nw_data.biased_inter_capacitance.magnitude)
        dnw_inter_pix_mask = np.isfinite(dnw_data.biased_inter_capacitance.magnitude)
        dnw_inter_cost_p, dnw_inter_m_p = detailed_fit(dnw_properties.Perimeter[dnw_inter_pix_mask],
                                                     dnw_data.biased_inter_capacitance.magnitude[dnw_inter_pix_mask],
                                                     dnw_data.biased_inter_capacitance.stat_error[dnw_inter_pix_mask],
                                                     quadratic_model, "Inter - Perimeter - DNW - LIN",
                                                     "$U$ / \\unit{{\\micro\\meter}}",
                                                     1, 0, 0, model_gradient=quadratic_model_grad)

        dnw_inter_cost_p_2, dnw_inter_m_p_2 = detailed_fit(dnw_properties.Perimeter[dnw_inter_pix_mask],
                                                         dnw_data.biased_inter_capacitance.magnitude[dnw_inter_pix_mask],
                                                         dnw_data.biased_inter_capacitance.stat_error[dnw_inter_pix_mask],
                                                         exponential_model, "Inter - Perimeter - DNW - EXP",
                                                         "$U$ / \\unit{{\\micro\\meter}}",
                                                         1, 0, model_gradient=exponential_model_grad)
        print("DNW hypothesis test")
        print(1 - chi2(dnw_inter_m_p_2.fmin.fval - dnw_inter_m_p.fmin.fval, 1))

        nw_inter_cost_p, nw_inter_m_p = detailed_fit(nw_properties.Perimeter[second_mask][nw_inter_pix_mask],
                                                     full_nw_data.biased_inter_capacitance.magnitude[nw_inter_pix_mask],
                                                     full_nw_data.biased_inter_capacitance.stat_error[nw_inter_pix_mask],
                                                     quadratic_model, "Inter - Perimeter - NW - LIN", "$U$ / \\unit{{\\micro\\meter}}",
                                                     1, 0, 0, model_gradient=quadratic_model_grad)

        nw_inter_cost_p_2, nw_inter_m_p_2 = detailed_fit(nw_properties.Perimeter[second_mask][nw_inter_pix_mask],
                                                         full_nw_data.biased_inter_capacitance.magnitude[nw_inter_pix_mask],
                                                         full_nw_data.biased_inter_capacitance.stat_error[nw_inter_pix_mask],
                                                         exponential_model, "Inter - Perimeter - NW - EXP",
                                               "$U$ / \\unit{{\\micro\\meter}}",
                                                         1, 0, model_gradient=exponential_model_grad)
        print("NW hypothesis test")
        print(1 - chi2(nw_inter_m_p_2.fmin.fval - nw_inter_m_p.fmin.fval, 1))

        fig, fig_full, ax, ax_std, ax_log, ax_loglog = get_new_axes()
        visualize_full((fig, fig_full), (ax, ax_std, ax_log, ax_loglog), pdf, pdf_log,
                       (nw_inter_m_p, nw_inter_m_p_2, dnw_inter_m_p, dnw_inter_m_p_2), "Inter - Perimeter",
                       "$U$ / \\unit{{\\micro\\meter}}", cost=(nw_inter_cost_p, nw_inter_cost_p_2, dnw_inter_cost_p,
                                                               dnw_inter_cost_p_2),)

        enhanced_full_visualisation(pdf_2,
                                    (nw_inter_m_p, nw_inter_m_p_2, dnw_inter_m_p, dnw_inter_m_p_2), "Inter - Perimeter",
                                    "$U$ / \\unit{{\\micro\\meter}}",
                                    get_data_plotter("biased_inter_capacitance", "Perimeter", 0),
                                    cost=(nw_inter_cost_p, nw_inter_cost_p_2, dnw_inter_cost_p,
                                          dnw_inter_cost_p_2),
                                    labels=("NW, planar, quadratisch", "NW, planar, exponentiell", "DNW, planar, quadratisch", "DNW, planar, exponentiell"),)

        print("INTER CAP PERIMETETER QUADRATIC DEPTH DEPENDENCE")
        print(np.polyfit([3, 5],[nw_inter_m_p.values['a'], dnw_inter_m_p.values['a']],1))

        print(np.polyfit([3, 5],[nw_inter_m_p.values['b'], dnw_inter_m_p.values['b']],1))

        print(np.polyfit([3, 5],[nw_inter_m_p.values['c'], dnw_inter_m_p.values['c']],1))

        print("INVESTIGATE SEPARATION DEPENDENCE FOR INTER-PIX")
        dnw_inter_cost_s, dnw_inter_m_s = detailed_fit(dnw_properties.pixel_separation_x[dnw_inter_pix_mask],
                                                     dnw_data.biased_inter_capacitance.magnitude[dnw_inter_pix_mask],
                                                     dnw_data.biased_inter_capacitance.stat_error[dnw_inter_pix_mask],
                                                     reciprocal_model, "Inter - Separation - DNW",
                                                     "$\\Delta$ / \\unit{{\\micro\\meter}}",
                                                     1, 0)

        dnw_inter_cost_s_2, dnw_inter_m_s_2 = detailed_fit(
            dnw_properties.pixel_separation_x[dnw_inter_pix_mask],
            dnw_data.biased_inter_capacitance.magnitude[dnw_inter_pix_mask],
            dnw_data.biased_inter_capacitance.stat_error[dnw_inter_pix_mask],
            exponential_model, "Inter - Separation - DNW - EXP",
            "$\\Delta$ / \\unit{{\\micro\\meter}}",
            1, 0, model_gradient=exponential_model_grad)
        nw_inter_cost_s, nw_inter_m_s = detailed_fit(nw_properties.pixel_separation_x[second_mask][nw_inter_pix_mask],
                                                     full_nw_data.biased_inter_capacitance.magnitude[nw_inter_pix_mask],
                                                     full_nw_data.biased_inter_capacitance.stat_error[nw_inter_pix_mask],
                                                     reciprocal_model, "Inter - Separation - NW",
                                               "$\\Delta$ / \\unit{{\\micro\\meter}}",
                                                     1, 0)

        nw_inter_cost_s_2, nw_inter_m_s_2 = detailed_fit(nw_properties.pixel_separation_x[second_mask][nw_inter_pix_mask],
                                                         full_nw_data.biased_inter_capacitance.magnitude[nw_inter_pix_mask],
                                                         full_nw_data.biased_inter_capacitance.stat_error[nw_inter_pix_mask],
                                                         exponential_model, "Inter - Separation - NW - EXP",
                                                   "$\\Delta$ / \\unit{{\\micro\\meter}}",
                                                         1, 0, model_gradient=exponential_model_grad)
        fig, fig_full, ax, ax_std, ax_log, ax_loglog = get_new_axes()
        visualize_full((fig, fig_full), (ax, ax_std, ax_log, ax_loglog), pdf, pdf_log,
                       (nw_inter_m_s, nw_inter_m_s_2, dnw_inter_m_s, dnw_inter_m_s_2), "Inter - Separation",
                       "$\\Delta$ / \\unit{{\\micro\\meter}}", cost=(nw_inter_cost_s, nw_inter_cost_s_2,
                                                                     dnw_inter_cost_s, dnw_inter_cost_s_2))

        enhanced_full_visualisation(pdf_2,
                                    (nw_inter_m_s, nw_inter_m_s_2, dnw_inter_m_s, dnw_inter_m_s_2),
                                    "Inter - Separation",
                                    "$\\Delta$ / \\unit{{\\micro\\meter}}",
                                    get_data_plotter("biased_inter_capacitance", "pixel_separation_x", 0),
                                    cost=(nw_inter_cost_s, nw_inter_cost_s_2, dnw_inter_cost_s, dnw_inter_cost_s_2),
                                    labels=("NW, planar, reziprok", "NW, planar, exponentiell", "DNW, planar, reziprok", "DNW, planar, exponentiell"))


        fig, ax = plt.subplots()
        get_data_plotter("biased_inter_capacitance", "implantation_area", 0)(ax)
        ax.set_xlabel("$A$ / \\unit{{\\micro\\meter\\squared}}")
        ax.set_ylabel("$C_\\text{{inter}}$ / \\unit{{\\femto\\farad}}")
        if not GENERATE_THESIS_PLOTS:
            ax.set_title("Inter - Area")
        ax.legend()
        pdf_2.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        with plt.rc_context(rc={'axes.prop_cycle': get_error_cycler()}):
            fig, ax = plt.subplots()
            ax.errorbar(full_properties.implantation_depth[dnw_planar_slice], full_data.biased_inter_capacitance.magnitude[dnw_planar_slice],
                        yerr=full_data.biased_inter_capacitance.stat_error[dnw_planar_slice], fmt='x', capsize=3.0)
            ax.set_xlabel("$d$ / \\unit{{\\micro\\meter}}")
            ax.set_ylabel("$C_\\text{{inter}}$ / \\unit{{\\femto\\farad}}")
            if not GENERATE_THESIS_PLOTS:
                ax.set_title("Inter:Dependence on the implantation depth.")
            pdf_2.savefig(fig, bbox_inches="tight")
            print("THE PROPERTY CYCLE IS!")
            print(plt.rcParams["axes.prop_cycle"])
        plt.close(fig)


        # the first results from detailed inter-pix fits imply a dependence on the implantation depth invisible in the
        # plots for the thesis by the effect of the hpk, fbk and sintef sensors;
        # deeper implants seems to imply higher capacitance!

        def active_inter_pix_model(xy,a0, a1, a2, a3, a4, a5):
            p, d = xy
            return quadratic_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5))

        def active_inter_pix_grad(xy,a0, a1, a2, a3, a4, a5):
            p, d = xy
            first = quadratic_model_grad(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5))
            second_temp = np.array([
                linear_model_grad(d, a0, a1),
                linear_model_grad(d, a2, a3),
                linear_model_grad(d, a4, a5)
            ])
            if len(second_temp.shape) == 2:
                second_temp = second_temp.reshape(second_temp.shape[0], second_temp.shape[1], 1)
            second = np.array([second_temp[:, :, i].flatten() for i in range(second_temp.shape[2])]).T
            return np.array([first[k // 2] * sec for k, sec in enumerate(second)])



        slc = slice(0, -5, 1)
        active_inter_pix_cost = LeastSquares((full_properties.Perimeter[inter_pix_mask][slc], full_properties.implantation_depth[inter_pix_mask][slc]),
                                             full_data.biased_inter_capacitance.magnitude[inter_pix_mask][slc],
                                             full_data.biased_inter_capacitance.stat_error[inter_pix_mask][slc],
                                             active_inter_pix_model)

        active_inter_pix_minuit = Minuit(active_inter_pix_cost, 0.06, 0.11, -0.0017, -0.00029, 1.57e-5, 3.9e-6)
        active_inter_pix_minuit.migrad()
        active_inter_pix_minuit.hesse()
        try:
            active_inter_pix_minuit.minos()
        except:
            pass
        print("RESULTS FROM THE ACTIVE INTER PIXEL CAP FIT")
        print(active_inter_pix_minuit.fmin)
        print(active_inter_pix_minuit.params)
        print(1 - chi2(active_inter_pix_minuit.fmin.fval, active_inter_pix_minuit.ndof))

        active_pix_dependents = (full_properties.implantation_area[lf_selection_slice], full_properties.implantation_depth[lf_selection_slice],
                                 full_properties.Perimeter[lf_selection_slice], full_properties.pixel_separation_x[lf_selection_slice],
                                 full_properties.pixel_separation_y[lf_selection_slice])
        active_pix_cost = LeastSquares(
            active_pix_dependents,
            full_data.biased_capacitance.magnitude[lf_selection_slice],
            full_data.biased_capacitance.systematic_dispersion[lf_selection_slice],
            extended_cap_model_5)

        active_pix_minuit = Minuit(active_pix_cost, 0.06, 0.11, -0.0017, -0.00029, 1.57e-5, 3.9e-6, -0.36e3, 140)
        active_pix_minuit.migrad()
        active_pix_minuit.hesse()
        try:
            active_pix_minuit.minos()
        except:
            pass
        print("RESULTS FROM THE ACTIVE PIXEL CAP FIT")
        print(active_pix_minuit.fmin)
        print(active_pix_minuit.params)
        print(1 - chi2(active_pix_minuit.fmin.fval, active_pix_minuit.ndof))
        active_pix_predictions = np.array([extended_cap_model_5(np.asarray(active_pix_dependents)[:, i], *active_pix_minuit.values.to_dict().values()) for i in range(len(active_pix_dependents[0]))])
        full_prediction_coordinates = np.vstack([full_properties.implantation_area,
                                                 full_properties.implantation_depth,
                                                 full_properties.Perimeter,
                                                 full_properties.pixel_separation_x,
                                                 full_properties.pixel_separation_y]).T
        full_total_predictions = np.array([extended_cap_model_5(coord,
                                                         *active_pix_minuit.values.to_dict().values()) for coord in full_prediction_coordinates])

        active_pix_residues = active_pix_predictions - full_data.biased_capacitance.magnitude[lf_selection_slice]
        print(active_pix_residues)
        print(active_pix_predictions)
        print(active_pix_minuit.ndof)
        fig, ax = plt.subplots()
        ax.errorbar(full_properties.pixel_separation_x[lf_selection_slice], active_pix_residues, fmt='.k')
        fig.savefig("residues_total_plot.png")
        plt.close(fig)


        # need to get an additonal values
        plt.close('all')
        inter_area = full_properties.implantation_depth[inter_pix_mask] * full_properties.Perimeter[inter_pix_mask]
        print(inter_area)
        fig, ax = plt.subplots()
        slc = slice(0, -5, 1)
        ax.errorbar(inter_area[slc], full_data.biased_inter_capacitance.magnitude[inter_pix_mask][slc], yerr=
        full_data.biased_inter_capacitance.stat_error[inter_pix_mask][slc], fmt='ok')
        for record, sensor in zip(full_data[inter_pix_mask][slc], full_properties[inter_pix_mask][slc]):
            xy = (sensor.implantation_depth * sensor.Perimeter, record.biased_inter_capacitance.magnitude)
            ax.annotate(record.sensor.decode(),xy)
        fig.savefig("inter_pix_area.png")
        fig, ax = plt.subplots()
        slc = slice(0, -5, 1)
        inter_area_divide = full_properties.implantation_depth[inter_pix_mask] * full_properties.Perimeter[inter_pix_mask] / \
                            full_properties.pixel_separation_x[inter_pix_mask]
        ax.errorbar(inter_area_divide[slc], full_data.biased_inter_capacitance.magnitude[inter_pix_mask][slc], yerr=
        full_data.biased_inter_capacitance.stat_error[inter_pix_mask][slc], fmt='ok')
        for record, sensor in zip(full_data[inter_pix_mask][slc], full_properties[inter_pix_mask][slc]):
            xy = (sensor.implantation_depth * sensor.Perimeter / \
                  sensor.pixel_separation_x, record.biased_inter_capacitance.magnitude)
            ax.annotate(record.sensor.decode(), xy)
        fig.savefig("inter_pix_area_divide_dist.png")
        fig, ax = plt.subplots()
        slc = slice(0, -5, 1)
        inter_area_divide = full_properties.Perimeter[inter_pix_mask] / full_properties.pixel_separation_x[inter_pix_mask]
        ax.errorbar(inter_area_divide[slc], full_data.biased_inter_capacitance.magnitude[inter_pix_mask][slc], yerr=
        full_data.biased_inter_capacitance.stat_error[inter_pix_mask][slc], fmt='ok')
        for record, sensor in zip(full_data[inter_pix_mask][slc], full_properties[inter_pix_mask][slc]):
            xy = (sensor.Perimeter / sensor.pixel_separation_x, record.biased_inter_capacitance.magnitude)
            ax.annotate(record.sensor.decode(), xy)
        fig.savefig("inter_pix_area_divide_dist_2.png")
        fig, ax = plt.subplots()
        slc = slice(0, -6, 1)
        inter_area_divide = full_properties.implantation_depth * full_properties.Perimeter / full_properties.pixel_separation_x
        ax.errorbar(inter_area_divide[slc], full_data.biased_capacitance.magnitude[slc], yerr=full_data.biased_inter_capacitance.stat_error[slc], fmt='ok')
        for record, sensor in zip(full_data[slc], full_properties[slc]):
            xy = (sensor.implantation_depth * sensor.Perimeter / sensor.pixel_separation_x, record.biased_capacitance.magnitude)
            ax.annotate(record.sensor.decode(), xy)
        fig.savefig("total_pix_area_divide_dist.png")

        print("Full prediction")
        print(full_total_predictions)
        print(full_data.biased_capacitance.magnitude)
        print(full_data.sensor)
        print(np.abs(full_total_predictions - full_data.biased_capacitance.magnitude))
        print(np.sum((np.abs(full_total_predictions[dnw_planar_slice] - full_data.biased_capacitance.magnitude[dnw_planar_slice]) / full_data.biased_capacitance.systematic_dispersion[dnw_planar_slice])**2))
        print(*active_pix_minuit.values.to_dict().values())
        print(active_pix_minuit.values)

        print("HPK PREDICTION FOR SEPARATION")
        print(reciprocal_model(nw_properties.pixel_separation_x[hpk_mask], *nw_m_s.values.to_dict().values()))
        print(nw_properties.sensor[hpk_mask])
        print(hpk_nw_data.biased_capacitance.magnitude)
        from scipy.optimize import newton
        from numpy.random import default_rng
        rng = default_rng(42)
        def root_model(x, ref, model, *args):
            return model(x, *args) - ref
        print("result from optimisation")
        number_bootstraps = 5000
        print(newton(root_model, 2.75, args=[hpk_nw_data.biased_capacitance.magnitude[-2], reciprocal_model, *nw_m_s.values.to_dict().values()], full_output=True))
        print(newton(root_model, 2.75, args=[hpk_nw_data.biased_capacitance.magnitude[-1], reciprocal_model, *nw_m_s.values.to_dict().values()], full_output=True))

        cap_parameter_values = nw_m_s.values.to_dict().values()

        # FIXME: bootstrapiing ignores the error on the parameters!
        def perform_bootstrap(rng, idx, data: np.recarray, init):
            guess = newton(root_model, init, args=[
                data.biased_capacitance.magnitude[idx],
                reciprocal_model,
                *cap_parameter_values
            ])

            bootstrap_data = rng.normal(loc=data.biased_capacitance.magnitude[idx], scale=data.biased_capacitance.stat_error[idx], size=number_bootstraps)
            bootstrap_result = np.array([newton(root_model, guess,
                                                args=[cap, reciprocal_model,
                                                      *nw_m_s.values.to_dict().values()]) for cap in bootstrap_data])
            return np.mean(bootstrap_result), np.std(bootstrap_result)


        x1_rev_separation = perform_bootstrap(rng, -2, hpk_nw_data, 2.75)
        x2_rev_separation = perform_bootstrap(rng, -1, hpk_nw_data, 2.75)
        print(x1_rev_separation)
        print(x2_rev_separation)
        print(50 - 4 - 2 * x1_rev_separation[0], np.abs(-2 * x1_rev_separation[1]))
        print(50 - 4 - 2 * x2_rev_separation[0], np.abs(-2 * x2_rev_separation[1]))
        from jacobi import propagate
        rev_separation_simple, rev_separation_simple_cov = propagate(lambda x: inverted_reciprocal_model(x, *cap_parameter_values),
                                                                     hpk_nw_data.biased_capacitance.magnitude,
                                                                     np.diag(hpk_nw_data.biased_capacitance.stat_error) ** 2)
        rev_separation_simple_err = np.diag(rev_separation_simple_cov) ** 0.5
        print(50 - 4 - 2 * rev_separation_simple[-2], np.abs(-2 * rev_separation_simple_err[-2]))
        print(50 - 4 - 2 * rev_separation_simple[-1], np.abs(-2 * rev_separation_simple_err[-1]))
        full_parameters = np.full(3, np.nan)
        full_parameters[0] = hpk_nw_data.biased_capacitance.magnitude[-2]
        full_parameters[1:] = [*cap_parameter_values]
        full_covariance = np.full((3, 3), 0)
        full_covariance[0, 0] = hpk_nw_data.biased_capacitance.stat_error[-2] ** 2
        full_covariance[1:, 1:] = nw_m_s.covariance[:]
        rev_separation_full, rev_separation_full_cov = propagate(
            lambda x: inverted_reciprocal_model(*x),
            full_parameters,
            full_covariance)
        rev_separation_full_err = rev_separation_full_cov ** 0.5
        print(50 - 4 - 2 * rev_separation_full, np.abs(-2 * rev_separation_full_err))
        # print(50 - 4 - 2 * rev_separation_full[-1], np.abs(-2 * rev_separation_full_err[-1]))
        # TODO: How to perform this calculation for multiple at once?
        # TODO: How to use bootstrapping to achieve the same result?
        guess = newton(root_model, 2.75, args=[
            hpk_nw_data.biased_capacitance.magnitude[-2],
            reciprocal_model,
            *cap_parameter_values
        ])

        bootstrap_data = rng.normal(loc=hpk_nw_data.biased_capacitance.magnitude[-2],
                                    scale=hpk_nw_data.biased_capacitance.stat_error[-2], size=number_bootstraps)
        print(type(cap_parameter_values))
        print(len(cap_parameter_values))
        print(np.shape(np.array([*cap_parameter_values], dtype=np.float64)))
        print(cap_parameter_values)
        bootstrap_parameter = rng.multivariate_normal(mean=np.array([*cap_parameter_values], dtype=np.float64), cov=nw_m_s.covariance, size=number_bootstraps)
        # print("The bootstrap parameters first.")
        # for parameters in bootstrap_parameter:
        #     print(parameters)
        # print("Finished bootstrapping")
        bootstrap_result = np.array([newton(root_model, guess,
                                            args=[cap, reciprocal_model,
                                                  *parameters]) for (cap, parameters) in zip(bootstrap_data, bootstrap_parameter)])
        rev_separation_full_boot, rev_separation_full_err_boot = np.mean(bootstrap_result), np.std(bootstrap_result)
        print(50 - 4 - 2 * rev_separation_full_boot, np.abs(-2 * rev_separation_full_err_boot))

        rev_separation, rev_separation_cov = propagate(lambda p: inverted_reciprocal_model(
            hpk_nw_data.biased_capacitance.magnitude, *p),
                                                       nw_m_s.values,
                                                       nw_m_s.covariance)
        print("Checkout error propagation")
        print(inverted_reciprocal_model(hpk_nw_data.biased_capacitance.magnitude, *cap_parameter_values))
        print(rev_separation, rev_separation_cov, np.diag(rev_separation_cov) ** 0.5)
        print(50 - 4 - 2 * rev_separation, -2 * np.diag(rev_separation_cov) ** 0.5)

        # at last calculate the predicitons for all the sensors:
