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
"""
This file contains some utilities needed for the analysis and the plotting.
In particular it should help with defining constants of values used quite often.
"""

import logging
import numpy as np
import tables as tb
try:
    # noinspection PyCompatibility
    from collections.abc import Sequence, Mapping
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Sequence
    from typing import Mapping
from tables import File
from tables.group import RootGroup
from typing import Union, Any

from pixcap65.utility.utils_2 import walk_to_node, UNITS_ATTRIBUTE_KEY, prevent_group_mix_up

logger = logging.getLogger(__name__)

# some utility constants to simplify expressions for the analysis
# region Analysis constants
GENERAL_TRANSFORMATION_MATRIX = np.array(
    [[1.e-12, 1.e-6, 1.e3, 1.e-6], [1.e-6, 1, 1.e9, 1], [1.e3, 1.e9, 1.e18, 1.e9], [1.e-6, 1, 1.e9, 1]])
GENERAL_TRANSFORMATION_MATRIX_TAU = np.array(
    [[1.e-12, 1.e-6, 1.e3, 1.e-6, 1e-6], [1.e-6, 1, 1.e9, 1, 1], [1.e3, 1.e9, 1.e18, 1.e9, 1e9], [1.e-6, 1, 1.e9, 1, 1],
     [1e-6, 1, 1.e9, 1, 1]])
ANALYSIS_GROUP_NAME = "analysis"
ANALYSIS_CORRECTED_GROUP_NAME = "analysis_correction"
TABLES_ARRAY_TYPE = Union[np.ndarray, tb.CArray]
TABLES_TABLE_TYPE = Union[tb.Table, Mapping[str, TABLES_ARRAY_TYPE]]
TABLES_LEAF_TYPE = Union[tb.Leaf, tb.Table, tb.Array]
TABLES_PART_LEAF_TYPE = Union[tb.Table, tb.Array, tb.CArray]
TABLES_LEAF_COMPAT_TYPE = Union[tb.Table, tb.Array, np.ndarray, tb.CArray]  # no Leaf as it must be an impl.
GENERAL_PIXCAP_SHAPE = (40, 41)
COVARIANCE_PIXCAP_SHAPE = (40, 41, 4, 4)
FULL_MODEL_LABEL = "I_\\text{{full}}"
SIMPLE_MODEL_LABEL = "I_\\text{{approximation}}"
FULL_MODEL_EXPRESSION = "\\frac{{{u0}\\cdot{c}\\cdot{freq}+{i}}}{{1+{r}\\cdot{c}\\cdot{freq}}}"
SIMPLE_MODEL_EXPRESSION = "{u0}\\cdot{c\\cdot{freq}+{i}}"
FULL_MODEL_PARAMETER_DICT = {"c": r"C", "r": "R", "i": r"I_\\text{{Leakage}}", "u0": "U_{{0}}", "freq": "\\nu"}
SIMPLE_MODEL_PARAMETER_DICT = {"c": r"C", "i": r"I", "u0": r"U_{0}", "freq": r"\nu"}
GLOBAL_FILTERS = tb.Filters(complevel=5, complib='blosc', shuffle=False, fletcher32=False)
FARAD_CONVERSION_FACTOR = 1e-6
CURRENT_CONVERSION_FACTOR = 1e9
PARASITIC_SUBTRACTION = "parasitic_subtraction"
# endregion

# some utility constants to simplify expressions for plotting handling.
# region Plotting constants
HIST_BIAS_MEAS_UNIT = "V"
HIST_LEAK_CURRENT_UNIT = "nA"
HIST_CAP_UNIT = "F"
HIST_CURRENT_MEAS_UNIT = "A"


# endregion: Plotting constants

# region Utility functions
def transform_covariance(cov):
    """
    transform_covariance

    Transforms the returned covariance such that the units match the specified ones.
    For the more advanced fits also the contributions of the fixed reference voltage parameter are removed
    from the covariance matrix.

    :param cov: covariance matrix to be transformed.
    :return: transformed covariance matrix.
    """
    cov = np.asarray(cov)

    # missing compatibility layer here!
    match cov.shape:
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
        case (5, 5):
            assert cov.shape == (5, 5)
            # here it is necessary to reduce the parts from the covariance of u0 to
            mask = np.array(
                [[True, True, True, False, True], [True, True, True, False, True], [True, True, True, False, True],
                 [False, False, False, False, False], [True, True, True, False, True]])
            return cov[mask].reshape((4, 4)) * GENERAL_TRANSFORMATION_MATRIX_TAU[mask].reshape((4, 4))
        case _:
            raise ValueError(
                "The dimension of the covariance matrix does not fit to any of the fitting functions and their "
                "parameters.")


def str_join(delimiter: str, *args) -> str:
    """
    str_join

    Modified function to join multiple strings separated by the specified delimiter.
    This function was necessary as the builtin implementation does not support variable args.
    :param delimiter: delimiter to use between the different strings when combining them.
    :param args: strings to be combined.
    :return: combined string.
    """
    # But what 'to do' if one of the var args is a list or in general an iterable of strings?
    return delimiter.join(args)


def check_leaf_unit(leaf: TABLES_LEAF_TYPE, unit: str) -> np.ndarray:
    """
    check_leaf_unit(leaf, unit)

    Verify that the table or array has a units attribute and the unit is the one expected.
    As last step return the data structure requested.

    This may not work for tables with multiple units (one per column)
    :param leaf: Data structure to be verified and extracted.
    :param unit: Expected unit for the data structure.
    :return: array_like of the data structures contents.
    """
    assert isinstance(leaf, TABLES_PART_LEAF_TYPE)
    if UNITS_ATTRIBUTE_KEY not in leaf.attrs or leaf.attrs[UNITS_ATTRIBUTE_KEY] != unit:
        print(f"The units key is {leaf.attrs[UNITS_ATTRIBUTE_KEY]}")
        raise AssertionError
    result = leaf[:]
    assert isinstance(result, np.ndarray)
    return result


# noinspection PyUnusedLocal
def handle_fitter_stub(fit_object, apply_contour, x_label, y_label, title, pdf, contours_title=None, fig_title=None):
    """
    handle_fitter_stub

    @author Dominik Fischer
    last update: 2026-08-24

    For the non-linear fitting frameworks are some further options available to directly plot the results or perform
    profiling.
    This is just a stub function, necessary to define the signature for all related functions.
    :param fit_object: object of the already performed fit.
    :param apply_contour: boolean, whether to apply contour profiling around the found optimum.
    :param x_label: label of the x-axis.
    :param y_label: label of the y-axis.
    :param title: title of the plot.
    :param pdf: PdfPages object to save the fit plots to.
    :param contours_title: title of the contour plot.
    :param fig_title: If provided, the super-title of the fit plot figures.
    """
    # This is just a stub method for simplifying the fit plotting
    pass


def handle_fitter_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf, contours_title=None,
                                   fig_title=None):
    """
    handle_fitter_advanced_options

    @author Dominik Fischer
    last update: 2026-08-24

    For the non-linear fitting frameworks are some further options available to directly plot the results or perform
    profiling.
    This is only a wrapper implementation as it will delegate immediately to the implementations specific for a
    particular fitting framework like `iminuit` or `kafe2`.
    :param fit_object: object of the already performed fit.
    :param apply_contour: boolean, whether to apply contour profiling around the found optimum.
    :param x_label: label of the x-axis.
    :param y_label: label of the y-axis.
    :param title: title of the plot.
    :param pdf: PdfPages object to save the fit plots to.
    :param contours_title: title of the contour plot.
    :param fig_title: If provided, the super-title of the fit plot figures.
    """
    from kafe2 import FitBase
    from iminuit import Minuit
    if isinstance(fit_object, FitBase):
        handle_kafe2_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf, contours_title,
                                      fig_title)
    elif isinstance(fit_object, Minuit):
        handle_minuit_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf, contours_title,
                                       fig_title)
    else:
        raise TypeError("Fitting object is from an unexpected type '{}'.".format(type(fit_object)))


def handle_kafe2_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf, contours_title=None,
                                  fit_title=None):
    """
    handle_kafe2_advanced_options

    For the non-linear fitting frameworks are some further options available to directly plot the results or perform
    profiling.
    This implementation is only for use with the kafe2 framework.

    :param fit_object: object of the already performed fit.
    :param apply_contour: boolean, whether to apply contour profiling around the found optimum.
    :param x_label: label of the x-axis.
    :param y_label: label of the y-axis.
    :param title: title of the plot.
    :param pdf: PdfPages object to save the fit plots to.
    :param contours_title: title of the contour plot.
    :param fit_title: If provided, the super-title of the fit plot figures.
    """
    from kafe2 import Plot, FitBase
    from matplotlib import pyplot as plt
    assert isinstance(fit_object, FitBase)
    fit_plot = Plot(fit_object)
    fit_plot.x_label = x_label
    fit_plot.y_label = y_label
    fit_plot.plot(residual=True)
    conv_invest = investigate_fit_convergence(fit_object)
    for (fig, axes) in zip(fit_plot.figures, fit_plot.axes):
        for ax_k, ax in enumerate(axes.values()):
            if ax_k >= 1:
                print("We are now at iteration ", ax_k)
            elif fit_title is not None:
                fig.suptitle(fit_title, fontsize=20)
            ax.set_title(title)
            ax.text(0, 0.9, f"Fit with cost={conv_invest['x']:.4n} and \np={conv_invest['p']:.4n}",
                    transform=ax.transAxes)
            break
        pdf.savefig(fig, bbox_inches='tight')
    for fig in fit_plot.figures:
        plt.close(fig)

    if apply_contour:
        from kafe2 import ContoursProfiler
        from matplotlib.figure import Figure
        cpf = ContoursProfiler(fit_object)
        cpf_figure = cpf.plot_profiles_contours_matrix()
        assert isinstance(cpf_figure, Figure)
        cpf_figure.axes[0].set_title(contours_title)
        pdf.savefig(cpf_figure, bbox_inches='tight')
        plt.close(cpf_figure)


# noinspection PyProtectedMember
def extract_iminuit_cost_object(fit):
    """
    Get the cost functions object from a `iminuit` framework `Minuit` instance.
    :param fit: `Minuit` instances from which to fetch the applied cost function.
    :return: cost function object.
    """
    from iminuit import Minuit
    assert isinstance(fit, Minuit)
    return fit.fcn._fcn


def handle_minuit_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf,
                                   contours_title=None, fig_title=None):
    """
    handle_iminuit_advanced_options

    For the non-linear fitting frameworks are some further options available to directly plot the results or perform
    profiling.
    This implementation is only for use with the iminuit framework.

    :param fit_object: object of the already performed fit.
    :param apply_contour: boolean, whether to apply contour profiling around the found optimum.
    :param x_label: label of the x-axis.
    :param y_label: label of the y-axis.
    :param title: title of the plot.
    :param pdf: PdfPages object to save the fit plots to.
    :param contours_title: title of the contour plot.
    :param fig_title: If provided, the super-title of the fit plot figures.
    """
    from matplotlib import pyplot as plt
    from iminuit import Minuit
    # noinspection PyProtectedMember
    from iminuit.minuit import _cl_to_errordef
    assert isinstance(fit_object, Minuit)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fit_object.visualize()
    conv_result = investigate_fit_convergence(fit_object)
    model_parameters = ""
    for key, value in fit_object.values.to_dict().items():
        model_parameters += f"{key} = \\num{{{value:.4f}}}\n"
    ax.legend(["model", "data"],
              title=f"{model_parameters}\nGoF={conv_result['x']:.4n} / ndf={conv_result['ndf']:n} = {conv_result["xn"]:.4n}\np={conv_result['p']:.4n}",
              frameon=False)

    # for error bands we must perform something similar
    if hasattr(extract_iminuit_cost_object(fit_object), "model"):
        try:
            from jacobi import propagate
            # noinspection PyProtectedMember
            x, _, _ = extract_iminuit_cost_object(fit_object)._masked.T
            # noinspection PyTypeChecker
            y, y_cov = propagate(lambda p: extract_iminuit_cost_object(fit_object).model(x, p), fit_object.values,
                                 fit_object.covariance)
            y_err_prop = np.diag(y_cov) ** 0.5
            plt.fill_between(x, y - y_err_prop, y + y_err_prop, facecolor="C1", alpha=0.5)
        except:
            pass

    if fig_title is not None:
        fig.suptitle(fig_title)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    cls = [0.68, 0.9, 0.99]

    if apply_contour and fit_object.valid:
        # will need 'to do' it on our selves
        pars = [p for p in fit_object.parameters if not fit_object.fixed[p]]
        n_par = len(pars)
        fig_size = None
        fig, ax = plt.subplots(
            n_par,
            n_par,
            figsize=fig_size,
            constrained_layout=True,
            squeeze=False,
        )

        for i, par1 in enumerate(pars):
            plt.sca(ax[i, i])
            f_max = 0
            for k, cl in enumerate(cls):
                f = _cl_to_errordef(cl, 1, 0.68)
                f_max = max(f_max, f)
                plt.axhline(f, color=f"C{k}")
            fit_object.draw_mnprofile(par1, subtract_min=True, bound=3)
            ax[i, i].set_ylabel("$\\Delta - 2\\log \\mathcal{{L}}$")

            for j in range(i):
                par2 = pars[j]
                plt.sca(ax[i, j])
                plt.plot(fit_object.values[par2], fit_object.values[par1], "+", color="k")
                try:
                    fit_object.draw_mncontour(par2, par1, cl=cls)
                    # contour_set = fit_object.draw_mncontour(par2, par1, cl=cls)
                    # from matplotlib.contour import ContourSet
                    # assert isinstance(contour_set, ContourSet)
                    # print("Levels")
                    # print(type(contour_set.levels))
                    #
                    # print("Segments")
                    # print(type(contour_set.allsegs))
                    # second_set = ContourSet(ax[i, j], levels=contour_set.levels, allsegs=contour_set.allsegs, allkinds=contour_set.allkinds, filled=True)
                    # ax[i, j].clabel(second_set)
                    # ax[i, j].colorbar(second_set)
                except:
                    fit_object.draw_contour(par2, par1)

                ax[j, i].set_visible(False)

        fig.suptitle(contours_title)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def investigate_fit_convergence(fitter):
    """
    investigate_fit_convergence(fitter)

    @author: Dominik Fischer
    last update: 2026-08-24

    Utility function to check convergence of the fit and calculate some GoF quantities.
    This function must not be used if neither the package `kafe2` nor the package `iminuit` is installed.

    In anyway the chi^2 estimator are fetched from the corresponding fitting object and written back into a dedicated
    mapping.
    If possible also the p-value for statistical hypothesis tests is calculated for the result of the fit and put
    into the mapping.
    If it is not possible to calculate the p-value, it will be set to -1.


    :param fitter: fitting object from a suitable framework.
    :return: convergence mapping of d.o.f. of the fit, chi2, reduced chi2 and p-Value.
    """
    try:
        from kafe2 import FitBase
    except ImportError:
        class FitBase(object):
            """
            Stub class in case kafe2 is not installed.
            """

            def __getattr__(self, name):
                def method(*args, **kwargs):
                    logger.warning(
                        "The package `kafe2` is not installed or at least the `kafe2.FitBase` class could not be imported. So no call to `%s` is possible.",
                        name)

                return method
    try:
        from iminuit import Minuit
    except ImportError:
        class Minuit(object):
            """
            Stub class in case iminuit is not installed.
            """

            def __getattr__(self, name):
                def method(*args, **kwargs):
                    logger.warning(
                        "The package `iminuit` is not installed or at least the `iminuit.Minuit` class could not be imported. So no call to `%s` is possible.",
                        name)

                return method
    if isinstance(fitter, Minuit):
        logger.debug("Fitter minimum found %f;%f and the reduced chi2 is %f",
                     fitter.fval, fitter.fmin.fval, fitter.fmin.reduced_chi2)
        cost_value = fitter.fval
        ndf = fitter.ndof
    elif isinstance(fitter, FitBase):
        cost_value = fitter.goodness_of_fit
        ndf = fitter.ndf
        logger.debug(fitter.chi2_probability)
    else:
        raise TypeError("Fitter must be either a Minuit or XYFit object.")

    if ndf == 0:
        ndf = -1
    regular_cost = cost_value / ndf if cost_value is not None else -1

    try:
        from scipy.stats.distributions import chi2
        p_value = 1 - chi2.cdf(regular_cost, df=ndf)
    except ImportError:
        p_value = -1
    convergence = dict(x=cost_value, xn=regular_cost, ndf=ndf, p=p_value)
    return convergence


def get_base_group(base_path, in_file_h5: File) -> Union[tb.Group, RootGroup]:
    """
    get_base_group

    Utility function to get the base_group from a hdf file and a hierarchy path for further use in analysis, plotting
    or measurements.

    :param base_path: path of the requested group in the hdf file.
    :param in_file_h5: hdf file with data for measurement and analysis
    :return: group from the hdf file.
    """
    if base_path is None:
        base_group = in_file_h5.root
    else:
        try:
            base_group, _ = walk_to_node(in_file_h5.root, base_path, verify_create=True)
            assert isinstance(base_group, tb.Group)
        except:
            print(in_file_h5)
            raise
    return base_group


def handle_analysis_mix_up(group):
    """
    handle_analysis_mix_up

    Utility function to prevent mix-up of different analysis runs on the same measurement data.
    It will remove any old analysis group from the file to prevent any confusion.

    :param group: group for which analysis mix-up has to be prevented.
    """
    prevent_group_mix_up(group, "analysis")


def extract_parasitic_capacitance(hist):
    """
    extract_parasitic_capacitance

    Utility function to extract the parasitic capacitance correction value from the data set.

    :param hist: capacitance data set to extract the parasitic capacitance correction value from.
    :return: correction value.
    """
    if PARASITIC_SUBTRACTION in hist.attrs:
        return hist.attrs[PARASITIC_SUBTRACTION]
    return 0.0


def get_analysis_group(base_group, **kwargs):
    """
    get_analysis_group

    @author Dominik Fischer
    @date 2026-08-11

    Get the correct analysis group for the given occasion.
    Distinguish between analysis and corrected analysis groups and provides the correct one.

    :param base_group: hdf files group where to look for the analysis groups.
    :keyword use_corrected: boolean, whether to use the corrected capacitance's for plotting.
    :keyword apply_correction: boolean, whether to use the corrected capacitance's for plotting/extraction.
    :return: analysis group from the hdf file.
    """
    if kwargs.get('use_corrected', False) or kwargs.get('apply_correction', False):
        return base_group.analysis_correction

    return base_group.analysis


class HandleFitterStubClass:
    """
    Handler/Wrapper class to implement the post-processing of further options to perform additional investigations on
    completed fits for different fitting frameworks like `kafe2` or `iminuit` following the same signature in the
    analysis code.

    For the non-linear fitting frameworks are some further options available to directly plot the results or perform
    profiling.
    But this is just a stub class (it is not abstract as it was necessary that this class instantiated).
    """

    def __call__(self, fit_object, apply_contour, x_label, y_label, title, pdf, contours_title=None, fig_title=None):
        """
        For the non-linear fitting frameworks are some further options available to directly plot the results or perform
        profiling.
        :param fit_object: object of the already performed fit.
        :param apply_contour: boolean, whether to apply contour profiling around the found optimum.
        :param x_label: label of the x-axis.
        :param y_label: label of the y-axis.
        :param title: title of the plot.
        :param pdf: PdfPages object to save the fit plots to.
        :param contours_title: title of the contour plot.
        :param fig_title: If provided, the super-title of the fit plot figures.
        """
        handle_fitter_stub(fit_object, apply_contour, x_label, y_label, title, pdf, contours_title, fig_title)


class HandleFitterGeneral(HandleFitterStubClass):
    """
    Explicit implementation of :class:`pixcap65.analysis_util.utility.HandleFitterStubClass` for a general case.
    It will decide appropropriately between `kafe2` and `iminuit`.
    """

    def __call__(self, fit_object, apply_contour, x_label, y_label, title, pdf, contours_title=None, fig_title=None):
        handle_fitter_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf, contours_title,
                                       fig_title)


class HandleFitterKafe2(HandleFitterStubClass):
    """
    Explicit implementation of :class:`pixcap65.analysis_util.utility.HandleFitterStubClass` for handling fits
    performed by the `kafe2` framework.
    """

    def __call__(self, fit_object, apply_contour, x_label, y_label, title, pdf, contours_title=None, fig_title=None):
        handle_kafe2_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf, contours_title,
                                      fig_title)


class HandleFitterMinuit(HandleFitterStubClass):
    """
    Explicit implementation of :class:`pixcap65.analysis_util.utility.HandleFitterStubClass` for handling fits
    performed by the `iminuit` framework.
    """

    def __call__(self, fit_object, apply_contour, x_label, y_label, title, pdf, contours_title=None, fig_title=None):
        handle_minuit_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf, contours_title,
                                       fig_title)


# endregion: Utility functions

# region Analysis Data structures
default_analysis_keyword_arguments = {
    "cap_name": "HistCap",
    "cap_title": "Capacitance Histogram",
    "cap_err_name": "HistCapErr",
    "cap_err_title": "Capacitance Error Histogram",
    "leak_name": "HistLeak",
    "leak_title": "Leakage Current Histogram",
    "leak_error_name": "HistLeakErr",
    "leak_error_title": "Leakage Current Error Histogram",
    "resistor_name": "HistRes",
    "resistor_title": "On-Resistance Histogram",
    "resistor_error_name": "HistResErr",
    "resistor_error_title": "On-Resistance Error Histogram",
    "cov_name": "HistFitCov",
    "cov_title": 'Fit Covariance Matrix'
}


class DepletionWidthData(tb.IsDescription):
    """
    :class:`pytables.IsDescription` data structure to store the results of the modelling of the depletion width profile
    for individual pixel on a sensor in a :class:`pytables.Table` object.

    Class variable are to be understand as fields/entries of the table.
    :cvar col: column coordinate of the investigated pixel.
    :cvar row: row coordinate of the investigated pixel.
    :cvar V: ??
    :cvar NAD: effective combined acceptor-donator density (defined like the reduced mass of the two)
    :cvar dep: estimated full depletion voltage
    :cvar sat: estimated saturation value for the full depletion width.

    For further general information see :py:class:`pytables.IsDescription`.
    """
    col = tb.Int64Col(pos=0)
    row = tb.Int64Col(pos=1)
    V = tb.Float32Col(pos=2)
    NAD = tb.Float32Col(pos=3)
    dep = tb.Float32Col(pos=4)
    sat = tb.Float32Col(pos=5)


class CVDistributionSimpleData(tb.IsDescription):
    """
    :py:class:`pytables.IsDescription` data structure to store the (averaged) fitting parameters of a sensor for a
    particular applied bias voltage in a :class:`pytables.Table` object.
    The bias voltage entry may also indicate a special kind of measurement.

    Class variable are to be understand as fields/entries of the table.
    :cvar bias: bias voltage used for this measurement (or the kind of the measurement).
    :cvar n_pixel: number of pixels which have contributed to these average values.
    :cvar capacitance: measured/averaged capacitance of the sensor at the given bias voltage.
    :cvar cap_err: uncertainty on the mean capacitance parameter of the sensor at the given bias voltage.
    :cvar cap_std: spread of the capacitance over the sensor at the given bias voltage.
    :cvar cap_std_err: uncertainty of the capacitance spread parameter at the given bias voltage.
    :cvar r_on: averaged/measured on-resistance of the measurement circuits.
    :cvar r_on_err: uncertainty of the on-resistance's mean parameter.
    :cvar r_on_std: spread of the on-resistance over the available measurement circuits/channels.
    :cvar r_on_std_err: uncertainty of the on-resistance spread parameter over the available measurement
    circuits/channels.
    :cvar cap_systematic_general: general systematic uncertainty of the capacitance measurement. Given by the spread of
    parasitic capacitance and the way the investigation/analysis is performed.
    :cvar cap_systematic_dispersion: systematic uncertainty of the capacitance measurement given by the dispersion of
    the parasitic capacitance of the measurement circuit between different pixcap65 chips.

    For further general information see :py:class:`pytables.IsDescription`.
    """
    bias = tb.Float64Col(pos=0)
    n_pixel = tb.Int64Col(pos=1)
    capacitance = tb.Float64Col(pos=2)
    cap_err = tb.Float64Col(pos=3)
    cap_std = tb.Float64Col(pos=4)
    cap_std_err = tb.Float64Col(pos=5)
    r_on = tb.Float64Col(pos=6)
    r_on_err = tb.Float64Col(pos=7)
    r_on_std = tb.Float64Col(pos=8)
    r_on_std_err = tb.Float64Col(pos=9)
    cap_systematic_general = tb.Float64Col(pos=10)
    cap_systematic_dispersion = tb.Float64Col(pos=11)


class CVDistributionData(tb.IsDescription):
    """
    :py:class:`pytables.IsDescription` data structure to store the (averaged) fitting parameters of a sensor for a
    particular applied bias voltage in a :class:`pytables.Table` object.
    The bias voltage entry may also indicate a special kind of measurement.
    In Addition to the implementation by :py:class:`pixcap65.analysis_util.utility.CVDistributionSimpleData`, this
    data structure also provides information about the corrected capacitance and the corresponding systematic
    uncertainties.

    Class variable are to be understand as fields/entries of the table.
    :cvar bias: bias voltage used for this measurement (or the kind of the measurement).
    :cvar n_pixel: number of pixels which have contributed to these average values.
    :cvar capacitance: measured/averaged capacitance of the sensor at the given bias voltage.
    :cvar cap_err: uncertainty on the mean capacitance parameter of the sensor at the given bias voltage.
    :cvar cap_std: spread of the capacitance over the sensor at the given bias voltage.
    :cvar cap_std_err: uncertainty of the capacitance spread parameter at the given bias voltage.
    :cvar r_on: averaged/measured on-resistance of the measurement circuits.
    :cvar r_on_err: uncertainty of the on-resistance's mean parameter.
    :cvar r_on_std: spread of the on-resistance over the available measurement circuits/channels.
    :cvar r_on_std_err: uncertainty of the on-resistance spread parameter over the available measurement
    circuits/channels.
    :cvar cap_corrected: measured/averaged capacitance of the sensor at the given bias voltage (based on the corrected
    capacitances).
    :cvar cap_corrected_err: spread of the capacitance over the sensor at the given bias voltage. (based on the corrected
    capacitances).
    :cvar cap_parasitic: value of the averaged parasitic capacitance used here.
    :cvar cap_systematic_error: general systematic uncertainty of the capacitance measurement. Given by the spread of
    parasitic capacitance and the way the investigation/analysis is performed.
    :cvar cap_corrected_est_error: uncertainty on the mean capacitance parameter of the sensor at the given
    bias voltage. (based on the corrected capacitances).
    :cvar cap_corrected_std_error: uncertainty of the capacitance spread parameter at the given bias voltage.
    (based on the corrected capacitances).
    :cvar cap_systematic_dispersion: systematic uncertainty of the capacitance measurement given by the dispersion of
    the parasitic capacitance of the measurement circuit between different pixcap65 chips.

    For further general information see :py:class:`pytables.IsDescription`.
    """
    bias = tb.Float64Col(pos=0)
    n_pixel = tb.Int64Col(pos=1)
    capacitance = tb.Float64Col(pos=2)
    cap_err = tb.Float64Col(pos=3)
    cap_std = tb.Float64Col(pos=4)
    cap_std_err = tb.Float64Col(pos=5)
    r_on = tb.Float64Col(pos=6)
    r_on_err = tb.Float64Col(pos=7)
    r_on_std = tb.Float64Col(pos=8)
    r_on_std_err = tb.Float64Col(pos=9)
    cap_corrected = tb.Float64Col(pos=10)
    cap_corrected_err = tb.Float64Col(pos=11)
    cap_parasitic = tb.Float64Col(pos=12)
    cap_systematic_error = tb.Float64Col(pos=13)
    cap_corrected_est_error = tb.Float64Col(pos=14)
    cap_corrected_std_error = tb.Float64Col(pos=15)
    cap_systematic_dispersion = tb.Float64Col(pos=16)


depletion_atomic_type = np.dtype([
    ("Ubi", np.float64),
    ("Ubi_error", np.float64),
    ("a", np.float64),
    ("a_error", np.float64),
    ("b", np.float64),
    ("b_error", np.float64),
    ("c", np.float64),
    ("c_error", np.float64),
    ("d", np.float64),
    ("d_error", np.float64),
])


class DepletionData(tb.IsDescription):
    """
    :py:class:`pytables.IsDescription` data structure to store the results of the depletion analysis in a
    :py:class:`pytables.Table` object.
    This will include in particular the results for the depletion voltage and their statistical and systematic
    uncertainties.

    As all fits are performed individually for corrected and uncorrected capacities, this data structure contains
    separate fields/entries for results based on uncorrected and corrected capacities.
    So not only the depletion voltage field is present twice but also the results for the fit parameters and their errors
    and the covariance matrix.

    Class variables are here to understand as the fields of the table.
    :cvar Ubi: estimation of the depletion voltage based on the uncorrected capacities.
    :cvar Ubi_error: (statistical) uncertainty of the depletion voltage based on the uncorrected capacities.
    :cvar a: slope parameter of the linear fit to the high voltage limit to estimate the depletion voltage.
    :cvar a_error: uncertainty of the slope parameter of the linear fit to the high voltage limit to estimate the
    depletion voltage.
    :cvar b: offset parameter of the linear fit to the high voltage limit to estimate the depletion voltage.
    :cvar b_error: uncertainty of the offset parameter of the linear fit to the high voltage limit to estimate the
    depletion voltage.
    :cvar c: slope parameter of the linear fit to the low voltage limit to estimate the depletion voltage.
    :cvar c_error: uncertainty of the slope parameter of the linear fit to the high voltage limit to estimate the
    depletion voltage.
    :cvar d: offset parameter of the linear fit to the low voltage limit to estimate the depletion voltage.
    :cvar d_error: uncertainty of the offset parameter of the linear fit to the low voltage limit to estimate the
    depletion voltage.
    :cvar Ubi_corrected: estimation of the depletion voltage based on the corrected capacities.
    :cvar Ubi_corrected_error: (statistical) uncertainty of the depletion voltage based on the corrected capacities.
    :cvar a_corrected: slope parameter of the linear fit to the high voltage limit to estimate the depletion voltage.
    :cvar a_corrected_error: uncertainty of the slope parameter of the linear fit to the high voltage limit to estimate the
    depletion voltage.
    :cvar b_corrected: offset parameter of the linear fit to the high voltage limit to estimate the depletion voltage.
    :cvar b_corrected_error: uncertainty of the offset parameter of the linear fit to the high voltage limit to estimate the
    depletion voltage.
    :cvar c_corrected: slope parameter of the linear fit to the low voltage limit to estimate the depletion voltage.
    :cvar c_corrected_error: uncertainty of the slope parameter of the linear fit to the high voltage limit to estimate the
    depletion voltage.
    :cvar d_corrected: offset parameter of the linear fit to the low voltage limit to estimate the depletion voltage.
    :cvar d_corrected_error: uncertainty of the offset parameter of the linear fit to the low voltage limit to estimate the
    depletion voltage.
    :cvar Ubi_systematic: systematic error of the (uncorrected) depletion voltage by the spread of the capacitances over
    the sensor and by the systematic effects of the analysis procedure like e.g. the fit ranges.
    :cvar Ubi_systematic_corrected: systematic error of the (corrected) depletion voltage by the spread of the capacitances over
    the sensor and by the systematic effects of the analysis procedure like e.g. the fit ranges.
    :cvar Ubi_systematic_dispersion: systematic error of the (uncorrected) depletion voltage by the dispersion of the
    parasitic capacitances' of the measurement circuit between multiple pixcap65 chips.
    :cvar Ubi_systematic_dispersion_corrected_error: systematic error of the (corrected) depletion voltage by tthe dispersion of the
    parasitic capacitances' of the measurement circuit between multiple pixcap65 chips.
    :cvar rho: estimation for the substrates specific resistivity
    :cvar rho_error: (statistical) uncertainty on the estimation of the substrates specific resistivity.
    :cvar rho_corrected: estimation for the substrates specific resistivity
    :cvar rho_corrected_error: (statistical) uncertainty on the estimation of the substrates specific resistivity.
    :cvar first_covariance: covariance matrix of the first fit region (high voltage limit)
    :cvar second_covariance: covariance matrix of the second fit region (low voltage limit)

    For further general information see :py:class:`pytables.IsDescription`.
    """
    Ubi = tb.Float64Col(pos=0)
    Ubi_error = tb.Float64Col(pos=1)
    a = tb.Float64Col(pos=2)
    a_error = tb.Float64Col(pos=3)
    b = tb.Float64Col(pos=4)
    b_error = tb.Float64Col(pos=5)
    c = tb.Float64Col(pos=6)
    c_error = tb.Float64Col(pos=7)
    d = tb.Float64Col(pos=8)
    d_error = tb.Float64Col(pos=9)
    Ubi_corrected = tb.Float64Col(pos=10)
    Ubi_corrected_error = tb.Float64Col(pos=11)
    a_corrected = tb.Float64Col(pos=12)
    a_corrected_error = tb.Float64Col(pos=13)
    b_corrected = tb.Float64Col(pos=14)
    b_corrected_error = tb.Float64Col(pos=15)
    c_corrected = tb.Float64Col(pos=16)
    c_corrected_error = tb.Float64Col(pos=17)
    d_corrected = tb.Float64Col(pos=18)
    d_corrected_error = tb.Float64Col(pos=19)
    Ubi_systematic = tb.Float64Col(pos=20)
    Ubi_systematic_corrected = tb.Float64Col(pos=21)
    Ubi_systematic_dispersion = tb.Float64Col(pos=22)
    Ubi_systematic_dispersion_corrected_error = tb.Float64Col(pos=23)
    rho = tb.Float64Col(pos=24)
    rho_error = tb.Float64Col(pos=25)
    rho_corrected = tb.Float64Col(pos=26)
    rho_corrected_error = tb.Float64Col(pos=27)
    first_covariance = tb.Float64Col(pos=28, shape=(2, 2))
    second_covariance = tb.Float64Col(pos=28, shape=(2, 2))

    def __new__(cls, classname: str, bases: Sequence, classdict: dict[str, Any]):
        print("Called new!")
        return tb.IsDescription.__new__(cls, classname, bases, classdict)

    def __init__(self):
        print("Called __init__!")
        self.avg_cap = tb.Float64Col(pos=10)


class CVDepletionCapacitanceData(tb.IsDescription):
    """
    :py:class:`pytables.IsDescription` data structure to store the results of the depletion analysis in a
    :py:class:`pytables.Table` object.
    This will include in particular the results for the depletion voltage and their statistical and systematic
    uncertainties.

    As all fits are performed individually for corrected and uncorrected capacities, this data structure contains
    separate fields/entries for results based on uncorrected and corrected capacities.
    So not only the depletion voltage field is present twice but also the results for the fit parameters and their errors
    and the covariance matrix.

    Class variables are here to understand as the fields of the table.
    :cvar Ubi: estimation of the depletion voltage based on the uncorrected capacities.
    :cvar Ubi_error: (statistical) uncertainty of the depletion voltage based on the uncorrected capacities.
    :cvar a: slope parameter of the linear fit to the high voltage limit to estimate the depletion voltage.
    :cvar a_error: uncertainty of the slope parameter of the linear fit to the high voltage limit to estimate the
    depletion voltage.
    :cvar b: offset parameter of the linear fit to the high voltage limit to estimate the depletion voltage.
    :cvar b_error: uncertainty of the offset parameter of the linear fit to the high voltage limit to estimate the
    depletion voltage.
    :cvar c: slope parameter of the linear fit to the low voltage limit to estimate the depletion voltage.
    :cvar c_error: uncertainty of the slope parameter of the linear fit to the high voltage limit to estimate the
    depletion voltage.
    :cvar d: offset parameter of the linear fit to the low voltage limit to estimate the depletion voltage.
    :cvar d_error: uncertainty of the offset parameter of the linear fit to the low voltage limit to estimate the
    depletion voltage.
    :cvar bias: bias voltage used for this measurement (or the kind of the measurement).
    :cvar n_pixel: number of pixels which have contributed to these average values.
    :cvar capacitance: measured/averaged capacitance of the sensor at the given bias voltage.
    :cvar cap_err: uncertainty on the mean capacitance parameter of the sensor at the given bias voltage.
    :cvar cap_std: spread of the capacitance over the sensor at the given bias voltage.
    :cvar cap_std_err: uncertainty of the capacitance spread parameter at the given bias voltage.
    :cvar r_on: averaged/measured on-resistance of the measurement circuits.
    :cvar r_on_err: uncertainty of the on-resistance's mean parameter.
    :cvar r_on_std: spread of the on-resistance over the available measurement circuits/channels.
    :cvar r_on_std_err: uncertainty of the on-resistance spread parameter over the available measurement
    circuits/channels.
    :cvar cap_corrected: measured/averaged capacitance of the sensor at the given bias voltage (based on the corrected
    capacitances).
    :cvar cap_corrected_err: spread of the capacitance over the sensor at the given bias voltage. (based on the corrected
    capacitances).
    :cvar cap_parasitic: value of the averaged parasitic capacitance used here.
    :cvar cap_systematic_error: general systematic uncertainty of the capacitance measurement. Given by the spread of
    parasitic capacitance and the way the investigation/analysis is performed.

    For further general information see :py:class:`pytables.IsDescription`.
    """
    bias = tb.Float64Col(pos=0)
    n_pixel = tb.Int64Col(pos=1)
    capacitance = tb.Float64Col(pos=2)
    cap_err = tb.Float64Col(pos=3)
    cap_std = tb.Float64Col(pos=4)
    cap_std_err = tb.Float64Col(pos=5)
    r_on = tb.Float64Col(pos=6)
    r_on_err = tb.Float64Col(pos=7)
    r_on_std = tb.Float64Col(pos=8)
    r_on_std_err = tb.Float64Col(pos=9)
    cap_corrected = tb.Float64Col(pos=10)
    cap_corrected_err = tb.Float64Col(pos=11)
    cap_parasitic = tb.Float64Col(pos=12)
    cap_systematic_error = tb.Float64Col(pos=13)
    Ubi = tb.Float64Col(pos=14)
    Ubi_error = tb.Float64Col(pos=15)
    a = tb.Float64Col(pos=16)
    a_error = tb.Float64Col(pos=17)
    b = tb.Float64Col(pos=18)
    b_error = tb.Float64Col(pos=19)
    c = tb.Float64Col(pos=20)
    c_error = tb.Float64Col(pos=21)
    d = tb.Float64Col(pos=22)
    d_error = tb.Float64Col(pos=23)
# endregion
