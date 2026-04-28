"""
This file contains some utilities needed for the analysis and the plotting.
"""

from typing import Union, Mapping, AnyStr, LiteralString

import numpy as np
import tables as tb
import time
from iminuit.minuit import _cl_to_errordef
from tables import File
from tables.group import RootGroup

from utility.utils_2 import walk_to_node, UNITS_ATTRIBUTE_KEY

# some utility constants to simplify expressions for the analysis
# region Analysis constants
GENERAL_TRANSFORMATION_MATRIX = np.array(
    [[1.e-12, 1.e-6, 1.e3, 1.e-6], [1.e-6, 1, 1.e9, 1], [1.e3, 1.e9, 1.e18, 1.e9], [1.e-6, 1, 1.e9, 1]])
ANALYSIS_GROUP_NAME = "analysis"
ANALYSIS_CORRECTED_GROUP_NAME = "analysis_correction"
TABLES_ARRAY_TYPE = Union[np.ndarray, tb.CArray]
TABLES_TABLE_TYPE = Union[tb.Table, Mapping[str, TABLES_ARRAY_TYPE]]
TABLES_LEAF_TYPE = Union[tb.Leaf, tb.Table, tb.Array]
TABLES_LEAF_COMPAT_TYPE = Union[tb.Leaf, tb.Table, tb.Array, np.ndarray, tb.CArray]
GENERAL_PIXCAP_SHAPE = (40, 40)
COVARIANCE_PIXCAP_SHAPE = (40, 40, 4, 4)
FULL_MODEL_LABEL = "I_\\text{{full}}"
SIMPLE_MODEL_LABEL = "I_\\text{{approximation}}"
FULL_MODEL_EXPRESSION = "\\frac{{{u0}\\cdot{c}\\cdot{freq}+{i}}}{{1+{r}\\cdot{c}\\cdot{freq}}}"
SIMPLE_MODEL_EXPRESSION = "{u0}\\cdot{c\\cdot{freq}+{i}}"
FULL_MODEL_PARAMETER_DICT = {"c": "C", "r": "R", "i": "I_\\text{{Leakage}}", "u0": "U_{{0}}", "freq": "\\nu"}
SIMPLE_MODEL_PARAMETER_DICT = {"c": "C", "i": "I", "u0": "U_{{0}}", "freq": "\\nu"}
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


def str_join(delimiter: AnyStr, *args: AnyStr) -> LiteralString | bytes:
    """
    str_join

    Modified function to join multiple strings separated by the specified delimiter.
    This function was necessary as the builtin implementation does not support variable args.
    :param delimiter: delimiter to use between the different strings when combining them.
    :param args: strings to be combined
    :return: combined string
    """
    # But what 'to do' if one of the var args is a list or in general an iterable of strings?
    return delimiter.join(args)


def check_leaf_unit(leaf: TABLES_LEAF_TYPE, unit: str) -> TABLES_LEAF_COMPAT_TYPE:
    """
    check_leaf_unit(leaf, unit)

    Verify that the table or array has a units attribute and the unit is the one expected.
    As last step return the data structure requested.

    This may not work for tables with multiple units (one per column)
    :param leaf: Data structure to be verified and extracted.
    :param unit: Expected unit for the data structure.
    :return: array_like of the data structures contents.
    """
    assert isinstance(leaf, TABLES_LEAF_TYPE)
    if UNITS_ATTRIBUTE_KEY not in leaf.attrs or leaf.attrs[UNITS_ATTRIBUTE_KEY] != unit:
        raise AssertionError
    result = leaf[:]
    assert isinstance(result, TABLES_LEAF_COMPAT_TYPE)
    return result


def handle_kafe2_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf, contours_title=None):
    """
    handle_kafe2_advanced_options

    For the non-linear fitting frameworks are some further options available to directly plot the results or perform
    profiling.
    This implementation is only for use with the kafe2 framework.

    :param fit_object: object of the already performed fit.
    :param apply_contour: boolean, whether to apply contour profiling around the found optimum.
    :param x_label: label of the x axis.
    :param y_label: label of the y axis.
    :param title: title of the plot.
    :param pdf: PdfPages object to save the fit plots to.
    :param contours_title: title of the contour plot.
    """
    from kafe2 import Plot, FitBase
    assert isinstance(fit_object, FitBase)
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
        cpf_figure.axes[0].set_title(contours_title)
        pdf.savefig(cpf_figure, bbox_inches='tight')


def handle_minuit_advanced_options(fit_object, apply_contour, x_label, y_label, title, pdf,
                                   contours_title=None):
    """
    handle_iminuit_advanced_options

    For the non-linear fitting frameworks are some further options available to directly plot the results or perform
    profiling.
    This implementation is only for use with the iminuit framework.

    :param fit_object: object of the already performed fit.
    :param apply_contour: boolean, whether to apply contour profiling around the found optimum.
    :param x_label: label of the x axis.
    :param y_label: label of the y axis.
    :param title: title of the plot.
    :param pdf: PdfPages object to save the fit plots to.
    :param contours_title: title of the contour plot.
    """
    from matplotlib import pyplot as plt
    from iminuit import Minuit
    assert isinstance(fit_object, Minuit)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fit_object.visualize()
    # for error bands we must perform something similar
    if hasattr(fit_object.fcn._fcn, "model"):
        try:
            from jacobi import propagate
            x, _, _ = fit_object.fcn._fcn._masked.T
            y, ycov = propagate(lambda p: fit_object.fcn._fcn.model(x, p), fit_object.values, fit_object.covariance)
            yerr_prop = np.diag(ycov) ** 0.5
            plt.fill_between(x, y - yerr_prop, y + yerr_prop, facecolor="C1", alpha=0.5)
        except:
            pass

    pdf.savefig(fig, bbox_inches='tight')
    cls = [0.68, 0.9, 0.99]

    if apply_contour:
        # will need 'to do' it on our selves
        pars = [p for p in fit_object.parameters if not fit_object.fixed[p]]
        npar = len(pars)
        figsize=None
        fig, ax = plt.subplots(
            npar,
            npar,
            figsize=figsize,
            constrained_layout=True,
            squeeze=False,
        )

        for i, par1 in enumerate(pars):
            plt.sca(ax[i, i])
            fmax = 0
            for k, cl in enumerate(cls):
                f = _cl_to_errordef(cl, 1, 0.68)
                fmax = max(fmax, f)
                plt.axhline(f, color=f"C{k}")
            fit_object.draw_mnprofile(par1, subtract_min=True, bound=3)
            ax[i, i].set_ylabel("$\\Delta - 2\\log \\mathcal{{L}}$")

            for j in range(i):
                par2 = pars[j]
                plt.sca(ax[i, j])
                plt.plot(fit_object.values[par2], fit_object.values[par1], "+", color="k")
                fit_object.draw_mncontour(par2, par1, cl=cls)
                ax[j, i].set_visible(False)

        fig.suptitle(contours_title)
        pdf.savefig(fig, bbox_inches='tight')
        fig, ax = fit_object.draw_mnmatrix(cl=cls)

        # fit_object.draw_mncontour()
        fig.suptitle(contours_title)
        pdf.savefig(fig, bbox_inches='tight')


def investigate_fit_convergence(fitter):
    from kafe2 import FitBase
    from iminuit import Minuit
    if isinstance(fitter, Minuit):
        print(fitter.fval, fitter.fmin.fval, fitter.fmin.reduced_chi2)
        cost_value = fitter.fval
        ndf = fitter.ndof
    elif isinstance(fitter, FitBase):
        cost_value = fitter.goodness_of_fit
        ndf = fitter.ndf
        print(fitter.chi2_probability)
    else:
        raise TypeError("Fitter must be either a Minuit or XYFit object.")
    from scipy.stats.distributions import chi2
    regular_cost = cost_value / ndf
    p_value = 1 - chi2.cdf(regular_cost, df=ndf)
    convergence = dict(x=cost_value, xn=regular_cost, ndf=ndf, p=p_value)
    print(convergence)
    # print(-2*np.log(cost_value))
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
            base_group = walk_to_node(in_file_h5.root, base_path)
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
    if "analysis" in group:
        # TODO: will need a handler to back previous analysis iterations away without loosing them.
        group.analysis._f_remove(recursive=True)
        time.sleep(1)


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
    get_analysis_grouo

    Get the correct analysis group for the given occaison.
    Distinghuish between analysis and corrected analysis groups and provides the correct one.

    :param base_group: hdf files group where to look for the analysis groups.
    :param use_corrected: boolean, whether to use the corrected capacitances for plotting.
    :return: analysis group from the hdf file.
    """
    if kwargs.get('use_corrected', False):
        return base_group.analysis_correction

    return base_group.analysis


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


class DepletionData(tb.IsDescription):
    col = tb.Int64Col(pos=0)
    row = tb.Int64Col(pos=1)
    V = tb.Float32Col(pos=2)
    NA = tb.Float32Col(pos=3)
    ND = tb.Float32Col(pos=4)
# endregion
