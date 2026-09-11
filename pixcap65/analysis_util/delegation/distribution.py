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
import numpy as np
import tables as tb
from typing import Optional

from pixcap65.analysis_util.constants import UNITS_ATTRIBUTE_KEY, REDUCED_SYSTEMATICS_SAMPLE_SIZE, \
    DISPERSION_PARASITIC_DEVIATION
from pixcap65.analysis_util.modelling.physics_modelling import gauss_model, extended_gauss_integral
from pixcap65.analysis_util.multiprocessing import __mp_init_distribution_delegate, __mp_handle_distribution_delegate
from pixcap65.analysis_util.stats.sampling import get_rng
from pixcap65.analysis_util.utility import check_leaf_unit, HIST_CAP_UNIT, handle_fitter_advanced_options, \
    investigate_fit_convergence, get_base_group
from pixcap65.pixcap.pixcap_structure import CAPACITANCE_CONVERSION_FACTOR, DEFAULT_BIN_NUMBER
from pixcap65.plotting_util.utility import evaluate_pixel_mask
from pixcap65.utility.tables_util import set_group_attribute


def analyze_capacitance_distribution_delegate(analysis_group: Optional[tb.Group], output_pdf, **kwargs):
    """
    analyse_capacitance_distribution_delegate

    @author: Dominik Fischer
    @date: 2026-08-11

    Implementation of the investigation in the distribution of the capacitance on the chip.
    It should predominantly be used to determine the intrinsic and parasitic capacitance of a bare PixCap65 chip.
    To achieve the distribution, the data is first binned and presented into a histogram.
    Next, a gaussian shape is fitted to the histogram to match its shape.
    This fit is either performed by the `iminuit` or the `kafe2` framework.

    Last the main results are written back to the analysis group as an attribute.

    :param analysis_group: hdf file's group where to find the capacitance to be analysed.
    :param output_pdf: PDF object to write the created figures to for long-term saving.
    :keyword capacitance: histogram of the capacitance to use instead of those extracted from the provided hdf files group.
    :keyword set_parasitic: boolean, whether to set the parasitic capacitance for this data set.
    :type set_parasitic: bool
    :keyword no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
    :keyword convert: boolean, whether to convert the capacitance to fF, or not (default: True)
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :keyword apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword fit_plot_pdf: PDF object to save the fit figures to.
    :keyword test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :keyword mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation. [array-like]
    :keyword mask_lower: threshold to mask all pixels below this value.
    :type mask_lower: float
    :keyword mask_upper: threshold to mask all pixels above this value.
    :type mask_upper: float
    :keyword hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :return tuple of (capacitance data used, mean value, mean parameter error, std parameter value, std parameter error)
    :rtype tuple
    """
    # none to expect, as these function runs fine without leaks when not called from an mp processing pool!
    from pixcap65.pixcap.pixcap_structure import COUNTS_HIST_LABEL
    from pixcap65.pixcap.pixcap_structure import HIST_PIX_CAP_LABEL
    from matplotlib import pyplot as plt
    # extract further arguments for the performance of the fitting
    use_kafe2 = kwargs.pop("use_kafe2", False)
    if output_pdf is None:
        output_pdf = kwargs.pop("fit_plot_pdf", None)
    set_parasitic = kwargs.pop("set_parasitic", analysis_group is not None)
    assert not set_parasitic or analysis_group is not None

    # we want to fit a binned distribution; but some bins might be empty
    cap_hist = kwargs.pop("capacitance", None)
    if cap_hist is None or not isinstance(cap_hist, np.ndarray):
        assert analysis_group is not None
        cap_hist = check_leaf_unit(analysis_group.HistCap, HIST_CAP_UNIT)

    fig, ax = plt.subplots()
    hist_cap_hist = evaluate_pixel_mask(cap_hist, **kwargs)
    temp_hist_back_data = hist_cap_hist[~np.isnan(hist_cap_hist)].reshape(-1) * CAPACITANCE_CONVERSION_FACTOR
    hist_data, bins, _ = ax.hist(temp_hist_back_data,
                                 bins=kwargs.get("hist_bins", DEFAULT_BIN_NUMBER), density=False, )
    bin_positions = bins[:-1] + np.diff(bins)

    # now fit this to a gauss function
    initial_estimator = {'u': float(np.mean(temp_hist_back_data)), 's': float(np.std(temp_hist_back_data))}
    extended_fitter = None
    if use_kafe2:
        from kafe2 import HistContainer, HistFit
        data_container = HistContainer(bin_edges=bins, fill_data=temp_hist_back_data)
        fitter = HistFit(data=data_container, density=False, model_function=gauss_model)
        fitter.set_parameter_values(**initial_estimator)
        fitter.do_fit()
        assert fitter.did_fit
        mean_value = fitter.parameter_values[0]
        mean_error = fitter.parameter_errors[0]
        std_value = fitter.parameter_values[1]
        std_error = fitter.parameter_errors[1]
        norm = fitter.parameter_values[2]
        fit_results = fitter.parameter_values
        fit_cov = fitter.parameter_cov_mat
    else:
        from iminuit import Minuit
        # noinspection PyProtectedMember
        from iminuit.cost import Model, BinnedNLL, ExtendedBinnedNLL
        from pixcap65.analysis_util.stats import distribution_norm as norm
        assert isinstance(gauss_model, Model)
        assert isinstance(extended_gauss_integral, Model)
        assert isinstance(norm.cdf, Model)
        initial_estimator_2 = initial_estimator.copy()
        initial_estimator_2["b"] = 1.0
        cost = BinnedNLL(hist_data, bins, cdf=norm.cdf, name=('u', 's'))
        cost_2 = ExtendedBinnedNLL(hist_data, bins, scaled_cdf=extended_gauss_integral, name=("b", 'u', 's'))
        extended_fitter = Minuit(cost_2, **initial_estimator_2)
        extended_fitter.migrad()
        extended_fitter.hesse()
        fitter = Minuit(cost, **initial_estimator)
        fitter.migrad()
        fitter.hesse()
        fit_results = fitter.values
        fit_cov = fitter.covariance
        norm = np.sum(hist_data)
        mean_value = fitter.values["u"]
        mean_error = fitter.errors["u"]
        std_value = fitter.values["s"]
        std_error = fitter.errors["s"]

    if output_pdf is not None:
        if extended_fitter is not None:
            handle_fitter_advanced_options(extended_fitter, False, "$C$ / \\unit{{\\femto\\farad}}", COUNTS_HIST_LABEL,
                                           "Capacitance distribution (EXTENDED)", output_pdf,
                                           "Contours for the capacitance distribution (EXTENDED)")
        handle_fitter_advanced_options(fitter, True, "$C$ / \\unit{{\\femto\\farad}}", COUNTS_HIST_LABEL,
                                       "Capacitance distribution", output_pdf,
                                       "Contours for the capacitance distribution")
    if set_parasitic:
        assert analysis_group is not None
        set_group_attribute(analysis_group, "parasitic", mean_value)
        set_group_attribute(analysis_group, "parasitic_error", std_value)
        set_group_attribute(analysis_group, UNITS_ATTRIBUTE_KEY, "fF")
    binning_span = np.linspace(bin_positions[0], bin_positions[-1], 1000)
    normalisation = norm * np.mean(np.diff(bin_positions))
    model_data = gauss_model(binning_span, mean_value, std_value, normalisation)
    ax.plot(binning_span, model_data, ":",
            label="C =  \\qty{{{c:.2f}\\pm{error:.2f}}}{{\\femto\\farad}}".format(c=mean_value, error=std_value))

    try:
        from jacobi import propagate
        # noinspection PyTypeChecker
        y, y_cov = propagate(lambda p: gauss_model(binning_span, p[0], p[1], normalisation), fit_results,
                             fit_cov)
        y_err_prop = np.diag(y_cov) ** 0.5
        ax.fill_between(binning_span, y - y_err_prop, y + y_err_prop, facecolor="C1", alpha=0.5)

    except ImportError:
        # noinspection PyUnusedLocal
        def propagate(*args):
            """
            This is just a template.
            :param args:
            """
            # This just a template
            pass
    except np.linalg.LinAlgError:
        pass
    except TypeError as e:
        if str(e).startswith("unsupported operand type(s) for *"):
            print("THE ERROR ESTIMATION FAILED!")
        else:
            raise

    # add some information about the model
    hypo_test = investigate_fit_convergence(fitter)
    ax.set_ylabel(COUNTS_HIST_LABEL)
    ax.set_xlabel(HIST_PIX_CAP_LABEL)
    ax.grid()
    if fit_cov is None:
        print(hist_data)
        print(bins)
        print(temp_hist_back_data)
        plt.show()
        raise ValueError("The fit failed by estimating the covariance matrix.")
    else:
        ax.legend(
            title=f"GoF = {hypo_test['x']:.4n}\nndf = {hypo_test['ndf']: .4n}\np = {hypo_test['p']: .4n}\nu = "
                  f"{mean_value:.6n}+-{mean_error:.6n}\ns = {std_value:.6n}+-{std_error:.6n}")
    if kwargs.get('no_plot', False):
        plt.close(fig)
    elif output_pdf is None:
        plt.show()
    else:
        output_pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    if not kwargs.pop("convert", True):
        mean_value /= CAPACITANCE_CONVERSION_FACTOR
        mean_error /= CAPACITANCE_CONVERSION_FACTOR
        std_value /= CAPACITANCE_CONVERSION_FACTOR
        std_error /= CAPACITANCE_CONVERSION_FACTOR

    return temp_hist_back_data.reshape(-1).shape[0], mean_value, mean_error, std_value, std_error


def analyze_capacitance_distribution(raw_data, base_path=None, corrected_distribution=False, **kwargs):
    """
    analyze_capacitance_distribution

    @author: Dominik Fischer
    @date: 2026-08-12 (originally earlier)
    last update: 2026-08-12

    Helper function to describe how the capacitance is distributed over the sensor.
    We are in particular interested in the average capacitance of the pixel and their spread/dispersion.
    This is just a wrapper function to perform the file handles and delegate the whole analysis to another
    function.

    :param raw_data: hdf file containing the measurements and investigation of a pixcap sample to obtain
        information about the capacitances.
    :param corrected_distribution: boolean, False, indicates whether the corrected capacitance distribution
        should be analyzed.
    :param base_path: hdf files group witht the measurement data.
    :keyword fit_plot_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure).
    :keyword output_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided,
        Only used for the advanced procedure).
    :keyword fit_plot_pdf_name:  Name of the PDF file to save fitting figures from the advanced procedures to
        (Only used for the advanced procedure).
    :keyword mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :keyword test_cap_exclusion: boolean, whether to exclude the test capacitator row from the histograms.
    :keyword hist_bins: integer, number of bins to use for the histogram.
    :keyword no_plot: boolean, whether to supress (interactive) plotting of the distribution of the capacitance.
    :keyword convert: boolean, whether to convert the capacitance to fF, or not (default: True)
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :keyword apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword mask_lower: threshold to mask all pixels below this value.
    :type mask_lower: float
    :keyword mask_upper: threshold to mask all pixels above this value.
    :type mask_upper: float
    """
    temp_plot_name = kwargs.pop("fit_plot_pdf_name", None)
    if temp_plot_name is not None:
        assert "fit_plot_pdf_name" not in kwargs
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(temp_plot_name) as pdf:
            analyze_capacitance_distribution(raw_data, base_path, corrected_distribution,
                                             fit_plot_pdf=pdf, **kwargs)
    else:
        output_pdf = kwargs.pop("output_pdf", None)
        if output_pdf is None:
            output_pdf = kwargs.pop("fit_plot_pdf", None)
        with tb.open_file(raw_data, mode='a') as in_file_h5:
            base_group = get_base_group(base_path, in_file_h5)
            if corrected_distribution:
                analyze_capacitance_distribution_delegate(base_group.total_cap.analysis_correction, output_pdf,
                                                          **kwargs)
            else:
                analyze_capacitance_distribution_delegate(base_group.total_cap.analysis, output_pdf, **kwargs)


def _get_sensor_distribution(ana_group: tb.Group, bias_voltage, cap_data, dist_entry, parasitic: float,
                             parasitic_error: float, **kwargs):
    """
    _get_sensor_distribution

    @author Dominik Fischer
    @date 2026-05-31
    last update: 2026-08-12

    Internal helper function to perform the fits to the capacitance distribution over the measured part of the sensor.
    The handler is introduced to also handle the estimation of systematic uncertainties on the capacitance values
    in the fits.
    Will first compute the distribution of the capacitance over the full sensor approximated by a gaussian pdf.
    The values from this fit will be used as the average values later on.
    The procedure is repeated with the corrected capacitance values, if necessary. Otherwise the whole fit is recomputed
    and treated as a corrected fit for all further computations.

    To extract the systematic uncertainties on this distribution, it is necessary to repeat the fit multiple times.
    As the uncertainties of the data points are not used when applying a binned nll fit, the systematic effects could
    only be approximated on a statistical basis.

    For the spread of the parasitics two fits at the extremes are used with appropriately modified capacitance arrays.
    For the dispersion effects random samples of the dispersion are generated and added to the corrected capacitance
    data.

    :param ana_group: hdf files' analysis group which stores the capacitances' for the pixels on which to compute the
        capacitance distribution.
    :param bias_voltage: hv voltage applied as reversed bias to the sensor for this capacitance measurement. Will be
        propagated without further processing to the table row.
    :type cap_data: numpy.ndarray
    :param cap_data: numpy array of the capacitance for each pixel (nan if the capacitance for a particular pixel is
        not measured)
    :param dist_entry: tables row to append the extracted information about the fit result and parasitic capacitance,
        as well, as some other data about systematic uncertainties to.
    :param parasitic: parasitic capacitance of the pixcap chip and the bump-bonds below the sensor.
    :param parasitic_error: spread of the parasitic capacitance over a pixcap chip
    :param kwargs: further keyword arguments (but which only used for the resistance distribution if estimators
        are present)
    :keyword hist_res_key: name/identifiert of the array/table which contains the on-resistance estimators if present.
    :keyword apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :keyword mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation. [array-like]
    :keyword mask_lower: threshold to mask all pixels below this value.
    :type mask_lower: float
    :keyword mask_upper: threshold to mask all pixels above this value.
    :type mask_upper: float
    :keyword hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    """
    # remove all unnecessary keywords
    kwargs.pop("plot", None)
    kwargs.pop("fit_plot_pdf", None)
    kwargs.pop("use_kafe2", None)
    kwargs.pop("output_pdf", None)
    kwargs['no_plot'] = True

    hist_res_key = kwargs.pop("hist_res_key", "HistRes")
    # Will continue using tuples for performance!
    # first get a histogram with all the data
    dist_result_list = []
    dist_result_list.append(analyze_capacitance_distribution_delegate(None,
                                                                      None,
                                                                      capacitance=cap_data,
                                                                      convert=False,
                                                                      **kwargs))
    # we need an independent fit for the corrected distributions
    # interestingly only the parasitic negatives are present here.
    cap_data_para = np.where(np.isfinite(cap_data), cap_data - parasitic, np.nan)
    rng = get_rng()
    sensor_distribution_type = np.dtype([
        ("n", np.uint64),
        ("C", np.float64),
        ("C_err", np.float64),
        ("C_std", np.float64),
        ("C_std_err", np.float64),
    ])
    dist_result_list.append(analyze_capacitance_distribution_delegate(None, None,
                                                                      capacitance=cap_data_para,
                                                                      convert=False,
                                                                      **kwargs))
    # this should not change the computation time.
    parasitic_advanced_samples = rng.normal(loc=0, scale=parasitic_error, size=REDUCED_SYSTEMATICS_SAMPLE_SIZE)
    # we could also optimise here by using multiprocessing iterators!
    dispersion_cap_sample = rng.normal(loc=0, scale=DISPERSION_PARASITIC_DEVIATION,
                                       size=REDUCED_SYSTEMATICS_SAMPLE_SIZE,)
    # FIXME: prevent this mp worker pool from leaking semaphore objects all around!
    if True:
        __mp_init_distribution_delegate(cap_data_para, kwargs)
        parasitic_adv_cap_est = np.rec.array([__mp_handle_distribution_delegate(item) for item in parasitic_advanced_samples], dtype=sensor_distribution_type)
        dispersion_estimator = np.rec.array([__mp_handle_distribution_delegate(item) for item in dispersion_cap_sample], dtype=sensor_distribution_type)
    else:
        # for current size of these iterables there is no improvement in time by using pooled execution!
        with mp.Pool(processes=8, initializer=__mp_init_distribution_delegate, initargs=(cap_data_para, kwargs)) as pool:
            try:
                parasitic_adv_cap_est = np.rec.array(pool.map(__mp_handle_distribution_delegate, parasitic_advanced_samples,
                                                              chunksize=None), dtype=sensor_distribution_type)
                dispersion_estimator = np.rec.array(pool.map(__mp_handle_distribution_delegate, dispersion_cap_sample,
                                                             chunksize=None), dtype=sensor_distribution_type)
            finally:
                pool.close()

    # with concurrent_futures.ProcessPoolExecutor(max_workers=5, mp_context=ctx) as executor:
    #     parasitic_futures = [executor.submit(analyze_capacitance_distribution_delegate, None, None, capacitance=cap_data_para + offset, convert=False, **kargs) for offset in parasitic_advanced_samples]
    #     dispersion_futures = [executor.submit(analyze_capacitance_distribution_delegate, None, None, capacitance=cap_data_para + iterat, convert=False, **kargs) for iterat in dispersion_cap_sample]
    #     parasitic_adv_cap_est = np.rec.array([future.result() for future in parasitic_futures], dtype=sensor_distribution_type)
    #     dispersion_estimator = np.rec.array([future.result() for future in dispersion_futures], dtype=sensor_distribution_type)

    # for usage of the map function it would be necessary to wrap the executable appropriately.
    # parasitic_adv_cap_est = np.rec.array([analyze_capacitance_distribution_delegate(None, None, capacitance=cap_data_para+offset,convert=False, **kargs) for offset in parasitic_advanced_samples], dtype=sensor_distribution_type)
    # dispersion_estimator = np.rec.array([analyze_capacitance_distribution_delegate(None, None, capacitance=cap_data_para+iterat, convert=False, **kargs) for iterat in dispersion_cap_sample], dtype=sensor_distribution_type)

    distribution_result = np.rec.array(dist_result_list, dtype=sensor_distribution_type)

    # Fehler von Fehlern werden i.d.R. unterdrückt.
    def assemble_systematic_propagation(data: np.recarray):
        assert data.dtype == sensor_distribution_type
        weights = np.reciprocal(data.C_err ** 2 + data.C_std ** 2)
        avg = np.average(data.C, weights=weights, keepdims=True)
        systematic_propagation = np.nanstd(data.C, mean=avg, dtype=np.float64)
        return systematic_propagation

    cap_corr_systematic = assemble_systematic_propagation(parasitic_adv_cap_est)

    # the procedure must also be performed for the literature value of chip dispersion effects for the
    # parasitic capacitances.
    # this function is not able to distinguish between corrected and uncorrected iterations!
    cap_dispersion_systematic = assemble_systematic_propagation(dispersion_estimator)

    # try to get to the on-resistance datasets to perform the same distribution handler!
    # only issue with this attempt the capacitance will be saved as a parasitic one!
    if hist_res_key in ana_group and np.any(np.isfinite(ana_group[hist_res_key])):
        resistance_array = ana_group[hist_res_key]
        assert isinstance(resistance_array, (tb.Array, np.ndarray, tb.Table))
        resistance_data = resistance_array[:]
        temp_tuple = analyze_capacitance_distribution_delegate(None,
                                                               kwargs.get(
                                                                   "distribution_output_pdf",
                                                                   None),
                                                               capacitance=resistance_data,
                                                               **kwargs)
        _, res, res_err, res_std, res_std_err = temp_tuple
    else:
        res = 0.0
        res_err = 0.0
        res_std_err = 0.0
        res_std = 0.0

    dist_entry['bias'] = kwargs.pop("alter_spec", bias_voltage)
    dist_entry['n_pixel'] = distribution_result.n[0]
    dist_entry['capacitance'] = distribution_result.C[0]
    dist_entry['cap_err'] = distribution_result.C_err[0]
    dist_entry['cap_std'] = distribution_result.C_std[0]
    dist_entry['cap_std_err'] = distribution_result.C_std_err[0]
    dist_entry['r_on'] = res
    dist_entry['r_on_err'] = res_err
    dist_entry['r_on_std'] = res_std
    dist_entry['r_on_std_err'] = res_std_err
    dist_entry['cap_systematic_dispersion'] = cap_dispersion_systematic
    dist_entry['cap_systematic_error'] = cap_corr_systematic
    dist_entry['cap_corrected'] = distribution_result.C[1]
    dist_entry['cap_corrected_err'] = distribution_result.C_std[1]
    dist_entry['cap_parasitic'] = parasitic
    dist_entry['cap_corrected_est_error'] = distribution_result.C_err[1]
    dist_entry['cap_corrected_std_error'] = distribution_result.C_std_err[1]

    if isinstance(dist_entry, tb.tableextension.Row):
        dist_entry.append()
    else:
        print("No append of the distribution!")
