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
from tqdm import tqdm

try:
    # noinspection PyCompatibility
    from collections.abc import Callable, Sized, Iterable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Callable, Sized, Iterable
finally:
    from typing import Union, Optional, List, Tuple, Any

from pixcap65.analysis_util.constants import DISPERSION_PARASITIC_DEVIATION
from pixcap65.analysis_util.utility import GENERAL_PIXCAP_SHAPE
from pixcap65.analysis_util.correction import _extract_table_data
from pixcap65.analysis_util.modelling.data_store import DepletionNumpyStore, DepletionTableStore, DepletionArrayStore, \
    DepletionDataStore
from pixcap65.analysis_util.modelling.physics_modelling import depletion_model
from pixcap65.analysis_util.utility import DepletionData, TABLES_LEAF_COMPAT_TYPE, handle_kafe2_advanced_options, \
    handle_minuit_advanced_options
from pixcap65.pixcap.pixcap_structure import CAPACITANCE_CONVERSION_FACTOR
from pixcap65.utility.tables_util import group_get_file


# perhaps extract the result table as an parameter in order to put the correct depletion values also in the table!
def __distribution_depletion_estimation(dist_table,
                                        first_boundaries: Union[tuple, Iterable[tuple]],
                                        second_boundaries: Union[tuple, Iterable[tuple]], depletion_data_table,
                                        is_corrected, **kwargs) -> Optional[List]:
    """
    distribution_depletion_estimation

    NO LOCKS IN USE HERE!

    Internal handler function to estimate the depletion voltage / reversed bias for full depletion of the sensor.
    This handler function uses the capacitance distribution over the full sensor (or at least the measured part)
    assuming a gaussian pdf.
    It takes the gaussian centered capacitances (more accuratley: the distributions c-v-curve), fits two straight lines
    to low voltage limit for the capacitance behaviour of the undepleted sensor and the high voltage limit for the
    capacitance behaviour for a depletion zone extending over the full physical dimensions of the sensor.
    The depletion voltage is estimated as the intersection of these two straight lines.
    Both the c-v-data is taken from a table and the results are written back to another table.

    These estimation attempt is redone with slightly adjust fit ranges to estimate the systematic uncertainty.
    Here we account for the measured distribution of parasitic capacitance over the Pixcap Chip (its spread), for the
    dispersion of the parasitic capacitance between multiple sensors/chips and the particular choice of the fit range.
    For the additional fits to extract the systematic uncertainties no plotting of the fits is available.
    The Estimation of the different systematic uncertainties is performed by again fitting the voltage ranges but with
    either varied fit range or with the systematic uncertainties provided as the measurement error submitted to the fit
    algorithm. In the latter two cases the extracted fit errors on the depletion voltage is further propagated through
    the analysis.

    For the systmatic estimation another depeltion table will be temporarily created and then removed.

    :type dist_table: numpy.ndarray, numpy.rec.array
    :param dist_table: table of the measures capacitance data in dependence on the applied reverse bias. The data must be
        provided by a structured numpy rec-array and not by a pytables table.
    :param first_boundaries: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will only be estimated if this argument is provided.
        It is also possible to provide an Iterable of boundaries for multiple depletion regions to fit.
    :param second_boundaries: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will be estimated if this argument is provided.
        It is also possible to provide an Iterable of boundaries for multiple depletion regions to fit.
    :type depletion_data_table: pytables.Table
    :param depletion_data_table: table to write the depletion voltage estimation data to.
    :type is_corrected: bool
    :param is_corrected: whether to use the corrected data for this operation.
    :param kwargs: further keyword arguments for the fits to estimate the depletion voltage.
    :keyword fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :keyword use_kafe2: boolean, False, indicates whether kafe2 is used for the fit.
    :keyword apply_contours: boolean, indicates whether to determine the contours and try to plot them. (No effect for
        the fit to estimate systematic effects)
    :type apply_contours: bool
    :keyword plot: boolean, False, indicates whether to plot the data. AN output PDF object could be submitted here
         instead of an explicitly created one. (No effect for the fit to estimate systematic effects)
    :keyword fit_plot_pdf: PDF object to save the fit figures to. (No effect for
        the fit to estimate systematic effects.)
    :keyword verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    :keyword cv_fit_plot_pdf: analog to `fit_plot_pdf` to activate the plotting for c-v- and depletion fits independent from
        the plotting for capacitance estimation fits. If this keyword argument is present also the `plot` arguments will
        be set automatically. (This keyword argument will be ignored for the fits to estimate systematic effects)
    :returns: optional list of figure holder objects.
    """
    pixel_cap_data = _extract_table_data('capacitance', is_corrected, dist_table)
    pixel_cap_error_data = _extract_table_data('cap_std', is_corrected, dist_table)
    voltage_data = _extract_table_data('bias', is_corrected, dist_table)
    systematic_offset = kwargs.pop("systematic_offset", 2)
    systematic_key_args = kwargs.copy()
    systematic_key_args.pop("plot", False)
    systematic_key_args.pop("fit_plot_pdf", None)
    systematic_key_args.pop("output_pdf", None)
    systematic_key_args.pop("cv_fit_plot_pdf", None)

    # is dist_table a ordinary numpy array instead of an rec-array?
    if is_corrected:
        try:
            temp_systematic = dist_table['cap_corr_systematic_error']
            temp_dispersion = dist_table['cap_dispersion_error']
            if np.all(temp_systematic == 0) or not np.any(np.isfinite(temp_systematic)):
                systematic_error = dist_table['cap_systematic_error']
            else:
                systematic_error = temp_systematic
            if np.all(temp_dispersion == 0) or not np.any(np.isfinite(temp_dispersion)):
                dispersion_error = dist_table['cap_systematic_dispersion']
            else:
                dispersion_error = temp_dispersion
        except (KeyError, ValueError):
            # just a safe-guard in case we encounter a table entry before the update!
            systematic_error = dist_table['cap_systematic_error']
            dispersion_error = dist_table['cap_systematic_dispersion']
    else:
        systematic_error = dist_table['cap_systematic_error']
        dispersion_error = dist_table['cap_systematic_dispersion']
    second_systematic_result_storage = DepletionNumpyStore(depletion_data_table.dtype)
    third_systematic_result_storage = DepletionNumpyStore(depletion_data_table.dtype)
    fourth_systematic_result_storage = DepletionNumpyStore(depletion_data_table.dtype)

    if isinstance(first_boundaries, tuple):
        first_boundaries = [first_boundaries]
        second_boundaries = [second_boundaries]

    assert isinstance(first_boundaries, Sized)
    n_depletion_regions = len(first_boundaries)
    fit_result_storage = DepletionTableStore(depletion_data_table, n_depletions=n_depletion_regions, )
    for k, (first_bound, second_bound) in enumerate(zip(first_boundaries, second_boundaries)):
        first_lower, first_upper = first_bound
        second_lower, second_upper = second_bound

        # maybe this could be speed up by a different implementation!
        # Now the real implementation starts.
        analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, pixel_cap_error_data,
                                second_lower,
                                second_upper, voltage_data, fit_result_storage,
                                fit_description_text=" Sensor Distribution", **kwargs)

        # What about the systematic uncertainties of the depletion voltage?
        # 1. fit range
        # 2. fit initial guess
        # 3. systematic uncertainties of the capacitances used for the fit
        # 4. systematic dispersion of the capacitances used for the fit (must be treaded separately from the others)
        # the later two are quite simple to implement, but the first two might be problem,
        # in particular when combining them!
        # Currently it is not even possible to determine the second effect at all.
        __extract_systematic_depletion_effects((first_lower, first_upper),
                                               (second_lower, second_upper), voltage_data,
                                               pixel_cap_data, pixel_cap_error_data, systematic_error, dispersion_error,
                                               systematic_offset, second_systematic_result_storage,
                                               third_systematic_result_storage,
                                               fourth_systematic_result_storage,
                                               systematic_key_args)

    depletion_data_table.flush()
    depletion_data_table.cols.Ubi_systematic[:] = np.sqrt(np.abs(
        depletion_data_table.cols.Ubi[:] - second_systematic_result_storage.table.Ubi[
            :]) ** 2 + third_systematic_result_storage.table.Ubi_error[:] ** 2)
    depletion_data_table.cols.Ubi_systematic_dispersion[:] = fourth_systematic_result_storage.table.Ubi_error[:]
    depletion_data_table.flush()
    group_get_file(depletion_data_table).flush()


def __extract_systematic_depletion_effects(lower_boundary: Tuple, upper_boundary: Tuple, voltage_data: np.ndarray,
                                           pixel_cap_data: np.ndarray,
                                           pixel_cap_error_data: np.ndarray, systematic_error: np.ndarray,
                                           dispersion_error: np.ndarray, systematic_offset: float,
                                           second_systematic_result_storage: DepletionNumpyStore,
                                           third_systematic_result_storage: DepletionNumpyStore,
                                           fourth_systematic_result_storage: DepletionNumpyStore,
                                           systematic_key_args: dict[str, Any]):
    """
    __extract_systematic_depletion_effects

    @author Dominik Fischer
    @date 2026-05-30

    NO LOCKS IN USE HERE

    Private/Internal helper function to compute the systematic uncertainties of the depletion voltage by the used fit
    range, the spread of the parasitic capacitance over a Pixcap Chip and the dispersion of parasitic effects over
    different Pixcap Chip samples.

    These estimation attempt is redone with slightly adjust fit ranges to estimate the systematic uncertainty.
    Here we account for the measured distribution of parasitic capacitance over the Pixcap Chip (its spread), for the
    dispersion of the parasitic capacitance between multiple sensors/chips and the particular choice of the fit range.
    For the additional fits to extract the systematic uncertainties no plotting of the fits is available.
    The Estimation of the different systematic uncertainties is performed by again fitting the voltage ranges but with
    either varied fit range or with the systematic uncertainties provided as the measurement error submitted to the fit
    algorithm. In the latter two cases the extracted fit errors on the depletion voltage is further propagated through
    the analysis.

    :param lower_boundary: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will only be estimated if this argument is provided.
        It is also possible to provide an Iterable of boundaries for multiple depletion regions to fit.
    :param upper_boundary: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will be estimated if this argument is provided.
        It is also possible to provide an Iterable of boundaries for multiple depletion regions to fit.
    :param voltage_data: array of the biasing HV voltages (with the correct sign)
    :param pixel_cap_data: capacitance data for one particular pixel on the sensor
    :param pixel_cap_error_data: uncertainties of the capacitance data for one particular pixel on the sensor.
    :param systematic_error: standard deviation of the parasitic capacitance of the Pixcap chip on a single sensor.
    :param dispersion_error: spread of the dispersion of the parasitic capacitance between different Pixcap chip
        samples. This will induce a systematic effect on the accuracy of the capacitance's and the depletion voltage of
        the investigated sensor.
    :param systematic_offset: enlargement in V for the fit range conditions applied for fitting. Needed to estiamte the
        systematic uncertainties by the fit range accurately. (default: 2)
    :param second_systematic_result_storage: numpy based result storage object for the results
        from varying the fit range.
    :param third_systematic_result_storage: numpy based result storage object for the results
        from accounting for the spread of the parasitic capacitance on the same pixcap chip.
    :param fourth_systematic_result_storage: numpy based result storage object for the results
        from accouting for the dispersion of the parasitic capacitance between different Pixcap chips.
    :param systematic_key_args: further keyword arguments to be propagated to the fits.
    """

    first_lower, first_upper = lower_boundary
    second_lower, second_upper = upper_boundary

    analyze_pixel_depletion(first_lower - systematic_offset, first_upper + systematic_offset, pixel_cap_data,
                            pixel_cap_error_data,
                            second_lower - systematic_offset,
                            second_upper + systematic_offset, voltage_data, second_systematic_result_storage,
                            fit_description_text=" Sensor Distribution Systematic", **systematic_key_args)
    analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, systematic_error, second_lower, second_upper,
                            voltage_data,
                            third_systematic_result_storage, fit_description_text=" Sensor Distribution Systematic",
                            **systematic_key_args)
    analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, dispersion_error, second_lower, second_upper,
                            voltage_data,
                            fourth_systematic_result_storage, fit_description_text=" Sensor Distribution Systematic",
                            **systematic_key_args)


def __depletion_iterator_implementation(index, cap_data, cap_error_data, lower_boundary,
                                        upper_boundary, voltage_data, storage, **kwargs):
    """
    :param index: tuple of columns and row of the pixel to be investigated.
    :param cap_data: matrix of pixel capacitance data with dependence on the applied bias voltage in the last index.
    :param cap_error_data: matrix of pixel capacitances' (statistical) uncertainties with dependence on the applied bias voltage in the last index.
    :param lower_boundary: boundaries for the fits to the high voltage asymptotic limit for determining the depletion voltage.
    :param upper_boundary: boundaries for the fits to the low voltage asymptotic limit for determining the depletion voltage.
    :param voltage_data: array of the biasing HV voltages (with the correct sign)
    :param storage: (specialised) object to store the results of the analysis while the mapping to individual pixels is done by the calling code.
    :keyword systematic_offset: (default: 2)
    :type systematic_offset: float
    :keyword para_dist: standard deviation of the parasitic capacitance of the Pixcap chip on a single sensor.
    :type para_dist: float
    :keyword systematic_dispersion: dispersion of the parasitic capacitance's between multiple PixCap65 chips
    :type systematic_dispersion: float
    :keyword fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :keyword fit_plot_pdf: PDF object to save the fit figures to.
    :keyword verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    :keyword cv_fit_plot_pdf: analog to `fit_plot_pdf` to activate the plotting for c-v- and depletion fits independent from
        the plotting for capacitance estimation fits. If this keyword argument is present also the `plot` arguments will
        be set automatically.
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :keyword apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword plot: indicates whether to plot the data. An output PDF object could be submitted here
         instead of an explicitly created one. (default: False)
    :type plot: bool
    """
    try:
        # this will require the usage of additonal arrays! But we could reuse the implementation for
        systematic_offset = kwargs.pop("systematic_offset", 2)
        systematic_key_args = kwargs.copy()
        systematic_key_args.pop("plot", False)
        systematic_key_args.pop("fit_plot_pdf", None)
        systematic_key_args.pop("output_pdf", None)
        systematic_key_args.pop("cv_fit_plot_pdf", None)

        ii, jj = index
        first_lower, first_upper = lower_boundary
        second_lower, second_upper = upper_boundary
        pixel_cap_data = cap_data[ii, jj, :]
        pixel_cap_error_data = cap_error_data[ii, jj, :]
        # could only perform the analysis for pixels with trustable measurements.
        if np.all(np.isfinite(pixel_cap_data)):
            storage.set_pixel(jj, ii)
            analyze_pixel_depletion(first_lower, first_upper, pixel_cap_data, pixel_cap_error_data, second_lower,
                                    second_upper, voltage_data, storage,
                                    fit_description_text=" for Pixel ({col},{row})".format(col=ii, row=jj),
                                    **kwargs)

            # needs to be fixed by the correct value!
            systematic_error = np.full_like(pixel_cap_data, fill_value=kwargs.get("para_dist", 1.e-18))
            dispersion_error = np.full_like(pixel_cap_data,
                                            fill_value=kwargs.get("systematic_dispersion",
                                                                  DISPERSION_PARASITIC_DEVIATION))

            second_systematic_result_storage = DepletionNumpyStore(tb.dtype_from_descr(DepletionData))
            third_systematic_result_storage = DepletionNumpyStore(tb.dtype_from_descr(DepletionData))
            fourth_systematic_result_storage = DepletionNumpyStore(tb.dtype_from_descr(DepletionData))
            assert isinstance(voltage_data, np.ndarray)
            __extract_systematic_depletion_effects((first_lower, first_upper),
                                                   (second_lower, second_upper), voltage_data,
                                                   pixel_cap_data, pixel_cap_error_data, systematic_error,
                                                   dispersion_error, systematic_offset,
                                                   second_systematic_result_storage,
                                                   third_systematic_result_storage,
                                                   fourth_systematic_result_storage,
                                                   systematic_key_args)
            dispersion_result = fourth_systematic_result_storage.table.Ubi_error[-1]
            if storage.depletion_reg is None:
                systematic_result = np.sqrt(np.abs(
                    storage.depletion_voltage[ii, jj] - second_systematic_result_storage.table.Ubi[-1]) ** 2 +
                                            third_systematic_result_storage.table.Ubi_error[-1] ** 2)
            else:
                systematic_result = np.sqrt(np.abs(
                    storage.depletion_voltage[ii, jj, storage.depletion_reg] -
                    second_systematic_result_storage.table.Ubi[-1]) ** 2 +
                                            third_systematic_result_storage.table.Ubi_error[
                                                -1] ** 2)

            storage.store_data("systematic", systematic_result)
            storage.store_data("dispersion", dispersion_result)
    except:
        print("Exception occured")
        import sys
        print(sys.exc_info())
        raise


def depletion_delegation_impl(cap_data, cap_error_data, first_lower, first_upper, second_lower, second_upper,
                              fit_result_storage: DepletionArrayStore, voltage_data: Union[np.ndarray, tb.CArray],
                              **kwargs) -> Optional[List]:
    """
    depletion_delegate_impl

    @author Dominik Fischer
    @date 2026-05-07
    last update: 2026-08-12

    Implementation of the pixel-wise depletion voltage estimation from a provided C-V characterization.
    For pixel the depletion voltage of the sensor is estimated from two fits to the C-V curve (more precise: 1/C^2)
    The intersection of this two straight lines then provides the depletion voltage.

    Also the systematic uncertainties of the depletion voltage are estimated.
    Here we account for the measured distribution of parasitic capacitance over the Pixcap Chip (its spread), for the
    dispersion of the parasitic capacitance between multiple sensors/chips and the particular choice of the fit range.
    For the additional fits to extract the systematic uncertainties no plotting of the fits is available.
    The Estimation of the different systematic uncertainties is performed by again fitting the voltage ranges but with
    either varied fit range or with the systematic uncertainties provided as the measurement error submitted to the fit
    algorithm. In the latter two cases the extracted fit errors on the depletion voltage is further propagated through
    the analysis.


    :param cap_data: capacitance data from the characterization.
    :param cap_error_data: uncertainties of the capacitance data from the characterization.
    :param first_lower: lower bound of the first fit section (high voltage limit) to estimate the depletion voltage of
        the pixel.
    :param first_upper: upper bound of the first fit section (high voltage limit) to estimate the depletion voltage of
        the pixel.
    :param second_lower: lower bound of the second fit section (low voltage limit) to estimate the depletion voltage of
        the pixel.
    :param second_upper: upper bound of the second fit section (low voltage limit) to estimate the depletion voltage of
        the pixel.
    :param fit_result_storage: DataStorage object for intermediate storage of fit results and depletion parameters
    :param voltage_data: data of the applied HV voltages
    :param kwargs: further keyword arguments to be propagated to functions/implementations.
    :keyword use_kafe2: boolean, False, indicates whether kafe2 is used for the fit.
    :keyword apply_contours: boolean, indicates whether to determine the contours and try to plot them.
    :keyword plot: boolean, False, indicates whether to plot the data. AN output PDF object could be submitted here
         instead of an explicitly created one.
    :keyword fit_plot_pdf: PDF object to save the fit figures to.
    :keyword verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    :keyword systematic_offset: enlargement in V for the fit range conditions applied for fitting. Needed to estiamte the
        systematic uncertainties by the fit range accurately. (default: 2)
    :keyword para_dist: standard deviation of the parasitic capacitance of the Pixcap chip on a single sensor.
    :keyword systematic_dispersion: spread of the dispersion of the parasitic capacitance between different Pixcap chip
        samples. This will induce a systematic effect on the accuracy of the capacitance's and the depletion voltage of
        the investigated sensor.
    :keyword systematic_offset: (default: 2)
    :type systematic_offset: float
    :keyword fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :keyword cv_fit_plot_pdf: analog to `fit_plot_pdf` to activate the plotting for c-v- and depletion fits independent from
        the plotting for capacitance estimation fits. If this keyword argument is present also the `plot` arguments will
        be set automatically.
    :return: optionally list of figure holder objects; the list elements could also be None themselves.
    """
    kwargs.setdefault("verbose", False)
    # here we have the largest optimisation potential by using multiprocessing, but it would be necessary to share
    # the data storage class among the different processes. We could not be sure that there wont be a data-race.

    lower_hv_boundary = (first_lower, first_upper)
    upper_hv_boundary = (second_lower, second_upper)
    iterator = tqdm(np.ndindex(GENERAL_PIXCAP_SHAPE), total=1600)

    # need to make sure that we will not leake any locks or other synchronization primitives!
    # in this branch, there seems to be no leakage of semaphore objects!
    # this is the old standard mode
    [__depletion_iterator_implementation(index, cap_data, cap_error_data,
                                         lower_hv_boundary, upper_hv_boundary, voltage_data,
                                         fit_result_storage, **kwargs) for index in iterator]


# For performace optimisation we must make sure that the implementation is thread-safe.
def analyze_pixel_depletion(first_lower, first_upper,
                            pixel_cap_data: np.ndarray,
                            pixel_cap_error_data: np.ndarray, second_lower, second_upper,
                            voltage_data: TABLES_LEAF_COMPAT_TYPE,
                            result: DepletionDataStore, **kwargs):
    """
    analyze_pixel_depletion

    @author Dominik Fischer
    @date 2026-05-07
    last update: 2026-08-12

    Helper function to perform the investigation of the depletion voltage for a single pixel.
    The function does not need to know which pixel is currently investigated which enables the usage of
    for distributions of the capacitance over the whole sensor.
    After performing the 'fits' the data will be stored in the provided storage object.

    This helper function does not make use of locks for synchronization of processes or threads.

    :param first_upper: upper limit of the first fit range for the high voltage limit of the capacitance behaviour
        to estimate the depletion voltage of the pixel.
    :param first_lower: lower limit of the first fit range for the high voltage limit of the capacitance behaviour
        to estimate the depletion voltage of the pixel.
    :param second_upper: upper limit of the second fit range for the low voltage limit of the capacitance behaviour
        to estimate the depletion voltage of the pixel.
    :param second_lower: lower limit of the second fit range for the low voltage limit of the capacitance behaviour
        to estimate the depletion voltage of the pixel.
    :param pixel_cap_data: capacitance data for one particular pixel on the sensor
    :param pixel_cap_error_data: uncertainties of the capacitance data for one particular pixel on the sensor.
    :param voltage_data: array of the biasing HV voltages (with the correct sign)
    :param result: data store container to write the results back
    :param kwargs: further keyword arguments for fitting and output.
    :keyword fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :keyword fit_plot_pdf: PDF object to save the fit figures to.
    :keyword verbose: boolean, indicating whether to use verbose output of the depletion voltages.
    :keyword cv_fit_plot_pdf: analog to `fit_plot_pdf` to activate the plotting for c-v- and depletion fits independent from
        the plotting for capacitance estimation fits. If this keyword argument is present also the `plot` arguments will
        be set automatically.
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :keyword apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword plot: indicates whether to plot the data. An output PDF object could be submitted here
         instead of an explicitly created one. (default: False)
    :type plot: bool
    """
    # extract the additional parameters for advanced fitting procedures
    verbose_output = kwargs.pop("verbose", False)
    if kwargs.get("enhanced", False):
        raise KeyError("Enhanced mode not implemented yet")

    # perhaps the implemenation could be improved in general if the data is first collected by a structured numpy array.
    (dep_voltage_2, dep_voltage_error_2, first_dep_errors, first_dep_parameters, first_covariance,
     second_dep_errors, second_dep_parameters, second_covariance) = _analyze_pixel_depletion_fit(
        pixel_cap_data, pixel_cap_error_data, voltage_data, first_lower, first_upper, second_lower, second_upper,
        **kwargs)
    if verbose_output:
        print("The depletion voltage is {voltage}+-{error}".format(voltage=dep_voltage_2,
                                                                   error=dep_voltage_error_2))

    result.store_data("depletion", dep_voltage_2)
    result.store_data("depletion_error", dep_voltage_error_2)
    result.store_data("fit_result_first", first_dep_parameters)
    result.store_data("fit_result_second", second_dep_parameters)
    result.store_data("fit_error_first", first_dep_errors)
    result.store_data("fit_error_second", second_dep_errors)
    result.store_data("fit_result_cov_first", first_covariance)
    result.store_data("fit_result_cov_second", second_covariance)
    result.flush_data()


def _analyze_pixel_depletion_fit(pixel_cap_data: np.ndarray,
                                 pixel_cap_error_data: np.ndarray,
                                 voltage_data: TABLES_LEAF_COMPAT_TYPE,
                                 first_lower, first_upper, second_lower, second_upper, **kwargs) -> tuple:
    """
    _analyze_pixel_depletion_fit

    @author Dominik Fischer
    @date 2026-08-11

    Helper function to analyze the properties of the depletion region and their dependence on the applied bias voltage (designed for reverse bias only).
    This helper function does not use any locks for synchronization.
    Here the analysis is handled for a single pixel or for average values over a whole sensor.

    The depletion voltage is estimated by two fits.
    One at high voltages and the other at low voltages in the asymptotic behaviour of the C-V characterization.
    The fits are performed either by `iminuit` or by `kafe2` depending on the choice of keyword arguments provided.
    The depletion voltage is then determined as the intersection these two straight lines.

    :param pixel_cap_data: array-like of the pixel-capacitance's measured for the given bias voltages and one
            particular pixel.
    :type pixel_cap_data: numpy.ndarray
    :param pixel_cap_error_data: array-like of the pixel capacitances' (statistical) uncertainties for the given
            bias voltages for one particular pixel.
    :type pixel_cap_error_data: numpy.ndarray
    :param voltage_data: array-like of applied bias voltages for the measurement.
    :param first_lower: lower bound for the fitting range in the high voltage limit of the C-V-curve to estimate the
            depletion voltage.
    :param first_upper: upper bound for the fitting range in the high voltage limit of the C-V-curve to estimate the
            depletion voltage.
    :param second_lower: lower bound for the fitting range in the low voltage limit of the C-V-curve to estimate the
            depletion voltage.
    :param second_upper: upper bound for the fitting range in the low voltage limit of the C-V-curve to estimate the
            depletion voltage.
    :keyword cv_fit_plot_pdf: analog to `fit_plot_pdf` to activate the plotting for c-v- and depletion fits independent from
        the plotting for capacitance estimation fits. If this keyword argument is present also the `plot` arguments will
        be set automatically.
    :keyword fit_plot_pdf: pdf object to write the all the fits control figures to.
    :keyword enhanced: Not clear what this key really does. It should not be used at all.
    :type enhanced: bool
    :keyword fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :keyword apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword plot: indicates whether to plot the data. An output PDF object could be submitted here
         instead of an explicitly created one. (default: False)
    :type plot: bool
    :keyword fit_plot_pdf: PDF object to save the fit figures to.
    :return: tuple (Udep, error of Udep, first fits parameters, first fits parameter errors, covariance matrix of the first fit, second fits parameters, second fits parameter errors, covariance matrix for the second fit). If the `enhanced` keyword is present the return type/values might differ.
    :rtype: tuple
    """
    cv_fit_pdf = kwargs.pop("cv_fit_plot_pdf", None)
    if cv_fit_pdf:
        kwargs["fit_plot_pdf"] = cv_fit_pdf
        kwargs["plot"] = True
    # extract the information about the depletion voltage
    assert not isinstance(voltage_data, tb.Leaf)
    first_section_upper_mask = voltage_data <= first_upper
    first_section_lower_mask = voltage_data >= first_lower
    first_section_mask = np.logical_and(first_section_upper_mask, first_section_lower_mask)

    second_section_upper_mask = voltage_data <= second_upper
    second_section_lower_mask = voltage_data >= second_lower
    second_section_mask = np.logical_and(second_section_upper_mask, second_section_lower_mask)

    # extract and select the usable voltage and capacitance data for the two fit ranges.
    effective_capacitance_data = np.reciprocal(pixel_cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 2
    effective_capacitance_error_data = np.reciprocal(
        pixel_cap_data * CAPACITANCE_CONVERSION_FACTOR) ** 3 * pixel_cap_error_data * CAPACITANCE_CONVERSION_FACTOR if np.all(
        np.isfinite(pixel_cap_error_data)) else pixel_cap_error_data
    first_voltage_data = voltage_data[first_section_mask]
    second_voltage_data = voltage_data[second_section_mask]
    first_cap_data = effective_capacitance_data[first_section_mask]
    second_cap_data = effective_capacitance_data[second_section_mask]
    first_cap_error_data = effective_capacitance_error_data[first_section_mask]
    second_cap_error_data = effective_capacitance_error_data[second_section_mask]

    # perform fits to the two boundary regions specified to estimate the two distinct behaviours.
    is_first = True
    try:
        first_dep_cov, first_dep_errors, first_dep_parameters = get_depletion_fit(first_cap_data, first_cap_error_data,
                                                                                  first_voltage_data, "First",
                                                                                  **kwargs)

        is_first = False

        second_dep_cov, second_dep_errors, second_dep_parameters = get_depletion_fit(second_cap_data,
                                                                                     second_cap_error_data,
                                                                                     second_voltage_data, "Second",
                                                                                     **kwargs)
    except AssertionError:
        print(first_lower, first_upper, second_lower, second_upper)
        print(is_first)
        if kwargs.get("enhanced", False):
            return np.nan, np.nan, np.nan, np.nan, \
                np.nan, np.nan, np.nan, np.nan, \
                np.nan, np.nan
        else:
            raise

    # estimate the depletion voltage
    d = first_dep_parameters[1]
    b = second_dep_parameters[1]
    c = first_dep_parameters[0]
    a = second_dep_parameters[0]
    dep_voltage_2 = (d - b) / (a - c)

    dep_voltage_jacobian = np.array([
        (b - d) / ((a - c) ** 2),
        1 / (c - a),
        (d - b) / ((a - c) ** 2),
        1 / (a - c)
    ])

    # combine both cov matrices into a single one:
    full_cov_2 = np.zeros((4, 4))  # cross correlations between the two fits are not known
    assert full_cov_2 is not None
    assert first_dep_cov is not None
    assert second_dep_cov is not None
    full_cov_2[:2, :2] = second_dep_cov
    full_cov_2[2:, 2:] = second_dep_cov

    # perform the full error calculation with matrix methods:
    dep_voltage_error_2 = np.sqrt(dep_voltage_jacobian @ full_cov_2 @ dep_voltage_jacobian)

    if kwargs.get("enhanced", False):
        return dep_voltage_2, dep_voltage_error_2, first_dep_parameters[0], first_dep_errors[0], first_dep_parameters[
            1], first_dep_errors[1], second_dep_parameters[0], second_dep_errors[0], second_dep_parameters[1], \
            second_dep_errors[1]

    return dep_voltage_2, dep_voltage_error_2, first_dep_errors, first_dep_parameters, first_dep_cov, second_dep_errors, second_dep_parameters, second_dep_cov


def get_depletion_fit(cap_data: np.ndarray, cap_error_data: np.ndarray, voltage_data: np.ndarray,
                      fit_reference, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    get_depletion_fit

    @author Dominik Fischer
    @date 2026-05-07

    last updated: 2026-08-11

    Helper function to perform the necessary fits to estimate the depletion voltage from linear fits to the capacitance
    characterization.
    For performing the fits either the `iminuit` or the `kafe2` framework are used with correct estimation of the
    parameter uncertainties.
    This function does not use locks for synchronization.


    :param cap_data: capacitance data from the characterization for this fit section.
    :param cap_error_data: uncertainties of the capacitance data from the characterization for this fit section.
    :param voltage_data: data of the applied HV voltages for this fit section.
    :param fit_reference: identifying the fit section for which the fit is performed.
    :keyword fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :keyword use_kafe2: indicates whether kafe2 is used for the fit. (default: False)
    :type use_kafe2: bool
    :keyword apply_contours: indicates whether to determine the contours and try to plot them. (default: False)
    :type apply_contours: bool
    :keyword plot: indicates whether to plot the data. An output PDF object could be submitted here
         instead of an explicitly created one. (default: False)
    :type plot: bool
    :keyword fit_plot_pdf: PDF object to save the fit figures to.
    :return: covariance_matrix, fit parameter errors, fit parameter values
    """
    # extract the additional parameters for advanced fitting procedures
    use_kafe2 = kwargs.pop("use_kafe2", False)
    apply_contours = kwargs.pop("apply_contours", False)
    plot = kwargs.pop("plot", False)
    output_pdf = kwargs.pop("fit_plot_pdf", None)
    fit_description_text = kwargs.pop("fit_description_text", "")
    if np.all(np.isfinite(cap_error_data)):
        if use_kafe2:
            # perform the fit
            from kafe2 import XYContainer, XYFit
            data_container = XYContainer(voltage_data, cap_data)
            data_container.add_error(axis='y', err_val=cap_error_data)
            m = XYFit(data_container, model_function=depletion_model)
            m.do_fit()

            # extract the fit parameters
            assert m.did_fit
            first_dep_parameters = np.array([m.parameter_values[0], m.parameter_values[1]])
            first_dep_errors = np.array([m.parameter_errors[0], m.parameter_errors[1]])
            first_dep_cov = m.parameter_cov_mat
            assert first_dep_cov is not None
            if plot:
                handle_kafe2_advanced_options(m, apply_contours, "U in V", "\\frac{{1}}{{C^2}}",
                                              "{} Fit{}".format(fit_reference, fit_description_text),
                                              output_pdf,
                                              "{} Contour{}".format(fit_reference, fit_description_text))

        else:
            # perform the fit
            from iminuit import Minuit
            # noinspection PyProtectedMember
            from iminuit.cost import LeastSquares, Model
            assert isinstance(depletion_model, Model)
            cost = LeastSquares(voltage_data, cap_data, cap_error_data, depletion_model)
            m = Minuit(cost, a=1, b=0)
            m.migrad()
            m.hesse()

            # extract the parameters
            first_dep_parameters = np.array([m.values['a'], m.values['b']])
            first_dep_errors = np.array([m.errors['a'], m.errors['b']])
            first_dep_cov = m.covariance
            if plot:
                from pixcap65.plotting_util.threaded_plotting import threading_lock
                with threading_lock:
                    handle_minuit_advanced_options(m, apply_contours, "$U$ / \\unit{{\\volt}}",
                                                   "$\\frac{{1}}{{C^2}}$ / \\unit{{\\per\\femto\\farad\\squared}}",
                                                   "{} Fit{}".format(fit_reference, fit_description_text),
                                                   output_pdf,
                                                   "{} Contour{}".format(fit_reference, fit_description_text))
            try:
                assert first_dep_cov is not None
            except AssertionError:
                print(m.valid)
                print(m.fmin)
                print(voltage_data)
                from matplotlib import pyplot as plt
                plt.close('all')
                plt.errorbar(voltage_data, cap_data, yerr=cap_error_data, )
                plt.show()
                print(voltage_data.shape)
                print(cap_data.shape)
                print(cap_error_data.shape)
                raise
    else:
        first_result = np.polyfit(voltage_data, cap_data, deg=1, cov=True)

        # extract the parameters and fit results.
        first_dep_parameters = np.asarray(first_result[0])
        first_dep_cov = np.asarray(first_result[1])
        first_dep_errors = np.sqrt(np.diag(first_dep_cov))
        assert first_dep_cov is not None
    assert isinstance(first_dep_parameters, np.ndarray)
    assert isinstance(first_dep_errors, np.ndarray)
    assert isinstance(first_dep_cov, np.ndarray)
    return first_dep_cov, first_dep_errors, first_dep_parameters
