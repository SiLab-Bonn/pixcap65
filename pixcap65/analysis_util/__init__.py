import numpy as np
import tables as tb

from pixcap65.analysis_util.utility import GENERAL_PIXCAP_SHAPE
from pixcap65.analysis_util.utility import TABLES_ARRAY_TYPE, TABLES_TABLE_TYPE, GENERAL_PIXCAP_SHAPE, \
    transform_covariance


def analyze_data_delegate(file: tb.File, group: tb.Group, current_hist: TABLES_ARRAY_TYPE,
                          scan_parameters: TABLES_TABLE_TYPE, **kwargs):
    """
    analyze_data_delegate

    @author: Dominik Fischer
    last update: 2026-08-12

    Implementation of the simple analysis strategy for the capacitance measurement of a pixel sensor.
    For determination of the capacitance values a simplified linear least-squares fit is used.
    The covariance information from the simplified fit is not trust worthy, as the even when the uncertainties/weights
    of the data are trust-worthy the covariance matrix from the fit is rescaled such that the Chi^2 / d.o.f. is 1.
    Therefore, the covariance matrix of these fits is not suitable to estimate the uncertainties of the capacitances.

    :param file: h5 file object containing the data to be analysed.
    :param group: hierarchy group of the opened hdf file to write the analysis results to.
    :param current_hist: 2D-Array for the current data to fit the model to.
    :param scan_parameters: table of the scan parameters used for each measurement point within the frequency and/or
        voltage scan.
    :key current_error_hist: 2D-Array for the errors of the current data. This keyword argument must be present
        for the advanced analysis strategy. (This argument has no effect by the current implementation).
    """
    # create array like objects to temporarily save the analysis results
    cap_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)  # capacitance for each pixel
    cap_error_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    leak_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    leak_error_hist = np.full(shape=GENERAL_PIXCAP_SHAPE, fill_value=np.nan)
    fit_cov = np.full(shape=(40, 40, 2, 2), fill_value=np.nan)

    weights = kwargs.pop("current_error_hist", None)
    if weights is not None:
        weights = np.reciprocal(np.asarray(weights) ** 2)

    # Fit pixel data in order to extract capacitance for each pixel
    for col in range(current_hist.shape[0]):
        for row in range(current_hist.shape[1]):
            if np.isfinite(current_hist[col, row, 0]):
                mask = np.isfinite(current_hist[col, row, :])
                res = np.polyfit(scan_parameters['frequency'][mask], current_hist[col, row][mask], deg=1, cov=True)
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

    # Store capacitance values
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
                                    name=kwargs.get("cov_name", "HistFitCov"),
                                    title=kwargs.get("cov_title", 'Fit Covariance Matrix'),
                                    obj=fit_cov, filters=tb.Filters(complib='blosc',
                                                                    complevel=5,
                                                                    fletcher32=False)
                                    )
    temp_array.attrs["Units"] = "{{F^2, F nA},{nA F, nA^2}}"
    temp_array.flush()
