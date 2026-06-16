# ----------------------------------------------------------
#  Copyright (c) 2026.
#   All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------
import contextvars
import queue
import threading

import pixcap65.plotting as plotting

thread_storage = queue.SimpleQueue()
threading_lock = contextvars.ContextVar("threading_lock").get(threading.RLock())


# for plotting we could do this as a file could be opened in reading mode multiple times, the only issue may be the backend.

def plot_data(interpreted_data, base_path=None, suffix="general_data", use_group=False, lock=threading_lock, **kwargs):
    """
    plot_data

    Helper function to graphical present/plot the analysis results of a simple pixel capacitance scan.
    The plotting is only performed for the pixels which contribute a usable capacitance measurement.
    In Addition to the fits for estimating the capacitance also the capacitance distribution and frequency is plotted.
    The name of the resulting PDF is derived from the file name with the measurement data.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    :key use_corrected: boolean, False, indicating whether to use the corrected capacitance for plotting.
    :key exclude_test_cap: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key extract_pixel: iterable of pixel positions on the grid to extract the figures from.
    :key distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor.
    """
    with lock:
        thread = threading.Thread(target=plotting.plot_data, args=(interpreted_data, base_path, suffix, use_group), kwargs=kwargs)
        thread.start()
        thread_storage.put(thread)
    return thread


def plot_inter_pix_data(interpreted_data, base_path=None, suffix="general_inter_pix_data", use_group=False,
                    total_path=None, total_data=None, lock=threading_lock, **kwargs):
    """
    plot_data

    Helper function to graphical present/plot the analysis results of an inter-pixel capacitance scan.
    The plotting is only performed for the pixels which contribute a usable capacitance measurement.
    In Addition to the fits for estimating the capacitance also the capacitance distribution and frequency is plotted.
    The name of the resulting PDF is derived from the file name with the measurement data.

    Besides the naming it is not just plotting but also a bit of analysis as the distribution of the capacitance
    over the pixel and in general for all three currents is investigated, as well.
    The current-frequency dependency will plot for each scanned pixel with finite currents.
    Also, the capacitance distribution over the whole sensor and the frequency of capacitance values are plotted for all
    three current measurements. Besides the naming of the plots the capacitance are not directly the inter-pixel or
    total-pixel capacitance values.

    The modelling process is automatically corrected to use the bare capacitance if corrected capacitance values are
    supplied. If the correction is not applied by the functions from the analysis module then this may not work.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    :param total_data: path to the hdf file which holds the analyzed data for the total capacitance scan.
    :param total_path: hdf group path inside the hdf file containing the total cap analysis results.
    :key exclude_test_cap: boolean, whether to exclude the test capacitator row from the histograms.
    :key hist_bins: integer, number of bins to use for the histogram.
    :key mask_pixel: iterable of pixel positions on the grid to ignore for evaluations.
    :key extract_pixel: iterable of pixel positions on the grid to extract the figures from.
    """
    with lock:
        if total_path is not None:
            kwargs["total_path"] = total_path
        if total_data is not None:
            kwargs["total_data"] = total_data
        thread = threading.Thread(target=plotting.plot_inter_pix_data, args=(interpreted_data, base_path, suffix, use_group,), kwargs=kwargs)
        thread.start()
        thread_storage.put(thread)
    return thread


def plot_bias_data(interpreted_data, base_path=None, suffix="bias_curve", use_group=False, lock=threading_lock, **kwargs):
    """
    plot_bias_data

    Plot the data acquired for the pixel-diodes I-V characterization.

    :param interpreted_data: path to the hdf file which holds the raw data.
    :param base_path: path within the files hierarchy for the base group.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    """
    with lock:
        thread = threading.Thread(target=plotting.plot_bias_data, args=(interpreted_data, base_path, suffix, use_group), kwargs=kwargs)
        thread.start()
        thread_storage.put(thread)
    return thread


def plot_cv_data(interpreted_data, base_path=None, suffix="C_V_characteristic", use_group=False, lock=threading_lock, **kwargs):
    """
    plot_cv_data

    Plot the results of the C-V characterization of the scanned pixels.
    To achieve this we need the different c-v-data.
    Then the C-V curve is plotted for every pixel.
    If requested also fits to the boundary regions of the c-v-curve are performed to determine
    the depletion voltage of the pixel.
    To do so, two fit ranges for the two boundaries with physically distinct behaviour needs to be supplied.


    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix:  additional suffix to use for naming the PDF containing the plots.
    :param use_group:   boolean, whether to append the group name of the measurements to the PDF name.
    :key verbose: boolean, indicating whether to use verbose output for depletion voltages
    :key distribution: boolean, indicating whether also the capacitance distribution of the whole sensor
        should be investigated.
    """
    with lock:
        thread = threading.Thread(target=plotting.plot_cv_data, args=(interpreted_data, base_path, suffix, use_group), kwargs=kwargs)
        thread.start()
        thread_storage.put(thread)
    return thread


def plot_combined_data(interpreted_data, base_path=None, suffix="combined_bias_cv_curve", use_group=False,
                       lock=threading_lock, **kwargs):
    """
    plot_combined_data

    Plot the data acquired for the pixel-diodes I-V characterization and the C-V characterization of the pixels.
    Plot the results of the C-V characterization of the scanned pixels.
    To achieve this we need the different c-v-data.
    Then the C-V curve is plotted for every pixel.
    If requested also fits to the boundary regions of the c-v-curve are performed to determine the
    depletion voltage of the pixel.
    To do so, two fit ranges for the two boundaries with physically distinct behaviour needs to be supplied.

    :param interpreted_data: path to the hdf file which holds the raw data and the analysis results.
    :param base_path: path to the base group in the hdf files hierarchy.
    :param suffix: additional suffix to use for naming the PDF containing the plots.
    :param use_group: boolean, whether to append the group name of the measurements to the PDF name.
    :key use_corrected: boolean, whether to use corrected data
    :key verbose: boolean, indicating whether to use verbose output for depletion voltages
    :key distribution: boolean, indicating whether also the capacitance distribution of the whole sensor
        should be investigated.
    """
    with lock:
        thread = threading.Thread(target=plotting.plot_combined_data, args=(interpreted_data, base_path, suffix, use_group), kwargs=kwargs)
        thread.start()
        thread_storage.put(thread)
    return thread

def joint_plotting(timeout=None, fetch_timeout=None, lock=threading_lock):
    with lock:
        while not thread_storage.empty():
            try:
                current_thread = thread_storage.get(timeout=fetch_timeout)
                current_thread.join(timeout=timeout)
            except queue.Empty:
                print("Possible timeout on fetching remaing thread encountered")
                break



