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
from tables import NaturalNameWarning
from typing import Optional, List
from warnings import warn, filterwarnings

from pixcap65.analysis_util.constants import BOUNDARY_TYPE, RANDOM_SEED
from pixcap65.analysis_util.general import _analyze_data
from pixcap65.analysis_util.multiprocessing import get_manager_keywords
from pixcap65.analysis_util.utility import GENERAL_PIXCAP_SHAPE, FARAD_CONVERSION_FACTOR, CURRENT_CONVERSION_FACTOR, \
    GLOBAL_FILTERS
from pixcap65.analysis_util.utility import TABLES_ARRAY_TYPE, TABLES_TABLE_TYPE, GENERAL_PIXCAP_SHAPE, \
    transform_covariance

filterwarnings("ignore", category=NaturalNameWarning)
try:
    from iminuit.warnings import IMinuitWarning

    filterwarnings("ignore", category=IMinuitWarning)
except ImportError:
    # no action required if iminuit is not present
    pass
filterwarnings("ignore", category=np.exceptions.RankWarning, module="jacobi")


def analyze_data(raw_data, base_path=None, is_advanced=False, is_cv=False,
                 first_boundaries: Optional[BOUNDARY_TYPE] = None,
                 second_boundaries: Optional[BOUNDARY_TYPE] = None,
                 is_inter_pixel=False, **kwargs) -> Optional[List]:
    """
    analyze_data

    @author: Dominik Fischer
    @date: 2026-08-12 (originally earlier)
    last update: 2026-08-12

    Implementation of the analysis strategy for the capacitance measurement of a pixel sensor.
    But keep in mind that this function serves as a wrapper to handle file access and modification around
    the actual analysis implementation.
    The capacitance of the pixels are measured and investigated individually. For determination of the
    capacitance values either a linear fit or non-linear least square fit algorithms are used.
    Depending on the choice of the ´is_advanced` parameter non-linear techniques are used.
    In this case either 'kafe2' or 'iminuit' are used for the least-squares minimization depending on the choice
    of parameters.
    When using the advanced least-squares procedure the fit results will be plotted to verify the convergence of the
    fit.
    Thus, it is possible to use this wrapper to handle the PDF file to save fit-plot figures to instead of doing this
    individually for each analysis call.

    Afterwards, it is possible to directly correct the results for the capacitance by the connection and the
    measurement circuit.

    In Addition, there is the special case of an inter-pixel capacitance measurement.
    If the data to be analyzed comes from such a measurement this needs to be specified.
    Thus, in this case it will be checked which of the total current or the two inter-pix current data sets exist.
    The analysis will be done for each existing data set.
    Combinations with a C-V-Characterization might still be an issue.

    For measurements of the C-V-Characteristic of a sensor, the fits will be applied for every bias voltage measured.
    If additional fit boundaries are supplied, it will be tried to also determine the depletion behaviour of the
    pixel sensor including the depletion voltage and the corresponding capacitance.
    When doing this, also the doping profile and some intrinsic properties could be investigated.
    Currently, the C-V characterization for the inter-pixel measurements is not implemented yet.

    For synchronization of the access to the data files a lock is used.
    It is strongly recommended to provide this locking object by a multiprocessing.Manager instance to transmit to objects handled by another process.


    :param raw_data: path to the hdf file containing the raw data.
    :param base_path: path to the base group to look for the data.
    :param is_advanced: boolean indicating if the advanced analysis strategy should be used
        or not (may require additional keyword arguments)
    :param is_cv: boolean, indicates whether this is a C-V characterization.
    :param first_boundaries: tuple of bounds for the high voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will only be estimated if this argument is provided.
    :param second_boundaries: tuple of bounds for the low voltage limit of the capacitance behaviour to estimate
        the depletion voltage of the pixel. Depletion voltage will be estimated if this argument is provided.
    :param is_inter_pixel: boolean, False, indicating whether the measurement to be analyzed is an inter-pixel
        capacitance measurement. In this case more fits will be applied, adjusted to the specific structure of this
        problem
    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file. IT IS STRONGLY RECOMMENDED TO EXPLICITLY SUPPLY A LOCK
        for synchronization.
    :keyword apply_correction: boolean, indicates whether the measured capacitance should be
        corrected immediately; Will require the presence of further arguments as information about
        the parasitic capacitance needs to be submitted. (data corrected for parasitic capacitances of PixCap65,
        default: False)
    :type apply_correction: bool
    :keyword fit_plot_pdf_name:  Name of the PDF file to save fitting figures from the advanced procedures to.
        (Only used for the advanced procedure)
    :type fit_plot_pdf_name:  str
    :keyword use_kafe2: boolean, indicates whether kafe2 is used for the fit. (default: False) (Only used for the advanced procedure)
    :type use_kafe2:  bool
    :keyword plot: indicates whether to plot the data. An output PDF object could be submitted here instead of an explicitly created one. (Default: False) (Only used for the advanced procedure)
    :type plot: bool
    :keyword apply_contour: boolean, indicates whether to determine the contours and try to plot them. (default: False)
        (Only used for the advanced procedure or 'apply_contours': there might be some inconsistencies.)
    :type apply_contour: bool
    :keyword fit_plot_pdf: PDF object to save the fit figures to.
    :keyword distribution: boolean, indicating whether to investigate the capacitance distribution over the whole sensor. (default: False)
    :type distribution: bool
    :keyword total_cap_file:
    :keyword total_cap_group:
    :keyword in_cap_file:
    :keyword in_cap_group:
    :keyword inter_pix_id: identifier of the kind of inter-pixel-capacitance measurement to be processed (default: 18000)
    :type inter_pix_id: int
    :keyword full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used. (Only used for the advanced procedure)
    :type full_model: bool
    :keyword bare_file: hdf file containing the measurements and investigation of a bare pix cap sample to obtain
        information about intrinsic and parasitic capacitance. (Only required for the correction procedure, but in
        this case it must be present)
    :type bare_file: str
    :keyword bare_hdf_path: hdf files hierarchy path to the group containing the bare pix cap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present) There might be some inconsistencies when using 'bare_path'.
    :type bare_hdf_path: str
    :keyword chip_group_name: HDF files hierarchy group (path to it) with the data/specifications of the pixels on the current sensor.
        (Will only be usd if the depletion behaviour is investigated)
    :type chip_group_name: str
    :keyword address: address of the socket of the multiprocessing.Manager object we want to connect to.
    :keyword authkey: authentication key necessary to connect to the socket. (It is recommended not to use this parameter as
        it is not pickable)
    :keyword full_model: boolean, True, indicates whether the full model for extended frequency range is to be used.
        Otherwise, the linear model is used. (Only used for the advanced procedure)
    :keyword output_pdf: PdfPages object, to save the fit plot figures to (will override the plot object if provided)
    :keyword hist_res_key: name/identifiert of the array/table which contains the on-resistance estimators if present.
    :keyword test_cap_exclusion: whether to exclude row 0 completely. (default: False)
    :type test_cap_exclusion: bool
    :keyword mask_pixel: array/iterable of tuple of pixel positions to be masked and therefore ignored for evaluation. [array-like]
    :keyword mask_lower: threshold to mask all pixels below this value.
    :type mask_lower: float
    :keyword mask_upper: threshold to mask all pixels above this value.
    :type mask_upper: float
    :keyword hist_bins: integer, number of bins to use for the histogram. (default: 50)
    :type hist_bins: int
    :keyword chip_group: HDF files hierarchy group (object) with the data/specifications of the pixels on the current sensor.
        (Will only be usd if the depletion behaviour is investigated)
    :keyword fit_description_text: text describing the fit performed for usage within the plot handler of the fits.
    :type fit_description_text: str
    :returns: optional list of figure holder objects to be processed later on.
    """
    manager_kargs = get_manager_keywords(**kwargs)
    if "lock" not in kwargs:
        from examples.mp_analysis import processed_manager
        with processed_manager(**manager_kargs) as (manager, lock):
            kwargs["lock"] = lock
            manager_kargs.update(address=manager.address)
            kwargs.update(**manager_kargs)
            _analyze_data(raw_data, base_path, is_advanced, is_cv, first_boundaries, second_boundaries, is_inter_pixel,
                          **kwargs)
        return
    lock = kwargs.get("lock", None)
    if lock is not None:
        print(type(lock))
        if hasattr(lock, "_token"):
            print("Token of the proxy object:", lock._token)
    correction_key_value = kwargs.pop("use_corrected", None)
    if correction_key_value is not None:
        msg = "keyword argument `use_corrected` is deprecated, use `apply_correction` instead. If `apply_correction` is also present this value will take precedence, otherwise the provided value will be used. This keyword argument will be removed in the future."
        warn(msg, DeprecationWarning, stacklevel=2)
        kwargs.setdefault("apply_correction", correction_key_value)

    # handle the additional PDF file in case of plotting enabled.
    fit_plot_pdf_name = kwargs.pop('fit_plot_pdf_name', None)
    if "plot" in kwargs and kwargs["plot"] and fit_plot_pdf_name is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        assert "fit_plot_pdf_name" not in kwargs
        with PdfPages(fit_plot_pdf_name) as pdf:
            kwargs['fit_plot_pdf'] = pdf
            _analyze_data(raw_data, base_path, is_advanced, is_cv, first_boundaries,
                          second_boundaries, is_inter_pixel, **kwargs)
            return

    _analyze_data(raw_data, base_path, is_advanced, is_cv, first_boundaries, second_boundaries, is_inter_pixel, **kwargs)
