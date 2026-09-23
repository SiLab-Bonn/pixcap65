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
Analysis and plotting script evaluate all the data taking from the beginning!
"""
from os.path import join as hdf

import logging
import multiprocessing as mp
import numpy as np
import tables as tb
import threading
from matplotlib.backends.backend_pdf import PdfPages

import pixcap65.plotting_util.threaded_plotting as threaded_plotting
from examples import data_constants as data_constants
from examples.data_constants import E1_SCAN_FILE, E1_2_SCAN_FILE, R13_2_SCAN_FILE, R11_SCAN_FILE
from examples.data_constants import X1_SCAN_2_FILE, X2_SCAN_2_FILE, X2_SCAN_FILE
from examples.data_constants import X4_SCAN_FILE, X5_SCAN_FILE, X6_SCAN_FILE, X7_SCAN_FILE
from pixcap65.analysis_util import analyze_data
from pixcap65.analysis_util.delegation.distribution import analyze_capacitance_distribution
from pixcap65.analysis_util.utility import get_base_group
from pixcap65.plotting_util import plot_data, plot_inter_pix_data, plot_bias_data, plot_cv_data, plot_combined_data
from pixcap65.utility import synchronized_process_open_file
from pixcap65.utility.homogenize_plots import set_params
from pixcap65.utility.tables_util import get_group_attribute

INTER_UNBIASED_MODEL = 'inter_unbiased_full_model'
UNBIASED_MODEL = 'unbiased_full_model'
INTER_MIX = "inter-mix"
UNBIASED = 'unbiased_full'
INTER_UNBIASED = 'inter_unbiased_full'
LOG_FINISHED_ANALYSIS = "Finished the Analysis for"
LOG_CV_ANALYSIS = "CV Analysis for"
CV_DATA_FOR_ = "CV Data for {}"
SECOND_LABEL = " Second Try."
AUTHKEY_OUTPUT = "Fetch new authkey:"

logger = logging.getLogger(__name__)


# define the analysis handling
def bare_analysis_handler(tb_lock):
    """
    Handles the plotting of the measurements with the sensor-less (bare) sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Then the Inter-Pixel-Capacitance analysis is investigated.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.

    The applied biasing states are:

    * 0V (unbiased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    with synchronized_process_open_file("packaged/Reference_Bare_renewed.h5", 'a', lock=tb_lock) as h5_file:
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_31_renew", newname="unbiased_31_renew_full_model",
                          overwrite=True, recursive=True)

    with synchronized_process_open_file("packaged/data/Bare_Sample_05_Extended_Scan.h5", 'a', lock=tb_lock) as h5_file:
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_model",
                          overwrite=True, recursive=True)
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_kafe2",
                          overwrite=True, recursive=True)
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_extended",
                          overwrite=True, recursive=True)
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_quad",
                          overwrite=True, recursive=True)

    analyze_data(raw_data="packaged/Reference_Bare_renewed.h5", base_path="Reference/Bare/unbiased_31_renew",
                 is_advanced=True, full_model=False, lock=tb_lock)
    analyze_data(raw_data="packaged/Reference_Bare_renewed.h5",
                 base_path="Reference/Bare/unbiased_31_renew_full_model", is_advanced=True,
                 full_model=True, lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5", base_path="Reference/Bare/unbiased_full",
                 is_advanced=True, full_model=False, lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_model", is_advanced=True,
                 full_model=True, lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_kafe2", is_advanced=True,
                 full_model=True, lock=tb_lock, use_kafe2=True)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_extended", is_advanced=True,
                 full_model='extended', lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_quad", is_advanced=True,
                 full_model='quad', lock=tb_lock)
    analyze_capacitance_distribution(raw_data="packaged/Reference_Bare_renewed.h5",
                                     base_path="Reference/Bare/unbiased_31_renew",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_renew_parasitic.pdf", lock=tb_lock)
    analyze_capacitance_distribution(raw_data="packaged/Reference_Bare_renewed.h5",
                                     base_path="Reference/Bare/unbiased_31_renew_full_model",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_renew_parasitic_full_model.pdf", lock=tb_lock)
    analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5',
                                     base_path="Reference/bare/unbiased_8_full_model",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_parasitic_full_model.pdf", lock=tb_lock)
    analyze_capacitance_distribution(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                                     base_path="Reference/Bare/unbiased_full",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_parasitic_extended_renew_full_model.pdf",
                                     lock=tb_lock)


def r1_analysator(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the R1/R11 (LF (CMOS)) sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Then the Inter-Pixel-Capacitance analysis is investigated.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased, full depletion?)


    In this special case also measurements for only considering top-bottom neighbours, only considering left-right
    neighbours or diagonal neighbours are investigated.

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    import pixcap65.concurrency
    from examples.mp_analysis import synchronize_full_model
    name = "R1"
    display_name = name
    top_ref = "Reference"
    print("Analyze", display_name)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    # this is not necessary for the ExtendedSyncManager as this accessed right here.
    # But this will only take effect as long as we are not spawning additional subprocesses.

    synchronize_full_model(R11_SCAN_FILE, top_ref, name, 80, tb_lock,)
    synchronize_full_model(R11_SCAN_FILE, top_ref, name, 80, tb_lock,
                           inter_biased_group="inter_biased_M_80_V_full_Extended",
                           inter_pix_extension_active=True,
                           api=False)

    # handle the full sensor analysis
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'),
                 is_advanced=True, distribution=True, full_model=False,
                 test_cap_exclusion=True, mask_pixel=data_constants.r1_pixel_mask, lock=tb_lock,
                 fit_plot_pdf_name="Fit References/{}/unbiased_reduced_model.pdf".format(name), plot=True,
                 **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'),
                 is_advanced=True, distribution=True, full_model=False,
                 fit_plot_pdf_name="Fit References/{}/biased_reduced_model.pdf".format(name), plot=True,
                 test_cap_exclusion=True, mask_pixel=data_constants.r1_pixel_mask, lock=tb_lock, **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'),
                 is_advanced=True, distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.r1_pixel_mask,
                 fit_plot_pdf_name="Fit References/{}/unbiased_full_model.pdf".format(name), plot=True,
                 lock=tb_lock, **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
                 is_advanced=True, distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.r1_pixel_mask,
                 fit_plot_pdf_name="Fit References/{}/biased_full_model.pdf".format(name), plot=True,
                 lock=tb_lock, **correction_args)

    # handle the inter-pix analysis
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 total_cap_file=R11_SCAN_FILE,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'), **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'), **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 total_cap_file=R11_SCAN_FILE,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'), **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'), **correction_args)

    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'), **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'), **correction_args)

    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__sides'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=R11_SCAN_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=20000, **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__sides'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=R11_SCAN_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=20000, **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__diagonals'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=R11_SCAN_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=18000, **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__diagonals'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=R11_SCAN_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=18000, **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__tops'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=R11_SCAN_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=19000, **correction_args)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__tops'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=R11_SCAN_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=19000, **correction_args)

    # handle the C-V-analysis
    r11_depletion_args = {
        "first_boundaries": (-250, -125),
        "second_boundaries": (-3.0, 0),
        "distribution": False,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    r11_depletion_args_refined = {
        "first_boundaries": (-250, -75),
        "second_boundaries": (-2.5, 0),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    r11_depletion_args.update(**correction_args)
    r11_depletion_args.update(**kwargs)
    r11_depletion_args_refined.update(**correction_args)
    r11_depletion_args_refined.update(**kwargs)

    print(LOG_CV_ANALYSIS, display_name)
    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                 test_cap_exclusion=True, mask_pixel=data_constants.r1_pixel_mask,
                 lock=tb_lock, **r11_depletion_args)

    analyze_data(raw_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                 test_cap_exclusion=True, mask_pixel=data_constants.r1_pixel_mask,
                 lock=tb_lock, **r11_depletion_args_refined)
    print(LOG_FINISHED_ANALYSIS, display_name)


def r13_analysator_first(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the R13 (LF (CMOS)) sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'. This implementation acts on the first run of measurements.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased, fully depleted?)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    import pixcap65.concurrency
    name = "R13"
    print("Analyze", name)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    correction_args = correction_args.copy()
    correction_args.update(lock=tb_lock)

    analyze_data(raw_data='pixcap65/Data/TEST_2.h5', is_advanced=False, lock=tb_lock, )
    analyze_data(raw_data='pixcap65/Data/r13-measurement/data.h5', is_advanced=False, **correction_args)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/TEST.h5', is_advanced=False, **correction_args)
    analyze_data(raw_data='Reference_R13_Scan.h5', base_path="Reference/R13/unbiased_12_full",
                 is_advanced=True, **correction_args)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', is_advanced=True,
                 **correction_args)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1",
                 is_advanced=True, **correction_args)
    analyze_data(raw_data='R13-Interpixel_Scan.h5',
                 base_path="Reference/R13/demo_measurement_65_unbiased_1_discharge",
                 is_inter_pixel=True, is_advanced=True, lock=tb_lock, )

    print(LOG_CV_ANALYSIS, name)
    # r13-measurements/R13_BIAS_CV_2.h5 could not be directly investigated as it is incomplete.
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_2.h5', is_advanced=False, is_cv=True,
                 use_corrected=True, apply_doping=True, chip_group_name="sensor",
                 **correction_args)
    # Data/r13-measurement/R13_BIAS_CV_COMBI_3.h5 no further investigation possible as data set is incomplete!
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_5.h5', is_advanced=False, is_cv=True,
                 first_boundaries=(-100, -40),
                 second_boundaries=(-8, -0), use_corrected=True, apply_doping=True, chip_group_name="sensor",
                 **correction_args)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', is_advanced=False, is_cv=True,
                 first_boundaries=(-100, -40),
                 second_boundaries=(-10, 0), apply_doping=True, chip_group_name="sensor", use_corrected=True,
                 **correction_args)
    print(LOG_FINISHED_ANALYSIS, name)


def r13_analysator_second(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the R13 (LF (CMOS)) sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.
    This implementation is used of the second measurement runs with full accuracy.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Then the Inter-Pixel-Capacitance analysis is investigated.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased, fully depleted?)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    from examples.mp_analysis import synchronize_full_model
    import pixcap65.concurrency
    name = "R13"
    display_name = "R13" + SECOND_LABEL
    top_ref = "Reference"
    print("Analyze", display_name)
    logger.debug(threading.get_native_id())
    logger.debug(mp.current_process().name)
    logger.debug(mp.current_process().pid)
    logger.debug(AUTHKEY_OUTPUT, mp.current_process().authkey)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    r13_depletion_args = {
        "first_boundaries": (-85, -25),
        "second_boundaries": (-3.4, 0),
        "distribution": True,
        "apply_contour": True,
        "apply_contours": True,
        "chip_group_name": hdf(top_ref, name, "sensor"),
        "apply_doping": True,
    }
    r13_depletion_args.update(**correction_args)

    synchronize_full_model(R13_2_SCAN_FILE, top_ref, name, 80, tb_lock,
                           unbiased_group="unbiased_1_full", api=True)
    synchronize_full_model(R13_2_SCAN_FILE, top_ref, name, 80, tb_lock,
                           unbiased_group="unbiased_1_full", inter_unbiased_group="inter_unbiased_full_renew",
                           inter_biased_group="inter_biased_M_{}_V_full_renew")
    synchronize_full_model(R13_2_SCAN_FILE, top_ref, name, 80, tb_lock,
                           inter_unbiased_group="inter_unbiased_renew_Extended_full",
                           inter_biased_group="inter_biased_M_{}_V_renew_Extended_full", api=True)

    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full'), is_advanced=True,
                 lock=tb_lock, test_cap_exclusion=True, distribution=True, full_model=False,
                 fit_plot_pdf_name="Fit References/{}/unbiased_reduced_model.pdf".format(name), plot=True,
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'), is_advanced=True,
                 lock=tb_lock, distribution=True, test_cap_exclusion=True, full_model=False, plot=True,
                 fit_plot_pdf_name="Fit References/{}/biased_reduced_model.pdf".format(name),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full_model'),
                 is_advanced=True,
                 lock=tb_lock, test_cap_exclusion=True, distribution=True,
                 fit_plot_pdf_name="Fit References/{}/unbiased_full_model.pdf".format(name), plot=True,
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
                 is_advanced=True,
                 lock=tb_lock, distribution=True, test_cap_exclusion=True,
                 fit_plot_pdf_name="Fit References/{}/biased_full_model.pdf".format(name), plot=True,
                 **correction_args)

    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 lock=tb_lock, distribution=True, test_cap_exclusion=True,
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_1_full_model/total_cap'),
                 **correction_args)

    analyze_data(
        raw_data=R13_2_SCAN_FILE,
        base_path=hdf(top_ref, name, "inter_biased_M_80_V_full"),
        lock=tb_lock,
        distribution=True,
        test_cap_exclusion=True,
        is_advanced=True,
        full_model=False,
        is_inter_pixel=True,
        total_cap_file=R13_2_SCAN_FILE,
        total_cap_group=hdf(top_ref, name, "biased_80_V_full_model/total_cap"),
        **correction_args
    )
    analyze_data(
        raw_data=R13_2_SCAN_FILE,
        base_path=hdf(top_ref, name, "inter_unbiased_full_model"),
        lock=tb_lock,
        distribution=True,
        test_cap_exclusion=True,
        is_advanced=True,
        full_model=True,
        is_inter_pixel=True,
        total_cap_file=R13_2_SCAN_FILE,
        total_cap_group=hdf(top_ref, name, "unbiased_1_full_model/total_cap"),
        **correction_args
    )
    analyze_data(
        raw_data=R13_2_SCAN_FILE,
        base_path=hdf(top_ref, name, "inter_biased_M_80_V_full_model"),
        lock=tb_lock,
        distribution=True,
        test_cap_exclusion=True,
        is_advanced=True,
        full_model=True,
        is_inter_pixel=True,
        total_cap_file=R13_2_SCAN_FILE,
        total_cap_group=hdf(top_ref, name, "biased_80_V_full_model/total_cap"),
        **correction_args
    )

    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_renew'),
                 lock=tb_lock, distribution=True, test_cap_exclusion=True,
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_1_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_renew'),
                 lock=tb_lock, distribution=True, test_cap_exclusion=True,
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_renew_model'),
                 lock=tb_lock, distribution=True, test_cap_exclusion=True,
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_1_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_renew_model'),
                 lock=tb_lock, distribution=True, test_cap_exclusion=True,
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 **correction_args)

    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_renew_Extended_full'),
                 lock=tb_lock, distribution=True, test_cap_exclusion=True,
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_1_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_renew_Extended_full'),
                 lock=tb_lock, distribution=True, test_cap_exclusion=True,
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_unbiased_renew_Extended_full_model'),
                 lock=tb_lock, distribution=True, test_cap_exclusion=True,
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_1_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_renew_Extended_full_model'),
                 lock=tb_lock, distribution=True, test_cap_exclusion=True,
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 **correction_args)

    print("CV -", display_name)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True,
                 lock=tb_lock,
                 **r13_depletion_args)

    print("Finished", display_name)


def e1_analysator_second(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the E1 (LF (CMOS)) sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.
    This implementation is based on the second measurement runs with full accuracy.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Then the Inter-Pixel-Capacitance analysis is investigated.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased, fully depleted?)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    import pixcap65.concurrency
    name = "E1"
    display_name = name + SECOND_LABEL
    top_ref = "Reference"
    print("Analyze", display_name)
    print(AUTHKEY_OUTPUT, mp.current_process().authkey)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    e1_depletion_args = {
        "first_boundaries": (-100, -35),
        "second_boundaries": (-5, 0),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "mask_pixel": data_constants.e1_pixel_mask,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    e1_depletion_args.update(**correction_args)
    synchronize_full_model(E1_2_SCAN_FILE, top_ref, name, 80, tb_lock)

    analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), is_advanced=True,
                 lock=tb_lock,
                 full_model=False, distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask,
                 **correction_args)
    analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'), is_advanced=True,
                 lock=tb_lock,
                 full_model=False, distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask,
                 **correction_args)
    analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'), is_advanced=True,
                 lock=tb_lock,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask, **correction_args)
    analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
                 is_advanced=True, lock=tb_lock,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask, **correction_args)

    analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True, lock=tb_lock,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask,
                 total_cap_file=E1_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full/total_cap'),
                 **correction_args)
    analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True, lock=tb_lock,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask,
                 total_cap_file=E1_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full/total_cap'),
                 **correction_args)
    analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True, lock=tb_lock,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask,
                 total_cap_file=E1_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full/total_cap'),
                 **correction_args)
    analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True, lock=tb_lock,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask,
                 total_cap_file=E1_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full/total_cap'),
                 **correction_args)

    analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                 lock=tb_lock, test_cap_exclusion=True,
                 **e1_depletion_args)

    for type_name in data_constants.e1_pixel_groups.keys():
        with synchronized_process_open_file(E1_2_SCAN_FILE, mode='a', lock=tb_lock) as h5_file:
            h5_file.copy_node(where="/Reference/E1", newname="unbiased_full_model_{}".format(type_name),
                              name="unbiased_full_{}".format(type_name), recursive=True, overwrite=True)
            h5_file.copy_node(where="/Reference/E1", newname="biased_80_V_full_model_{}".format(type_name),
                              name="biased_80_V_full_{}".format(type_name), recursive=True, overwrite=True)
            h5_file.copy_node(where="/Reference/E1", newname="inter_unbiased_full_model_{}".format(type_name),
                              name="inter_unbiased_full_{}".format(type_name), recursive=True, overwrite=True)
            h5_file.copy_node(where="/Reference/E1", newname="inter_biased_M_80_V_full_model_{}".format(type_name),
                              name="inter_biased_M_80_V_full_{}".format(type_name), recursive=True, overwrite=True)

        analyze_data(raw_data=E1_2_SCAN_FILE, base_path="{}/{}/unbiased_full_{}".format(top_ref, name, type_name),
                     is_advanced=True, lock=tb_lock, full_model=False, distribution=True, test_cap_exclusion=True,
                     mask_pixel=data_constants.e1_pixel_mask,
                     **correction_args)
        analyze_data(raw_data=E1_2_SCAN_FILE, base_path="{}/{}/biased_80_V_full_{}".format(top_ref, name, type_name),
                     is_advanced=True, lock=tb_lock, full_model=False, distribution=True, test_cap_exclusion=True,
                     mask_pixel=data_constants.e1_pixel_mask,
                     **correction_args)
        analyze_data(raw_data=E1_2_SCAN_FILE, base_path="{}/{}/unbiased_full_model_{}".format(top_ref, name, type_name),
                     is_advanced=True, lock=tb_lock, distribution=True, test_cap_exclusion=True,
                     mask_pixel=data_constants.e1_pixel_mask,
                     **correction_args)
        analyze_data(raw_data=E1_2_SCAN_FILE,
                     base_path="{}/{}/biased_80_V_full_model_{}".format(top_ref, name, type_name),
                     is_advanced=True, lock=tb_lock, distribution=True, test_cap_exclusion=True,
                     mask_pixel=data_constants.e1_pixel_mask,
                     **correction_args)

        analyze_data(raw_data=E1_2_SCAN_FILE, base_path="{}/{}/inter_unbiased_full_{}".format(top_ref, name, type_name),
                     is_advanced=True, full_model=False, is_inter_pixel=True, lock=tb_lock,
                     distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask,
                     total_cap_file=E1_2_SCAN_FILE,
                     total_cap_group="{}/{}/unbiased_full_model_{}/total_cap".format(top_ref, name, type_name),
                     **correction_args)
        analyze_data(raw_data=E1_2_SCAN_FILE,
                     base_path="{}/{}/inter_biased_M_80_V_full_{}".format(top_ref, name, type_name),
                     is_advanced=True, full_model=False, is_inter_pixel=True, lock=tb_lock,
                     distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask,
                     total_cap_file=E1_2_SCAN_FILE,
                     total_cap_group="{}/{}/biased_80_V_full_model_{}/total_cap".format(top_ref, name, type_name),
                     **correction_args)
        analyze_data(raw_data=E1_2_SCAN_FILE,
                     base_path="{}/{}/inter_unbiased_full_model_{}".format(top_ref, name, type_name),
                     is_advanced=True, full_model=True, is_inter_pixel=True, lock=tb_lock,
                     distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask,
                     total_cap_file=E1_2_SCAN_FILE,
                     total_cap_group="{}/{}/unbiased_full_model_{}/total_cap".format(top_ref, name, type_name),
                     **correction_args)
        analyze_data(raw_data=E1_2_SCAN_FILE,
                     base_path="{}/{}/inter_biased_M_80_V_full_model_{}".format(top_ref, name, type_name),
                     is_advanced=True, full_model=True, is_inter_pixel=True, lock=tb_lock,
                     distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask,
                     total_cap_file=E1_2_SCAN_FILE,
                     total_cap_group="{}/{}/biased_80_V_full_model_{}/total_cap".format(top_ref, name, type_name),
                     **correction_args)

        actual_depletion_args = data_constants.e1_pixel_depletion_args[type_name]
        actual_depletion_args.update(**correction_args)
        actual_depletion_args.update(chip_group_name=hdf(top_ref, name, 'sensor'), apply_doping=True)

        # with PdfPages("Fit References/{}/Renew_C_V_Verify_{}.pdf".format(name, type_name)) as pdf:
        analyze_data(raw_data=E1_2_SCAN_FILE,
                     base_path="{}/{}/C_V_Characteristic_refined_{}".format(top_ref, name, type_name),
                     is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                     test_cap_exclusion=True,
                     lock=tb_lock, **actual_depletion_args)

    print("Finished", name)


def x1_analysator_first(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the X1 (HPK) sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.
    This implementation used the first measurement runs with reduced accuracy for the analysis.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Last the C-V-Characterization is investigated.
    This in particular includes the estimation of depletion voltage.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased, fully depleted?)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    import pixcap65.concurrency
    name = "X1"
    print("Analyze", name)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    correction_args = correction_args.copy()
    correction_args.update(lock=tb_lock)

    analyze_data(raw_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_1",
                 is_advanced=False, **correction_args)
    analyze_data(raw_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_2",
                 is_advanced=False, **correction_args)
    analyze_data(raw_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_1",
                 is_advanced=False, **correction_args)
    analyze_data(raw_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_2",
                 is_advanced=False, **correction_args)
    analyze_data(raw_data='pixcap65/Data/ATLAS ITk/New_1_Initial_Scan.h5', is_advanced=True, **correction_args)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_1",
                 is_advanced=False, **correction_args)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_2", full_model=False,
                 is_advanced=True, **correction_args)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
                 is_advanced=True,
                 **correction_args)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_4",
                 is_advanced=False, **correction_args)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V",
                 is_advanced=True,
                 **correction_args)

    print(LOG_CV_ANALYSIS, name)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic",
                 is_advanced=False, is_cv=True, use_corrected=True, apply_doping=True,
                 chip_group_name="ATLAS ITk/sensor",
                 first_boundaries=[(-60, -40), (-80, -75)], second_boundaries=[(-5, 0), (-70, -65)],
                 **correction_args)
    print(LOG_FINISHED_ANALYSIS, name)


def x1_analysator_second(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the X1 (HPK) sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.
    This implementation used the second measurement runs with full accuracy for the analysis.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Then the Inter-Pixel-Capacitance analysis is investigated.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.

    Last the C-V-Characterization is investigated.
    This in particular includes the estimation of depletion voltage.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased, fully depleted?)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    import pixcap65.concurrency
    from examples.mp_analysis import synchronize_full_model
    name = "X1"
    display_name = name + SECOND_LABEL
    top_ref = "Thesis/ATLAS_ITk"
    print("Analyze", display_name)
    print(threading.get_native_id())
    print(mp.current_process().name)
    print(mp.current_process().pid)
    print(AUTHKEY_OUTPUT, mp.current_process().authkey)

    primary_manager = pixcap65.concurrency.get_manager(**kwargs)
    x1_depletion_args = {
        "first_boundaries": [(-60, -20), (-83, -77.5)],
        "second_boundaries": [(-0.6, 0), (-77.5, -67.5)],
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True
    }
    x1_depletion_args.update(**correction_args)
    print("The manager address:", primary_manager.address)

    synchronize_full_model(X1_SCAN_2_FILE, top_ref, name, 80, tb_lock, unbiased_group="unbiased_61_full", api=True)
    synchronize_full_model(X1_SCAN_2_FILE, top_ref, name, 200, tb_lock, unbiased_group="unbiased_61_full",
                           inter_unbiased_group="inter_unbiased_full_renew_Extended",
                           inter_biased_group="inter_biased_M_{}_V_full_renew_Extended", api=True)
    synchronize_full_model(X1_SCAN_2_FILE, top_ref, name, 80, tb_lock,
                           inter_biased_group="inter_biased_M_{}_V_full_Extended",
                           inter_pix_extension_active=True, api=True)
    synchronize_full_model(X1_SCAN_2_FILE, top_ref, name, 80, tb_lock,
                           inter_biased_group="inter_biased_M_{}_V_full_Extended_2",
                           inter_pix_extension_active=True, api=True)

    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_61_full'), is_advanced=True,
                 full_model=False,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock, fit_plot_pdf_name="Fit References/{}/unbiased_reduced_model.pdf".format(name), plot=True,
                 **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'), is_advanced=True,
                 full_model=False,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 fit_plot_pdf_name="Fit References/{}/biased_reduced_model.pdf".format(name), plot=True,
                 lock=tb_lock, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'biased_200_V_full'), is_advanced=True,
                 full_model=False,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock, **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_61_full_model'),
                 is_advanced=True,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock, fit_plot_pdf_name="Fit References/{}/unbiased_full_model.pdf".format(name), plot=True,
                 **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
                 is_advanced=True,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 fit_plot_pdf_name="Fit References/{}/biased_full_model.pdf".format(name), plot=True,
                 lock=tb_lock, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'biased_200_V_full_model'),
                 is_advanced=True,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock, **correction_args)

    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_61_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True, distribution=True, test_cap_exclusion=True,
                 mask_pixel=data_constants.x1_second_pixel_mask, total_cap_file=X1_SCAN_2_FILE,
                 lock=tb_lock, total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_61_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True, distribution=True, test_cap_exclusion=True,
                 mask_pixel=data_constants.x1_second_pixel_mask, total_cap_file=X1_SCAN_2_FILE,
                 lock=tb_lock, total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 **correction_args)

    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_unbiased_full_renew_Extended'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_61_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_unbiased_full_model_renew_Extended'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_61_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_200_V_full_renew_Extended'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file="packaged/data/X1_12_Renew_Scan.h5",
                 total_cap_group=hdf(top_ref, name, 'biased_200_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_200_V_full_model_renew_Extended'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file="packaged/data/X1_12_Renew_Scan.h5",
                 total_cap_group=hdf(top_ref, name, 'biased_200_V_full_model/total_cap'),
                 **correction_args)

    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__sides'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=20000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__sides'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=20000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__diagonals'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=18000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__diagonals'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=18000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__tops'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=19000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__tops'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=19000, **correction_args)

    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended_2__sides'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=20000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended_2__sides'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=20000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended_2__diagonals'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=18000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended_2__diagonals'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=18000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended_2__tops'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=19000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended_2__tops'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=19000, **correction_args)

    print("CV -", display_name)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                 test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 **x1_depletion_args)
    x1_depletion_args = {
        "first_boundaries": [(-60, -20), (-350, -150)],
        "second_boundaries": [(-2.6, 0), (-72, -62)],
        "distribution": False,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True
    }
    x1_depletion_args.update(**correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_Second_Extended'),
                 is_advanced=True, full_model=False, is_inter_pixel=False, is_cv=True, use_corrected=True,
                 test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock,
                 **x1_depletion_args)
    x1_depletion_args = {
        "first_boundaries": [(-60, -20), (-350, -200), (-85, -75)],
        "second_boundaries": [(-0.6, 0), (-69.8, -64), (-70, -60)],
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True
    }
    x1_depletion_args.update(**correction_args)
    with PdfPages("Fit References/X1/Combined_reference_fits_Extended_combined.pdf") as cv_pdf:
        analyze_data(raw_data=X1_SCAN_2_FILE,
                     base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_Extended_Combined'),
                     is_advanced=True, full_model=False, is_inter_pixel=False, is_cv=True, use_corrected=True,
                     test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                     cv_fit_plot_pdf=cv_pdf,
                     lock=tb_lock, **x1_depletion_args)

    x1_depletion_args = {
        "first_boundaries": [(-60, -20), (-350, -150)],
        "second_boundaries": [(-2.6, 0), (-72, -66)],
        "distribution": False,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True
    }
    x1_depletion_args.update(**correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_Second_Extended'),
                 is_advanced=True, full_model=False, is_inter_pixel=False, is_cv=True, use_corrected=True,
                 test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask,
                 lock=tb_lock, **x1_depletion_args)
    print("Finished", display_name)


def x2_analysator_first(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the X2 (HPK) sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.
    This implementation used the first measurement runs with reduced accuracy for the analysis.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Last the C-V-Characterization is investigated.
    This in particular includes the estimation of depletion voltage.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased, fully depleted?)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    import pixcap65.concurrency
    name = "X2"
    print("Analyze", name)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    correction_args = correction_args.copy()
    correction_args.update(lock=tb_lock)

    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1", is_advanced=True,
                 **correction_args)
    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/biased_80_V", is_advanced=True,
                 **correction_args)

    print(LOG_CV_ANALYSIS, name)
    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic", is_advanced=True,
                 is_cv=True,
                 first_boundaries=[(-60, -40), (-80, -75)], second_boundaries=[(-5, 0), (-70, -65)],
                 use_corrected=True,
                 **correction_args)
    with PdfPages("../New_2_Scan_CV_refined_distribution.pdf") as pdf:
        analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic_refined",
                     is_advanced=True,
                     is_cv=True,
                     first_boundaries=(-60, -20), second_boundaries=(-5, 0), use_corrected=True, apply_doping=True,
                     chip_group_name="ATLAS_Itk/X2/sensor", distribution=True, set_parasitic=False,
                     distribution_output_pdf=pdf, **correction_args)
    print(LOG_FINISHED_ANALYSIS, name)


def x2_analysator_second(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the X2 (HPK) sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.
    This implementation used the second measurement runs with full accuracy for the analysis.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Last the C-V-Characterization is investigated.
    This in particular includes the estimation of depletion voltage.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased, fully depleted?)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    from examples.mp_analysis import synchronize_full_model
    import pixcap65.concurrency
    name = "X2"
    display_name = name + SECOND_LABEL
    top_ref = "Thesis/ATLAS_ITk"
    print("Analyze", display_name)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    x2_depletion_args = {
        "first_boundaries": [(-59.5, -15), (-100, -60)],
        "second_boundaries": [(-0.5, 0), (-75, -50)],
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    x2_depletion_refined_args = {
        "first_boundaries": [(-59.5, -15), (-350, -150), (-85, -75)],
        "second_boundaries": [(-0.5, 0), (-70, -54), (-70, -60)],
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    x2_depletion_args.update(**correction_args)
    x2_depletion_refined_args.update(**correction_args)

    synchronize_full_model(X2_SCAN_2_FILE, top_ref, name, 80, tb_lock, unbiased_group="unbiased_1_full")
    synchronize_full_model(X2_SCAN_2_FILE, top_ref, name, 200, tb_lock)
    with synchronized_process_open_file(X2_SCAN_2_FILE, mode='a', lock=tb_lock) as h5_file:
        h5_file.copy_node(where="/Thesis/ATLAS_ITk/X2", newname="unbiased_1_full_model", name="unbiased_1_full",
                          recursive=True, overwrite=True)
        h5_file.copy_node(where="/Thesis/ATLAS_ITk/X2", newname="biased_80_V_full_model", name="biased_80_V_full",
                          recursive=True, overwrite=True)

    analyze_data(raw_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full'), is_advanced=True,
                 lock=tb_lock,
                 full_model=False, distribution=True, **correction_args)
    analyze_data(raw_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'), is_advanced=True,
                 lock=tb_lock,
                 full_model=False, distribution=True, **correction_args)
    analyze_data(raw_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_200_V_full'), is_advanced=True,
                 lock=tb_lock,
                 full_model=False, distribution=True, **correction_args)
    analyze_data(raw_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full_model'),
                 is_advanced=True, lock=tb_lock,
                 distribution=True, **correction_args)
    analyze_data(raw_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
                 is_advanced=True, lock=tb_lock,
                 distribution=True, **correction_args)
    analyze_data(raw_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_200_V_full_model'),
                 is_advanced=True, lock=tb_lock,
                 distribution=True, **correction_args)

    print("CV -", display_name)
    # are both arguments available use_corrected and apply_correction!
    analyze_data(raw_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=True, lock=tb_lock,
                 **x2_depletion_args)
    with PdfPages("Fit References/X2/Scan_extended_reference_fits.pdf") as cv_pdf:
        analyze_data(raw_data=X2_SCAN_2_FILE,
                     base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_extended_renew_retry'),
                     is_advanced=True, full_model=False, is_cv=True, use_corrected=True, lock=tb_lock,
                     # cv_fit_plot_pdf=cv_pdf,
                     **x2_depletion_refined_args)
    # After all they could not be measured at all.
    print("Finished -", display_name)


def x4_analysator(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the X4 (HPK) 3D-sample.

    :author: Dominik Fischer
    :date: 2026-07-30

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Then the Inter-Pixel-Capacitance analysis is investigated.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.

    Last the C-V-Characterization is investigated.
    This in particular includes the estimation of depletion voltage.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased, fully depleted?)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    import pixcap65.concurrency
    from examples.mp_analysis import synchronize_full_model
    name = "X4"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Analyze", display_name)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    # this is not necessary for the ExtendedSyncManager as this accessed right here.
    # But this will only take effect as long as we are not spawning additional subprocesses.

    synchronize_full_model(X4_SCAN_FILE, top_ref, name, 80, tb_lock, )

    # handle the full sensor analysis
    analyze_data(raw_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'),
                 is_advanced=True, distribution=True, full_model=False,
                 test_cap_exclusion=True, mask_pixel=data_constants.x4_pixel_mask, lock=tb_lock,
                 **correction_args)
    analyze_data(raw_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'),
                 is_advanced=True, distribution=True, full_model=False,
                 test_cap_exclusion=True, mask_pixel=data_constants.x4_pixel_mask, lock=tb_lock, **correction_args)
    analyze_data(raw_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'),
                 is_advanced=True, distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x4_pixel_mask,
                 lock=tb_lock, **correction_args)
    analyze_data(raw_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
                 is_advanced=True, distribution=True, test_cap_exclusion=True, mask_pixel=data_constants.x4_pixel_mask,
                 lock=tb_lock, **correction_args)

    # handle the inter-pix analysis
    analyze_data(raw_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 total_cap_file=X4_SCAN_FILE,
                 lock=tb_lock, mask_pixel=data_constants.x4_pixel_mask,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'), **correction_args)
    analyze_data(raw_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x4_pixel_mask,
                 total_cap_file=X4_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'), **correction_args)
    analyze_data(raw_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 total_cap_file=X4_SCAN_FILE,
                 lock=tb_lock, mask_pixel=data_constants.x4_pixel_mask,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'), **correction_args)
    analyze_data(raw_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x4_pixel_mask,
                 total_cap_file=X4_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'), **correction_args)

    # handle the C-V-analysis
    x4_depletion_args = {
        "first_boundaries": [(-15, -4.5), (-70, -50), (-70, -50)],
        "second_boundaries": [(-1, 0), (-42, -39), (-32, -25)],
        "distribution": False,
        "apply_contour": False,
        "apply_contours": False,
        # "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    x4_depletion_args_refined = {
        "first_boundaries": [(-15, -4.5), (-70, -50), (-70, -50)],
        "second_boundaries": [(-0.75, 0), (-42, -39), (-37, -29)],
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        # "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    x4_depletion_args.update(**correction_args)
    x4_depletion_args.update(**kwargs)
    x4_depletion_args_refined.update(**correction_args)
    x4_depletion_args_refined.update(**kwargs)

    print(LOG_CV_ANALYSIS, display_name)
    # with PdfPages("Fit References/X4/coarse_reference_fits.pdf") as pdf:
    analyze_data(raw_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_trial'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                 test_cap_exclusion=True, mask_pixel=data_constants.x4_pixel_mask,
                 lock=tb_lock, **x4_depletion_args)

    with PdfPages("Fit References/X4/fine_reference_fits.pdf") as pdf:
        analyze_data(raw_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_Extended'),
                     is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                     test_cap_exclusion=True, mask_pixel=data_constants.x4_pixel_mask,
                     cv_fit_plot_pdf=pdf,
                     lock=tb_lock, **x4_depletion_args_refined)
    print(LOG_FINISHED_ANALYSIS, display_name)


def x5_analysator(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the X5 (HPK) 3D-sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Then the Inter-Pixel-Capacitance analysis is investigated.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.

    Last the C-V-Characterization is investigated.
    This in particular includes the estimation of depletion voltage.

    The applied biasing states are:

    * 0V (unbiased)
    * -40V (biased, fully depleted?)
    * -90V (biased, fully depleted?)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    from examples.mp_analysis import synchronize_full_model
    import pixcap65.concurrency
    name = "X5"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Analyze", display_name)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    # new data store for FBK:
    np.unique(np.concat((
        np.geomspace(0.1, 15, 15),
        np.geomspace(15, 60, 35)
    )))
    np.unique(np.concat((
        np.geomspace(0.1, 15, 15),
        np.geomspace(15, 40, 20),
        np.arange(40, 100.1, 1.25)
    )))

    synchronize_full_model(X5_SCAN_FILE, top_ref, name, 40, tb_lock)
    synchronize_full_model(X5_SCAN_FILE, top_ref, name, 90, tb_lock)

    # handle the full sensor analysis
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), is_advanced=True,
                 distribution=True, full_model=False, mask_pixel=data_constants.x5_second_pixel_mask,
                 test_cap_exclusion=True,
                 lock=tb_lock, plot=True,
                 fit_plot_pdf_name="Fit References/{}/unbiased_reduced_model_reference_fits.pdf".format(name),
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40_V_full'),
                 is_advanced=True,
                 distribution=True, full_model=False, mask_pixel=data_constants.x5_second_pixel_mask,
                 test_cap_exclusion=True,
                 lock=tb_lock, plot=True,
                 fit_plot_pdf_name="Fit References/{}/biased_reduced_model_reference_fits.pdf".format(name),
                 mask_lower=50e-15,
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_90_V_full'),
                 is_advanced=True,
                 distribution=True, full_model=False, mask_pixel=data_constants.x5_second_pixel_mask,
                 test_cap_exclusion=True,
                 lock=tb_lock,
                 # mask_lower=50e-15,
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'),
                 is_advanced=True,
                 distribution=True, mask_pixel=data_constants.x5_second_pixel_mask, test_cap_exclusion=True,
                 lock=tb_lock, plot=True,
                 fit_plot_pdf_name="Fit References/{}/unbiased_full_model_reference_fits.pdf".format(name),
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40_V_full_model'),
                 is_advanced=True, full_model=True,
                 distribution=True, mask_pixel=data_constants.x5_second_pixel_mask, test_cap_exclusion=True,
                 lock=tb_lock, mask_lower=51.5e-15,
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_90_V_full_model'),
                 is_advanced=True,
                 distribution=True, mask_pixel=data_constants.x5_second_pixel_mask, test_cap_exclusion=True,
                 lock=tb_lock,
                 # mask_lower=50e-15,
                 **correction_args)

    # handle the inter-pix analysis
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 mask_pixel=data_constants.x5_second_pixel_mask, test_cap_exclusion=True, distribution=True,
                 total_cap_file=X5_SCAN_FILE,
                 lock=tb_lock,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_40_V_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 mask_pixel=data_constants.x5_pixel_mask, test_cap_exclusion=True, distribution=True,
                 lock=tb_lock,
                 total_cap_file=X5_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_40_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 mask_pixel=data_constants.x5_second_pixel_mask, test_cap_exclusion=True, distribution=True,
                 total_cap_file=X5_SCAN_FILE,
                 lock=tb_lock,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_40_V_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 mask_pixel=data_constants.x5_pixel_mask, test_cap_exclusion=True, distribution=True,
                 lock=tb_lock,
                 total_cap_file=X5_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_40_V_full/total_cap'),
                 **correction_args)

    # handle the C-V-analysis
    x5_depletion_args = {
        "first_boundaries": [(-15, -4.5), (-57.5, -40), (-57.5, -40)],
        "second_boundaries": [(-3, 0), (-35, -25), (-23, -20)],
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    x5_depletion_args_refined = {
        "first_boundaries": [(-15, -4.5), (-57, -36), (-57, -36)],
        "second_boundaries": [(-0.75, 0), (-35, -26), (-23.5, -18)],
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    x5_depletion_args.update(**correction_args)
    x5_depletion_args_refined.update(**correction_args)

    print("CV Analysis for X5")
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 mask_pixel=data_constants.x5_second_pixel_mask,
                 test_cap_exclusion=True,
                 lock=tb_lock, **x5_depletion_args)
    print("Finished first CV")
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 mask_pixel=data_constants.x5_pixel_mask,
                 test_cap_exclusion=True,
                 lock=tb_lock, **x5_depletion_args_refined)
    print("Finished second CV")
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_Extended'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 mask_pixel=data_constants.x5_pixel_mask,
                 test_cap_exclusion=True,
                 lock=tb_lock, **x5_depletion_args_refined)
    print("Finished third CV")

    # add here the additonal CV analysis used for extended range!
    print("Finished the Analysis for X5")


def x6_analysator(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the X6 (HPK) 3D-sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Then the Inter-Pixel-Capacitance analysis is investigated.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.

    Last the C-V-Characterization is investigated.
    This in particular includes the estimation of depletion voltage.

    The applied biasing states are:

    * 0V (unbiased)
    * -45V (biased, fully depleted?)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    import pixcap65.concurrency
    name = "X6"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Analyze", display_name)
    print(threading.get_native_id())
    print(mp.current_process().name)
    print(mp.current_process().pid)
    print(AUTHKEY_OUTPUT, mp.current_process().authkey)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    x6_depletion_args = {
        "first_boundaries": (-100, -35),
        "second_boundaries": (-2.8, -0.1),
        "distribution": False,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    # most of the refined scan is missing for this sensor!
    x6_depletion_args_refined = {
        "first_boundaries": (-100, -40),
        "second_boundaries": (-2.0, -0.6),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    x6_depletion_args.update(**correction_args)
    x6_depletion_args_refined.update(**correction_args)
    synchronize_full_model(X6_SCAN_FILE, top_ref, name, 45, tb_lock)

    analyze_data(raw_data=data_constants.X6_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'),
                 is_advanced=True, distribution=True, full_model=False, test_cap_exclusion=True,
                 lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask,
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_45_V_full'),
                 is_advanced=True,
                 distribution=True, full_model=False, mask_pixel=data_constants.x6_second_pixel_mask,
                 test_cap_exclusion=True,
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'),
                 is_advanced=True, distribution=True, test_cap_exclusion=True,
                 lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask,
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_45_V_full_model'),
                 is_advanced=True,
                 distribution=True, mask_pixel=data_constants.x6_second_pixel_mask, test_cap_exclusion=True,
                 **correction_args)

    # handle the inter-pix analysis
    analyze_data(raw_data=data_constants.X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 total_cap_file=data_constants.X6_SCAN_FILE,
                 lock=tb_lock,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 total_cap_file=data_constants.X6_SCAN_FILE,
                 lock=tb_lock,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_45_V_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock,
                 total_cap_file=data_constants.X6_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_45_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_biased_M_45_V_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True,
                 lock=tb_lock,
                 total_cap_file=data_constants.X6_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_45_V_full_model/total_cap'),
                 **correction_args)

    # CV analysis
    print("Cv analysis for X6")
    analyze_data(raw_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 test_cap_exclusion=True, mask_pixel=data_constants.x6_second_pixel_mask,
                 lock=tb_lock, **x6_depletion_args)

    analyze_data(raw_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask,
                 test_cap_exclusion=True, **x6_depletion_args_refined)
    print("Finished the CV analysis for X6")


def x7_analysator(tb_lock, correction_args, **kwargs):
    """
    Handles the plotting of the measurements with the X7 (HPK) 3D-sample.

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17


    First all the measurements are duplicated to apply both the linear model and the full enhanced model for analysis
    and estimation of the capacitances'.

    In a second step all the total pixel capacitances (unbiased as well as fully depleted (biased)) are investigated.
    In this step the correction for the circuits parasitic capacitances' is applied directly.
    Also it is tried to characterize the distribution of these capacitances' over the measured sensor.

    Then the Inter-Pixel-Capacitance analysis is investigated.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.

    Last the C-V-Characterization is investigated.
    This in particular includes the estimation of depletion voltage.

    The applied biasing states are:

    * 0V (unbiased)
    * -40V (biased, fully depleted?)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param correction_args: keyword arguments/dict to define how to correct for parasitic capacitances'.
    """
    import pixcap65.concurrency
    name = "X7"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Analyze", display_name)
    print(threading.get_native_id())
    print(mp.current_process().name)
    print(mp.current_process().pid)
    print(AUTHKEY_OUTPUT, mp.current_process().authkey)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    synchronize_full_model(X7_SCAN_FILE, top_ref, name, 40.0, tb_lock)

    analyze_data(raw_data=data_constants.X7_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'),
                 is_advanced=True,
                 distribution=True, full_model=False, test_cap_exclusion=True, lock=tb_lock,
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40.0_V_full'),
                 is_advanced=True, distribution=True, full_model=False, test_cap_exclusion=True, lock=tb_lock,
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'),
                 is_advanced=True, distribution=True, test_cap_exclusion=True, lock=tb_lock,
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40.0_V_full_model'),
                 is_advanced=True, distribution=True, test_cap_exclusion=True, lock=tb_lock,
                 **correction_args)

    # inter-pixel capacitance analysis
    analyze_data(raw_data=data_constants.X7_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True, lock=tb_lock,
                 total_cap_file=data_constants.X7_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_biased_M_40.0_V_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True, lock=tb_lock,
                 total_cap_file=data_constants.X7_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_40.0_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True, lock=tb_lock,
                 total_cap_file=data_constants.X7_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_biased_M_40.0_V_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 test_cap_exclusion=True, distribution=True, lock=tb_lock,
                 total_cap_file=data_constants.X7_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_40.0_V_full/total_cap'),
                 **correction_args)

    x7_depletion_args = {
        "first_boundaries": (-100, -30),
        "second_boundaries": (-1.8, -0.5),
        "distribution": False,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    x7_depletion_args_refined = {
        "first_boundaries": (-100, -40),
        "second_boundaries": (-2.0, -0.6),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    x7_depletion_args.update(correction_args)
    x7_depletion_args_refined.update(correction_args)
    print("CV Analysis for", display_name)
    analyze_data(raw_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 test_cap_exclusion=True, lock=tb_lock,
                 **x7_depletion_args)

    analyze_data(raw_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 test_cap_exclusion=True, lock=tb_lock,
                 **x7_depletion_args_refined)
    print("Finished the Analysis for", display_name)


# define the plotting handler
def bare_sample_plotter_first(tb_lock):
    """
    Handles the plotting of the measurements with the sensor-less bare sample in the first run group.

    In a first run overview pdf of the unbiased are created including the distributions of these
    capacitances over the whole sensor.

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "bare sample"
    print("Plotting", name)
    plot_data(interpreted_data='pixcap65/Data/bare-measurement/TEST.h5', suffix="general_bare_data_1-1",
              use_group=False, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/advanced-bare-measurement/TEST.h5', suffix="general_bare_data_2-1",
              use_group=False, lock=tb_lock)
    plot_data(interpreted_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
              suffix="general_bare_data_3", use_group=True,
              test_cap_exclusion=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/TEST_2.h5', suffix="test_run", use_group=False, lock=tb_lock)
    print("Finished -", name)


def bare_sample_plotter_second(tb_lock):
    """
    Handles the plotting of the measurements with the sensor-less bare sample in the second run group.

    In a first run overview pdf of the unbiased measurements are created including the distributions of these
    capacitances over the whole sensor.

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "bare"
    display_name = name + "Second Try"
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    plot_data(interpreted_data="packaged/Reference_Bare_renewed.h5",
              base_path="Reference/Bare/unbiased_31_renew_full_model",
              suffix="general_data_bare-2", use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data="packaged/Reference_Bare_renewed.h5", base_path="Reference/Bare/unbiased_31_renew",
              suffix="general_data_bare-2", use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8_full_model",
              suffix="general_data_bare-2", use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
              suffix="general_data_bare-2", use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock)

    masking = [
        [16, 20],
        [10, 25],
        [32, 17],
        [5, 16],
        [10, 16],
        [18, 16],
        [34, 16],
        [37, 37],
    ]
    plot_data(interpreted_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
              base_path="Reference/Bare/unbiased_full_model",
              suffix="general_data_bare-2", use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=masking)
    plot_data(interpreted_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
              base_path="Reference/Bare/unbiased_full",
              suffix="general_data_bare-2", use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=masking)
    plot_data(interpreted_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
              base_path="Reference/Bare/unbiased_full_kafe2",
              suffix="general_data_bare-2", use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=masking)
    plot_data(interpreted_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
              base_path="Reference/Bare/unbiased_full_extended",
              suffix="general_data_bare-2", use_group=True, test_cap_exclusion=False, distribution=True, lock=tb_lock,
              mask_pixel=masking)
    plot_data(interpreted_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
              base_path="Reference/Bare/unbiased_full_quad",
              suffix="general_data_bare-2", use_group=True, test_cap_exclusion=False, distribution=True, lock=tb_lock,
              mask_pixel=masking)
    print("Finished -", display_name)


def general_plotter(tb_lock, file_name, bias, **kwargs):
    """
    Handles the plotting of the measurements with the 3D-Sensor X5 (FBK sample).

    :author: Dominik Fischer
    :date: 2026-06-22

    last update: 2026-09-17

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -40V (biased, exact value will depend here on the provided parameters.)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    :param file_name: name of the hdf file, where the measurement data is stored together with the results of the
     analysis.
    :param bias: bias voltage applied for measurements with a fully depleted sensor.
    :keyword name: name of the sensor for which the results should be plotted. This must be part of the correct hdf
     groups path. (default: X5)
    :keyword appendix: appendix to `name` for the display name of this sensor in outputs.
    :keyword reference: path of the hdf files' group under which the sensors measurements are stored
     (default: Thesis/ATLAS_ITk).
    :keyword bias_sets: set of measurement group names containing purely I-V characterization measurements.
    :keyword simple_sets: set of measurement group names containing c-v characterization measurement (simple case)
    :keyword refined_sets: set of measurement group names containing c-v characterization measurements (refined case)
    """
    # note: the pixel mask must be submitted explicitly by 'mask_pixel'
    from os.path import join as hdf

    name = kwargs.pop('name', 'X5')
    display_name = name + kwargs.pop('appendix', '')
    top_ref = kwargs.pop('reference', "Thesis/ATLAS_ITk")
    bias_names = kwargs.pop('bias_sets', ["I_V_Characteristic"])
    simple_cv_names = kwargs.pop('simple_sets', ["C_V_Characteristic"])
    refined_cv_names = kwargs.pop('refined_sets', ["C_V_Characteristic_refined"])
    rounded_bias = int(bias)
    if np.abs(rounded_bias - bias) < 0.05:
        bias = rounded_bias

    print("Plotting", display_name)
    plot_data(interpreted_data=file_name, base_path=hdf(top_ref, name, UNBIASED), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock, **kwargs)
    plot_data(interpreted_data=file_name, base_path=hdf(top_ref, name, UNBIASED), use_group=True,
              use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock, **kwargs)
    plot_data(interpreted_data=file_name, base_path=hdf(top_ref, name, 'biased_{}_V_full'.format(bias)),
              use_group=True, test_cap_exclusion=True, distribution=True,
              lock=tb_lock, **kwargs)
    plot_data(interpreted_data=file_name, base_path=hdf(top_ref, name, 'biased_{}_V_full'.format(bias)),
              use_group=True, use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock, **kwargs)
    plot_data(interpreted_data=file_name, base_path=hdf(top_ref, name, UNBIASED_MODEL),
              use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock, **kwargs)
    plot_data(interpreted_data=file_name, base_path=hdf(top_ref, name, UNBIASED_MODEL),
              use_group=True, use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock, **kwargs)
    plot_data(interpreted_data=file_name, base_path=hdf(top_ref, name, 'biased_{}_V_full_model'.format(bias)),
              use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock, **kwargs)
    plot_data(interpreted_data=file_name, base_path=hdf(top_ref, name, 'biased_{}_V_full_model'.format(bias)),
              use_group=True, use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock, **kwargs)

    for bias_set in bias_names:
        plot_bias_data(interpreted_data=file_name, base_path=hdf(top_ref, name, bias_set),
                       use_group=True, lock=tb_lock)

    plot_inter_pix_data(interpreted_data=file_name, base_path=hdf(top_ref, name, INTER_UNBIASED),
                        use_group=True, distribution=True,
                        test_cap_exclusion=True, total_data=X5_SCAN_FILE,
                        total_path=hdf(top_ref, name, UNBIASED), lock=tb_lock, **kwargs)
    plot_inter_pix_data(interpreted_data=file_name,
                        base_path=hdf(top_ref, name, INTER_UNBIASED_MODEL),
                        use_group=True, distribution=True,
                        test_cap_exclusion=True, suffix=INTER_MIX, total_data=X5_SCAN_FILE,
                        total_path=hdf(top_ref, name, UNBIASED), lock=tb_lock, **kwargs)
    plot_inter_pix_data(interpreted_data=file_name, base_path=hdf(top_ref, name, INTER_UNBIASED),
                        use_group=True, distribution=True,
                        test_cap_exclusion=True, suffix=INTER_MIX, total_data=X5_SCAN_FILE,
                        total_path=hdf(top_ref, name, UNBIASED_MODEL), lock=tb_lock, **kwargs)
    plot_inter_pix_data(interpreted_data=file_name,
                        base_path=hdf(top_ref, name, INTER_UNBIASED_MODEL), use_group=True,
                        distribution=True, test_cap_exclusion=True,
                        total_data=file_name, total_path=hdf(top_ref, name, UNBIASED_MODEL),
                        lock=tb_lock, **kwargs)

    plot_inter_pix_data(interpreted_data=file_name,
                        base_path=hdf(top_ref, name, 'inter_biased_M_{}_V_full'.format(bias)), use_group=True,
                        test_cap_exclusion=True, distribution=True,
                        total_data=file_name, total_path=hdf(top_ref, name, 'biased_{}_V_full'.format(bias)),
                        lock=tb_lock, **kwargs)
    plot_inter_pix_data(interpreted_data=file_name,
                        base_path=hdf(top_ref, name, 'inter_biased_M_{}_V_full_model'.format(bias)), use_group=True,
                        test_cap_exclusion=True, distribution=True,
                        total_data=file_name, total_path=hdf(top_ref, name, 'biased_{}_V_full_model'.format(bias)),
                        lock=tb_lock, **kwargs)
    plot_inter_pix_data(interpreted_data=file_name,
                        base_path=hdf(top_ref, name, 'inter_biased_M_{}_V_full_model'.format(bias)), use_group=True,
                        test_cap_exclusion=True, distribution=True,
                        suffix=INTER_MIX, total_data=file_name,
                        total_path=hdf(top_ref, name, 'biased_{}_V_full'.format(bias)), lock=tb_lock, **kwargs)
    plot_inter_pix_data(interpreted_data=file_name,
                        base_path=hdf(top_ref, name, 'inter_biased_M_{}_V_full'.format(bias)), use_group=True,
                        test_cap_exclusion=True, distribution=True,
                        suffix=INTER_MIX, total_data=file_name,
                        total_path=hdf(top_ref, name, 'biased_{}_V_full_model'.format(bias)), lock=tb_lock, **kwargs)

    print(display_name, "- CV")
    for cv_name in simple_cv_names:
        threaded_plotting.plot_combined_data(interpreted_data=file_name,
                                             base_path=hdf(top_ref, name, cv_name),
                                             use_group=True, distribution=False, lock=tb_lock, apply_doping=False,
                                             **kwargs)
        threaded_plotting.plot_combined_data(interpreted_data=file_name,
                                             base_path=hdf(top_ref, name, cv_name),
                                             use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                             lock=tb_lock, **kwargs)

    for cv_name in refined_cv_names:
        threaded_plotting.plot_combined_data(interpreted_data=file_name,
                                             base_path=hdf(top_ref, name, cv_name),
                                             use_group=True, distribution=True, lock=tb_lock, apply_doping=False,
                                             **kwargs)
        threaded_plotting.plot_combined_data(interpreted_data=file_name,
                                             base_path=hdf(top_ref, name, cv_name),
                                             use_group=True, use_corrected=True, apply_doping=False, distribution=True,
                                             lock=tb_lock, **kwargs)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def r1_plotter(tb_lock):
    """
    Handles the plotting of the measurements with the planar Sensor R1/R11 (LF (CMOS) sample).

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "R1"
    display_name = name
    top_ref = "Reference"
    print("Plotting", display_name)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), use_group=True,
              use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'),
              use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'),
              use_group=True, use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'), use_group=True,
              use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
              use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
              use_group=True, use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    plot_bias_data(interpreted_data=R11_SCAN_FILE, base_path=hdf(top_ref, name, 'I_V_Characteristic'),
                   use_group=True, lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask)

    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full'), use_group=True,
                        distribution=True, test_cap_exclusion=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full'), use_group=True,
                        distribution=True, test_cap_exclusion=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full_model'), lock=tb_lock, suffix="inter-mix",
                        mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model'), use_group=True,
                        distribution=True, test_cap_exclusion=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full'), lock=tb_lock, suffix="inter-mix",
                        mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model'), use_group=True,
                        distribution=True, test_cap_exclusion=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full_model'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'), lock=tb_lock,
                        suffix="inter-mix", mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        suffix="inter-mix", mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask)

    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'), lock=tb_lock,
                        suffix="inter-mix", mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        suffix="inter-mix", mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask)

    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__diagonals'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=18000)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__sides'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended'),
                        grouped_inter_pix_id=20000)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__tops'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended'),
                        grouped_inter_pix_id=19000)

    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__diagonals'),
                        use_group=True, test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=18000)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__sides'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=20000)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__tops'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=19000)

    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__diagonals'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=18000)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__sides'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended'),
                        grouped_inter_pix_id=20000)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__tops'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended'),
                        grouped_inter_pix_id=19000)

    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__diagonals'),
                        use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=18000)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__sides'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=20000)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__tops'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask,
                        inter_data=R11_SCAN_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=19000)

    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data=R11_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, distribution=False, lock=tb_lock, apply_doping=False,
                                         mask_pixel=data_constants.r1_pixel_mask)
    threaded_plotting.plot_combined_data(interpreted_data=R11_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                         lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask)
    threaded_plotting.plot_combined_data(interpreted_data=R11_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, distribution=True, lock=tb_lock, apply_doping=False,
                                         mask_pixel=data_constants.r1_pixel_mask)
    threaded_plotting.plot_combined_data(interpreted_data=R11_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=True,
                                         lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def r13_plotter_first(tb_lock):
    """
    Handles the plotting of the measurements with the planar Sensor R13 (LF (CMOS) sample) in the first measurement
    run group.

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "R13"
    display_name = name
    top_ref = "Reference"
    print("Plotting", display_name)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/data.h5', suffix="test_run", use_group=False)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/data.h5', suffix="test_run", use_group=False,
              use_corrected=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/Test.h5', suffix="test-general_run", use_group=False,
              lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/Test.h5', suffix="test-general_run", use_group=False,
              use_corrected=True, lock=tb_lock)
    plot_data(interpreted_data='Reference_R13_Scan.h5', suffix="test-general_run", use_group=False,
              base_path="Reference/R13/unbiased_12_full", test_cap_exclusion=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data='Reference_R13_Scan.h5', suffix="test-general_run", use_group=False,
              use_corrected=True, base_path="Reference/R13/unbiased_12_full", test_cap_exclusion=True,
              distribution=True, lock=tb_lock)
    threaded_plotting.plot_bias_data(interpreted_data='pixcap65/Data/r13-measurement/R13_BIAS_2.h5', lock=tb_lock)
    # Data/r13-measurement/R13_BIAS_CV_COMBI_2.h5 no further investigation possible as data set is incomplete!
    # Data/r13-measurement/R13_BIAS_CV_COMBI_3.h5 no further investigation possible as data set is incomplete!
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5',
                                suffix="general_data",
                                use_group=False, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data",
              use_group=False, use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Initial_3_Scan.h5',
              base_path="ATLAS ITk/unbiased_1", suffix="unbiased_full_measurement", use_group=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Initial_3_Scan.h5',
              base_path="ATLAS ITk/unbiased_1", suffix="unbiased_full_measurement", use_group=True,
              use_corrected=True, lock=tb_lock)
    plot_inter_pix_data(interpreted_data='R13-Interpixel_Scan.h5',
                        base_path=hdf(top_ref, name, 'demo_measurement_65_unbiased_1_discharge'),
                        use_group=True, suffix="inter_pix_65", total_data="Data/r13-measurement/TEST.h5",
                        distribution=True, set_parasitic=False, lock=tb_lock)

    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_5.h5',
                                         lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_5.h5',
                                         use_corrected=True, apply_doping=True, lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5',
                                         lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5',
                                         use_corrected=True, apply_doping=True, lock=tb_lock)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def r13_plotter_second(tb_lock):
    """
    Handles the plotting of the measurements with the planar Sensor R13 (LF (CMOS) sample) in the second measurement
    run group with enhanced accuracy.

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "R13"
    display_name = name + " Second Try."
    top_ref = "Reference"
    print("Plotting", display_name)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full'),
              use_group=True, lock=tb_lock, test_cap_exclusion=True, distribution=True)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full'),
              use_group=True, lock=tb_lock, test_cap_exclusion=True, distribution=True, use_corrected=True)
    noqa: S1192
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'),
              use_group=True, lock=tb_lock, distribution=True)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'),
              use_group=True, lock=tb_lock, distribution=True, use_corrected=True)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full_model'),
              use_group=True, lock=tb_lock, test_cap_exclusion=True, distribution=True)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full_model'),
              use_group=True, lock=tb_lock, test_cap_exclusion=True, distribution=True, use_corrected=True)
    noqa: S1192
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
              use_group=True, lock=tb_lock, distribution=True)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
              use_group=True, lock=tb_lock, distribution=True, use_corrected=True)

    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full'), lock=tb_lock, use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_1_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full'), lock=tb_lock, use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path=hdf(top_ref, name, 'unbiased_1_full_model'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model'), lock=tb_lock,
                        use_group=True, test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        suffix="inter-mix", total_path=hdf(top_ref, name, 'unbiased_1_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model'), lock=tb_lock,
                        use_group=True, test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_1_full_model'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path=hdf(top_ref, name, 'biased_80_V_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'))

    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_renew'), lock=tb_lock, use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_1_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_renew'), lock=tb_lock, use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path=hdf(top_ref, name, 'unbiased_1_full_model'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_renew_model'), lock=tb_lock,
                        use_group=True, test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        suffix="inter-mix", total_path=hdf(top_ref, name, 'unbiased_1_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_renew_model'), lock=tb_lock,
                        use_group=True, test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_1_full_model'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_renew'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_renew'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_renew_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path=hdf(top_ref, name, 'biased_80_V_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_renew_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'))

    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_renew_Extended_full'), lock=tb_lock,
                        use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_1_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_renew_Extended_full'), lock=tb_lock,
                        use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path=hdf(top_ref, name, 'unbiased_1_full_model'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_renew_Extended_full_model'), lock=tb_lock,
                        use_group=True, test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        suffix="inter-mix", total_path=hdf(top_ref, name, 'unbiased_1_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_renew_Extended_full_model'), lock=tb_lock,
                        use_group=True, test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_1_full_model'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_renew_Extended_full'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_renew_Extended_full'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_renew_Extended_full_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path=hdf(top_ref, name, 'biased_80_V_full'))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_renew_Extended_full_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'))

    # the uncertainties of the capacitance estimators looks quite large.
    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data=R13_2_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'), lock=tb_lock,
                                         use_group=True, test_cap_exclusion=True, distribution=True,
                                         apply_doping=False,)
    threaded_plotting.plot_combined_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock, apply_doping=True,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, test_cap_exclusion=True, distribution=True, use_corrected=True)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def e1_plotter_first(tb_lock):
    """
    Handles the plotting of the measurements with the planar Sensor E1 (LF (CMOS) sample) in the second measurement
    run group with enhanced accuracy.

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.
    This is the implementation for the first measurements and therefore ignores the different sizes of the pixel
    implants.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "E1"
    display_name = name
    top_ref = "Reference"
    print("Plotting", display_name)
    plot_data(interpreted_data=E1_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_4_full'),
              use_group=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock, )
    plot_data(interpreted_data=E1_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_4_full'),
              use_group=True, use_corrected=True, distribution=True, test_cap_exclusion=True,
              mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock, )
    plot_data(interpreted_data=E1_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_1_test'),
              use_group=True, lock=tb_lock)
    plot_data(interpreted_data=E1_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_2_test'),
              use_group=True, lock=tb_lock)
    plot_data(interpreted_data=E1_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_3_test'),
              use_group=True, lock=tb_lock)
    threaded_plotting.plot_bias_data(interpreted_data=E1_SCAN_FILE,
                                     base_path=hdf(top_ref, name, 'I_V_Characteristic'),
                                     use_group=True, lock=tb_lock)
    print(display_name, "- CV")
    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data=E1_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, distribution=True, lock=tb_lock, )
    threaded_plotting.plot_combined_data(interpreted_data=E1_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                         lock=tb_lock, )
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def e1_plotter_second(tb_lock):
    """
    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17

    Handles the plotting of the measurements with the planar Sensor E1 (LF (CMOS) sample) in the second measurement
    run group with enhanced accuracy.
    This special in the sense that it has different pixel implantation sizes and depths.
    So we will iterate over the different regions to generate the plots.

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "E1"
    display_name = name + "Second Try."
    top_ref = "Reference"
    print("Plotting", display_name)
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
              base_path=hdf(top_ref, name, 'unbiased_full'), test_cap_exclusion=True,
              mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock, )
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
              base_path=hdf(top_ref, name, 'unbiased_full'), test_cap_exclusion=True,
              mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, lock=tb_lock, )
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
              base_path=hdf(top_ref, name, 'biased_80_V_full'),
              mask_pixel=data_constants.e1_pixel_mask, test_cap_exclusion=True, lock=tb_lock, )
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
              base_path=hdf(top_ref, name, 'biased_80_V_full'),
              mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, test_cap_exclusion=True,
              lock=tb_lock, )
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
              base_path=hdf(top_ref, name, 'unbiased_full_model'), test_cap_exclusion=True,
              mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock, )
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
              base_path=hdf(top_ref, name, 'unbiased_full_model'), test_cap_exclusion=True,
              mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, lock=tb_lock, )
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
              base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
              mask_pixel=data_constants.e1_pixel_mask, test_cap_exclusion=True, lock=tb_lock, )
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
              base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
              mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, test_cap_exclusion=True,
              lock=tb_lock, )
    plot_bias_data(interpreted_data=E1_2_SCAN_FILE,
                   base_path=hdf(top_ref, name, 'I_V_Characteristic'), use_group=True,
                   lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                        use_group=True, total_data=E1_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full'), lock=tb_lock,
                        mask_pixel=data_constants.e1_pixel_mask, )
    plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        use_group=True, total_data=E1_2_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.e1_pixel_mask, )

    threaded_plotting.plot_combined_data(interpreted_data=E1_2_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, distribution=True, mask_pixel=data_constants.e1_pixel_mask,
                                         lock=tb_lock, )
    threaded_plotting.plot_combined_data(interpreted_data=E1_2_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, distribution=True, mask_pixel=data_constants.e1_pixel_mask,
                                         use_corrected=True, lock=tb_lock, )

    print("E1 - Switching to individual pixel regions.")
    for type_name in data_constants.e1_pixel_groups.keys():
        path_name = hdf(top_ref, name, 'unbiased_full_' + type_name)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path=path_name, test_cap_exclusion=True,
                  mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock, distribution=True)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path=hdf(top_ref, name, 'unbiased_full_' + type_name), test_cap_exclusion=True,
                  mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, lock=tb_lock, distribution=True)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path=hdf(top_ref, name, 'biased_80_V_full_' + type_name), distribution=True,
                  mask_pixel=data_constants.e1_pixel_mask, test_cap_exclusion=True, lock=tb_lock, )
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path=hdf(top_ref, name, 'biased_80_V_full_' + type_name),
                  mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, test_cap_exclusion=True,
                  lock=tb_lock, distribution=True)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path=hdf(top_ref, name, 'unbiased_full_model_' + type_name), test_cap_exclusion=True,
                  mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock, distribution=True)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path=hdf(top_ref, name, 'unbiased_full_model_' + type_name), test_cap_exclusion=True,
                  mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, lock=tb_lock, distribution=True)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path=hdf(top_ref, name, 'biased_80_V_full_model_' + type_name), distribution=True,
                  mask_pixel=data_constants.e1_pixel_mask, test_cap_exclusion=True, lock=tb_lock, )
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path=hdf(top_ref, name, 'biased_80_V_full_model_' + type_name),
                  mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, test_cap_exclusion=True,
                  lock=tb_lock, distribution=True)

        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path=hdf(top_ref, name, 'inter_unbiased_full_' + type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE,
                            total_path=hdf(top_ref, name, 'unbiased_full_' + type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask, )
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path=hdf(top_ref, name, 'inter_unbiased_full_model_' + type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE, suffix="inter-mix",
                            total_path=hdf(top_ref, name, 'unbiased_full_' + type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask, )
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path=hdf(top_ref, name, 'inter_unbiased_full_' + type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE, suffix="inter-mix",
                            total_path=hdf(top_ref, name, 'unbiased_full_model_' + type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask, )
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path=hdf(top_ref, name, 'inter_unbiased_full_model_' + type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE,
                            total_path=hdf(top_ref, name, 'unbiased_full_model_' + type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask, )

        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_' + type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE,
                            total_path=hdf(top_ref, name, 'biased_80_V_full_' + type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask, )
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_' + type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE, suffix="inter-mix",
                            total_path=hdf(top_ref, name, 'biased_80_V_full_model_' + type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask, )
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_' + type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE, suffix="inter-mix",
                            total_path=hdf(top_ref, name, 'biased_80_V_full_' + type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask, )
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_' + type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE,
                            total_path=hdf(top_ref, name, 'biased_80_V_full_model_' + type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask, )

        threaded_plotting.plot_combined_data(interpreted_data=E1_2_SCAN_FILE,
                                             base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_' + type_name),
                                             use_group=True, distribution=True, mask_pixel=data_constants.e1_pixel_mask,
                                             lock=tb_lock, apply_doping=False)
        threaded_plotting.plot_combined_data(interpreted_data=E1_2_SCAN_FILE,
                                             base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_' + type_name),
                                             use_group=True, distribution=True, mask_pixel=data_constants.e1_pixel_mask,
                                             use_corrected=True, lock=tb_lock, apply_doping=False)

    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def x1_plotter_first(tb_lock):
    """
    Handles the plotting of the measurements with the planar Sensor X1 (HPK sample) in the first measurement
    run group.

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X1"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5',
              base_path="run_1", suffix="test_run", use_group=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5',
              base_path="run_1", suffix="test_run", use_group=True, use_corrected=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5',
              base_path="run_2", suffix="test_run", use_group=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5',
              base_path="run_2", suffix="test_run", use_group=True, use_corrected=True,
              lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5',
              base_path="ATLAS ITk/run_1", suffix="test_run", use_group=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5',
              base_path="ATLAS ITk/run_1", suffix="test_run", use_group=True, use_corrected=True,
              lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5',
              base_path="ATLAS ITk/run_2", suffix="test_run", use_group=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5',
              base_path="ATLAS ITk/run_2", suffix="test_run", use_group=True, use_corrected=True,
              lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_Scan.h5', suffix="test_run",
              use_group=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_Scan.h5', suffix="test_run",
              use_group=True, use_corrected=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
              base_path="ATLAS ITk/run_1", suffix="general_data_test_test", use_group=True,
              lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
              base_path="ATLAS ITk/run_1", suffix="general_data_test_test", use_group=True,
              use_corrected=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
              base_path="ATLAS ITk/run_2", suffix="general_data", use_group=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
              base_path="ATLAS ITk/run_2", suffix="general_data", use_group=True,
              use_corrected=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
              suffix="general_data", use_group=True, test_cap_exclusion=True,
              mask_pixel=data_constants.x1_second_pixel_mask, distribution=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
              suffix="general_data", use_group=True, use_corrected=True,
              mask_pixel=data_constants.x1_second_pixel_mask, test_cap_exclusion=True, distribution=True,
              lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
              base_path="ATLAS ITk/unbiased_4", suffix="general_data", use_group=True,
              lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
              base_path="ATLAS ITk/unbiased_4", suffix="general_data", use_group=True,
              use_corrected=True, lock=tb_lock)
    threaded_plotting.plot_bias_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                                     base_path="ATLAS ITk/I_V_Characteristic", use_group=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V",
              suffix="general_data", use_group=True, exclude_test_cap=True,
              mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V",
              suffix="general_data", use_group=True, use_corrected=True, apply_doping=True, test_cap_exclusion=True,
              mask_pixel=data_constants.x1_second_pixel_mask, distribution=True, lock=tb_lock)

    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                                         base_path="ATLAS ITk/C_V_Characteristic", use_group=True, lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                                         base_path="ATLAS ITk/C_V_Characteristic", use_group=True, use_corrected=True,
                                         apply_doping=True, lock=tb_lock)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def x1_plotter(tb_lock):
    """
    Handles the plotting of the measurements with the planar Sensor X1 (HPK sample) in the second measurement
    run group with enhanced accuracy.

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased)


    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X1"
    display_name = name + " Second Try."
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_61_full'), use_group=True,
              test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask, distribution=True, lock=tb_lock,)
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_61_full'), use_group=True,
              test_cap_exclusion=True, use_corrected=True, mask_pixel=data_constants.x1_second_pixel_mask,
              distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'), use_group=True,
              test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask, distribution=True,
              lock=tb_lock, )
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'), use_group=True,
              test_cap_exclusion=True, use_corrected=True, mask_pixel=data_constants.x1_second_pixel_mask,
              distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_61_full_model'), use_group=True,
              test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask, distribution=True, lock=tb_lock,)
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_61_full_model'), use_group=True,
              test_cap_exclusion=True, use_corrected=True, mask_pixel=data_constants.x1_second_pixel_mask,
              distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'), use_group=True,
              test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask, distribution=True,
              lock=tb_lock, )
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'), use_group=True,
              test_cap_exclusion=True, use_corrected=True, mask_pixel=data_constants.x1_second_pixel_mask,
              distribution=True, lock=tb_lock, )

    plot_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5", base_path=hdf(top_ref, name, 'biased_200_V_full'),
              use_group=True,
              test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask, distribution=True,
              lock=tb_lock, )
    plot_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5", base_path=hdf(top_ref, name, 'biased_200_V_full'),
              use_group=True,
              test_cap_exclusion=True, use_corrected=True, mask_pixel=data_constants.x1_second_pixel_mask,
              distribution=True, lock=tb_lock, )
    plot_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
              base_path=hdf(top_ref, name, 'biased_200_V_full_model'), use_group=True,
              test_cap_exclusion=True, mask_pixel=data_constants.x1_second_pixel_mask, distribution=True,
              lock=tb_lock, )
    plot_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
              base_path=hdf(top_ref, name, 'biased_200_V_full_model'), use_group=True,
              test_cap_exclusion=True, use_corrected=True, mask_pixel=data_constants.x1_second_pixel_mask,
              distribution=True, lock=tb_lock,)

    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                        use_group=True, test_cap_exclusion=True, distribution=True,
                        total_data=X1_SCAN_2_FILE, total_path=hdf(top_ref, name, 'unbiased_61_full'),
                        mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                        use_group=True, test_cap_exclusion=True, distribution=True,
                        total_data=X1_SCAN_2_FILE, total_path=hdf(top_ref, name, 'unbiased_61_full_model'),
                        mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock, suffix="inter-mix")
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_61_full_model'),
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_61_full'),
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock, suffix="inter-mix")
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'),
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        use_group=True, test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'),
                        mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock, suffix="inter-mix")
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'),
                        mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'),
                        use_group=True, test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'),
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock, suffix="inter-mix")

    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_renew_Extended'),
                        use_group=True, test_cap_exclusion=True, distribution=True,
                        total_data=X1_SCAN_2_FILE, total_path=hdf(top_ref, name, 'unbiased_61_full'),
                        mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock, )
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_renew_Extended'),
                        use_group=True, test_cap_exclusion=True, distribution=True,
                        total_data=X1_SCAN_2_FILE, total_path=hdf(top_ref, name, 'unbiased_61_full_model'),
                        mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock, suffix="inter-mix")
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model_renew_Extended'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_61_full_model'),
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock, )
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model_renew_Extended'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_61_full'),
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock, suffix="inter-mix")

    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_200_V_full_renew_Extended'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data="packaged/data/X1_12_Renew_Scan.h5",
                        total_path=hdf(top_ref, name, 'biased_200_V_full'),
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock, )
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_200_V_full_renew_Extended'),
                        use_group=True, test_cap_exclusion=True, distribution=True,
                        total_data="packaged/data/X1_12_Renew_Scan.h5",
                        total_path=hdf(top_ref, name, 'biased_200_V_full_model'),
                        mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock, suffix="inter-mix")
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_200_V_full_model_renew_Extended'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data="packaged/data/X1_12_Renew_Scan.h5",
                        total_path=hdf(top_ref, name, 'biased_200_V_full_model'),
                        mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock, )
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_200_V_full_model_renew_Extended'),
                        use_group=True, test_cap_exclusion=True, distribution=True,
                        total_data="packaged/data/X1_12_Renew_Scan.h5",
                        total_path=hdf(top_ref, name, 'biased_200_V_full'),
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock, suffix="inter-mix")

    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__diagonals'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=18000)
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__sides'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=20000)
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__tops'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=19000)

    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__diagonals'),
                        use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=18000)
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__sides'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=20000)
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__tops'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=19000)

    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended_2__diagonals'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=18000)
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended_2__sides'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=20000)
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended_2__tops'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=19000)

    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended_2__diagonals'),
                        use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=18000)
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended_2__sides'),
                        use_group=True, test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=20000)
    plot_inter_pix_data(interpreted_data="packaged/data/X1_12_Renew_Scan.h5",
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended_2__tops'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x1_second_pixel_mask,
                        inter_data=X1_SCAN_2_FILE, inter_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                        grouped_inter_pix_id=19000)

    threaded_plotting.plot_bias_data(interpreted_data=X1_SCAN_2_FILE,
                                     base_path=hdf(top_ref, name, 'I_V_Characteristic'),
                                     use_group=True, lock=tb_lock, )

    print(display_name, "- CV")
    plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                       base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                       use_group=True, mask_pixel=data_constants.x1_second_pixel_mask,
                       distribution=True, apply_doping=False,
                       lock=tb_lock, )
    plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                       base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                       use_group=True, use_corrected=True,
                       apply_doping=False, distribution=True,
                       mask_pixel=data_constants.x1_second_pixel_mask,
                       lock=tb_lock, )
    plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                       base_path=hdf(top_ref, name, 'C_V_Characteristic_Second_Extended'),
                       use_group=True, mask_pixel=data_constants.x1_second_pixel_mask,
                       distribution=False, apply_doping=False,
                       lock=tb_lock, )
    plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                       base_path=hdf(top_ref, name, 'C_V_Characteristic_Second_Extended'),
                       use_group=True, use_corrected=True,
                       apply_doping=False, distribution=False,
                       mask_pixel=data_constants.x1_second_pixel_mask,
                       lock=tb_lock, )
    plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                       base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_Extended_Combined'),
                       use_group=True, mask_pixel=data_constants.x1_second_pixel_mask,
                       distribution=True, apply_doping=False,
                       lock=tb_lock)
    plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                       base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_Extended_Combined'),
                       use_group=True, use_corrected=True,
                       apply_doping=True, distribution=True,
                       mask_pixel=data_constants.x1_second_pixel_mask,
                       lock=tb_lock)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def x2_plotter_first(tb_lock):
    """
    Handles the plotting of the measurements with the planar Sensor X2 (HPK sample) in the first
    measurement run group.

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:
    * 0V (unbiased)
    * -80V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X2"
    display_name = name
    top_ref = "ATLAS_ITk"
    print("Plotting", display_name)
    threaded_plotting.plot_data(interpreted_data='New_2_Scan.h5',
                                base_path=hdf(top_ref, name, 'unbiased_1'), suffix="general_data_80V",
                                use_group=True, test_cap_exclusion=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='New_2_Scan.h5',
                                base_path=hdf(top_ref, name, 'unbiased_1'), suffix="general_data_80V",
                                use_group=True, use_corrected=True, test_cap_exclusion=True, lock=tb_lock)
    threaded_plotting.plot_bias_data(interpreted_data='New_2_Scan.h5',
                                     base_path=hdf(top_ref, name, 'I_V_Characteristic'), use_group=True,
                                     lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='New_2_Scan.h5',
                                base_path=hdf(top_ref, name, 'biased_80_V'), suffix="general_data_80V",
                                use_group=True, test_cap_exclusion=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='New_2_Scan.h5',
                                base_path=hdf(top_ref, name, 'biased_80_V'), suffix="general_data_80V",
                                use_group=True, use_corrected=True, apply_doping=True, test_cap_exclusion=True,
                                lock=tb_lock)

    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data='New_2_Scan.h5',
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data='New_2_Scan.h5',
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, use_corrected=True, lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data='New_2_Scan.h5',
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data='New_2_Scan.h5',
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, use_corrected=True,
                                         apply_doping=True, distribution=True, lock=tb_lock)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def x2_plotter_second(tb_lock):
    """
    Handles the plotting of the measurements with the planar Sensor X2 (HPK sample) in the second measurments
    run group with enhanced accuracy.

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:
    * 0V (unbiased)
    * -80V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X2"
    display_name = name + "Second Try."
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full'), use_group=True,
              test_cap_exclusion=True, use_corrected=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'), use_group=True,
              test_cap_exclusion=True, use_corrected=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_200_V_full'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_200_V_full'), use_group=True,
              test_cap_exclusion=True, use_corrected=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full_model'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full_model'), use_group=True,
              test_cap_exclusion=True, use_corrected=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'), use_group=True,
              test_cap_exclusion=True, use_corrected=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_200_V_full_model'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_200_V_full_model'), use_group=True,
              test_cap_exclusion=True, use_corrected=True, distribution=True, lock=tb_lock, )
    plot_bias_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'I_V_Characteristic_2'),
                   use_group=True, lock=tb_lock, )

    # plot inter-pix capacitance data

    # plot the c-v data
    print(display_name, "- CV")
    plot_combined_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                       use_group=True, lock=tb_lock, test_cap_exclusion=True, distribution=True, apply_doping=False, )
    plot_combined_data(interpreted_data=X2_SCAN_2_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                       use_group=True, use_corrected=True,
                       apply_doping=False, distribution=True, lock=tb_lock, test_cap_exclusion=True, )
    plot_combined_data(interpreted_data=X2_SCAN_2_FILE,
                       base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_extended_renew_retry'),
                       use_group=True, lock=tb_lock, distribution=True, test_cap_exclusion=True, apply_doping=False, )
    plot_combined_data(interpreted_data=X2_SCAN_2_FILE,
                       base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_extended_renew_retry'),
                       use_group=True, use_corrected=True,
                       apply_doping=False, distribution=True, lock=tb_lock, test_cap_exclusion=True)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def x4_plotter(tb_lock):
    """
    Handles the plotting of the measurements with the planar Sensor R1/R11 (LF (CMOS) sample).

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -80V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X4"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    plot_data(interpreted_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock, mask_pixel=data_constants.x4_pixel_mask)
    plot_data(interpreted_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), use_group=True,
              use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.x4_pixel_mask)
    plot_data(interpreted_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'),
              use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.x4_pixel_mask)
    plot_data(interpreted_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'),
              use_group=True, use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.x4_pixel_mask)
    plot_data(interpreted_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'), use_group=True,
              use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.x4_pixel_mask)
    plot_data(interpreted_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
              use_group=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.x4_pixel_mask)
    plot_data(interpreted_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
              use_group=True, use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.x4_pixel_mask)
    plot_bias_data(interpreted_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'I_V_Characteristic_Extended_8'),
                   use_group=True, lock=tb_lock, mask_pixel=data_constants.x4_pixel_mask)
    plot_bias_data(interpreted_data=X4_SCAN_FILE, base_path=hdf(top_ref, name, 'I_V_Characteristic_Extended_20'),
                   use_group=True, lock=tb_lock, mask_pixel=data_constants.x4_pixel_mask)

    plot_inter_pix_data(interpreted_data=X4_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full'), use_group=True,
                        distribution=True, test_cap_exclusion=True, total_data=X4_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full'), lock=tb_lock,
                        mask_pixel=data_constants.x4_pixel_mask)
    plot_inter_pix_data(interpreted_data=X4_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full'), use_group=True,
                        distribution=True, test_cap_exclusion=True, total_data=X4_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full_model'), lock=tb_lock, suffix="inter-mix",
                        mask_pixel=data_constants.x4_pixel_mask)
    plot_inter_pix_data(interpreted_data=X4_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model'), use_group=True,
                        distribution=True, test_cap_exclusion=True, total_data=X4_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full'), lock=tb_lock, suffix="inter-mix",
                        mask_pixel=data_constants.x4_pixel_mask)
    plot_inter_pix_data(interpreted_data=X4_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model'), use_group=True,
                        distribution=True, test_cap_exclusion=True, total_data=X4_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full_model'), lock=tb_lock,
                        mask_pixel=data_constants.x4_pixel_mask)
    plot_inter_pix_data(interpreted_data=X4_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X4_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'), lock=tb_lock,
                        suffix="inter-mix", mask_pixel=data_constants.x4_pixel_mask)
    plot_inter_pix_data(interpreted_data=X4_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X4_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        suffix="inter-mix", mask_pixel=data_constants.x4_pixel_mask)
    plot_inter_pix_data(interpreted_data=X4_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X4_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full'), lock=tb_lock,
                        mask_pixel=data_constants.x4_pixel_mask)
    plot_inter_pix_data(interpreted_data=X4_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'), use_group=True,
                        test_cap_exclusion=True, distribution=True, total_data=X4_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_80_V_full_model'), lock=tb_lock,
                        mask_pixel=data_constants.x4_pixel_mask)

    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data=X4_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_trial'),
                                         use_group=True, distribution=False, lock=tb_lock, apply_doping=False,
                                         mask_pixel=data_constants.x4_pixel_mask)
    threaded_plotting.plot_combined_data(interpreted_data=X4_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_trial'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                         lock=tb_lock, mask_pixel=data_constants.x4_pixel_mask)
    threaded_plotting.plot_combined_data(interpreted_data=X4_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_Extended'),
                                         use_group=True, distribution=True, lock=tb_lock, apply_doping=False,
                                         mask_pixel=data_constants.x4_pixel_mask)
    threaded_plotting.plot_combined_data(interpreted_data=X4_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_Extended'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=True,
                                         lock=tb_lock, mask_pixel=data_constants.x4_pixel_mask)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def x5_plotter(tb_lock):
    """
    Handles the plotting of the measurements with the 3D-Sensor X5 (FBK sample).

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -40V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X5"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), use_group=True,
              test_cap_exclusion=True, mask_pixel=data_constants.x5_second_pixel_mask, distribution=True, lock=tb_lock,)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), use_group=True,
              use_corrected=True, test_cap_exclusion=True, mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40_V_full'),
              use_group=True, test_cap_exclusion=True, mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock, mask_lower=50e-15)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40_V_full'),
              use_group=True, use_corrected=True, test_cap_exclusion=True,
              mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock, mask_lower=50e-15)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_90_V_full'),
              use_group=True, test_cap_exclusion=True, mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_90_V_full'),
              use_group=True, use_corrected=True, test_cap_exclusion=True,
              mask_pixel=data_constants.x5_second_pixel_mask, distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'),
              use_group=True, test_cap_exclusion=True, mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'), use_group=True,
              use_corrected=True, test_cap_exclusion=True, mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40_V_full_model'),
              use_group=True, test_cap_exclusion=True, mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock, mask_lower=51.5e-15)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40_V_full_model'),
              use_group=True, use_corrected=True, test_cap_exclusion=True,
              mask_pixel=data_constants.x5_second_pixel_mask, distribution=True, lock=tb_lock, mask_lower=51.5e-15)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_90_V_full_model'),
              use_group=True, test_cap_exclusion=True, mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_90_V_full_model'),
              use_group=True, use_corrected=True, test_cap_exclusion=True,
              mask_pixel=data_constants.x5_second_pixel_mask, distribution=True, lock=tb_lock)
    plot_bias_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'I_V_Characteristic'),
                   use_group=True, lock=tb_lock, )
    plot_bias_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'I_V_Characteristic_Extended'),
                   use_group=True, lock=tb_lock, )

    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                        use_group=True, distribution=True, mask_pixel=data_constants.x5_second_pixel_mask,
                        test_cap_exclusion=True, total_data=X5_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                        use_group=True, distribution=True, mask_pixel=data_constants.x5_second_pixel_mask,
                        test_cap_exclusion=True, suffix="inter-mix", total_data=X5_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                        use_group=True, distribution=True, mask_pixel=data_constants.x5_second_pixel_mask,
                        test_cap_exclusion=True, suffix="inter-mix", total_data=X5_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'unbiased_full_model'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_unbiased_full_model'), use_group=True,
                        distribution=True, mask_pixel=data_constants.x5_second_pixel_mask, test_cap_exclusion=True,
                        total_data=X5_SCAN_FILE, total_path=hdf(top_ref, name, 'unbiased_full_model'),
                        lock=tb_lock, )

    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_40_V_full'), use_group=True,
                        mask_pixel=data_constants.x5_pixel_mask, test_cap_exclusion=True, distribution=True,
                        total_data=X5_SCAN_FILE, total_path=hdf(top_ref, name, 'biased_40_V_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_40_V_full_model'), use_group=True,
                        mask_pixel=data_constants.x5_pixel_mask, test_cap_exclusion=True, distribution=True,
                        total_data=X5_SCAN_FILE, total_path=hdf(top_ref, name, 'biased_40_V_full_model'),
                        lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_40_V_full_model'), use_group=True,
                        mask_pixel=data_constants.x5_pixel_mask, test_cap_exclusion=True, distribution=True,
                        suffix="inter-mix", total_data=X5_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_40_V_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path=hdf(top_ref, name, 'inter_biased_M_40_V_full'), use_group=True,
                        mask_pixel=data_constants.x5_pixel_mask, test_cap_exclusion=True, distribution=True,
                        suffix="inter-mix", total_data=X5_SCAN_FILE,
                        total_path=hdf(top_ref, name, 'biased_40_V_full_model'), lock=tb_lock, )

    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data=X5_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, distribution=False, lock=tb_lock, apply_doping=False,
                                         mask_pixel=data_constants.x5_second_pixel_mask, )
    threaded_plotting.plot_combined_data(interpreted_data=X5_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                         lock=tb_lock,
                                         mask_pixel=data_constants.x5_second_pixel_mask, )
    threaded_plotting.plot_combined_data(interpreted_data=X5_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, distribution=True, lock=tb_lock, apply_doping=False,
                                         mask_pixel=data_constants.x5_second_pixel_mask, )
    threaded_plotting.plot_combined_data(interpreted_data=X5_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=True,
                                         lock=tb_lock, mask_pixel=data_constants.x5_second_pixel_mask, )
    threaded_plotting.plot_combined_data(interpreted_data=X5_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_Extended'),
                                         use_group=True, distribution=True, lock=tb_lock, apply_doping=False,
                                         mask_pixel=data_constants.x5_pixel_mask, )
    threaded_plotting.plot_combined_data(interpreted_data=X5_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_Extended'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=True,
                                         lock=tb_lock, mask_pixel=data_constants.x5_pixel_mask, )
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def x6_plotter(tb_lock):
    """
    Handles the plotting of the measurements with the 3D-Sensor X6 (Sintef sample).

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -40V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X6"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), use_group=True,
              test_cap_exclusion=True, lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask, distribution=True,)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), use_group=True,
              use_corrected=True, test_cap_exclusion=True, lock=tb_lock,
              mask_pixel=data_constants.x6_second_pixel_mask, distribution=True, )
    plot_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'), use_group=True,
              test_cap_exclusion=True, lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask, distribution=True,)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'), use_group=True,
              use_corrected=True, test_cap_exclusion=True, lock=tb_lock,
              mask_pixel=data_constants.x6_second_pixel_mask, distribution=True, )
    plot_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_45_V_full'), use_group=True,
              test_cap_exclusion=True, lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask, distribution=True,)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_45_V_full'), use_group=True,
              use_corrected=True, test_cap_exclusion=True, lock=tb_lock,
              mask_pixel=data_constants.x6_second_pixel_mask, distribution=True, )
    plot_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_45_V_full_model'), use_group=True,
              test_cap_exclusion=True, lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask, distribution=True,)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_45_V_full_model'), use_group=True,
              use_corrected=True, test_cap_exclusion=True, lock=tb_lock,
              mask_pixel=data_constants.x6_second_pixel_mask, distribution=True, )

    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                        use_group=True, distribution=True,
                        test_cap_exclusion=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        total_data=X6_SCAN_FILE, total_path=hdf(top_ref, name, 'unbiased_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                        use_group=True, distribution=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        test_cap_exclusion=True, suffix="inter-mix",
                        total_data=X6_SCAN_FILE, total_path=hdf(top_ref, name, 'unbiased_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                        use_group=True, distribution=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        test_cap_exclusion=True, suffix="inter-mix",
                        total_data=X6_SCAN_FILE, total_path=hdf(top_ref, name, 'unbiased_full_model'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                        use_group=True, distribution=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        test_cap_exclusion=True,
                        total_data=X6_SCAN_FILE, total_path=hdf(top_ref, name, 'unbiased_full_model'), lock=tb_lock, )

    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_45_V_full'),
                        use_group=True, test_cap_exclusion=True,
                        distribution=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        total_data=X6_SCAN_FILE, total_path=hdf(top_ref, name, 'biased_45_V_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_45_V_full_model'),
                        use_group=True, test_cap_exclusion=True,
                        distribution=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        total_data=X6_SCAN_FILE, total_path=hdf(top_ref, name, 'biased_45_V_full_model'),
                        lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_45_V_full_model'),
                        use_group=True, test_cap_exclusion=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        distribution=True, suffix="inter-mix",
                        total_data=X6_SCAN_FILE, total_path=hdf(top_ref, name, 'biased_45_V_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_45_V_full'),
                        use_group=True, test_cap_exclusion=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        distribution=True, suffix="inter-mix",
                        total_data=X6_SCAN_FILE, total_path=hdf(top_ref, name, 'biased_45_V_full_model'),
                        lock=tb_lock, )

    print(display_name, "- CV")
    threaded_plotting.plot_bias_data(interpreted_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'I_V_Characteristic'),
                                     use_group=True,
                                     lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data=X6_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, distribution=False, lock=tb_lock, apply_doping=False, )
    threaded_plotting.plot_combined_data(interpreted_data=X6_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                         lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data=X6_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, distribution=True, lock=tb_lock, apply_doping=False,
                                         mask_pixel=data_constants.x6_second_pixel_mask, test_cap_exclusion=True, )
    threaded_plotting.plot_combined_data(interpreted_data=X6_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=True,
                                         mask_pixel=data_constants.x6_second_pixel_mask, test_cap_exclusion=True, )
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def x7_plotter(tb_lock):
    """
    Handles the plotting of the measurements with the 3D-Sensor X7 (Sintef sample).

    In a first run overview pdf of the unbiased and biased measurements are created including the distributions of these
    capacitances over the whole sensor.

    Next the analysis of the leakage current over the a large range of reversed biasing voltages.
    Then the Inter-Pixel-Capacitance analysis is plotted.
    There are two methods/models for extracting the capacitance applicable for both the in-pix-capacitance and the
    total-pixel capacitance. So for each biasing state, there four possibilities to combine these two methods to
    extract the inter-pixel capacitance.
    The plots are then created for each possibility for each biasing state.

    The applied biasing states are:

    * 0V (unbiased)
    * -40V (biased)

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X7"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), use_group=True,
              use_corrected=True, test_cap_exclusion=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'), use_group=True,
              use_corrected=True, test_cap_exclusion=True,
              distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40.0_V_full'),
              use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40.0_V_full'),
              use_group=True,
              use_corrected=True, test_cap_exclusion=True,
              distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40.0_V_full_model'), use_group=True,
              test_cap_exclusion=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40.0_V_full_model'), use_group=True,
              use_corrected=True, test_cap_exclusion=True,
              distribution=True, lock=tb_lock)

    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                        use_group=True, distribution=True,
                        test_cap_exclusion=True,
                        total_data=X7_SCAN_FILE, total_path=hdf(top_ref, name, 'unbiased_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                        use_group=True, distribution=True,
                        test_cap_exclusion=True, suffix="inter-mix",
                        total_data=X7_SCAN_FILE, total_path=hdf(top_ref, name, 'unbiased_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                        use_group=True, distribution=True,
                        test_cap_exclusion=True, suffix="inter-mix",
                        total_data=X7_SCAN_FILE, total_path=hdf(top_ref, name, 'unbiased_full_model'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                        use_group=True, distribution=True,
                        test_cap_exclusion=True,
                        total_data=X7_SCAN_FILE, total_path=hdf(top_ref, name, 'unbiased_full_model'), lock=tb_lock, )

    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_40.0_V_full'),
                        use_group=True, test_cap_exclusion=True,
                        distribution=True,
                        total_data=X7_SCAN_FILE, total_path=hdf(top_ref, name, 'biased_40.0_V_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_40.0_V_full_model'),
                        use_group=True, test_cap_exclusion=True,
                        distribution=True,
                        total_data=X7_SCAN_FILE, total_path=hdf(top_ref, name, 'biased_40.0_V_full_model'),
                        lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_40.0_V_full_model'),
                        use_group=True, test_cap_exclusion=True,
                        distribution=True, suffix="inter-mix",
                        total_data=X7_SCAN_FILE, total_path=hdf(top_ref, name, 'biased_40.0_V_full'), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_40.0_V_full'),
                        use_group=True, test_cap_exclusion=True,
                        distribution=True, suffix="inter-mix",
                        total_data=X7_SCAN_FILE, total_path=hdf(top_ref, name, 'biased_40.0_V_full_model'),
                        lock=tb_lock, )

    print(display_name, "- CV")
    threaded_plotting.plot_bias_data(interpreted_data=X7_SCAN_FILE,
                                     base_path=hdf(top_ref, name, 'I_V_Characteristic'), use_group=True,
                                     lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data=X7_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, distribution=False, lock=tb_lock, apply_doping=False, )
    threaded_plotting.plot_combined_data(interpreted_data=X7_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                         lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data=X7_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, distribution=True, lock=tb_lock, test_cap_exclusion=True,
                                         apply_doping=False, )
    threaded_plotting.plot_combined_data(interpreted_data=X7_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=True,
                                         lock=tb_lock, test_cap_exclusion=True)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def presentation_plotter(tb_lock):
    """
    presentation_plotter

    :author: Dominik Fischer
    :date: 2026-06-16

    last update: 2026-09-17

    Utility function to generate the (final) plots for my bachelors' thesis.+
    This only applies to the plots for c-v characterization.
    In this case the different c-v-curves of the different sensors are combined into a single plot.

    For the I-V Characterization the leakage current will be normalised onto the area of the pixel implantations.

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    e1_full_size = 0
    for key, content in data_constants.e1_pixel_groups.items():
        size_parameter, _ = key.removeprefix("dnw").removeprefix("nw").split("_", 1)
        implant_size = float(size_parameter)
        n_pixels = 0
        for col_reg, row_reg in zip(content["columns"], content["rows"]):
            n_pixels = (col_reg[1] - col_reg[0]) * (row_reg[1] - row_reg[0])

        e1_full_size += n_pixels * implant_size * implant_size
    print("Plot Presentable")
    print("The estimated E1 size is:")
    print(e1_full_size)
    with (tb_lock):
        # collect all our IV groups
        iv_file_names = [
            'pixcap65/Data/r13-measurement/R13_BIAS_2.h5',
            'pixcap65/Data/New_1_Initial_6_Scan.h5',
            X2_SCAN_FILE,
            X1_SCAN_2_FILE,
            "packaged/E1_Renew_Scan.h5",
            X2_SCAN_2_FILE,
            X2_SCAN_2_FILE,
            X1_SCAN_2_FILE,
            R13_2_SCAN_FILE,
            X2_SCAN_FILE,
            X2_SCAN_FILE,
            'pixcap65/Data/New_1_Initial_6_Scan.h5',
            R11_SCAN_FILE,
        ]
        iv_group_names = [
            None,
            "ATLAS ITk/I_V_Characteristic",
            "ATLAS_Itk/X2/I_V_Characteristic",
            "ATLAS_ITk/X1/I_V_Characteristic",
            "Reference/E1/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X2/I_V_Characteristic_2",
            "ATLAS_ITk/X2/C_V_Characteristic_refined",
            "ATLAS_ITk/X1/C_V_Characteristic_refined",
            "Reference/R13/C_V_Characteristic_refined",
            "ATLAS_Itk/X2/C_V_Characteristic",
            "ATLAS_Itk/X2/C_V_Characteristic_refined",
            "ATLAS ITk/C_V_Characteristic",
            "Reference/R11/I_V_Characteristic",
        ]
        iv_labels = [
            'R13',
            "(HPK) X1",
            "(HPK) X2",
            "X1 (second)",
            "E1 (second, CV)",
            "X2 (third?)",
            "X2 (second, CV)",
            "X1 (second, CV)",
            "R13 (second, CV)",
            "X2 (CV)",
            "X2 (CV, refined)",
            "X1 (CV)",
            "R1 (R11)",
        ]
        assert len(iv_file_names) == len(iv_group_names)
        assert len(iv_file_names) == len(iv_labels)
        normalisation = [
            40 * 40 * 30 * 30,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            e1_full_size,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 30 * 30,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 6 * 81.,
        ]
        plot_bias_data(iv_file_names, iv_group_names, pdf_name="I-V Combination.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_labels])
        plot_bias_data(iv_file_names, iv_group_names, pdf_name="I-V Combination-2.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_labels], area_normalisation=normalisation)

        iv_file_names_final = [
            R11_SCAN_FILE,
            R13_2_SCAN_FILE,
            X1_SCAN_2_FILE,
            X2_SCAN_2_FILE,
            E1_2_SCAN_FILE,
        ]
        iv_group_names_final = [
            "Reference/R11/I_V_Characteristic",
            "Reference/R13/C_V_Characteristic_refined",
            "ATLAS_ITk/X1/I_V_Characteristic",
            "Thesis/ATLAS_ITk/X2/I_V_Characteristic_2",
            "Reference/E1/C_V_Characteristic_refined",
        ]
        iv_labels_final = [
            "R1 (LF, 25x100)",
            "R13 (LF, 50x50)",
            "X1 (HPK)",
            "X2 (HPK)",
            "E1 (LF)",
        ]
        assert len(iv_file_names_final) == len(iv_group_names_final)
        assert len(iv_file_names_final) == len(iv_labels_final)
        normalisation_final = [
            40 * 40 * 45 * 45,
            e1_full_size,
            40 * 40 * 45 * 45,
            40 * 40 * 30 * 30,
            40 * 40 * 81 * 6,
        ]
        plot_bias_data(iv_file_names_final, iv_group_names_final, pdf_name="I-V Combination_final.pdf",
                       labels=iv_labels_final)
        plot_bias_data(iv_file_names_final, iv_group_names_final, pdf_name="I-V Combination-2_final.pdf",
                       labels=iv_labels_final,
                       area_normalisation=normalisation_final)

        print("CV Combination")
        cv_file_names = [
            X2_SCAN_FILE,
            X2_SCAN_FILE,
            X2_SCAN_2_FILE,
            X1_SCAN_2_FILE,
            R13_2_SCAN_FILE,
            X1_SCAN_2_FILE,
            "packaged/E1_Renew_Scan.h5",
            E1_SCAN_FILE,
            'pixcap65/Data/New_1_Initial_6_Scan.h5',
            R11_SCAN_FILE,
            X1_SCAN_2_FILE,
            X2_SCAN_2_FILE,
        ]
        cv_groups = [
            "ATLAS_Itk/X2/C_V_Characteristic",
            "ATLAS_Itk/X2/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X2/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X1/C_V_Characteristic_refined",
            "Reference/R13/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X1/C_V_Characteristic_Second_Extended",
            "Reference/E1/C_V_Characteristic_refined",
            "Reference/E1/C_V_Characteristic",
            "ATLAS ITk/C_V_Characteristic",
            "Reference/R1/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X1/C_V_Characteristic_Second_Extended",
            "Thesis/ATLAS_ITk/X2/C_V_Characteristic_refined_extended_renew_retry",
        ]
        cv_labels = [
            'X2_1_1',
            "X2_1_2",
            "X2_2",
            "X1 (second)",
            "R13 (second)",
            "X1 (second, extended)",
            "E1 (second)",
            "E1",
            "X1",
            "R1/R11",
            "X1 (combined)",
            "X2 (extended)",
        ]
        assert len(cv_file_names) == len(cv_groups)
        assert len(cv_file_names) == len(cv_labels)
        plot_cv_data(cv_file_names, cv_groups,
                     pdf_name="C-V Combination.pdf", labels=[CV_DATA_FOR_.format(item) for item in cv_labels],
                     use_corrected=True, distribution=True)
        plot_cv_data(cv_file_names, cv_groups,
                     pdf_name="C-V Combination_raw.pdf", labels=[CV_DATA_FOR_.format(item) for item in cv_labels],
                     use_corrected=False, distribution=True)

        cv_file_names_final = [
            R11_SCAN_FILE,
            R13_2_SCAN_FILE,
            X1_SCAN_2_FILE,
            X2_SCAN_2_FILE,
            E1_2_SCAN_FILE,
            E1_2_SCAN_FILE,
            E1_2_SCAN_FILE,
            E1_2_SCAN_FILE,
            E1_2_SCAN_FILE,
            E1_2_SCAN_FILE,
            E1_2_SCAN_FILE,
            E1_2_SCAN_FILE,
        ]
        cv_groups_final = [
            "Reference/R1/C_V_Characteristic_refined",
            "Reference/R13/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X1/C_V_Characteristic_refined_Extended_Combined",
            "Thesis/ATLAS_ITk/X2/C_V_Characteristic_refined_extended_renew_retry",
            "Reference/E1/C_V_Characteristic_refined_nw15_50",
            "Reference/E1/C_V_Characteristic_refined_nw20_50",
            "Reference/E1/C_V_Characteristic_refined_nw25_50",
            "Reference/E1/C_V_Characteristic_refined_nw30_50",
            "Reference/E1/C_V_Characteristic_refined_dnw15_50",
            "Reference/E1/C_V_Characteristic_refined_dnw20_50",
            "Reference/E1/C_V_Characteristic_refined_dnw25_50",
            "Reference/E1/C_V_Characteristic_refined_dnw30_50",
        ]
        cv_labels_final = [
            "R1 (LF, 6x81)",
            "R13 (LF, 30x30)",
            "X1 (HPK)",
            "X2 (HPK)",
            "E1nw15 (LF, 15x15)",
            "E1nw20 (LF, 20x20)",
            "E1nw25 (LF, 25x25)",
            "E1nw30 (LF, 30x30)",
            "E1dnw15 (LF, 15x15)",
            "E1dnw20 (LF, 20x20)",
            "E1dnw25 (LF, 25x25)",
            "E1dnw30 (LF, 30x30)",
        ]
        assert len(cv_file_names_final) == len(cv_groups_final)
        assert len(cv_file_names_final) == len(cv_labels_final)
        plot_cv_data(cv_file_names_final, cv_groups_final,
                     pdf_name="C-V Combination_final.pdf", labels=cv_labels_final,
                     use_corrected=True, distribution=True)
        plot_cv_data(cv_file_names_final, cv_groups_final,
                     pdf_name="C-V Combination_raw_final.pdf", labels=cv_labels_final,
                     use_corrected=False, distribution=True)

        iv_3d_file_names = [
            X5_SCAN_FILE,
            X5_SCAN_FILE,
            X7_SCAN_FILE,
            X7_SCAN_FILE,
            X6_SCAN_FILE,
            X6_SCAN_FILE,
            X5_SCAN_FILE,
            X4_SCAN_FILE,
            X4_SCAN_FILE,
        ]
        iv_3d_group_names = [
            "Thesis/ATLAS_ITk/X5/I_V_Characteristic",
            "Thesis/ATLAS_ITk/X5/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X7/I_V_Characteristic",
            "Thesis/ATLAS_ITk/X7/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X6/I_V_Characteristic",
            "Thesis/ATLAS_ITk/X6/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X5/I_V_Characteristic_Extended",
            "Thesis/ATLAS_ITk/X4/I_V_Characteristic_Extended_8",
            "Thesis/ATLAS_ITk/X4/I_V_Characteristic_Extended_10",
        ]
        iv_3d_labels = [
            "X5 (221-W6-J, FBK)",
            "X5 (CV, 221-W6-J, FBK)",
            "X7 (Sintef)",
            "X7 (CV, Sintef)",
            "X6 (Sintef)",
            "X6 (CV, coarse, Sintef)",
            "X5 (extended, 221-W6-J)",
            "X4 (breakdown, 221-W5-S, fehlerhaft)",
            "X4 (221-W5-S, fehlerhaft)",
        ]
        assert len(iv_3d_file_names) == len(iv_3d_group_names)
        assert len(iv_3d_file_names) == len(iv_3d_labels)
        # this is using the pixel size but not the electrodes size!
        normalisation = [
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
        ]
        plot_bias_data(iv_3d_file_names, iv_3d_group_names, pdf_name="I-V 3D Combination.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_3d_labels])
        plot_bias_data(iv_3d_file_names, iv_3d_group_names, pdf_name="I-V 3D Combination-2.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_3d_labels],
                       area_normalisation=normalisation)

        iv_3d_file_names_final = [
            X5_SCAN_FILE,
            X6_SCAN_FILE,
            X7_SCAN_FILE,
            X4_SCAN_FILE,
        ]
        iv_3d_group_names_final = [
            "Thesis/ATLAS_ITk/X5/I_V_Characteristic_Extended",
            "Thesis/ATLAS_ITk/X6/I_V_Characteristic",
            "Thesis/ATLAS_ITk/X7/I_V_Characteristic",
            "Thesis/ATLAS_ITk/X4/I_V_Characteristic_Extended_8",
        ]
        iv_3d_labels_final = [
            "X5 (FBK, W6_J)",
            "X6 (Sintef)",
            "X7 (Sintef)",
            "X4 (FBK, W5_S, fehlerhaft)",
        ]
        assert len(iv_3d_file_names_final) == len(iv_3d_group_names_final)
        assert len(iv_3d_file_names_final) == len(iv_3d_labels_final)
        normalisation_final = [
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50,
            40 * 40 * 50 * 50
        ]
        plot_bias_data(iv_3d_file_names_final, iv_3d_group_names_final, pdf_name="I-V 3D Combination_final.pdf",
                       labels=iv_3d_labels_final)
        plot_bias_data(iv_3d_file_names_final, iv_3d_group_names_final, pdf_name="I-V 3D Combination-2_final.pdf",
                       labels=iv_3d_labels_final,
                       area_normalisation=normalisation_final)

        print("CV Combination")
        cv_3d_file_names = [
            X5_SCAN_FILE,
            X5_SCAN_FILE,
            X6_SCAN_FILE,
            X7_SCAN_FILE,
            X7_SCAN_FILE,
            X5_SCAN_FILE,
            X6_SCAN_FILE,
        ]
        cv_3d_groups = [
            "Thesis/ATLAS_ITk/X5/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X5/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X6/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X7/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X7/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X5/C_V_Characteristic_refined_Extended",
            "Thesis/ATLAS_ITk/X6/C_V_Characteristic_refined",
        ]
        cv_3d_labels = [
            'X5',
            'X5 (refined)',
            "X6",
            "X7",
            "X7 (refined)",
            "X5 (extended)",
            "X6 (refined)",
        ]
        assert len(cv_3d_file_names) == len(cv_3d_groups)
        assert len(cv_3d_file_names) == len(cv_3d_labels)
        plot_cv_data(cv_3d_file_names, cv_3d_groups,
                     pdf_name="C-V 3D Combination_raw.pdf", labels=[CV_DATA_FOR_.format(item) for item in cv_3d_labels],
                     use_corrected=False, distribution=True)
        plot_cv_data(cv_3d_file_names, cv_3d_groups,
                     pdf_name="C-V 3D Combination.pdf", labels=[CV_DATA_FOR_.format(item) for item in cv_3d_labels],
                     use_corrected=True, distribution=True)

        cv_3d_file_names_final = [
            X5_SCAN_FILE,
            X6_SCAN_FILE,
            X7_SCAN_FILE,
            X4_SCAN_FILE,
        ]
        cv_3d_groups_final = [
            "Thesis/ATLAS_ITk/X5/C_V_Characteristic_refined_Extended",
            "Thesis/ATLAS_ITk/X6/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X7/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X4/C_V_Characteristic_refined_Extended",
        ]
        cv_3d_labels_final = [
            'X5 (FBK)',
            "X6 (Sintef)",
            "X7 (Sintef)",
            'X4 (FBK)'
        ]
        assert len(cv_3d_file_names_final) == len(cv_3d_groups_final)
        assert len(cv_3d_file_names_final) == len(cv_3d_labels_final)
        plot_cv_data(cv_3d_file_names_final, cv_3d_groups_final,
                     pdf_name="C-V 3D Combination_raw_final.pdf", labels=cv_3d_labels_final,
                     use_corrected=False, distribution=True)
        plot_cv_data(cv_3d_file_names_final, cv_3d_groups_final,
                     pdf_name="C-V 3D Combination_final.pdf", labels=cv_3d_labels_final,
                     use_corrected=True, distribution=True)


def mp_plotting_init():
    """
    Make sure that all sub-processes are initialized such that the matplotlib PDF backend is used anyway.
    """
    import matplotlib
    matplotlib.use('PDF')


if __name__ == '__main__':
    # additional setup
    # noinspection GrazieInspectionRunner
    set_params(latex=True,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                           r"stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                           r"retain-zero-uncertainty}", fig_height=8.26772, fig_width=11.69291, )
    bare_correction_args = {
        "apply_correction": True,
        "bare_file": "Bare_Repeat_2_Scan.h5",
        "bare_hdf_path": "Reference/bare/unbiased_8/total_cap",
    }

    doping_investigation_args = {

    }

    global_processing_lock = threading.RLock()


    def example_analysator(tb_lock, correction_args, logger, file_name, dep_bias, reference="Reference"):
        import multiprocessing as mp
        import threading
        name = "Y?"
        print("Analyzing ", name)
        print(threading.get_native_id())
        print(mp.current_process().name)
        print(mp.current_process().pid)

        # duplicate for the full sensor analysis to use it independently for both
        with synchronized_process_open_file(file_name, mode='a', lock=tb_lock) as h5_file:
            h5_file.copy_node(where="{}/{}".format(reference, name), newname="unbiased_full_model",
                              name="unbiased_full", recursive=True, overwrite=True)
            h5_file.copy_node(where="{}/{}".format(reference, name),
                              newname="biased_{}_V_full_model".format(dep_bias),
                              name="biased_{}_V_full".format(dep_bias), recursive=True, overwrite=True)

        # general analysis of the full sensor
        # this general example would make it necessary to transmit masks on some way.
        analyze_data(raw_data=file_name, base_path=hdf(reference, name, 'unbiased_full'),
                     is_advanced=True,
                     distribution=True, full_model=False, mask_pixel=data_constants.x5_second_pixel_mask,
                     exclude_cap_test=True,
                     lock=tb_lock,
                     **correction_args)
        analyze_data(raw_data=file_name, base_path=hdf(reference, name, 'biased_40_V_full'),
                     is_advanced=True,
                     distribution=True, full_model=False, mask_pixel=data_constants.x5_second_pixel_mask,
                     exclude_cap_test=True,
                     lock=tb_lock,
                     **correction_args)
        analyze_data(raw_data=file_name, base_path=hdf(reference, name, 'unbiased_full_model'),
                     is_advanced=True,
                     distribution=True, mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True,
                     lock=tb_lock,
                     **correction_args)
        analyze_data(raw_data=file_name, base_path=hdf(reference, name, 'biased_40_V_full_model'),
                     is_advanced=True,
                     distribution=True, mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True,
                     lock=tb_lock,
                     **correction_args)

        # Inter-Pixel Cap analysis


    # investigate the bare pixcap ship and it's distribution!
    print("Analyze the Bare samples for calibration of the pixcap chips")
    analyze_data(raw_data='pixcap65/Data/bare-measurement/TEST.h5', is_advanced=False)

    analyze_data(raw_data='pixcap65/Data/advanced-bare-measurement/TEST.h5', is_advanced=True, is_cv=False)
    analyze_data(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8", is_advanced=True, is_cv=False,
                 full_model=False)
    analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
                                     corrected_distribution=False,
                                     test_cap_exclusion=True, use_kafe2=True,
                                     fit_plot_pdf_name="Bare_analysis_parasitic_kafe2.pdf")
    analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
                                     corrected_distribution=False,
                                     test_cap_exclusion=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_parasitic.pdf")
    # investigate the bump capacitance
    with tb.open_file("../Bare_Repeat_2_Scan.h5", 'r') as f:
        base_group = get_base_group("Reference/bare/unbiased_8", f)
        bump_caps = np.concat(
            (base_group.total_cap.analysis.HistCap[:5, 0], base_group.total_cap.analysis.HistCap[35:, 0],))
        bump_errors = np.concat(
            (base_group.total_cap.analysis.HistCapErr[:5, 0], base_group.total_cap.analysis.HistCapErr[35:, 0],))
        weights = np.reciprocal(bump_errors ** 2)
        average_bump_cap = np.average(bump_caps, weights=weights)
        statistical_bump_error = np.shape(bump_errors)[0] / np.sum(weights)
        parasitic = get_group_attribute(base_group.total_cap.analysis, "parasitic")
        parasitic_error = get_group_attribute(base_group.total_cap.analysis, "parasitic_error")
        bump_capacitance = parasitic - average_bump_cap
        logger.info("bump capacitance: ({}+-{}+-{})".format(bump_capacitance,
                                                            statistical_bump_error,
                                                            parasitic_error))
        logger.info(np.std(bump_caps))

    # analysis section/calibration
    print("Analyze the R13 reference sample.")
    r13_analysator_first(global_processing_lock, bare_correction_args)


    def e1_analysator_first(tb_lock, correction_args, **kwargs):
        import pixcap65.concurrency
        name = "E1 (First Ref)"
        print("Analyze", name)
        print(threading.get_native_id())
        print(mp.current_process().name)
        print(mp.current_process().pid)

        _ = pixcap65.concurrency.get_manager(**kwargs)
        correction_args = correction_args.copy()
        correction_args.update(lock=tb_lock)

        analyze_data(raw_data=E1_SCAN_FILE, base_path="Reference/E1/unbiased_4_full", is_advanced=True,
                     **correction_args)

        e1_depletion_args = {
            "first_boundaries": (-100, -35),
            "second_boundaries": (-5, 0),
            "distribution": False,
            "apply_contour": False,
            "apply_contours": False,
            "pixel_mask": [[39, 1]]
        }
        e1_depletion_args.update(**correction_args)
        print(LOG_CV_ANALYSIS, name)
        with PdfPages("../Fit References/E1_C_V_Verify.pdf") as pdf:
            analyze_data(raw_data=E1_SCAN_FILE, base_path="Reference/E1/C_V_Characteristic",
                         is_advanced=True, full_model=False, is_cv=True, use_corrected=True, cv_fit_plot_pdf=pdf,
                         **e1_depletion_args)
        print(LOG_FINISHED_ANALYSIS, name)

        print(LOG_FINISHED_ANALYSIS, name)


    from examples.mp_analysis import SECOND_LABEL, AUTHKEY_OUTPUT, synchronize_full_model

    print("Analyze the ATLAS ITk samples.")
    x1_analysator_first(global_processing_lock, bare_correction_args)

    print("Analyze the ATLAS sample X2")
    x2_analysator_first(global_processing_lock, bare_correction_args)

    e1_analysator_first(global_processing_lock, bare_correction_args)

    # Second Try R13
    bare_correction_args = {
        "apply_correction": True,
        "bare_file": "packaged/Reference_Bare_renewed.h5",
        "bare_hdf_path": "Reference/Bare/unbiased_31_renew/total_cap",
    }
    r13_analysator_second(global_processing_lock, bare_correction_args)

    print("Analyze X1")
    x1_analysator_second(global_processing_lock, bare_correction_args)

    # Second Try X2
    # I do not think that we could use this data set properly as the C-V-curve ends to soon! will need
    # to perform a new measurement for higher voltages.
    print("Analyze X2")
    x2_analysator_second(global_processing_lock, bare_correction_args)

    e1_analysator_second(global_processing_lock, bare_correction_args)

    print("Analyze X5")
    x5_analysator(global_processing_lock, bare_correction_args)

    print("Analyze X6")
    x6_analysator(global_processing_lock, bare_correction_args)

    # plotting section
    print("Plots for the reference sample BARE 5 BUMPS")
    bare_sample_plotter_first(global_processing_lock)

    print("Plots for the reference sample R13")
    r13_plotter_first(global_processing_lock)

    print("Plots for the reference sample E1")
    e1_plotter_first(global_processing_lock)

    print("Plots for ATLAS sample X1")
    x1_plotter_first(global_processing_lock)

    print("Plots for ATLAS sample X2")
    x2_plotter_first(global_processing_lock)

    # second try Bare
    print("Plot Bare Second Try")
    bare_sample_plotter_second(global_processing_lock)

    print("Plot R13 Second Try.")
    r13_plotter_second(global_processing_lock)

    # second try X1
    print("Plot X1 Second Try.")
    x1_plotter(global_processing_lock)

    print("Plot X2 Second Try.")
    x2_plotter_second(global_processing_lock)

    # E1 first run
    print("Plot E1")
    e1_plotter_second(global_processing_lock)

    # X5 first run
    print("Plot X5")
    x5_plotter(global_processing_lock)

    print("Plot X6")
    x6_plotter(global_processing_lock)

    print("Plot X7")
    x7_plotter(global_processing_lock)
