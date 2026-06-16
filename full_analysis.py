"""
Analysis and plotting script evaluate all the data taking from the beginning!
"""

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

from os.path import join as hdf

import multiprocessing as mp
import numpy as np
import tables as tb
import threading
from matplotlib.backends.backend_pdf import PdfPages

import pixcap65.data_constants as data_constants
import pixcap65.threaded_plotting as threaded_plotting
from pixcap65.analysis import analyze_data, analyze_capacitance_distribution
from pixcap65.analysis_util.utility import get_base_group
from pixcap65.data_constants import E1_SCAN_FILE, E1_2_SCAN_FILE, R13_2_SCAN_FILE, R11_SCAN_FILE
from pixcap65.data_constants import X1_SCAN_2_FILE, X2_SCAN_2_FILE, X2_SCAN_FILE
from pixcap65.data_constants import X5_SCAN_FILE, X6_SCAN_FILE, X7_SCAN_FILE
from pixcap65.plotting import plot_data, plot_combined_data, plot_bias_data, plot_inter_pix_data, \
    plot_cv_data, CV_DATA_FOR_
from pixcap65.utility import synchronized_process_open_file
from pixcap65.utility.homogenize_plots import set_params
from pixcap65.utility.tables_util import get_group_attribute


# define the analysis handling
def r1_analysator(tb_lock, correction_args, **kwargs):
    import pixcap65.concurrency
    name = "R1"
    display_name = name
    top_ref = "Reference"
    print("Analyze", name)
    print(threading.get_native_id())
    print(mp.current_process().name)
    print(mp.current_process().pid)

    _ = pixcap65.concurrency.get_manager(**kwargs)

    with synchronized_process_open_file(R11_SCAN_FILE, mode='a', lock=tb_lock) as h5_file:
        h5_file.copy_node(where="/Reference/R11", newname="unbiased_full_model", name="unbiased_full",
                          recursive=True, overwrite=True)
        h5_file.copy_node(where="/Reference/R1", newname="biased_80_V_full_model", name="biased_80_V_full",
                          recursive=True, overwrite=True)
        h5_file.copy_node(where="/Reference/R1", newname="inter_unbiased_full_model", name="inter_unbiased_full",
                          recursive=True, overwrite=True)
        h5_file.copy_node(where="/Reference/R1", newname="inter_biased_M_80_V_full_model", name="inter_biased_M_80_V_full",
                          recursive=True, overwrite=True)

    # handle the full sensor analysis
    # name = "R11"
    analyze_data(raw_data=R11_SCAN_FILE, base_path="{}/{}/unbiased_full".format(top_ref, name),
                 is_advanced=True, distribution=True, full_model=False,
                 exclude_cap_test=True, mask_pixel=data_constants.r1_pixel_mask, lock=tb_lock,
                 fit_plot_pdf_name="Fit References/{}/unbiased_reduced_model.pdf".format(name), plot=True,
                 **correction_args)
    # name = "R1"
    analyze_data(raw_data=R11_SCAN_FILE, base_path="{}/{}/biased_80_V_full".format(top_ref, name),
                 is_advanced=True, distribution=True, full_model=False,
                 fit_plot_pdf_name="Fit References/{}/biased_reduced_model.pdf".format(name), plot=True,
                 exclude_cap_test=True, mask_pixel=data_constants.r1_pixel_mask, lock=tb_lock, **correction_args)
    # name = "R11"
    analyze_data(raw_data=R11_SCAN_FILE, base_path="{}/{}/unbiased_full_model".format(top_ref, name),
                 is_advanced=True, distribution=True, exclude_cap_test=True, mask_pixel=data_constants.r1_pixel_mask,
                 fit_plot_pdf_name="Fit References/{}/unbiased_full_model.pdf".format(name), plot=True,
                 lock=tb_lock, **correction_args)
    # name = "R1"
    analyze_data(raw_data=R11_SCAN_FILE, base_path="{}/{}/biased_80_V_full_model".format(top_ref, name),
                 is_advanced=True, distribution=True, exclude_cap_test=True, mask_pixel=data_constants.r1_pixel_mask,
                 fit_plot_pdf_name="Fit References/{}/biased_full_model.pdf".format(name), plot=True,
                 lock=tb_lock, **correction_args)

    # handle the inter-pix analysis
    # we will need a safe-guard
    # TODO: remove the safe-guard after fixing the naming convention
    # with synchronized_process_open_file(R11_SCAN_FILE, mode='a', lock=tb_lock) as h5_file:
    #     h5_file.copy_node(where=h5_file.root.Reference.R11, newparent=h5_file.root.Reference.R1, name="unbiased_full",
    #                       recursive=True, overwrite=False, createparents=True)
    analyze_data(raw_data=R11_SCAN_FILE, base_path="{}/{}/inter_unbiased_full".format(top_ref, name),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 total_cap_file=R11_SCAN_FILE,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_group="{}/{}/unbiased_full/total_cap".format(top_ref, name))
    analyze_data(raw_data=R11_SCAN_FILE, base_path="{}/{}/inter_biased_M_80_V_full".format(top_ref, name),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group="{}/{}/biased_80_V_full/total_cap".format(top_ref, name))
    analyze_data(raw_data=R11_SCAN_FILE, base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 total_cap_file=R11_SCAN_FILE,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_group="{}/{}/unbiased_full/total_cap".format(top_ref, name))
    analyze_data(raw_data=R11_SCAN_FILE, base_path="{}/{}/inter_biased_M_80_V_full_model".format(top_ref, name),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask,
                 total_cap_file=R11_SCAN_FILE,
                 total_cap_group="{}/{}/biased_80_V_full/total_cap".format(top_ref, name))

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
    r11_depletion_args_refined.update(**correction_args)

    print("CV Analysis for", display_name)
    # name = "R11"
    analyze_data(raw_data=R11_SCAN_FILE, base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                 exclude_cap_test=True, mask_pixel=data_constants.r1_pixel_mask,
                 lock=tb_lock, **r11_depletion_args)

    # name = "R1"
    analyze_data(raw_data=R11_SCAN_FILE, base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                 exclude_cap_test=True, mask_pixel=data_constants.r1_pixel_mask,
                 lock=tb_lock, **r11_depletion_args_refined)
    print("Finished the Analysis for", display_name)


def r13_analysator_first(tb_lock, correction_args, **kwargs):
    import pixcap65.concurrency
    name = "R13"
    print("Analyze", name)
    print(threading.get_native_id())
    print(mp.current_process().name)
    print(mp.current_process().pid)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    correction_args = correction_args.copy()
    correction_args.update(lock=tb_lock)

    analyze_data(raw_data='pixcap65/Data/TEST_2.h5', is_advanced=False, lock=tb_lock, )
    analyze_data(raw_data='pixcap65/Data/r13-measurement/data.h5', is_advanced=False, **correction_args)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/TEST.h5', is_advanced=False, **correction_args)
    analyze_data(raw_data='Reference_R13_Scan.h5', base_path="Reference/R13/unbiased_12_full", **correction_args,
                 is_advanced=True)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', is_advanced=True,
                 **correction_args)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1",
                 is_advanced=True, **correction_args)
    analyze_data(raw_data='R13-Interpixel_Scan.h5',
                 base_path="Reference/R13/demo_measurement_65_unbiased_1_discharge",
                 is_inter_pixel=True, is_advanced=True, lock=tb_lock, )

    print("CV Analysis for", name)
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
    print("Finished the Analysis for", name)


def e1_plotter_first(tb_lock):
    name = "E1"
    display_name = name
    top_ref = "Reference"
    print("Plotting", display_name)
    threaded_plotting.plot_data(interpreted_data=E1_SCAN_FILE, base_path="{}/{}/unbiased_1_test".format(top_ref, name),
                                use_group=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data=E1_SCAN_FILE, base_path="{}/{}/unbiased_2_test".format(top_ref, name),
                                use_group=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data=E1_SCAN_FILE, base_path="{}/{}/unbiased_3_test".format(top_ref, name),
                                use_group=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data=E1_SCAN_FILE, base_path="{}/{}/unbiased_4_full".format(top_ref, name),
                                use_group=True, lock=tb_lock)
    plot_data(interpreted_data=E1_SCAN_FILE, base_path="{}/{}/unbiased_4_full".format(top_ref, name),
              use_group=True, use_corrected=True, distribution=True, exclude_test_cap=True, lock=tb_lock)
    threaded_plotting.plot_bias_data(interpreted_data=E1_SCAN_FILE,
                                     base_path="{}/{}/I_V_Characteristic".format(top_ref, name),
                                     use_group=True, lock=tb_lock)
    print(display_name, "- CV")
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def x1_analysator_first(tb_lock, correction_args, **kwargs):
    import pixcap65.concurrency
    name = "X1"
    print("Analyze", name)
    print(threading.get_native_id())
    print(mp.current_process().name)
    print(mp.current_process().pid)

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

    print("CV Analysis for", name)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic",
                 is_advanced=False, is_cv=True, use_corrected=True, apply_doping=True,
                 chip_group_name="ATLAS ITk/sensor",
                 first_boundaries=[(-60, -40), (-80, -75)], second_boundaries=[(-5, 0), (-70, -65)],
                 **correction_args)
    print("Finished the Analysis for", name)


def x2_analysator_first(tb_lock, correction_args, **kwargs):
    import pixcap65.concurrency
    name = "X2"
    print("Analyze", name)
    print(threading.get_native_id())
    print(mp.current_process().name)
    print(mp.current_process().pid)

    _ = pixcap65.concurrency.get_manager(**kwargs)
    correction_args = correction_args.copy()
    correction_args.update(lock=tb_lock)

    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1", is_advanced=True,
                 **correction_args)
    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/biased_80_V", is_advanced=True,
                 **correction_args)

    print("CV Analysis for", name)
    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic", is_advanced=True,
                 is_cv=True,
                 first_boundaries=[(-60, -40), (-80, -75)], second_boundaries=[(-5, 0), (-70, -65)],
                 use_corrected=True,
                 **correction_args)
    with PdfPages("New_2_Scan_CV_refined_distribution.pdf") as pdf:
        analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic_refined",
                     is_advanced=True,
                     is_cv=True,
                     first_boundaries=(-60, -20), second_boundaries=(-5, 0), use_corrected=True, apply_doping=True,
                     chip_group_name="ATLAS_Itk/X2/sensor", distribution=True, set_parasitic=False,
                     distribution_output_pdf=pdf, **correction_args)
    print("Finished the Analysis for", name)


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
              exclude_test_cap=True, lock=tb_lock)
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
    plot_data(interpreted_data="packaged/Reference_Bare_renewed.h5", base_path="Reference/Bare/unbiased_31_renew",
              suffix="general_data_bare-2", use_group=True, exclude_test_cap=True, distribution=True, lock=tb_lock)


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
    .. list-table:: Table Bias States
        :widths: 20 50 30
        :header-rows: 1

    * - HV
      - Column B
      - Column C
    * - 0V (unbiased)
      - A
      - A
    * - -80V (biased)
      - B
      - C

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "R1"
    display_name = name
    top_ref = "Reference"
    print("Plotting", display_name)
    # TODO: unify the naming convention here.
    name = "R11"
    plot_data(interpreted_data=R11_SCAN_FILE, base_path="{}/{}/unbiased_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, distribution=True, lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path="{}/{}/unbiased_full".format(top_ref, name), use_group=True,
              use_corrected=True, exclude_test_cap=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    name = "R1"
    plot_data(interpreted_data=R11_SCAN_FILE, base_path="{}/{}/biased_80_V_full".format(top_ref, name),
              use_group=True, exclude_test_cap=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path="{}/{}/biased_80_V_full".format(top_ref, name),
              use_group=True, use_corrected=True, exclude_test_cap=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    name = "R11"
    plot_data(interpreted_data=R11_SCAN_FILE, base_path="{}/{}/unbiased_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, distribution=True, lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path="{}/{}/unbiased_full_model".format(top_ref, name), use_group=True,
              use_corrected=True, exclude_test_cap=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    name = "R1"
    plot_data(interpreted_data=R11_SCAN_FILE, base_path="{}/{}/biased_80_V_full_model".format(top_ref, name),
              use_group=True, exclude_test_cap=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    plot_data(interpreted_data=R11_SCAN_FILE, base_path="{}/{}/biased_80_V_full_model".format(top_ref, name),
              use_group=True, use_corrected=True, exclude_test_cap=True, distribution=True, lock=tb_lock,
              mask_pixel=data_constants.r1_pixel_mask)
    name = "R11"
    plot_bias_data(interpreted_data=R11_SCAN_FILE, base_path="{}/{}/I_V_Characteristic".format(top_ref, name),
                   use_group=True, lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask)

    name = "R1"
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path="{}/{}/inter_unbiased_full".format(top_ref, name), use_group=True,
                        distribution=True, exclude_cap_test=True, total_data=R11_SCAN_FILE,
                        total_path="{}/{}/unbiased_full".format(top_ref, name), lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path="{}/{}/inter_unbiased_full".format(top_ref, name), use_group=True,
                        distribution=True, exclude_cap_test=True, total_data=R11_SCAN_FILE,
                        total_path="{}/{}/unbiased_full_model".format(top_ref, name), lock=tb_lock, suffix="inter-mix",
                        mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name), use_group=True,
                        distribution=True, exclude_cap_test=True, total_data=R11_SCAN_FILE,
                        total_path="{}/{}/unbiased_full".format(top_ref, name), lock=tb_lock, suffix="inter-mix",
                        mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name), use_group=True,
                        distribution=True, exclude_cap_test=True, total_data=R11_SCAN_FILE,
                        total_path="{}/{}/unbiased_full_model".format(top_ref, name), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path="{}/{}/inter_biased_M_80_V_full".format(top_ref, name), use_group=True,
                        exclude_cap_test=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path="{}/{}/biased_80_V_full_model".format(top_ref, name), lock=tb_lock,
                        suffix="inter-mix", mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path="{}/{}/inter_biased_M_80_V_full_model".format(top_ref, name), use_group=True,
                        exclude_cap_test=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path="{}/{}/biased_80_V_full".format(top_ref, name), lock=tb_lock,
                        suffix="inter-mix", mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path="{}/{}/inter_biased_M_80_V_full".format(top_ref, name), use_group=True,
                        exclude_cap_test=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path="{}/{}/biased_80_V_full".format(top_ref, name), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask)
    plot_inter_pix_data(interpreted_data=R11_SCAN_FILE,
                        base_path="{}/{}/inter_biased_M_80_V_full_model".format(top_ref, name), use_group=True,
                        exclude_cap_test=True, distribution=True, total_data=R11_SCAN_FILE,
                        total_path="{}/{}/biased_80_V_full_model".format(top_ref, name), lock=tb_lock,
                        mask_pixel=data_constants.r1_pixel_mask)

    print(display_name, "- CV")
    name = "R11"
    threaded_plotting.plot_combined_data(interpreted_data=R11_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                                         use_group=True, distribution=False, lock=tb_lock,
                                         mask_pixel=data_constants.r1_pixel_mask)
    threaded_plotting.plot_combined_data(interpreted_data=R11_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                         lock=tb_lock, mask_pixel=data_constants.r1_pixel_mask)
    threaded_plotting.plot_combined_data(interpreted_data=R11_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                                         use_group=True, distribution=True, lock=tb_lock,
                                         mask_pixel=data_constants.r1_pixel_mask)
    threaded_plotting.plot_combined_data(interpreted_data=R11_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
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
    .. list-table:: Table Bias States
        :widths: 20 50 30
        :header-rows: 1

    * - HV
      - Column B
      - Column C
    * - 0V (unbiased)
      - A
      - A
    * - -80V (biased)
      - B
      - C

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "R13"
    display_name = name
    top_ref = "Reference"
    print("Plotting", display_name)
    # TODO: Combine all these hdf files into a single file!
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/data.h5', suffix="test_run", use_group=False)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/data.h5', suffix="test_run", use_group=False,
              use_corrected=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/Test.h5', suffix="test-general_run", use_group=False,
              lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/Test.h5', suffix="test-general_run", use_group=False,
              use_corrected=True, lock=tb_lock)
    plot_data(interpreted_data='Reference_R13_Scan.h5', suffix="test-general_run", use_group=False,
              base_path="Reference/R13/unbiased_12_full", exclude_test_cap=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data='Reference_R13_Scan.h5', suffix="test-general_run", use_group=False,
              use_corrected=True, base_path="Reference/R13/unbiased_12_full", exclude_test_cap=True,
              distribution=True, lock=tb_lock)
    threaded_plotting.plot_bias_data(interpreted_data='pixcap65/Data/r13-measurement/R13_BIAS_2.h5', lock=tb_lock)
    # Data/r13-measurement/R13_BIAS_CV_COMBI_2.h5 no further investigation possible as data set is incomplete!
    # Data/r13-measurement/R13_BIAS_CV_COMBI_3.h5 no further investigation possible as data set is incomplete!
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5',
                                suffix="general_data",
                                use_group=False, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data",
              use_group=False, use_corrected=True, exclude_test_cap=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Initial_3_Scan.h5',
              base_path="ATLAS ITk/unbiased_1", suffix="unbiased_full_measurement", use_group=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Initial_3_Scan.h5',
              base_path="ATLAS ITk/unbiased_1", suffix="unbiased_full_measurement", use_group=True,
              use_corrected=True, lock=tb_lock)
    plot_inter_pix_data(interpreted_data='R13-Interpixel_Scan.h5',
                        base_path="{}/{}/demo_measurement_65_unbiased_1_discharge".format(top_ref, name),
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
    .. list-table:: Table Bias States
        :widths: 20 50 30
        :header-rows: 1

    * - HV
      - Column B
      - Column C
    * - 0V (unbiased)
      - A
      - A
    * - -80V (biased)
      - B
      - C

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "R13"
    display_name = name + "Second Try."
    top_ref = "Reference"
    print("Plotting", display_name)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path="{}/{}/unbiased_1_full".format(top_ref, name),
              use_group=True, lock=tb_lock, exclude_test_cap=True, distribution=True)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path="{}/{}/unbiased_1_full".format(top_ref, name),
              use_group=True, lock=tb_lock, exclude_test_cap=True, distribution=True, use_corrected=True)
    # noqa: S1192
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path="{}/{}/biased_80_V_full".format(top_ref, name),
              use_group=True, lock=tb_lock, distribution=True)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path="{}/{}/biased_80_V_full".format(top_ref, name),
              use_group=True, lock=tb_lock, distribution=True, use_corrected=True)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path="{}/{}/unbiased_1_full_model".format(top_ref, name),
              use_group=True, lock=tb_lock, exclude_test_cap=True, distribution=True)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path="{}/{}/unbiased_1_full_model".format(top_ref, name),
              use_group=True, lock=tb_lock, exclude_test_cap=True, distribution=True, use_corrected=True)
    # noqa: S1192
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path="{}/{}/biased_80_V_full_model".format(top_ref, name),
              use_group=True, lock=tb_lock, distribution=True)
    plot_data(interpreted_data=R13_2_SCAN_FILE, base_path="{}/{}/biased_80_V_full_model".format(top_ref, name),
              use_group=True, lock=tb_lock, distribution=True, use_corrected=True)

    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path="{}/{}/inter_unbiased_full".format(top_ref, name), lock=tb_lock, use_group=True,
                        exclude_test_cap=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path="{}/{}/unbiased_1_full".format(top_ref, name))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path="{}/{}/inter_unbiased_full".format(top_ref, name), lock=tb_lock, use_group=True,
                        exclude_test_cap=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path="{}/{}/unbiased_1_full_model".format(top_ref, name))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name), lock=tb_lock,
                        use_group=True, exclude_test_cap=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        suffix="inter-mix", total_path="{}/{}/unbiased_1_full".format(top_ref, name))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE,
                        base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name), lock=tb_lock,
                        use_group=True, exclude_test_cap=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path="{}/{}/unbiased_1_full_model".format(top_ref, name))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path="{}/{}/inter_biased_M_80_V_full".format(top_ref, name), use_group=True,
                        exclude_test_cap=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path="{}/{}/biased_80_V_full".format(top_ref, name))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path="{}/{}/inter_biased_M_80_V_full".format(top_ref, name), use_group=True,
                        exclude_test_cap=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path="{}/{}/biased_80_V_full_model".format(top_ref, name))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path="{}/{}/inter_biased_M_80_V_full_model".format(top_ref, name), use_group=True,
                        exclude_test_cap=True, distribution=True, total_data=R13_2_SCAN_FILE, suffix="inter-mix",
                        total_path="{}/{}/biased_80_V_full".format(top_ref, name))
    plot_inter_pix_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                        base_path="{}/{}/inter_biased_M_80_V_full_model".format(top_ref, name), use_group=True,
                        exclude_test_cap=True, distribution=True, total_data=R13_2_SCAN_FILE,
                        total_path="{}/{}/biased_80_V_full_model".format(top_ref, name))

    # the uncertainties of the capacitance estimators looks quite large.
    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data=R13_2_SCAN_FILE,
                       base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name), lock=tb_lock,
                       use_group=True, exclude_test_cap=True, distribution=True)
    threaded_plotting.plot_combined_data(interpreted_data=R13_2_SCAN_FILE, lock=tb_lock,
                       base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                       use_group=True, exclude_test_cap=True, distribution=True, use_corrected=True)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def e1_plotter_first(tb_lock):
    name = "X5"
    display_name = name
    top_ref = "Reference"
    print("Plotting", display_name)
    threaded_plotting.plot_data(interpreted_data=E1_SCAN_FILE, base_path="{}/{}/unbiased_4_full".format(top_ref, name),
                                use_group=True, exclude_test_cap=True, mask_pixel=data_constants.e1_pixel_mask,
                                lock=tb_lock,)
    plot_data(interpreted_data=E1_SCAN_FILE, base_path="{}/{}/unbiased_4_full".format(top_ref, name), use_group=True,
              use_corrected=True, distribution=True, exclude_test_cap=True, mask_pixel=data_constants.e1_pixel_mask,
              lock=tb_lock,)
    threaded_plotting.plot_bias_data(interpreted_data=E1_SCAN_FILE,
                                     base_path="{}/{}/I_V_Characteristic".format(top_ref, name), use_group=True,
                                     lock=tb_lock,)

    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data=E1_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                                         use_group=True, distribution=True, lock=tb_lock,)
    threaded_plotting.plot_combined_data(interpreted_data=E1_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                       use_group=True, use_corrected=True, apply_doping=False, distribution=False, lock=tb_lock, )
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def e1_plotter_second(tb_lock):
    name = "E1"
    display_name = name + "Second Try."
    top_ref = "Reference"
    print("Plotting", display_name)
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                                base_path="{}/{}/unbiased_full".format(top_ref, name), exclude_test_cap=True,
                                mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock,)
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                                base_path="{}/{}/unbiased_full".format(top_ref, name), exclude_test_cap=True,
                                mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, lock=tb_lock,)
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                                base_path="{}/{}/biased_80_V_full".format(top_ref, name),
                                mask_pixel=data_constants.e1_pixel_mask, exclude_test_cap=True, lock=tb_lock,)
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                                base_path="{}/{}/biased_80_V_full".format(top_ref, name),
                                mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, exclude_test_cap=True,
                                lock=tb_lock,)
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                                base_path="{}/{}/unbiased_full_model".format(top_ref, name), exclude_test_cap=True,
                                mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock, )
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                                base_path="{}/{}/unbiased_full_model".format(top_ref, name), exclude_test_cap=True,
                                mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, lock=tb_lock, )
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                                base_path="{}/{}/biased_80_V_full_model".format(top_ref, name),
                                mask_pixel=data_constants.e1_pixel_mask, exclude_test_cap=True, lock=tb_lock, )
    plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                                base_path="{}/{}/biased_80_V_full_model".format(top_ref, name),
                                mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, exclude_test_cap=True,
                                lock=tb_lock,)
    plot_bias_data(interpreted_data=E1_2_SCAN_FILE,
                                     base_path="{}/{}/I_V_Characteristic".format(top_ref, name), use_group=True,
                                     lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                                          base_path="{}/{}/inter_unbiased_full".format(top_ref, name),
                                          use_group=True, total_data=E1_2_SCAN_FILE,
                                          total_path="{}/{}/unbiased_full".format(top_ref, name), lock=tb_lock,
                                          mask_pixel=data_constants.e1_pixel_mask,)
    plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                                          base_path="{}/{}/inter_biased_M_80_V_full".format(top_ref, name),
                                          use_group=True, total_data=E1_2_SCAN_FILE,
                                          total_path="{}/{}/biased_80_V_full".format(top_ref, name), lock=tb_lock,
                                          mask_pixel=data_constants.e1_pixel_mask,)

    threaded_plotting.plot_combined_data(interpreted_data=E1_2_SCAN_FILE,
                       base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                       use_group=True, distribution=True, mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock,)
    threaded_plotting.plot_combined_data(interpreted_data=E1_2_SCAN_FILE,
                       base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                       use_group=True, distribution=True, mask_pixel=data_constants.e1_pixel_mask,
                       use_corrected=True, lock=tb_lock,)

    threaded_plotting.joint_plotting()
    print("E1 - Switching to individual pixel regions.")
    for type_name in data_constants.e1_pixel_groups.keys():
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path="{}/{}/unbiased_full_{}".format(top_ref, name, type_name), exclude_test_cap=True,
                  mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock, distribution=True)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path="{}/{}/unbiased_full_{}".format(top_ref, name, type_name), exclude_test_cap=True,
                  mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, lock=tb_lock, distribution=True)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path="{}/{}/biased_80_V_full_{}".format(top_ref, name, type_name), distribution=True,
                  mask_pixel=data_constants.e1_pixel_mask, exclude_test_cap=True, lock=tb_lock,)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path="{}/{}/biased_80_V_full_{}".format(top_ref, name, type_name),
                  mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, exclude_test_cap=True,
                  lock=tb_lock, distribution=True)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path="{}/{}/unbiased_full_model_{}".format(top_ref, name, type_name), exclude_test_cap=True,
                  mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock, distribution=True)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path="{}/{}/unbiased_full_model_{}".format(top_ref, name, type_name), exclude_test_cap=True,
                  mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, lock=tb_lock, distribution=True)
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path="{}/{}/biased_80_V_full_model_{}".format(top_ref, name, type_name), distribution=True,
                  mask_pixel=data_constants.e1_pixel_mask, exclude_test_cap=True, lock=tb_lock, )
        plot_data(interpreted_data=E1_2_SCAN_FILE, use_group=True,
                  base_path="{}/{}/biased_80_V_full_model_{}".format(top_ref, name, type_name),
                  mask_pixel=data_constants.e1_pixel_mask, use_corrected=True, exclude_test_cap=True,
                  lock=tb_lock, distribution=True)

        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path="{}/{}/inter_unbiased_full_{}".format(top_ref, name, type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE,
                            total_path="{}/{}/unbiased_full_{}".format(top_ref, name, type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask,)
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path="{}/{}/inter_unbiased_full_model_{}".format(top_ref, name, type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE, suffix="inter-mix",
                            total_path="{}/{}/unbiased_full_{}".format(top_ref, name, type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask,)
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path="{}/{}/inter_unbiased_full_{}".format(top_ref, name, type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE, suffix="inter-mix",
                            total_path="{}/{}/unbiased_full_model_{}".format(top_ref, name, type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask,)
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path="{}/{}/inter_unbiased_full_model_{}".format(top_ref, name, type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE,
                            total_path="{}/{}/unbiased_full_model_{}".format(top_ref, name, type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask,)


        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path="{}/{}/inter_biased_M_80_V_full_{}".format(top_ref, name, type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE,
                            total_path="{}/{}/biased_80_V_full_{}".format(top_ref, name, type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask,)
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path="{}/{}/inter_biased_M_80_V_full_{}".format(top_ref, name, type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE, suffix="inter-mix",
                            total_path="{}/{}/biased_80_V_full_model_{}".format(top_ref, name, type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask,)
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path="{}/{}/inter_biased_M_80_V_full_model_{}".format(top_ref, name, type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE, suffix="inter-mix",
                            total_path="{}/{}/biased_80_V_full_{}".format(top_ref, name, type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask,)
        plot_inter_pix_data(interpreted_data=E1_2_SCAN_FILE,
                            base_path="{}/{}/inter_biased_M_80_V_full_model_{}".format(top_ref, name, type_name),
                            use_group=True, total_data=E1_2_SCAN_FILE,
                            total_path="{}/{}/biased_80_V_full_model_{}".format(top_ref, name, type_name),
                            lock=tb_lock, distribution=True, mask_pixel=data_constants.e1_pixel_mask,)

        threaded_plotting.plot_combined_data(interpreted_data=E1_2_SCAN_FILE,
                           base_path="{}/{}/C_V_Characteristic_refined_{}".format(top_ref, name, type_name),
                           use_group=True, distribution=True, mask_pixel=data_constants.e1_pixel_mask,
                           lock=tb_lock,)
        threaded_plotting.plot_combined_data(interpreted_data=E1_2_SCAN_FILE,
                           base_path="{}/{}/C_V_Characteristic_refined_{}".format(top_ref, name, type_name),
                           use_group=True, distribution=True, mask_pixel=data_constants.e1_pixel_mask,
                           use_corrected=True, lock=tb_lock,)

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
    .. list-table:: Table Bias States
        :widths: 20 50 30
        :header-rows: 1

    * - HV
      - Column B
      - Column C
    * - 0V (unbiased)
      - A
      - A
    * - -80V (biased)
      - B
      - C

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X1"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    # TODO: reformat this files such that the general structure is used!
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5',
                                base_path="run_1", suffix="test_run", use_group=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5',
                                base_path="run_1", suffix="test_run", use_group=True, use_corrected=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5',
                                base_path="run_2", suffix="test_run", use_group=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5',
                                base_path="run_2", suffix="test_run", use_group=True, use_corrected=True,
                                lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5',
                                base_path="ATLAS ITk/run_1", suffix="test_run", use_group=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5',
                                base_path="ATLAS ITk/run_1", suffix="test_run", use_group=True, use_corrected=True,
                                lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5',
                                base_path="ATLAS ITk/run_2", suffix="test_run", use_group=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5',
                                base_path="ATLAS ITk/run_2", suffix="test_run", use_group=True, use_corrected=True,
                                lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_Scan.h5', suffix="test_run",
                                use_group=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_Scan.h5', suffix="test_run",
                                use_group=True, use_corrected=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                                base_path="ATLAS ITk/run_1", suffix="general_data_test_test", use_group=True,
                                lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                                base_path="ATLAS ITk/run_1", suffix="general_data_test_test", use_group=True,
                                use_corrected=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                                base_path="ATLAS ITk/run_2", suffix="general_data", use_group=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                                base_path="ATLAS ITk/run_2", suffix="general_data", use_group=True,
                                use_corrected=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
              suffix="general_data", use_group=True, exclude_test_cap=True,
              mask_pixel=data_constants.x1_second_pixel_mask, distribution=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
              suffix="general_data", use_group=True, use_corrected=True,
              mask_pixel=data_constants.x1_second_pixel_mask, exclude_test_cap=True, distribution=True,
              lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                                base_path="ATLAS ITk/unbiased_4", suffix="general_data", use_group=True,
                                lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                                base_path="ATLAS ITk/unbiased_4", suffix="general_data", use_group=True,
                                use_corrected=True, lock=tb_lock)
    threaded_plotting.plot_bias_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                                     base_path="ATLAS ITk/I_V_Characteristic", use_group=True, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V",
              suffix="general_data", use_group=True, exclude_test_cap=True,
              mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V",
              suffix="general_data", use_group=True, use_corrected=True, apply_doping=True, exclude_test_cap=True,
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
    .. list-table:: Table Bias States
        :widths: 20 50 30
        :header-rows: 1

    * - HV
      - Column B
      - Column C
    * - 0V (unbiased)
      - A
      - A
    * - -80V (biased)
      - B
      - C

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X1"
    display_name = name + " Second Try."
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    top_ref = "ATLAS_ITk"
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/unbiased_61_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, mask_pixel=data_constants.x1_second_pixel_mask, distribution=True, lock=tb_lock,)
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/unbiased_61_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, use_corrected=True, mask_pixel=data_constants.x1_second_pixel_mask,
              distribution=True, lock=tb_lock,)
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/biased_80_V_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, mask_pixel=data_constants.x1_second_pixel_mask, distribution=True,
              lock=tb_lock,)
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/biased_80_V_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, use_corrected=True, mask_pixel=data_constants.x1_second_pixel_mask,
              distribution=True, lock=tb_lock,)
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/unbiased_61_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, mask_pixel=data_constants.x1_second_pixel_mask, distribution=True, lock=tb_lock,)
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/unbiased_61_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, use_corrected=True, mask_pixel=data_constants.x1_second_pixel_mask,
              distribution=True, lock=tb_lock,)
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/biased_80_V_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, mask_pixel=data_constants.x1_second_pixel_mask, distribution=True,
              lock=tb_lock,)
    plot_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/biased_80_V_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, use_corrected=True, mask_pixel=data_constants.x1_second_pixel_mask,
              distribution=True, lock=tb_lock,)

    top_ref = "Thesis/ATLAS_ITk"
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/inter_unbiased_full".format(top_ref, name),
                        use_group=True, exclude_test_cap=True, distribution=True,
                        total_data=X1_SCAN_2_FILE, total_path="{}/{}/unbiased_61_full".format(top_ref, name),
                        mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/inter_unbiased_full".format(top_ref, name),
                        use_group=True, exclude_test_cap=True, distribution=True,
                        total_data=X1_SCAN_2_FILE, total_path="{}/{}/unbiased_61_full_model".format(top_ref, name),
                        mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock, suffix="inter-mix")
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE,
                        base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name), use_group=True,
                        exclude_test_cap=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path="{}/{}/unbiased_61_full_model".format(top_ref, name), mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE,
                        base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name), use_group=True,
                        exclude_test_cap=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path="{}/{}/unbiased_61_full".format(top_ref, name), mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock, suffix="inter-mix")
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE,
                        base_path="{}/{}/inter_biased_M_80_V_full".format(top_ref, name), use_group=True,
                        exclude_test_cap=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path="{}/{}/biased_80_V_full".format(top_ref, name), mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/inter_biased_M_80_V_full".format(top_ref, name),
                        use_group=True, exclude_test_cap=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path="{}/{}/biased_80_V_full_model".format(top_ref, name),
                        mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock, suffix="inter-mix")
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE,
                        base_path="{}/{}/inter_biased_M_80_V_full_model".format(top_ref, name), use_group=True,
                        exclude_test_cap=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path="{}/{}/biased_80_V_full_model".format(top_ref, name), mask_pixel=data_constants.x1_second_pixel_mask, lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X1_SCAN_2_FILE,
                        base_path="{}/{}/inter_biased_M_80_V_full_model".format(top_ref, name),
                        use_group=True, exclude_test_cap=True, distribution=True, total_data=X1_SCAN_2_FILE,
                        total_path="{}/{}/biased_80_V_full".format(top_ref, name), mask_pixel=data_constants.x1_second_pixel_mask,
                        lock=tb_lock, suffix="inter-mix")
    top_ref = "ATLAS_ITk"
    threaded_plotting.plot_bias_data(interpreted_data=X1_SCAN_2_FILE, base_path="{}/{}/I_V_Characteristic".format(top_ref, name),
                                     use_group=True, lock=tb_lock,)

    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                                         use_group=True, mask_pixel=data_constants.x1_second_pixel_mask,
                                         distribution=True,
                                         lock=tb_lock,)
    threaded_plotting.plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                                         use_group=True, use_corrected=True,
                                         apply_doping=True, distribution=True,
                                         mask_pixel=data_constants.x1_second_pixel_mask,
                                         lock=tb_lock,)
    top_ref = "Thesis/ATLAS_ITk"
    threaded_plotting.plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                                         base_path="{}/{}/C_V_Characteristic_Second_Extended".format(top_ref, name),
                                         use_group=True, mask_pixel=data_constants.x1_second_pixel_mask,
                                         distribution=False, apply_doping=True,
                                         lock=tb_lock,)
    threaded_plotting.plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                                         base_path="{}/{}/C_V_Characteristic_Second_Extended".format(top_ref, name),
                                         use_group=True, use_corrected=True,
                                         apply_doping=True, distribution=False,
                                         mask_pixel=data_constants.x1_second_pixel_mask,
                                         lock=tb_lock,)
    threaded_plotting.plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined_Extended_Combined".format(top_ref, name),
                                         use_group=True, mask_pixel=data_constants.x1_second_pixel_mask,
                                         distribution=True, apply_doping=True,
                                         lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data=X1_SCAN_2_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined_Extended_Combined".format(top_ref,name),
                                         use_group=True, use_corrected=True,
                                         apply_doping=False, distribution=True,
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
    .. list-table:: Table Bias States
        :widths: 20 50 30
        :header-rows: 1

    * - HV
      - Column B
      - Column C
    * - 0V (unbiased)
      - A
      - A
    * - -80V (biased)
      - B
      - C

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X2"
    display_name = name
    top_ref = "ATLAS_ITk"
    print("Plotting", display_name)
    threaded_plotting.plot_data(interpreted_data='New_2_Scan.h5',
                                base_path="{}/{}/unbiased_1".format(top_ref, name), suffix="general_data_80V",
                                use_group=True, exclude_test_cap=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='New_2_Scan.h5',
                                base_path="{}/{}/unbiased_1".format(top_ref, name), suffix="general_data_80V",
                                use_group=True, use_corrected=True, exclude_test_cap=True, lock=tb_lock)
    threaded_plotting.plot_bias_data(interpreted_data='New_2_Scan.h5',
                                     base_path="{}/{}/I_V_Characteristic".format(top_ref, name), use_group=True,
                                     lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='New_2_Scan.h5',
                                base_path="{}/{}/biased_80_V".format(top_ref, name), suffix="general_data_80V",
                                use_group=True, exclude_test_cap=True, lock=tb_lock)
    threaded_plotting.plot_data(interpreted_data='New_2_Scan.h5',
                                base_path="{}/{}/biased_80_V".format(top_ref, name), suffix="general_data_80V",
                                use_group=True, use_corrected=True, apply_doping=True, exclude_test_cap=True,
                                lock=tb_lock)

    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data='New_2_Scan.h5',
                                         base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                                         use_group=True, lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data='New_2_Scan.h5',
                                         base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                                         use_group=True, use_corrected=True, lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data='New_2_Scan.h5',
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                                         use_group=True, lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data='New_2_Scan.h5',
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
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
    .. list-table:: Table Bias States
        :widths: 20 50 30
        :header-rows: 1

    * - HV
      - Column B
      - Column C
    * - 0V (unbiased)
      - A
      - A
    * - -80V (biased)
      - B
      - C

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X2"
    display_name = name + "Second Try."
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    top_ref = "ATLAS_ITk"
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path="{}/{}/unbiased_1_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path="{}/{}/unbiased_1_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, use_corrected=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path="{}/{}/biased_80_V_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path="{}/{}/biased_80_V_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, use_corrected=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path="{}/{}/unbiased_1_full_model".format(top_ref, name), use_group=True,
             exclude_test_cap=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path="{}/{}/unbiased_1_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, use_corrected=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path="{}/{}/biased_80_V_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, distribution=True, lock=tb_lock, )
    plot_data(interpreted_data=X2_SCAN_2_FILE, base_path="{}/{}/biased_80_V_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, use_corrected=True, distribution=True, lock=tb_lock,)
    top_ref = "Thesis/ATLAS_ITk"
    plot_bias_data(interpreted_data=X2_SCAN_2_FILE, base_path="{}/{}/I_V_Characteristic_2".format(top_ref, name),
                   use_group=True, lock=tb_lock,)

    # plot inter-pix capacitance data

    # plot the c-v data
    print(display_name, "- CV")
    top_ref = "ATLAS_ITk"
    plot_combined_data(interpreted_data=X2_SCAN_2_FILE, base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                       use_group=True, lock=tb_lock, exclude_test_cap=True, distribution=True, )
    plot_combined_data(interpreted_data=X2_SCAN_2_FILE, base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                       use_group=True, use_corrected=True,
                       apply_doping=False, distribution=True, lock=tb_lock, exclude_test_cap=True,)
    top_ref = "Thesis/ATLAS_ITk"
    plot_combined_data(interpreted_data=X2_SCAN_2_FILE,
                       base_path="{}/{}/C_V_Characteristic_refined_extended_renew_retry".format(top_ref, name),
                       use_group=True, lock=tb_lock, distribution=True, exclude_test_cap=True)
    plot_combined_data(interpreted_data=X2_SCAN_2_FILE,
                       base_path="{}/{}/C_V_Characteristic_refined_extended_renew_retry".format(top_ref, name),
                       use_group=True, use_corrected=True,
                       apply_doping=False, distribution=True, lock=tb_lock, exclude_test_cap=True)
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
    .. list-table:: Table Bias States
        :widths: 20 50 30
        :header-rows: 1

    * - HV
      - Column B
      - Column C
    * - 0V (unbiased)
      - A
      - A
    * - -40V (biased)
      - B
      - C

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X5"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path="{}/{}/unbiased_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, mask_pixel=data_constants.x5_second_pixel_mask, distribution=True, lock=tb_lock,)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path="{}/{}/unbiased_full".format(top_ref, name), use_group=True,
              use_corrected=True, exclude_test_cap=True, mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock,)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path="{}/{}/biased_40_V_full".format(top_ref, name),
              use_group=True, exclude_test_cap=True, mask_pixel=data_constants.x5_second_pixel_mask, distribution=True,
              lock=tb_lock,)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path="{}/{}/biased_40_V_full".format(top_ref, name),
              use_group=True, use_corrected=True, exclude_test_cap=True, mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock,)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path="{}/{}/unbiased_full_model".format(top_ref, name),
              use_group=True, exclude_test_cap=True, mask_pixel=data_constants.x5_second_pixel_mask, distribution=True,
              lock=tb_lock)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path="{}/{}/unbiased_full_model".format(top_ref, name), use_group=True,
              use_corrected=True, exclude_test_cap=True, mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path="{}/{}/biased_40_V_full_model".format(top_ref, name),
              use_group=True, exclude_test_cap=True, mask_pixel=data_constants.x5_second_pixel_mask, distribution=True,
              lock=tb_lock)
    plot_data(interpreted_data=X5_SCAN_FILE, base_path="{}/{}/biased_40_V_full_model".format(top_ref, name),
              use_group=True, use_corrected=True, exclude_test_cap=True, mask_pixel=data_constants.x5_second_pixel_mask,
              distribution=True, lock=tb_lock)
    plot_bias_data(interpreted_data=X5_SCAN_FILE, base_path="{}/{}/I_V_Characteristic".format(top_ref, name),
                   use_group=True, lock=tb_lock,)

    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE, base_path="{}/{}/inter_unbiased_full".format(top_ref, name),
                        use_group=True, distribution=True, mask_pixel=data_constants.x5_second_pixel_mask,
                        exclude_cap_test=True, total_data=X5_SCAN_FILE,
                        total_path="{}/{}/unbiased_full".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name),
                        use_group=True, distribution=True, mask_pixel=data_constants.x5_second_pixel_mask,
                        exclude_cap_test=True, suffix="inter-mix", total_data=X5_SCAN_FILE,
                        total_path="{}/{}/unbiased_full".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE, base_path="{}/{}/inter_unbiased_full".format(top_ref, name),
                        use_group=True, distribution=True, mask_pixel=data_constants.x5_second_pixel_mask,
                        exclude_cap_test=True, suffix="inter-mix", total_data=X5_SCAN_FILE,
                        total_path="{}/{}/unbiased_full_model".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name), use_group=True,
                        distribution=True, mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True,
                        total_data=X5_SCAN_FILE, total_path="{}/{}/unbiased_full_model".format(top_ref, name),
                        lock=tb_lock,)

    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path="{}/{}/inter_biased_M_40_V_full".format(top_ref, name), use_group=True,
                        mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True, distribution=True,
                        total_data=X5_SCAN_FILE, total_path="{}/{}/biased_40_V_full".format(top_ref, name), lock=tb_lock, )
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path="{}/{}/inter_biased_M_40_V_full_model".format(top_ref, name), use_group=True,
                        mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True, distribution=True,
                        total_data=X5_SCAN_FILE, total_path="{}/{}/biased_40_V_full_model".format(top_ref, name),
                        lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path="{}/{}/inter_biased_M_40_V_full_model".format(top_ref, name), use_group=True,
                        mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True, distribution=True,
                        suffix="inter-mix", total_data=X5_SCAN_FILE,
                        total_path="{}/{}/biased_40_V_full".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X5_SCAN_FILE,
                        base_path="{}/{}/inter_biased_M_40_V_full".format(top_ref, name), use_group=True,
                        mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True, distribution=True,
                        suffix="inter-mix", total_data=X5_SCAN_FILE,
                        total_path="{}/{}/biased_40_V_full_model".format(top_ref, name), lock=tb_lock,)

    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data=X5_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                                         use_group=True, distribution=False, lock=tb_lock,
                                         mask_pixel=data_constants.x5_second_pixel_mask,)
    threaded_plotting.plot_combined_data(interpreted_data=X5_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                         lock=tb_lock,
                                         mask_pixel=data_constants.x5_second_pixel_mask,)
    threaded_plotting.plot_combined_data(interpreted_data=X5_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                                         use_group=True, distribution=True, lock=tb_lock,
                                         mask_pixel=data_constants.x5_second_pixel_mask,)
    threaded_plotting.plot_combined_data(interpreted_data=X5_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=True,
                                         lock=tb_lock, mask_pixel=data_constants.x5_second_pixel_mask,)
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
    .. list-table:: Table Bias States
        :widths: 20 50 30
        :header-rows: 1

    * - HV
      - Column B
      - Column C
    * - 0V (unbiased)
      - A
      - A
    * - -40V (biased)
      - B
      - C

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X6"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    # declare here a set of constants to use additonally!
    print("Plotting", display_name)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/unbiased_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask, distribution=True,)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/unbiased_full".format(top_ref, name), use_group=True,
              use_corrected=True, exclude_test_cap=True, lock=tb_lock,
              mask_pixel=data_constants.x6_second_pixel_mask, distribution=True,)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/unbiased_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask, distribution=True,)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/unbiased_full_model".format(top_ref, name), use_group=True,
              use_corrected=True, exclude_test_cap=True, lock=tb_lock,
              mask_pixel=data_constants.x6_second_pixel_mask, distribution=True,)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/biased_45_V_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask, distribution=True,)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/biased_45_V_full".format(top_ref, name), use_group=True,
              use_corrected=True, exclude_test_cap=True, lock=tb_lock,
              mask_pixel=data_constants.x6_second_pixel_mask, distribution=True,)
    plot_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/biased_45_V_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask, distribution=True, )
    plot_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/biased_45_V_full_model".format(top_ref, name), use_group=True,
              use_corrected=True, exclude_test_cap=True, lock=tb_lock,
              mask_pixel=data_constants.x6_second_pixel_mask, distribution=True,)

    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/inter_unbiased_full".format(top_ref, name),
                        use_group=True, distribution=True,
                        exclude_cap_test=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        total_data=X6_SCAN_FILE, total_path="{}/{}/unbiased_full".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name),
                        use_group=True, distribution=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        exclude_cap_test=True, suffix="inter-mix",
                        total_data=X6_SCAN_FILE, total_path="{}/{}/unbiased_full".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/inter_unbiased_full".format(top_ref, name),
                        use_group=True, distribution=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        exclude_cap_test=True, suffix="inter-mix",
                        total_data=X6_SCAN_FILE, total_path="{}/{}/unbiased_full_model".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name),
                        use_group=True, distribution=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        exclude_cap_test=True,
                        total_data=X6_SCAN_FILE, total_path="{}/{}/unbiased_full_model".format(top_ref, name), lock=tb_lock,)

    # TODO: use os.path.join (or at least a alias of it to shorten the lines here)
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/inter_biased_M_45_V_full".format(top_ref, name),
                        use_group=True, exclude_cap_test=True,
                        distribution=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        total_data=X6_SCAN_FILE, total_path="{}/{}/biased_45_V_full".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/inter_biased_M_45_V_full_model".format(top_ref, name),
                        use_group=True, exclude_cap_test=True,
                        distribution=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        total_data=X6_SCAN_FILE, total_path="{}/{}/biased_45_V_full_model".format(top_ref, name),
                        lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/inter_biased_M_45_V_full_model".format(top_ref, name),
                        use_group=True, exclude_cap_test=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        distribution=True, suffix="inter-mix",
                        total_data=X6_SCAN_FILE, total_path="{}/{}/biased_45_V_full".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/inter_biased_M_45_V_full".format(top_ref, name),
                        use_group=True, exclude_cap_test=True, mask_pixel=data_constants.x6_second_pixel_mask,
                        distribution=True, suffix="inter-mix",
                        total_data=X6_SCAN_FILE, total_path="{}/{}/biased_45_V_full_model".format(top_ref, name),
                        lock=tb_lock,)

    print(display_name, "- CV")
    threaded_plotting.plot_bias_data(interpreted_data=X6_SCAN_FILE, base_path="{}/{}/I_V_Characteristic".format(top_ref, name),
                                     use_group=True,
                                     lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data=X6_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                                         use_group=True, distribution=False, lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data=X6_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                         lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data=X6_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                                         use_group=True, distribution=True, lock=tb_lock,
                                         mask_pixel=data_constants.x6_second_pixel_mask, exclude_test_cap=True,)
    threaded_plotting.plot_combined_data(interpreted_data=X6_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=True,
                                         mask_pixel=data_constants.x6_second_pixel_mask, exclude_test_cap=True,)
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
    .. list-table:: Table Bias States
        :widths: 20 50 30
        :header-rows: 1

    * - HV
      - Column B
      - Column C
    * - 0V (unbiased)
      - A
      - A
    * - -40V (biased)
      - B
      - C

    :param tb_lock: multiprocessing lock to make operations on the hdf files process- and thread-safe.
    """
    name = "X7"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Plotting", display_name)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/unbiased_full".format(top_ref, name), use_group=True,
              exclude_test_cap=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/unbiased_full".format(top_ref, name), use_group=True,
              use_corrected=True, exclude_test_cap=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/unbiased_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/unbiased_full_model".format(top_ref, name), use_group=True,
              use_corrected=True, exclude_test_cap=True,
              distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/biased_40.0_V_full".format(top_ref, name),
              use_group=True,
              exclude_test_cap=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/biased_40.0_V_full".format(top_ref, name),
              use_group=True,
              use_corrected=True, exclude_test_cap=True,
              distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/biased_40.0_V_full_model".format(top_ref, name), use_group=True,
              exclude_test_cap=True, distribution=True, lock=tb_lock)
    plot_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/biased_40.0_V_full_model".format(top_ref, name), use_group=True,
              use_corrected=True, exclude_test_cap=True,
              distribution=True, lock=tb_lock)

    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/inter_unbiased_full".format(top_ref, name),
                        use_group=True, distribution=True,
                        exclude_cap_test=True,
                        total_data=X7_SCAN_FILE, total_path="{}/{}/unbiased_full".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name),
                        use_group=True, distribution=True,
                        exclude_cap_test=True, suffix="inter-mix",
                        total_data=X7_SCAN_FILE, total_path="{}/{}/unbiased_full".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/inter_unbiased_full".format(top_ref, name),
                        use_group=True, distribution=True,
                        exclude_cap_test=True, suffix="inter-mix",
                        total_data=X7_SCAN_FILE, total_path="{}/{}/unbiased_full_model".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/inter_unbiased_full_model".format(top_ref, name),
                        use_group=True, distribution=True,
                        exclude_cap_test=True,
                        total_data=X7_SCAN_FILE, total_path="{}/{}/unbiased_full_model".format(top_ref, name), lock=tb_lock,)

    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/inter_biased_M_40.0_V_full".format(top_ref, name),
                        use_group=True, exclude_cap_test=True,
                        distribution=True,
                        total_data=X7_SCAN_FILE, total_path="{}/{}/biased_40.0_V_full".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/inter_biased_M_40.0_V_full_model".format(top_ref, name),
                        use_group=True, exclude_cap_test=True,
                        distribution=True,
                        total_data=X7_SCAN_FILE, total_path="{}/{}/biased_40.0_V_full_model".format(top_ref, name),
                        lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/inter_biased_M_40.0_V_full_model".format(top_ref, name),
                        use_group=True, exclude_cap_test=True,
                        distribution=True, suffix="inter-mix",
                        total_data=X7_SCAN_FILE, total_path="{}/{}/biased_40.0_V_full".format(top_ref, name), lock=tb_lock,)
    plot_inter_pix_data(interpreted_data=X7_SCAN_FILE, base_path="{}/{}/inter_biased_M_40.0_V_full".format(top_ref, name),
                        use_group=True, exclude_cap_test=True,
                        distribution=True, suffix="inter-mix",
                        total_data=X7_SCAN_FILE, total_path="{}/{}/biased_40.0_V_full_model".format(top_ref, name),
                        lock=tb_lock,)

    print(display_name, "- CV")
    threaded_plotting.plot_bias_data(interpreted_data=X7_SCAN_FILE,
                                     base_path="{}/{}/I_V_Characteristic".format(top_ref, name), use_group=True,
                                     lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data=X7_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic".format(top_ref, name),
                                         use_group=True, distribution=False, lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data=X7_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic",
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                         lock=tb_lock)
    threaded_plotting.plot_combined_data(interpreted_data=X7_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                                         use_group=True, distribution=True, lock=tb_lock, exclude_cap_test=True,)
    threaded_plotting.plot_combined_data(interpreted_data=X7_SCAN_FILE,
                                         base_path="{}/{}/C_V_Characteristic_refined".format(top_ref, name),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=True,
                                         lock=tb_lock, exclude_cap_test=True,)
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)


def presentation_plotter(tb_lock):
    print("Plot Presentable")
    with (tb_lock):
        # noqa: S125
        # examples
        # plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data",
        #           use_group=False)
        # plot_inter_pix_data(interpreted_data='R13-Interpixel_Scan.h5',
        #                     base_path="Reference/R13/demo_measurement_65_unbiased_1_discharge",
        #                     use_group=True, suffix="inter_pix_65", total_data='Reference_R13_Scan.h5',
        #                     distribution=True, set_parasitic=False, total_path="Reference/R13/unbiased_12_full")
        # plot_bias_data(interpreted_data='Data/r13-measurement/R13_BIAS_2.h5')
        # plot_combined_data(interpreted_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', first_lower=-100,first_upper=-40, second_lower=-10, second_upper=0)

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
            64 * 64 * 30 * 30,
            384 * 400 * 30 * 30,
            384 * 400 * 30 * 30,
            384 * 400 * 30 * 30,
            1*1 * 30 * 30,
            384 * 400 * 30 * 30,
            384 * 400 * 30 * 30,
            384 * 400 * 30 * 30,
            64 * 64 * 30 * 30,
            384 * 400 * 30 * 30,
            384 * 400 * 30 * 30,
            64 * 64 * 8 * 81.,
                         ]
        plot_bias_data(iv_file_names, iv_group_names, pdf_name="I-V Combination.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_labels])
        plot_bias_data(iv_file_names, iv_group_names, pdf_name="I-V Combination-2.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_labels], area_normalisation=normalisation)

        iv_file_names_final = [
            X1_SCAN_2_FILE,
            E1_2_SCAN_FILE,
            X2_SCAN_2_FILE,
            R13_2_SCAN_FILE,
            R11_SCAN_FILE,
        ]
        iv_group_names_final = [
            "ATLAS_ITk/X1/I_V_Characteristic",
            "Reference/E1/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X2/I_V_Characteristic_2",
            "Reference/R13/C_V_Characteristic_refined",
            "Reference/R11/I_V_Characteristic",
        ]
        iv_labels_final = [
            "X1 (second)",
            "E1 (second, CV)",
            "X2 (third?)",
            "R13 (second, CV)",
            "R1 (R11)",
        ]
        assert len(iv_file_names_final) == len(iv_group_names_final)
        assert len(iv_file_names_final) == len(iv_labels_final)
        normalisation_final = [
            384 * 400 * 30 * 30,
            1 * 1 * 30 * 30,
            384 * 400 * 30 * 30,
            64 * 64 * 30 * 30,
            64 * 64 * 81 * 100,
        ]
        plot_bias_data(iv_file_names_final, iv_group_names_final, pdf_name="I-V Combination_final.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_labels_final])
        plot_bias_data(iv_file_names_final, iv_group_names_final, pdf_name="I-V Combination-2_final.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_labels_final],
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
            "ATLAS_ITk/X2/C_V_Characteristic_refined",
            "ATLAS_ITk/X1/C_V_Characteristic_refined",
            "Reference/R13/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X1/C_V_Characteristic_Second_Extended",
            "Reference/E1/C_V_Characteristic_refined",
            "Reference/E1/C_V_Characteristic",
            "ATLAS ITk/C_V_Characteristic",
            "Reference/R11/C_V_Characteristic",
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
            R13_2_SCAN_FILE,
            X1_SCAN_2_FILE,
            E1_2_SCAN_FILE,
            R11_SCAN_FILE,
            X2_SCAN_2_FILE,
        ]
        cv_groups_final = [
            "Reference/R13/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X1/C_V_Characteristic_Second_Extended",
            "Reference/E1/C_V_Characteristic_refined",
            "Reference/R11/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X2/C_V_Characteristic_refined_extended_renew_retry",
        ]
        cv_labels_final = [
            "R13 (second)",
            "X1 (second, extended)",
            "E1 (second)",
            "R1/R11",
            "X2 (extended)",
        ]
        assert len(cv_file_names_final) == len(cv_groups_final)
        assert len(cv_file_names_final) == len(cv_labels_final)
        plot_cv_data(cv_file_names_final, cv_groups_final,
                     pdf_name="C-V Combination_final.pdf", labels=[CV_DATA_FOR_.format(item) for item in cv_labels_final],
                     use_corrected=True, distribution=True)
        plot_cv_data(cv_file_names_final, cv_groups_final,
                     pdf_name="C-V Combination_raw_final.pdf", labels=[CV_DATA_FOR_.format(item) for item in cv_labels_final],
                     use_corrected=False, distribution=True)

        iv_3d_file_names = [
            X5_SCAN_FILE,
            X5_SCAN_FILE,
            X7_SCAN_FILE,
            X7_SCAN_FILE,
            X6_SCAN_FILE,
            X6_SCAN_FILE,
        ]
        iv_3d_group_names = [
            "Thesis/ATLAS_ITk/X5/I_V_Characteristic",
            "Thesis/ATLAS_ITk/X5/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X7/I_V_Characteristic",
            "Thesis/ATLAS_ITk/X7/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X6/I_V_Characteristic",
            "Thesis/ATLAS_ITk/X6/C_V_Characteristic",
        ]
        iv_3d_labels = [
            "X5",
            "X5 (CV)",
            "X7",
            "X7 (CV)",
            "X6",
            "X6 (CV, coarse)",
        ]
        assert len(iv_3d_file_names) == len(iv_3d_group_names)
        assert len(iv_3d_file_names) == len(iv_3d_labels)
        normalisation = [
            384 * 400 * 50 * 50,
            384 * 400 * 50 * 50,
            384 * 400 * 50 * 50,
            384 * 400 * 50 * 50,
            384 * 400 * 50 * 50,
        ]
        plot_bias_data(iv_3d_file_names, iv_3d_group_names, pdf_name="I-V 3D Combination.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_3d_labels])
        plot_bias_data(iv_3d_file_names, iv_3d_group_names, pdf_name="I-V 3D Combination-2.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_3d_labels],
                       area_normalisation=normalisation)

        iv_3d_file_names_final = [
            X5_SCAN_FILE,
            X7_SCAN_FILE,
            X6_SCAN_FILE,
        ]
        iv_3d_group_names_final = [
            "Thesis/ATLAS_ITk/X5/I_V_Characteristic",
            "Thesis/ATLAS_ITk/X7/I_V_Characteristic",
            "Thesis/ATLAS_ITk/X6/I_V_Characteristic",
        ]
        iv_3d_labels_final = [
            "X5",
            "X7",
            "X6",
        ]
        assert len(iv_3d_file_names_final) == len(iv_3d_group_names_final)
        assert len(iv_3d_file_names_final) == len(iv_3d_labels_final)
        normalisation_final = [
            384 * 400 * 50 * 50,
            384 * 400 * 50 * 50,
            384 * 400 * 50 * 50,
        ]
        plot_bias_data(iv_3d_file_names_final, iv_3d_group_names_final, pdf_name="I-V 3D Combination_final.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_3d_labels_final])
        plot_bias_data(iv_3d_file_names_final, iv_3d_group_names_final, pdf_name="I-V 3D Combination-2_final.pdf",
                       labels=["Bias Data for {}".format(item) for item in iv_3d_labels_final],
                       area_normalisation=normalisation_final)

        print("CV Combination")
        cv_3d_file_names = [
            X5_SCAN_FILE,
            X5_SCAN_FILE,
            X6_SCAN_FILE,
            X7_SCAN_FILE,
            X7_SCAN_FILE,
        ]
        cv_3d_groups = [
            "Thesis/ATLAS_ITk/X5/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X5/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X6/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X7/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X7/C_V_Characteristic_refined",
        ]
        cv_3d_labels = [
            'X5',
            'X5 (refined)',
            "X6",
            "X7",
            "X7 (refined)",
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
        ]
        cv_3d_groups_final = [
            "Thesis/ATLAS_ITk/X5/C_V_Characteristic_refined",
            "Thesis/ATLAS_ITk/X6/C_V_Characteristic",
            "Thesis/ATLAS_ITk/X7/C_V_Characteristic_refined",
        ]
        cv_3d_labels_final = [
            'X5 (refined)',
            "X6",
            "X7 (refined)",
        ]
        assert len(cv_3d_file_names_final) == len(cv_3d_groups_final)
        assert len(cv_3d_file_names_final) == len(cv_3d_labels_final)
        plot_cv_data(cv_3d_file_names_final, cv_3d_groups_final,
                     pdf_name="C-V 3D Combination_raw_final.pdf", labels=[CV_DATA_FOR_.format(item) for item in cv_3d_labels_final],
                     use_corrected=False, distribution=True)
        plot_cv_data(cv_3d_file_names_final, cv_3d_groups_final,
                     pdf_name="C-V 3D Combination_final.pdf", labels=[CV_DATA_FOR_.format(item) for item in cv_3d_labels_final],
                     use_corrected=True, distribution=True)


def mp_plotting_init():
    import matplotlib
    matplotlib.use('Agg')


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
        analyze_data(raw_data=file_name, base_path="{}/{}/unbiased_full".format(reference, name),
                     is_advanced=True,
                     distribution=True, full_model=False, mask_pixel=data_constants.x5_second_pixel_mask,
                     exclude_cap_test=True,
                     lock=tb_lock,
                     **correction_args)
        analyze_data(raw_data=file_name, base_path="{}/{}/biased_40_V_full".format(reference, name),
                     is_advanced=True,
                     distribution=True, full_model=False, mask_pixel=data_constants.x5_second_pixel_mask,
                     exclude_cap_test=True,
                     lock=tb_lock,
                     **correction_args)
        analyze_data(raw_data=file_name, base_path="{}/{}/unbiased_full_model".format(reference, name),
                     is_advanced=True,
                     distribution=True, mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True,
                     lock=tb_lock,
                     **correction_args)
        analyze_data(raw_data=file_name, base_path="{}/{}/biased_40_V_full_model".format(reference, name),
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
                                     exclude_test_cap=True, use_kafe2=True,
                                     fit_plot_pdf_name="Bare_analysis_parasitic_kafe2.pdf")
    analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_parasitic.pdf")
    # investigate the bump capacitance
    with tb.open_file("Bare_Repeat_2_Scan.h5", 'r') as f:
        base_group = get_base_group("Reference/bare/unbiased_8", f)
        try:
            bump_caps = np.concat(
                (base_group.total_cap.analysis.HistCap[:5, 0], base_group.total_cap.analysis.HistCap[35:, 0],))
            bump_errors = np.concat(
                (base_group.total_cap.analysis.HistCapErr[:5, 0], base_group.total_cap.analysis.HistCapErr[35:, 0],))
            weights = np.reciprocal(bump_errors ** 2)
            average_bump_cap = np.average(bump_caps, weights=weights)
            statistical_bump_error = np.shape(bump_errors)[0] / np.sum(weights)
            parasitic = get_group_attribute(base_group.total_cap.analysis, "parasitic")
            parasitic_error = get_group_attribute(base_group.total_cap.analysis, "parasitic_error")
            print(f"bump capacitance: ({parasitic - average_bump_cap}+-{statistical_bump_error}+-{parasitic_error})")
            print(np.std(bump_caps))
        except:
            print(f)
            raise

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
        print("CV Analysis for", name)
        with PdfPages("Fit References/E1_C_V_Verify.pdf") as pdf:
            analyze_data(raw_data=E1_SCAN_FILE, base_path="Reference/E1/C_V_Characteristic",
                         is_advanced=True, full_model=False, is_cv=True, use_corrected=True, cv_fit_plot_pdf=pdf,
                         **e1_depletion_args)
        print("Finished the Analysis for", name)

        print("Finished the Analysis for", name)

    from mp_analysis import e1_analysator_second
    from mp_analysis import r13_analysator_second
    from mp_analysis import x1_analysator as x1_analysator_second, x2_analysator as x2_analysator_second
    from mp_analysis import x5_analysator, x6_analysator

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
    # I do not think that we could use this data set properly as the C-V-curve ends to soon! will need to perform a new measurement for higher voltages.
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
