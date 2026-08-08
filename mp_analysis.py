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

import datetime
import logging
import multiprocessing as mp
import numpy as np
import threading
import time
from contextlib import contextmanager
from matplotlib.backends.backend_pdf import PdfPages

import pixcap65.data_constants as data_constants
from full_analysis import x4_analysator, r1_analysator
from pixcap65.analysis import analyze_data
from pixcap65.data_constants import E1_2_SCAN_FILE, R13_2_SCAN_FILE
from pixcap65.data_constants import X1_SCAN_2_FILE, X2_SCAN_2_FILE
from pixcap65.data_constants import X6_SCAN_FILE, X7_SCAN_FILE, X5_SCAN_FILE
from pixcap65.utility import synchronized_process_open_file

AUTHKEY_OUTPUT = "Fetch new authkey:"

SECOND_LABEL = " Second Try."

bare_correction_args = {
    "apply_correction": True,
    "bare_file": "packaged/data/Bare_Sample_05_Extended_Scan.h5",
    "bare_hdf_path": "Reference/Bare/unbiased_full/total_cap",
}

logger = logging.getLogger(__name__)


def synchronize_full_model(file, reference, name, bias, p_lock, **kwargs):
    unbiased_name = kwargs.get("unbiased_group", "unbiased_full")
    inter_unbiased_name = kwargs.get("inter_unbiased_group", "inter_unbiased_full")
    biased_name = kwargs.get("biased_group", "biased_{}_V_full")
    inter_biased_name = kwargs.get("inter_biased_group", "inter_biased_M_{}_V_full")

    biased_name = biased_name.format(bias)
    inter_biased_name = inter_biased_name.format(bias)
    with synchronized_process_open_file(file, mode='a', lock=p_lock) as h5_file:
        reference_node = h5_file._get_or_create_path("/{}/{}".format(reference, name), create=False)
        if unbiased_name in reference_node:
            h5_file.copy_node(where=reference_node, newname=unbiased_name + "_model", name=unbiased_name,
                              recursive=True, overwrite=True)
        if biased_name in reference_node:
            h5_file.copy_node(where=reference_node, newname=biased_name + "_model", name=biased_name,
                              recursive=True, overwrite=True)
        if inter_biased_name in reference_node:
            h5_file.copy_node(where=reference_node, newname=inter_unbiased_name + "_model", name=inter_unbiased_name,
                              recursive=True, overwrite=True)
        if inter_biased_name in reference_node:
            h5_file.copy_node(where=reference_node, newname=inter_biased_name + "_model", name=inter_biased_name,
                              recursive=True, overwrite=True)


def r13_analysator_second(tb_lock, correction_args, **kwargs):
    import pixcap65.concurrency
    name = "R13"
    display_name = "R13" + SECOND_LABEL
    top_ref = "Reference"
    print("Analyze", display_name)
    print(threading.get_native_id())
    print(mp.current_process().name)
    print(mp.current_process().pid)
    print(AUTHKEY_OUTPUT, mp.current_process().authkey)

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
                           unbiased_group="unbiased_1_full")

    # TODO: make this use some common functions instead!
    with synchronized_process_open_file(R13_2_SCAN_FILE, mode='a', lock=tb_lock) as h5_file:
        h5_file.copy_node(where="/Reference/R13", newname="inter_unbiased_full_renew_model",
                          name="inter_unbiased_full_renew",
                          recursive=True, overwrite=True)
        h5_file.copy_node(where="/Reference/R13", newname="inter_biased_M_80_V_full_renew_model",
                          name="inter_biased_M_80_V_full_renew",
                          recursive=True, overwrite=True)
        h5_file.copy_node(where="/Reference/R13", newname="inter_unbiased_renew_Extended_full_model",
                          name="inter_unbiased_renew_Extended_full",
                          recursive=True, overwrite=True)
        h5_file.copy_node(where="/Reference/R13", newname="inter_biased_M_80_V_renew_Extended_full_model",
                          name="inter_biased_M_80_V_renew_Extended_full",
                          recursive=True, overwrite=True)

    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full'), is_advanced=True,
                 lock=tb_lock, exclude_test_cap=True, distribution=True, full_model=False,
                 fit_plot_pdf_name="Fit References/{}/unbiased_reduced_model.pdf".format(name), plot=True,
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'), is_advanced=True,
                 lock=tb_lock, distribution=True, exclude_test_cap=True, full_model=False, plot=True,
                 fit_plot_pdf_name="Fit References/{}/biased_reduced_model.pdf".format(name),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_1_full_model'),
                 is_advanced=True,
                 lock=tb_lock, exclude_test_cap=True, distribution=True,
                 fit_plot_pdf_name="Fit References/{}/unbiased_full_model.pdf".format(name), plot=True,
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
                 is_advanced=True,
                 lock=tb_lock, distribution=True, exclude_test_cap=True,
                 fit_plot_pdf_name="Fit References/{}/biased_full_model.pdf".format(name), plot=True,
                 **correction_args)

    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 lock=tb_lock, distribution=True, exclude_test_cap=True,
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_1_full_model/total_cap'),
                 **correction_args)
    analyze_data(
        raw_data=R13_2_SCAN_FILE,
        base_path=hdf(top_ref, name, "inter_biased_M_80_V_full"),
        lock=tb_lock,
        distribution=True,
        exclude_test_cap=True,
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
        exclude_test_cap=True,
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
        exclude_test_cap=True,
        is_advanced=True,
        full_model=True,
        is_inter_pixel=True,
        total_cap_file=R13_2_SCAN_FILE,
        total_cap_group=hdf(top_ref, name, "biased_80_V_full_model/total_cap"),
        **correction_args
    )

    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_renew'),
                 lock=tb_lock, distribution=True, exclude_test_cap=True,
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_1_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_renew'),
                 lock=tb_lock, distribution=True, exclude_test_cap=True,
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_renew_model'),
                 lock=tb_lock, distribution=True, exclude_test_cap=True,
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_1_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_renew_model'),
                 lock=tb_lock, distribution=True, exclude_test_cap=True,
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 **correction_args)

    analyze_data(raw_data=R13_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_renew_Extended_full'),
                 lock=tb_lock, distribution=True, exclude_test_cap=True,
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_1_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_renew_Extended_full'),
                 lock=tb_lock, distribution=True, exclude_test_cap=True,
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_unbiased_renew_Extended_full_model'),
                 lock=tb_lock, distribution=True, exclude_test_cap=True,
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 total_cap_file=R13_2_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_1_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=R13_2_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_renew_Extended_full_model'),
                 lock=tb_lock, distribution=True, exclude_test_cap=True,
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
        "pixel_mask": data_constants.e1_pixel_mask,
        "chip_group_name": hdf(top_ref, name, 'sensor'),
        "apply_doping": True,
    }
    e1_depletion_args.update(**correction_args)
    synchronize_full_model(E1_2_SCAN_FILE, top_ref, name, 80, tb_lock)

    # analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full'), is_advanced=True,
    #              lock=tb_lock,
    #              full_model=False, distribution=True, exclude_cap_test=True, mask_pixel=data_constants.e1_pixel_mask,
    #              **correction_args)
    # analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'), is_advanced=True,
    #              lock=tb_lock,
    #              full_model=False, distribution=True, exclude_cap_test=True, mask_pixel=data_constants.e1_pixel_mask,
    #              **correction_args)
    # analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'), is_advanced=True,
    #              lock=tb_lock,
    #              distribution=True, exclude_cap_test=True, mask_pixel=data_constants.e1_pixel_mask, **correction_args)
    # analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
    #              is_advanced=True, lock=tb_lock,
    #              distribution=True, exclude_cap_test=True, mask_pixel=data_constants.e1_pixel_mask, **correction_args)
    #
    # analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
    #              is_advanced=True, full_model=False, is_inter_pixel=True, lock=tb_lock,
    #              distribution=True, exclude_test_cap=True, mask_pixel=data_constants.e1_pixel_mask,
    #              total_cap_file=E1_2_SCAN_FILE,
    #              total_cap_group=hdf(top_ref, name, 'unbiased_full/total_cap'),
    #              **correction_args)
    # analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
    #              is_advanced=True, full_model=False, is_inter_pixel=True, lock=tb_lock,
    #              distribution=True, exclude_test_cap=True, mask_pixel=data_constants.e1_pixel_mask,
    #              total_cap_file=E1_2_SCAN_FILE,
    #              total_cap_group=hdf(top_ref, name, 'biased_80_V_full/total_cap'),
    #              **correction_args)
    # analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
    #              is_advanced=True, full_model=True, is_inter_pixel=True, lock=tb_lock,
    #              distribution=True, exclude_test_cap=True, mask_pixel=data_constants.e1_pixel_mask,
    #              total_cap_file=E1_2_SCAN_FILE,
    #              total_cap_group=hdf(top_ref, name, 'unbiased_full/total_cap'),
    #              **correction_args)
    # analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'),
    #              is_advanced=True, full_model=True, is_inter_pixel=True, lock=tb_lock,
    #              distribution=True, exclude_test_cap=True, mask_pixel=data_constants.e1_pixel_mask,
    #              total_cap_file=E1_2_SCAN_FILE,
    #              total_cap_group=hdf(top_ref, name, 'biased_80_V_full/total_cap'),
    #              **correction_args)
    #
    # analyze_data(raw_data=E1_2_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
    #              is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
    #              lock=tb_lock, exclude_test_cap=True, mask_pixel=data_constants.e1_pixel_mask,
    #              **e1_depletion_args)

    for type_name in data_constants.e1_pixel_groups.keys():
        # with synchronized_process_open_file(E1_2_SCAN_FILE, mode='a', lock=tb_lock) as h5_file:
        #     h5_file.copy_node(where="/Reference/E1", newname="unbiased_full_model_{}".format(type_name),
        #                       name="unbiased_full_{}".format(type_name), recursive=True, overwrite=True)
        #     h5_file.copy_node(where="/Reference/E1", newname="biased_80_V_full_model_{}".format(type_name),
        #                       name="biased_80_V_full_{}".format(type_name), recursive=True, overwrite=True)
        #     h5_file.copy_node(where="/Reference/E1", newname="inter_unbiased_full_model_{}".format(type_name),
        #                       name="inter_unbiased_full_{}".format(type_name), recursive=True, overwrite=True)
        #     h5_file.copy_node(where="/Reference/E1", newname="inter_biased_M_80_V_full_model_{}".format(type_name),
        #                       name="inter_biased_M_80_V_full_{}".format(type_name), recursive=True, overwrite=True)

        # analyze_data(raw_data=E1_2_SCAN_FILE, base_path="{}/{}/unbiased_full_{}".format(top_ref, name, type_name),
        #              is_advanced=True, lock=tb_lock, full_model=False, distribution=True, exclude_cap_test=True,
        #              mask_pixel=data_constants.e1_pixel_mask,
        #              **correction_args)
        # analyze_data(raw_data=E1_2_SCAN_FILE, base_path="{}/{}/biased_80_V_full_{}".format(top_ref, name, type_name),
        #              is_advanced=True, lock=tb_lock, full_model=False, distribution=True, exclude_cap_test=True,
        #              mask_pixel=data_constants.e1_pixel_mask,
        #              **correction_args)
        # analyze_data(raw_data=E1_2_SCAN_FILE, base_path="{}/{}/unbiased_full_model_{}".format(top_ref, name, type_name),
        #              is_advanced=True, lock=tb_lock, distribution=True, exclude_cap_test=True,
        #              mask_pixel=data_constants.e1_pixel_mask,
        #              **correction_args)
        # analyze_data(raw_data=E1_2_SCAN_FILE,
        #              base_path="{}/{}/biased_80_V_full_model_{}".format(top_ref, name, type_name),
        #              is_advanced=True, lock=tb_lock, distribution=True, exclude_cap_test=True,
        #              mask_pixel=data_constants.e1_pixel_mask,
        #              **correction_args)
        #
        # analyze_data(raw_data=E1_2_SCAN_FILE, base_path="{}/{}/inter_unbiased_full_{}".format(top_ref, name, type_name),
        #              is_advanced=True, full_model=False, is_inter_pixel=True, lock=tb_lock,
        #              distribution=True, exclude_cap_test=True, mask_pixel=data_constants.e1_pixel_mask,
        #              total_cap_file=E1_2_SCAN_FILE,
        #              total_cap_group="{}/{}/unbiased_full_model_{}/total_cap".format(top_ref, name, type_name),
        #              **correction_args)
        # analyze_data(raw_data=E1_2_SCAN_FILE,
        #              base_path="{}/{}/inter_biased_M_80_V_full_{}".format(top_ref, name, type_name),
        #              is_advanced=True, full_model=False, is_inter_pixel=True, lock=tb_lock,
        #              distribution=True, exclude_cap_test=True, mask_pixel=data_constants.e1_pixel_mask,
        #              total_cap_file=E1_2_SCAN_FILE,
        #              total_cap_group="{}/{}/biased_80_V_full_model_{}/total_cap".format(top_ref, name, type_name),
        #              **correction_args)
        # analyze_data(raw_data=E1_2_SCAN_FILE,
        #              base_path="{}/{}/inter_unbiased_full_model_{}".format(top_ref, name, type_name),
        #              is_advanced=True, full_model=True, is_inter_pixel=True, lock=tb_lock,
        #              distribution=True, exclude_cap_test=True, mask_pixel=data_constants.e1_pixel_mask,
        #              total_cap_file=E1_2_SCAN_FILE,
        #              total_cap_group="{}/{}/unbiased_full_model_{}/total_cap".format(top_ref, name, type_name),
        #              **correction_args)
        # analyze_data(raw_data=E1_2_SCAN_FILE,
        #              base_path="{}/{}/inter_biased_M_80_V_full_model_{}".format(top_ref, name, type_name),
        #              is_advanced=True, full_model=True, is_inter_pixel=True, lock=tb_lock,
        #              distribution=True, exclude_cap_test=True, mask_pixel=data_constants.e1_pixel_mask,
        #              total_cap_file=E1_2_SCAN_FILE,
        #              total_cap_group="{}/{}/biased_80_V_full_model_{}/total_cap".format(top_ref, name, type_name),
        #              **correction_args)

        actual_depletion_args = data_constants.e1_pixel_depletion_args[type_name]
        actual_depletion_args.update(**correction_args)
        actual_depletion_args.update(chip_group_name=hdf(top_ref, name, 'sensor'), apply_doping=True)

        # with PdfPages("Fit References/{}/Renew_C_V_Verify_{}.pdf".format(name, type_name)) as pdf:
        analyze_data(raw_data=E1_2_SCAN_FILE,
                     base_path="{}/{}/C_V_Characteristic_refined_{}".format(top_ref, name, type_name),
                     is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                     exclude_cap_test=True,
                     lock=tb_lock, **actual_depletion_args)

    print("Finished", name)


def x1_analysator(tb_lock, correction_args, **kwargs):
    import pixcap65.concurrency
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

    synchronize_full_model(X1_SCAN_2_FILE, top_ref, name, 80, tb_lock, unbiased_group="unbiased_61_full", )

    with synchronized_process_open_file("packaged/data/X1_12_Renew_Scan.h5", 'a', lock=tb_lock) as h5_file:
        h5_file.copy_node(where="/Thesis/ATLAS_ITk/X1", name="biased_200_V_full", newname="biased_200_V_full_model",
                          overwrite=True, recursive=True)
        h5_file.copy_node(where="/Thesis/ATLAS_ITk/X1", name="inter_unbiased_full_renew_Extended",
                          newname="inter_unbiased_full_model_renew_Extended", overwrite=True, recursive=True)
        h5_file.copy_node(where="/Thesis/ATLAS_ITk/X1", name="inter_biased_M_200_V_full_renew_Extended",
                          newname="inter_biased_M_200_V_full_model_renew_Extended", overwrite=True, recursive=True)
        h5_file.copy_node(where="/Thesis/ATLAS_ITk/X1", newname="inter_biased_M_80_V_full_model_Extended__sides",
                     name="inter_biased_M_80_V_full_Extended__sides", overwrite=True, recursive=True)
        h5_file.copy_node(where="/Thesis/ATLAS_ITk/X1", newname="inter_biased_M_80_V_full_model_Extended__diagonals",
                     name="inter_biased_M_80_V_full_Extended__diagonals", overwrite=True, recursive=True)
        h5_file.copy_node(where="/Thesis/ATLAS_ITk/X1", newname="inter_biased_M_80_V_full_model_Extended__tops",
                     name="inter_biased_M_80_V_full_Extended__tops", overwrite=True, recursive=True)
        # h5_file.copy_node(where="/Thesis/ATLAS_ITk/X1", newname="inter_biased_M_80_V_full_model_Extended",
        #              name="inter_biased_M_80_V_full_Extended", overwrite=True, recursive=True)
        h5_file.copy_node(where="/Thesis/ATLAS_ITk/X1", newname="inter_biased_M_80_V_full_model_Extended_2__sides",
                          name="inter_biased_M_80_V_full_Extended_2__sides", overwrite=True, recursive=True)
        h5_file.copy_node(where="/Thesis/ATLAS_ITk/X1", newname="inter_biased_M_80_V_full_model_Extended_2__diagonals",
                          name="inter_biased_M_80_V_full_Extended_2__diagonals", overwrite=True, recursive=True)
        h5_file.copy_node(where="/Thesis/ATLAS_ITk/X1", newname="inter_biased_M_80_V_full_model_Extended_2__tops",
                          name="inter_biased_M_80_V_full_Extended_2__tops", overwrite=True, recursive=True)
        # h5_file.copy_node(where="/Thesis/ATLAS_ITk/X1", newname="inter_biased_M_80_V_full_model_Extended",
        #                   name="inter_biased_M_80_V_full_Extended", overwrite=True, recursive=True)

    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_61_full'), is_advanced=True,
                 full_model=False,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 lock=tb_lock, fit_plot_pdf_name="Fit References/{}/unbiased_reduced_model.pdf".format(name), plot=True,
                 **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full'), is_advanced=True,
                 full_model=False,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 fit_plot_pdf_name="Fit References/{}/biased_reduced_model.pdf".format(name), plot=True,
                 lock=tb_lock, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'biased_200_V_full'), is_advanced=True,
                 full_model=False,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 lock=tb_lock, **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'unbiased_61_full_model'),
                 is_advanced=True,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 lock=tb_lock, fit_plot_pdf_name="Fit References/{}/unbiased_full_model.pdf".format(name), plot=True,
                 **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'biased_80_V_full_model'),
                 is_advanced=True,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 fit_plot_pdf_name="Fit References/{}/biased_full_model.pdf".format(name), plot=True,
                 lock=tb_lock, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'biased_200_V_full_model'),
                 is_advanced=True,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 lock=tb_lock, **correction_args)

    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_61_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True, distribution=True, exclude_test_cap=True,
                 pixel_mask=data_constants.x1_second_pixel_mask, total_cap_file=X1_SCAN_2_FILE,
                 lock=tb_lock, total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_61_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True, distribution=True, exclude_test_cap=True,
                 pixel_mask=data_constants.x1_second_pixel_mask, total_cap_file=X1_SCAN_2_FILE,
                 lock=tb_lock, total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 **correction_args)

    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_unbiased_full_renew_Extended'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_61_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_unbiased_full_model_renew_Extended'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_61_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_200_V_full_renew_Extended'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file="packaged/data/X1_12_Renew_Scan.h5",
                 total_cap_group=hdf(top_ref, name, 'biased_200_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_200_V_full_model_renew_Extended'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 distribution=True, exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 lock=tb_lock,
                 total_cap_file="packaged/data/X1_12_Renew_Scan.h5",
                 total_cap_group=hdf(top_ref, name, 'biased_200_V_full_model/total_cap'),
                 **correction_args)

    # analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5", base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended'),
    #              is_advanced=True, full_model=False, is_inter_pixel=True,
    #              exclude_cap_test=True, distribution=True,
    #              lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
    #              total_cap_file=X1_SCAN_2_FILE,
    #              total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'), **correction_args)
    # analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5", base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended'),
    #              is_advanced=True, full_model=True, is_inter_pixel=True,
    #              exclude_cap_test=True, distribution=True,
    #              lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
    #              total_cap_file=X1_SCAN_2_FILE,
    #              total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'), **correction_args)

    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5", base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__sides'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=20000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5", base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__sides'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=20000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__diagonals'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=18000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5", base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__diagonals'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=18000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5", base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended__tops'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=19000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5", base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended__tops'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=19000, **correction_args)

    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended_2__sides'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=20000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended_2__sides'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=20000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended_2__diagonals'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=18000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended_2__diagonals'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=18000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_model_Extended_2__tops'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=19000, **correction_args)
    analyze_data(raw_data="packaged/data/X1_12_Renew_Scan.h5",
                 base_path=hdf(top_ref, name, 'inter_biased_M_80_V_full_Extended_2__tops'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock, mask_pixel=data_constants.x1_second_pixel_mask,
                 total_cap_file=X1_SCAN_2_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_80_V_full_model/total_cap'),
                 in_cap_file=X1_SCAN_2_FILE, in_cap_group=hdf(top_ref, name, 'inter_biased_M_80_V_full/inter_cap'),
                 inter_pix_id=19000, **correction_args)



    print("CV -", display_name)
    analyze_data(raw_data=X1_SCAN_2_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=True,
                 exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
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
                 exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask, lock=tb_lock,
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
                     exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
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
                 exclude_test_cap=True, pixel_mask=data_constants.x1_second_pixel_mask,
                 lock=tb_lock, **x1_depletion_args)
    print("Finished", display_name)


def x2_analysator(tb_lock, correction_args, **kwargs):
    import pixcap65.concurrency
    name = "X2"
    display_name = name + SECOND_LABEL
    top_ref = "Thesis/ATLAS_ITk"
    print("Analyze", display_name)
    print(threading.get_native_id())
    print(mp.current_process().name)
    print(mp.current_process().pid)
    print(AUTHKEY_OUTPUT, mp.current_process().authkey)

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


def x5_analysator(tb_lock, correction_args, **kwargs):
    import pixcap65.concurrency
    name = "X5"
    display_name = name
    top_ref = "Thesis/ATLAS_ITk"
    print("Analyze", display_name)
    print(threading.get_native_id())
    print(mp.current_process().name)
    print(mp.current_process().pid)
    print(AUTHKEY_OUTPUT, mp.current_process().authkey)

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
                 exclude_cap_test=True,
                 lock=tb_lock, plot=True, fit_plot_pdf_name="Fit References/{}/unbiased_reduced_model_reference_fits.pdf".format(name),
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40_V_full'),
                 is_advanced=True,
                 distribution=True, full_model=False, mask_pixel=data_constants.x5_second_pixel_mask,
                 exclude_cap_test=True,
                 lock=tb_lock, plot=True,
                 fit_plot_pdf_name="Fit References/{}/biased_reduced_model_reference_fits.pdf".format(name),
                 mask_lower=50e-15,
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_90_V_full'),
                 is_advanced=True,
                 distribution=True, full_model=False, mask_pixel=data_constants.x5_second_pixel_mask,
                 exclude_cap_test=True,
                 lock=tb_lock,
                 # mask_lower=50e-15,
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'),
                 is_advanced=True,
                 distribution=True, mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True,
                 lock=tb_lock, plot=True,
                 fit_plot_pdf_name="Fit References/{}/unbiased_full_model_reference_fits.pdf".format(name),
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40_V_full_model'),
                 is_advanced=True, full_model=True,
                 distribution=True, mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True,
                 lock=tb_lock, mask_lower=51.5e-15,
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_90_V_full_model'),
                 is_advanced=True,
                 distribution=True, mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True,
                 lock=tb_lock,
                 # mask_lower=50e-15,
                 **correction_args)

    # handle the inter-pix analysis
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True, distribution=True,
                 total_cap_file=X5_SCAN_FILE,
                 lock=tb_lock,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_40_V_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 mask_pixel=data_constants.x5_pixel_mask, exclude_cap_test=True, distribution=True,
                 lock=tb_lock,
                 total_cap_file=X5_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_40_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 mask_pixel=data_constants.x5_second_pixel_mask, exclude_cap_test=True, distribution=True,
                 total_cap_file=X5_SCAN_FILE,
                 lock=tb_lock,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full/total_cap'),
                 **correction_args)
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_40_V_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 mask_pixel=data_constants.x5_pixel_mask, exclude_cap_test=True, distribution=True,
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
                 exclude_cap_test=True,
                 lock=tb_lock, **x5_depletion_args)
    print("Finished first CV")
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 mask_pixel=data_constants.x5_pixel_mask,
                 exclude_cap_test=True,
                 lock=tb_lock, **x5_depletion_args_refined)
    print("Finished second CV")
    analyze_data(raw_data=X5_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined_Extended'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 mask_pixel=data_constants.x5_pixel_mask,
                 exclude_cap_test=True,
                 lock=tb_lock, **x5_depletion_args_refined)
    print("Finished third CV")

    # add here the additonal CV analysis used for extended range!
    print("Finished the Analysis for X5")


def x6_analysator(tb_lock, correction_args, **kwargs):
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
                 is_advanced=True, distribution=True, full_model=False, exclude_cap_test=True,
                 lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask,
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_45_V_full'),
                 is_advanced=True,
                 distribution=True, full_model=False, mask_pixel=data_constants.x6_second_pixel_mask,
                 exclude_cap_test=True,
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'),
                 is_advanced=True, distribution=True, exclude_cap_test=True,
                 lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask,
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_45_V_full_model'),
                 is_advanced=True,
                 distribution=True, mask_pixel=data_constants.x6_second_pixel_mask, exclude_cap_test=True,
                 **correction_args)

    # handle the inter-pix analysis
    analyze_data(raw_data=data_constants.X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 total_cap_file=data_constants.X6_SCAN_FILE,
                 lock=tb_lock,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 total_cap_file=data_constants.X6_SCAN_FILE,
                 lock=tb_lock,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_biased_M_45_V_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock,
                 total_cap_file=data_constants.X6_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_45_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X6_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_biased_M_45_V_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True,
                 lock=tb_lock,
                 total_cap_file=data_constants.X6_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_45_V_full_model/total_cap'),
                 **correction_args)

    # CV analysis
    print("Cv analysis for X6")
    analyze_data(raw_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 exclude_cap_test=True, mask_pixel=data_constants.x6_second_pixel_mask,
                 lock=tb_lock, **x6_depletion_args)

    analyze_data(raw_data=X6_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 lock=tb_lock, mask_pixel=data_constants.x6_second_pixel_mask,
                 exclude_cap_test=True, **x6_depletion_args_refined)
    print("Finished the CV analysis for X6")


def x7_analysator(tb_lock, correction_args, **kwargs):
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
                 distribution=True, full_model=False, exclude_cap_test=True, lock=tb_lock,
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40.0_V_full'),
                 is_advanced=True, distribution=True, full_model=False, exclude_cap_test=True, lock=tb_lock,
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_full_model'),
                 is_advanced=True, distribution=True, exclude_cap_test=True, lock=tb_lock,
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE, base_path=hdf(top_ref, name, 'biased_40.0_V_full_model'),
                 is_advanced=True, distribution=True, exclude_cap_test=True, lock=tb_lock,
                 **correction_args)

    # inter-pixel capacitance analysis
    analyze_data(raw_data=data_constants.X7_SCAN_FILE, base_path=hdf(top_ref, name, 'inter_unbiased_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True, lock=tb_lock,
                 total_cap_file=data_constants.X7_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_biased_M_40.0_V_full'),
                 is_advanced=True, full_model=False, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True, lock=tb_lock,
                 total_cap_file=data_constants.X7_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'biased_40.0_V_full_model/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_unbiased_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True, lock=tb_lock,
                 total_cap_file=data_constants.X7_SCAN_FILE,
                 total_cap_group=hdf(top_ref, name, 'unbiased_full/total_cap'),
                 **correction_args)
    analyze_data(raw_data=data_constants.X7_SCAN_FILE,
                 base_path=hdf(top_ref, name, 'inter_biased_M_40.0_V_full_model'),
                 is_advanced=True, full_model=True, is_inter_pixel=True,
                 exclude_cap_test=True, distribution=True, lock=tb_lock,
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
                 exclude_cap_test=True, lock=tb_lock,
                 **x7_depletion_args)

    analyze_data(raw_data=X7_SCAN_FILE, base_path=hdf(top_ref, name, 'C_V_Characteristic_refined'),
                 is_advanced=True, full_model=False, is_cv=True, use_corrected=False,
                 exclude_cap_test=True, lock=tb_lock,
                 **x7_depletion_args_refined)
    print("Finished the Analysis for", display_name)


def error_handler(exc):
    logger.error("While performing the analysis in multiple processes an error occured.", exc_info=exc)

@contextmanager
def processed_manager(**kwargs):
    import pixcap65.concurrency
    import gc
    with pixcap65.concurrency.get_context_manager(**kwargs) as manager_ctx:
        try:
            lock = manager_ctx.RLock()
            yield (manager_ctx, lock)
        finally:
            del lock
            gc.collect()

    gc.collect()


if __name__ == "__main__":
    # perhaps it is necessary to provide the different locks as arguments to the
    import pixcap65.concurrency

    start_time = time.time()

    process_handles = [
        x1_analysator,
        x2_analysator,
        x5_analysator,
        x6_analysator,
        x7_analysator,
        e1_analysator_second,
        r13_analysator_second,
        r1_analysator,
        x4_analysator,
    ]


    with processed_manager() as (manager, tables_lock):
        print("started the primary manager")
        # necessary to connect to the manager from the additional processes correctly
        authkey = mp.current_process().authkey
        manager_args = {
            # "authkey": authkey,
            "address": manager.address,
        }
        print("The following authkey is in use:", authkey)
        print("Fetched the authkey from the manager:", manager._authkey)
        # bare_analysis_handler(tables_lock)
        # handle all the processes
        # this could not be transformed to process handled as we could not submit authkeys!
        processes = { process.__name__: mp.Process(target=process, name=process.__name__, args=(tables_lock, bare_correction_args,), kwargs=manager_args) for process in process_handles }

        # bare_analysis_handler(tables_lock)

        for p in processes.values():
            p.start()

        while len(processes) > 0:
            remove_names = []
            for name, p in processes.items():
                p.join()
                if p.exitcode is None:
                    print("There was a process which has not terminated after join. The process is", name)
                else:
                    p.close()
                    remove_names.append(name)

            while len(remove_names) > 0:
                name = remove_names.pop()
                del processes[name]

        print(processes)
        print("Elapsed time: ", time.time() - start_time)


    import gc
    gc.collect()
    print("Finished collection!")
    with open('depletion_manager_information_??.txt', 'a') as f:
        date_obj = datetime.datetime.now()
        full_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        date = date_obj.strftime("%Y-%m-%d")
        time_str = date_obj.strftime("%H:%M:%S")
        print(date, mp.current_process().pid, time_str,
              mp.current_process().name, mp.current_process().authkey, "FINISH - MARK", file=f)


# list of updated sensors
# x4
# r1
# x5
# x6
# x7
# r13
# e1
