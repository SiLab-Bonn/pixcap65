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
Usages of the :py:mod:`pixcap65.utility.converter` module for the bachelor thesis.
"""

from pixcap65.utility.converter import *
from .data_constants import *


def internal_node_copy(grouping: tb.Group, h5_file: tb.File, parent: tb.Group,
                       exclude_iv=False, exclude_cv=False):
    """
    internal copy function of nodes used at the wrong group in the initial measurement definition.

    :author: Dominik Fischer
    :date: 2026-09-17

    last updated: 2026-09-17

    :param grouping: group under which the measurement or anaylsis is currently sitting
    :param h5_file: hdf file where the data is stored.
    :param parent: new node/group under which to put the data.
    :param exclude_iv: whether to exlude sole i-v characterization measurements.
    :param exclude_cv: whether to exlude sole c-v characterization measurements.
    """
    for group in grouping._f_iter_nodes():
        if not isinstance(group, tb.Group):
            continue
        if group == grouping:
            continue
        if group._v_name == "C_V_Characteristic_refined" and exclude_cv:
            continue
        if group._v_name == "I_V_Characteristic" and exclude_iv:
            continue
        print(group)
        h5_file.copy_node(where=grouping, name=group._v_name,
                          newname=group._v_name,
                          newparent=parent, recursive=True, overwrite=True)


if __name__ == '__main__':
    with tb.open_file(X1_SCAN_2_FILE, "a") as h5_file:
        generate_bias_table(h5_file.root.ATLAS_ITk.X1.I_V_Characteristic.biasing.measurements, transform_api=True)
        regenerate_c_v_errors(h5_file.root.ATLAS_ITk.X1.C_V_Characteristic_refined.biasing.measurements)
        regenerate_inter_pix_errors(h5_file.root.Thesis.ATLAS_ITk.X1.inter_unbiased_full.inter_cap.measurements)
        regenerate_inter_pix_errors(h5_file.root.Thesis.ATLAS_ITk.X1.inter_biased_M_80_V_full.inter_cap.measurements)
        combine_cv_measurements(h5_file.root.ATLAS_ITk.X1.C_V_Characteristic_refined,
                                h5_file.root.Thesis.ATLAS_ITk.X1.C_V_Characteristic_refined_Extended,)
        h5_file.copy_node(where=h5_file.root.ATLAS_ITk.X1, newparent=h5_file.root.Thesis.ATLAS_ITk.X1,
                          name="C_V_Characteristic_refined_Extended_Combined",
                          newname="C_V_Characteristic_refined_Extended_Combined",
                          overwrite=True, recursive=True)
        old_x1 = h5_file.root.ATLAS_ITk.X1
        internal_node_copy(h5_file.root.ATLAS_ITk.X1, h5_file, h5_file.root.Thesis.ATLAS_ITk.X1)
        generate_pixel_dimensions(h5_file.root.Thesis.ATLAS_ITk.X1)

    with tb.open_file(X2_SCAN_2_FILE, "a") as h5_file:
        # generate_bias_table(h5_file.root.ATLAS_ITk.X2.I_V_Characteristic.biasing.measurements)
        generate_bias_table(h5_file.root.ATLAS_ITk.X2.C_V_Characteristic_refined.biasing.measurements)
        regenerate_c_v_errors(h5_file.root.ATLAS_ITk.X2.C_V_Characteristic_refined.biasing.measurements)
        generate_pixel_dimensions(h5_file.root.ATLAS_ITk.X2)
        h5_file.root.ATLAS_ITk.X2.C_V_Characteristic_refined.biasing.measurements.BiasVoltageHist.attrs["Units"] = "V"
        old_x2 = h5_file.root.ATLAS_ITk.X2
        assert isinstance(old_x2, tb.Group)
        internal_node_copy(h5_file.root.ATLAS_ITk.X2, h5_file, h5_file.root.Thesis.ATLAS_ITk.X2)

        with tb.open_file("packaged/data/X2_12_Renew_Scan.h5") as backing_file:
            backing_file.copy_children(backing_file.root.Thesis.ATLAS_ITk.X2, h5_file.root.Thesis.ATLAS_ITk.X2,
                                       recursive=True, overwrite=True)

    with tb.open_file(R11_SCAN_FILE, "a") as h5_file:
        generate_pixel_dimensions(h5_file.root.Reference.R1)
        old_sensor = h5_file.root.Reference.R11
        internal_node_copy(h5_file.root.Reference.R11, h5_file, h5_file.root.Reference.R1,
                           exclude_cv=True, exclude_iv=True)

        # noinspection DuplicatedCode
        physical_dimensions = h5_file.root.Reference.R1.sensor.PhysicalDimensions[:]
        physical_dimensions[:, :] = np.asarray([6, 81], dtype=np.float64)
        physical_dimensions[:, 0] = np.nan
        h5_file.root.Reference.R1.sensor.PhysicalDimensions[:] = physical_dimensions
        h5_file.root.Reference.R1.sensor.PhysicalDimensions.flush()
        regenerate_i_v_errors(h5_file.root.Reference.R1.I_V_Characteristic.biasing.measurements)
        regenerate_i_v_errors(h5_file.root.Reference.R1.C_V_Characteristic.biasing.measurements)
        regenerate_i_v_errors(h5_file.root.Reference.R1.C_V_Characteristic_refined.biasing.measurements)
        full_regenerate_bias_table(h5_file.root.Reference.R1.I_V_Characteristic.biasing.measurements,
                                   force_recalculation=True)
        full_regenerate_bias_table(h5_file.root.Reference.R1.C_V_Characteristic.biasing.measurements,
                                   force_recalculation=True)
        full_regenerate_bias_table(h5_file.root.Reference.R1.C_V_Characteristic_refined.biasing.measurements,
                                   force_recalculation=True)
    with tb.open_file("packaged/data/R13_2_Scan.h5", "a") as h5_file:
        h5_file.copy_children(h5_file.root.ATLAS_ITk.X2, h5_file.root.Reference.R13, recursive=True, overwrite=True)
        h5_file.flush()
        with tb.open_file(R13_2_SCAN_FILE, "a") as second_file:
            h5_file.copy_children(h5_file.root, second_file.root, recursive=True, overwrite=True)

    with tb.open_file("packaged/data/R13_3_Scan.h5", "a") as h5_file:
        with tb.open_file(R13_2_SCAN_FILE, "a") as second_file:
            h5_file.copy_children(h5_file.root.Reference.R13, second_file.root.Reference.R13,
                                  recursive=True, overwrite=True)

    with tb.open_file(R13_2_SCAN_FILE, 'a') as h5_file:
        generate_pixel_dimensions(h5_file.root.Reference.R13, 30)
        generate_bias_table(h5_file.root.Reference.R13.C_V_Characteristic_refined.biasing.measurements)
        regenerate_c_v_errors(h5_file.root.Reference.R13.C_V_Characteristic_refined.biasing.measurements)
        regenerate_inter_pix_errors(h5_file.root.Reference.R13.inter_unbiased_full.inter_cap.measurements)
        regenerate_inter_pix_errors(h5_file.root.Reference.R13.inter_biased_M_80_V_full.inter_cap.measurements)

        with tb.open_file(X2_SCAN_2_FILE, 'a') as old_file:
            old_file.copy_node(where=old_file.root.Reference.R13, newparent=h5_file.root.Reference.R13,
                               name="inter_unbiased_full_renew_Extended",
                               newname="inter_unbiased_renew_Extended_full", recursive=True, overwrite=True)
            old_file.copy_node(where=old_file.root.Reference.R13, newparent=h5_file.root.Reference.R13,
                               name="inter_biased_M_80_V_full_renew_Extended",
                               newname="inter_biased_M_80_V_renew_Extended_full", recursive=True, overwrite=True)

    with tb.open_file(E1_2_SCAN_FILE, "a") as h5_file:
        generate_pixel_dimensions(h5_file.root.Reference.E1)
        physical_dimensions = h5_file.root.Reference.E1.sensor.PhysicalDimensions[:]
        physical_dimensions[:, 0] = np.nan
        for key, region in data_constants.e1_pixel_groups.items():
            for col_range, row_range in zip(region["columns"], region["rows"]):
                physical_dimensions[col_range[0]:col_range[1], row_range[0]:row_range[1]]\
                    = [data_constants.e1_pixel_dimensions[key],
                       data_constants.e1_pixel_dimensions[key]]

        h5_file.root.Reference.E1.sensor.PhysicalDimensions[:] = physical_dimensions
        h5_file.root.Reference.E1.sensor.PhysicalDimensions.flush()

        # still need to generate all the measurement errors
        regenerate_c_v_errors(h5_file.root.Reference.E1.C_V_Characteristic_refined.biasing.measurements)
        regenerate_inter_pix_errors(h5_file.root.Reference.E1.inter_unbiased_full.inter_cap.measurements)
        regenerate_inter_pix_errors(h5_file.root.Reference.E1.inter_biased_M_80_V_full.inter_cap.measurements)

        split_sensor_group(h5_file.root.Reference.E1.inter_biased_M_80_V_full)
        split_sensor_group(h5_file.root.Reference.E1.C_V_Characteristic_refined)
        split_sensor_group(h5_file.root.Reference.E1.biased_80_V_full)
        split_sensor_group(h5_file.root.Reference.E1.unbiased_full)
        split_sensor_group(h5_file.root.Reference.E1.inter_unbiased_full)

    with tb.open_file(X4_SCAN_FILE, "a") as h5_file:
        generate_pixel_dimensions(h5_file.root.Thesis.ATLAS_ITk.X4)
        h5_file.copy_node(where=h5_file.root.Thesis.ATLAS_ITk.X4, name="inter_unbiased_full_Extended_Second",
                          newname="inter_unbiased_full", overwrite=True, recursive=True)
        h5_file.copy_node(where=h5_file.root.Thesis.ATLAS_ITk.X4, name="inter_biased_M_80_V_full_Extended",
                          newname="inter_biased_M_80_V_full", overwrite=True, recursive=True)

    with tb.open_file(X5_SCAN_FILE, "a") as h5_file:
        generate_pixel_dimensions(h5_file.root.Thesis.ATLAS_ITk.X5)

    with tb.open_file(X6_SCAN_FILE, "a") as h5_file:
        wrong_parent = h5_file.root.Thesis.ATLAS_ITk.X7
        right_parent = h5_file.root.Thesis.ATLAS_ITk.X6
        if "inter_unbiased_full" in wrong_parent:
            h5_file.move_node(where=wrong_parent, name="inter_unbiased_full", newparent=right_parent,)
            h5_file.move_node(where=wrong_parent, name="inter_biased_M_45.0_V_full", newparent=right_parent,
                              newname="inter_biased_M_45_V_full")
            h5_file.move_node(where=wrong_parent, name="biased_45.0_V_full", newparent=right_parent,
                              newname="biased_45_V_full")
            h5_file.move_node(where=wrong_parent, name="C_V_Characteristic_refined", newparent=right_parent,
                              overwrite=True)

        generate_pixel_dimensions(h5_file.root.Thesis.ATLAS_ITk.X6)

    with tb.open_file(X7_SCAN_FILE, 'a') as h5_file:
        generate_pixel_dimensions(h5_file.root.Thesis.ATLAS_ITk.X7)
