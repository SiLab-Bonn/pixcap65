# ----------------------------------------------------------
#  Copyright (c) .
#   All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------
from os import PathLike

import logging
import numpy as np
import tables as tb
import time
from collections.abc import Iterable

from pixcap65.analysis import get_test_capacitance_data
from pixcap65.analysis_util.data_store import SummaryTable
from pixcap65.data_constants import R11_SCAN_FILE, R13_2_SCAN_FILE, E1_2_SCAN_FILE
from pixcap65.data_constants import X1_SCAN_2_FILE, X2_SCAN_2_FILE
from pixcap65.data_constants import X5_SCAN_FILE, X6_SCAN_FILE, X7_SCAN_FILE
from pixcap65.utility import synchronized_process_open_file
from pixcap65.utility.utils_2 import walk_to_node

SUMMARY_FILE = 'conclude_summary.h5'

logger = logging.getLogger(__name__)

demo_type_a = np.dtype([
    ("str1", np.str_, 8),
    ("str2", np.str_, 16),
    ("str3", np.str_, 32),
])

class DemoDescription(tb.IsDescription):
    str1 = tb.StringCol(itemsize=8)
    str2 = tb.StringCol(itemsize=16)
    str3 = tb.StringCol(itemsize=32)


field_names = [name.replace(" ", "_") for name in [
    "0 w_o bump", "1 w_o bump", "2 w_o bump", "3 w_o bump", "4 w_o bump", "5",
    "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "17",
    "18", "19", "20", "21", "22", "23", "24", "25", "26", "27 w_ bump",
    "28 w_ bump", "29 w_ bump", "30 w_ bump", "31 w_ bump", "32 w_ bump", "33 w_ bump",
    "34 w_ bump", "35 w_o bump", "36 w_o bump", "37 w_o bump", "38 w_o bump", "39 w_o bump"
]]

field_design_values = {
    "0 w_o bump": 0,
    "1 w_o bump": 0,
    "2 w_o bump": 0,
    "3 w_o bump": 0,
    "4 w_o bump": 0,
    "5": 15.41,
    "6": 30.33,
    "7": 60.14,
    "8": 119.39,
    "9": 237.87,
    "10": 30.82,
    "11": 61.64,
    "12": 123.28,
    "13": 2.51,
    "14": 1.63,
    "15": 5.51,
    "17": 8.53,
    "18": 8.53,
    "19": 8.53,
    "20": 8.53,
    "21": 8.53,
    "22": 8.53,
    "23": 8.53,
    "24": 8.53,
    "25": 8.53,
    "26": 8.53,
    "27 w_ bump": 0,
    "28 w_ bump": 0,
    "29 w_ bump": 0,
    "30 w_ bump": 0,
    "31 w_ bump": 0,
    "32 w_ bump": 0,
    "33 w_ bump": 0,
    "34 w_ bump": 0,
    "35 w_o bump": 0,
    "36 w_o bump": 0,
    "37 w_o bump": 0,
    "38 w_o bump": 0,
    "39 w_o bump": 0
}

test_design_values = {name.replace(" ", "_") : field_design_values[name] for name in [
"0 w_o bump", "1 w_o bump", "2 w_o bump", "3 w_o bump", "4 w_o bump", "5",
"6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "17",
"18", "19", "20", "21", "22", "23", "24", "25", "26", "27 w_ bump",
"28 w_ bump", "29 w_ bump", "30 w_ bump", "31 w_ bump", "32 w_ bump", "33 w_ bump",
"34 w_ bump", "35 w_o bump", "36 w_o bump", "37 w_o bump", "38 w_o bump", "39 w_o bump"] }

spatial_identifier = ["X{}".format(i) for i in range(3, 9)]

def generate_test_summary(files: Iterable[PathLike], groups: Iterable[PathLike], sensors: Iterable[str], summary_file: PathLike) -> None:
    with synchronized_process_open_file(summary_file, mode='a') as summary_file:
        table_description = [("Sensor", 'S16'), ]
        for name in field_names:
            table_description.extend([("test_{}".format(name), np.float64), ])
            table_description.extend([("test_{}_error".format(name), np.float64), ])

        if "TestCap" in summary_file.root:
            summary_file.root.TestCap.remove()
            summary_file.root.DemoCao.remove()
            time.sleep(10)

        table_type = np.dtype(table_description)
        table = summary_file.create_table(where=summary_file.root, name="TestCap",
                                          title="Test Capacitances from the different sensors",
                                          description=table_type)
        summary_file.create_table(where=summary_file.root, name="DemoCao", description=SummaryTable)
        for file, group_path, sensor in zip(files, groups, sensors):
            with synchronized_process_open_file(file, mode='r') as h5_file:
                group, _ = walk_to_node(h5_file.root, group_path, create=False, verify_create=True)
                test_cap, test_cap_err = get_test_capacitance_data(group, print_result=False)
                assert isinstance(sensor, str)
                result_data = [sensor.encode()]
                # result_data = [int(sensor.encode().hex(), base=16)]
                for cap, cap_err in zip(test_cap, test_cap_err):
                    result_data.append(cap)
                    result_data.append(cap_err)
                table.append([result_data, ])
        table.flush()
        table.cols.Sensor.create_csindex()
        table.flush()

        if "DemoTableDescription" in summary_file.root:
            summary_file.root.DemoTableDescription.remove()
            summary_file.root.DemoTableRecArray.remove()
            time.sleep(10)

        table = summary_file.create_table(where=summary_file.root, name="DemoTableDescription", description=DemoDescription)
        entry = table.row
        entry['str1'] = "E1_dnw_15_50_abcdefghijklm"
        entry['str2'] = "E1_dnw_15_50_abcdefghijklm"
        entry['str3'] = "E1_dnw_15_50_abcdefghijklm"
        entry.append()
        entry['str1'] = "E1_dnw_20_50_abcdefghijklm"
        entry['str2'] = "E1_dnw_20_50_abcdefghijklm"
        entry['str3'] = "E1_dnw_20_50_abcdefghijklm"
        entry.append()
        entry['str1'] = "E1_dnw_25_50_abcdefghijklm"
        entry['str2'] = "E1_dnw_25_50_abcdefghijklm"
        entry['str3'] = "E1_dnw_25_50_abcdefghijklm"
        entry.append()
        entry['str1'] = "E1_dnw_30_50_abcdefghijklm"
        entry['str2'] = "E1_dnw_30_50_abcdefghijklm"
        entry['str3'] = "E1_dnw_30_50_abcdefghijklm"
        entry.append()
        table.flush()

        print(table)

        rec_array_sample = np.rec.array([
            ("E1_dnw_15_50_abcdefghijklm", "E1_dnw_15_50_abcdefghijklm", "E1_dnw_15_50_abcdefghijklm"),
            ("E1_dnw_20_50_abcdefghijklm", "E1_dnw_20_50_abcdefghijklm", "E1_dnw_20_50_abcdefghijklm"),
            ("E1_dnw_25_50_abcdefghijklm", "E1_dnw_25_50_abcdefghijklm", "E1_dnw_25_50_abcdefghijklm"),
            ("E1_dnw_30_50_abcdefghijklm", "E1_dnw_30_50_abcdefghijklm", "E1_dnw_30_50_abcdefghijklm"),
        ], dtype=demo_type_a)
        print(rec_array_sample)
        print(hasattr(rec_array_sample, "dtype"))
        try:
            summary_file.create_table(where=summary_file.root, name="DemoTableRecArray",obj=rec_array_sample).flush()
        except:
            eff_table = summary_file.root.DemoTableRecArray
            print(eff_table.description._v_is_nested)
            print(rec_array_sample.dtype == eff_table.dtype)
            print(eff_table.dtype)
            print(rec_array_sample.dtype)
            print(eff_table._v_dtype)
            print(tb.descr_from_dtype(demo_type_a))
            # np.rec.array(rec_array_sample, dtype=eff_table.dtype)
        print(rec_array_sample)
        print("Type")
        print(demo_type_a)
        print(rec_array_sample.dtype)
        print(table.dtype)
        print(tb.dtype_from_descr(DemoDescription))
        print(DemoDescription)
        print(tb.descr_from_dtype(demo_type_a))
        print(DemoDescription().columns)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    summary_files = ["packaged/Reference_Bare_renewed.h5", R13_2_SCAN_FILE, R11_SCAN_FILE, X1_SCAN_2_FILE, X2_SCAN_2_FILE, X5_SCAN_FILE, X6_SCAN_FILE,
                     X7_SCAN_FILE, E1_2_SCAN_FILE]
    summary_groups = [
        "Reference/Bare/unbiased_31_renew/total_cap/analysis",
        "Reference/R13/unbiased_1_full_model/total_cap/analysis",
        "Reference/R11/unbiased_full_model/total_cap/analysis",
        "Thesis/ATLAS_ITk/X1/unbiased_61_full_model/total_cap/analysis",
        "Thesis/ATLAS_ITk/X2/unbiased_1_full_model/total_cap/analysis",
        "Thesis/ATLAS_ITk/X5/unbiased_full_model/total_cap/analysis",
        "Thesis/ATLAS_ITk/X6/unbiased_full_model/total_cap/analysis",
        "Thesis/ATLAS_ITk/X7/unbiased_full_model/total_cap/analysis",
        "Reference/E1/unbiased_full_model/total_cap/analysis",
    ]
    summary_sensors = ["Bare", "R13", "R1/R11", "X1", "X2", "X5", "X6", "X7", "E1"]

    logger.info("generate summary")
    generate_test_summary(summary_files, summary_groups, summary_sensors, SUMMARY_FILE)

    with tb.open_file(R11_SCAN_FILE) as h5_file:
        get_test_capacitance_data(h5_file.root.Reference.R11.unbiased_full.total_cap.analysis)


    # but how to format such a table?
    # we will need a master table 'to do' so.
    class MasterTableCapacitanceEntry(tb.IsDescription):
        capacitance = tb.Float64Col(pos=0)
        capacitance_err = tb.Float64Col(pos=1)
        capacitance_systematic_general = tb.Float64Col(pos=2)
        capacitance_systematic_dispersion = tb.Float64Col(pos=3)


    class MasterTableVoltageEntry(tb.IsDescription):
        dep_voltage = tb.Float64Col(pos=0)
        dep_voltage_err = tb.Float64Col(pos=1)
        dep_voltage_systematic_general = tb.Float64Col(pos=2)
        dep_voltage_systematic_dispersion = tb.Float64Col(pos=3)


    class MasterExtractionTable(tb.IsDescription):
        sensor = tb.StringCol(pos=0, itemsize=16)
        unbiased_capacitance = MasterTableCapacitanceEntry()
        unbiased_inter_capacitance = MasterTableCapacitanceEntry()
        biased_capacitance = MasterTableCapacitanceEntry()
        bias_voltage = tb.Float64Col(pos=1)
        biased_inter_capacitance = MasterTableCapacitanceEntry()
        depletion_voltage = MasterTableVoltageEntry()


    class GroupProviderScheme(tb.IsDescription):
        file = tb.StringCol(itemsize=220, pos=0)
        unbiased_group = tb.StringCol(itemsize=100, pos=1)
        unbiased_inter_pix_group = tb.StringCol(itemsize=100, pos=2)
        biased_group = tb.StringCol(itemsize=100, pos=3)
        biased_inter_pix_group = tb.StringCol(itemsize=100, pos=4)
        cv_group = tb.StringCol(itemsize=100, pos=5)


    # generate the table with summarieses all the data!
    DEFAULT_ENTRY = (np.nan, np.nan, np.nan, np.nan)

    def generate_ba_summary(extraction: np.recarray, target_file):
        # here we need to access the distribution tables from the data sets
        # for the inter-pix: Which of the different distribution tables should be used in the end?
        # nevertheless we need to iterate the recarray

        def generate_cap_entry(res_table):
            return (res_table.cap_corrected[0],
                    res_table.cap_corrected_err[0],
                    res_table.cap_systematic_error[0],
                    res_table.cap_systematic_dispersion[0])

        with tb.open_file(target_file, 'a') as out_file:
            if "GeneralSummaryTable" in out_file.root:
                out_file.root.GeneralSummaryTable.remove()
            table = out_file.create_table(where=out_file.root, name="GeneralSummaryTable",
                                          description=MasterExtractionTable)
            current_row = table.row
            for entry in extraction:
                assert isinstance(entry, np.record)
                with tb.open_file(str(entry.file)) as h5_file:
                    current_entry = DEFAULT_ENTRY
                    if entry.unbiased_group is not None and 'None' not in entry.unbiased_group:
                        group = h5_file._get_or_create_path("/" + entry.unbiased_group, False)
                        if "DistResultfF" in group:
                            result_table = np.rec.array(group.DistResultfF.read(), dtype=group.DistResultfF.dtype)
                            assert isinstance(result_table, np.recarray)
                            if result_table.shape[0] > 0:
                                current_entry = generate_cap_entry(result_table)

                    current_row['unbiased_capacitance'] = current_entry
                    current_entry  = DEFAULT_ENTRY

                    if entry.biased_group is not None and 'None' not in entry.biased_group:
                        group = h5_file._get_or_create_path("/" + entry.biased_group, False)
                        if "DistResultfF" in group:
                            result_table = np.rec.array(group.DistResultfF.read(), dtype=group.DistResultfF.dtype)
                            assert isinstance(result_table, np.recarray)
                            if result_table.shape[0] > 0:
                                current_entry = generate_cap_entry(result_table)

                    current_row['biased_capacitance'] = current_entry
                    current_entry = DEFAULT_ENTRY

                    if entry.unbiased_inter_pix_group is not None and 'None' not in entry.unbiased_inter_pix_group:
                        group = h5_file._get_or_create_path("/" + entry.unbiased_inter_pix_group, False)
                        if "DistResultfF" in group:
                            # we must read here at the right position
                            result_table = np.rec.array(group.DistResultfF.read_where("""(bias == {})""".format(14000)),
                                                        dtype=group.DistResultfF.dtype)
                            assert isinstance(result_table, np.recarray)
                            if result_table.shape[0] > 0:
                                current_entry = generate_cap_entry(result_table)

                    current_row["unbiased_inter_capacitance"] = current_entry
                    current_entry = DEFAULT_ENTRY

                    if entry.biased_inter_pix_group is not None and 'None' not in entry.biased_inter_pix_group:
                        group = h5_file._get_or_create_path("/" + entry.biased_inter_pix_group, False)
                        if "DistResultfF" in group:
                            # we must read here at the right position
                            result_table = np.rec.array(group.DistResultfF.read_where("""(bias == {})""".format(14000)),
                                                        dtype=group.DistResultfF.dtype)
                            assert isinstance(result_table, np.recarray)
                            if result_table.shape[0] > 0:
                                current_entry = generate_cap_entry(result_table)

                    current_row["biased_inter_capacitance"] = current_entry
                    current_entry = DEFAULT_ENTRY

                    # handle the C-V-Parameterisation
                    if entry.cv_group is not None and 'None' not in entry.cv_group:
                        group = h5_file._get_or_create_path("/" + entry.cv_group, False)
                        if "SensorDepletionRaw" in group:
                            result_table = np.rec.array(group.SensorDepletionRaw.read(),
                                                        dtype=group.SensorDepletionRaw.dtype)
                            assert isinstance(result_table, np.recarray)
                            if result_table.shape[0] > 0:
                                current_entry = (result_table.Ubi_corrected[0],
                                                                    result_table.Ubi_corrected_error[0],
                                                                    result_table.Ubi_systematic_corrected[0],
                                                                    result_table.Ubi_systematic_dispersion_corrected_error[
                                                                        0])

                    current_row["depletion_voltage"] = current_entry

                current_row['sensor'] = entry.sensor
                current_row.append()

            table.flush()
            table.cols.sensor.create_csindex()
            table.flush()


    group_provider_dtype = np.dtype([('file', str, 220), ('unbiased_group', str, 100),
                                     ('unbiased_inter_pix_group', str, 100), ('biased_group', str, 100),
                                     ('biased_inter_pix_group', str, 100), ('cv_group', str, 100),
                                     ('sensor', str, 100)])
    extraction_files = [
        X1_SCAN_2_FILE,
        R13_2_SCAN_FILE,
        X5_SCAN_FILE,
        X7_SCAN_FILE,
        X2_SCAN_2_FILE,
        R11_SCAN_FILE,
        X6_SCAN_FILE,
        E1_2_SCAN_FILE,
        E1_2_SCAN_FILE,
        E1_2_SCAN_FILE,
        E1_2_SCAN_FILE,
        E1_2_SCAN_FILE,
        E1_2_SCAN_FILE,
        E1_2_SCAN_FILE,
        E1_2_SCAN_FILE,
    ]
    extract_unbiased_full_groups = [
        "Thesis/ATLAS_ITk/X1/unbiased_61_full_model/total_cap/analysis_correction",
        "Reference/R13/unbiased_1_full_model/total_cap/analysis_correction",
        "Thesis/ATLAS_ITk/X5/unbiased_full_model/total_cap/analysis_correction",
        "Thesis/ATLAS_ITk/X7/unbiased_full_model/total_cap/analysis_correction",
        "Thesis/ATLAS_ITk/X2/unbiased_1_full_model/total_cap/analysis_correction",
        "Reference/R1/unbiased_full_model/total_cap/analysis_correction",
        "Thesis/ATLAS_ITk/X6/unbiased_full_model/total_cap/analysis_correction",
        "Reference/E1/unbiased_full_model_nw15_50/total_cap/analysis_correction",
        "Reference/E1/unbiased_full_model_nw20_50/total_cap/analysis_correction",
        "Reference/E1/unbiased_full_model_nw25_50/total_cap/analysis_correction",
        "Reference/E1/unbiased_full_model_nw30_50/total_cap/analysis_correction",
        "Reference/E1/unbiased_full_model_dnw15_50/total_cap/analysis_correction",
        "Reference/E1/unbiased_full_model_dnw20_50/total_cap/analysis_correction",
        "Reference/E1/unbiased_full_model_dnw25_50/total_cap/analysis_correction",
        "Reference/E1/unbiased_full_model_dnw30_50/total_cap/analysis_correction",
    ]
    extract_biased_full_groups = [
        "Thesis/ATLAS_ITk/X1/biased_80_V_full_model/total_cap/analysis_correction",
        "Reference/R13/biased_80_V_full_model/total_cap/analysis_correction",
        "Thesis/ATLAS_ITk/X5/biased_40_V_full_model/total_cap/analysis_correction",
        "Thesis/ATLAS_ITk/X7/biased_40.0_V_full_model/total_cap/analysis_correction",
        "Thesis/ATLAS_ITk/X2/biased_80_V_full_model/total_cap/analysis_correction",
        "Reference/R1/biased_80_V_full_model/total_cap/analysis_correction",
        "Thesis/ATLAS_ITk/X6/biased_45_V_full_model/total_cap/analysis_correction",
        "Reference/E1/biased_80_V_full_model_nw15_50/total_cap/analysis_correction",
        "Reference/E1/biased_80_V_full_model_nw20_50/total_cap/analysis_correction",
        "Reference/E1/biased_80_V_full_model_nw25_50/total_cap/analysis_correction",
        "Reference/E1/biased_80_V_full_model_nw30_50/total_cap/analysis_correction",
        "Reference/E1/biased_80_V_full_model_dnw15_50/total_cap/analysis_correction",
        "Reference/E1/biased_80_V_full_model_dnw20_50/total_cap/analysis_correction",
        "Reference/E1/biased_80_V_full_model_dnw25_50/total_cap/analysis_correction",
        "Reference/E1/biased_80_V_full_model_dnw30_50/total_cap/analysis_correction",
    ]
    extract_unbiased_inter_groups = [
        "Thesis/ATLAS_ITk/X1/inter_unbiased_full/inter_cap/analysis",
        # "Reference/R13/inter_unbiased_full/inter_cap/analysis",
        None,
        "Thesis/ATLAS_ITk/X5/inter_unbiased_full/inter_cap/analysis",
        "Thesis/ATLAS_ITk/X7/inter_unbiased_full/inter_cap/analysis",
        # "Thesis/ATLASK_ITk/X2/inter_unbiased_full/inter_cap/analysis",
        None,
        "Reference/R1/inter_unbiased_full/inter_cap/analysis",
        "Thesis/ATLAS_ITk/X6/inter_unbiased_full/inter_cap/analysis",
        "Reference/E1/inter_unbiased_full_nw15_50/inter_cap/analysis",
        "Reference/E1/inter_unbiased_full_nw20_50/inter_cap/analysis",
        "Reference/E1/inter_unbiased_full_nw25_50/inter_cap/analysis",
        "Reference/E1/inter_unbiased_full_nw30_50/inter_cap/analysis",
        "Reference/E1/inter_unbiased_full_dnw15_50/inter_cap/analysis",
        "Reference/E1/inter_unbiased_full_dnw20_50/inter_cap/analysis",
        "Reference/E1/inter_unbiased_full_dnw25_50/inter_cap/analysis",
        "Reference/E1/inter_unbiased_full_dnw30_50/inter_cap/analysis",
    ]
    extract_biased_inter_groups = [
        "Thesis/ATLAS_ITk/X1/inter_biased_M_80_V_full/inter_cap/analysis",
        # "Reference/R13/inter_biased_M_80_V_full/inter_cap/analysis",
        None,
        "Thesis/ATLAS_ITk/X5/inter_biased_M_40_V_full/inter_cap/analysis",
        "Thesis/ATLAS_ITk/X7/inter_biased_M_40.0_V_full/inter_cap/analysis",
        # "Thesis/ATLASK_ITk/X2/inter_unbiased_full/inter_cap/analysis",
        None,
        "Reference/R1/inter_biased_M_80_V_full/inter_cap/analysis",
        "Thesis/ATLAS_ITk/X6/inter_biased_M_45_V_full/inter_cap/analysis",
        "Reference/E1/inter_biased_M_80_V_full_nw15_50/inter_cap/analysis",
        "Reference/E1/inter_biased_M_80_V_full_nw20_50/inter_cap/analysis",
        "Reference/E1/inter_biased_M_80_V_full_nw25_50/inter_cap/analysis",
        "Reference/E1/inter_biased_M_80_V_full_nw30_50/inter_cap/analysis",
        "Reference/E1/inter_biased_M_80_V_full_dnw15_50/inter_cap/analysis",
        "Reference/E1/inter_biased_M_80_V_full_dnw20_50/inter_cap/analysis",
        "Reference/E1/inter_biased_M_80_V_full_dnw25_50/inter_cap/analysis",
        "Reference/E1/inter_biased_M_80_V_full_dnw30_50/inter_cap/analysis",
    ]
    extract_cv_groups = [
        "Thesis/ATLAS_ITk/X1/C_V_Characteristic_refined_Extended_Combined/biasing/analysis_correction",
        "Reference/R13/C_V_Characteristic_refined/biasing/analysis_correction",
        "Thesis/ATLAS_ITk/X5/C_V_Characteristic_refined_Extended/biasing/analysis_correction",
        "Thesis/ATLAS_ITk/X7/C_V_Characteristic_refined/biasing/analysis_correction",
        "Thesis/ATLAS_ITk/X2/C_V_Characteristic_refined_extended_renew_retry/biasing/analysis_correction",
        "Reference/R1/C_V_Characteristic_refined/biasing/analysis_correction",
        "Thesis/ATLAS_ITk/X6/C_V_Characteristic_refined/biasing/analysis_correction",
        "Reference/E1/C_V_Characteristic_refined_nw15_50/biasing/analysis_correction",
        "Reference/E1/C_V_Characteristic_refined_nw20_50/biasing/analysis_correction",
        "Reference/E1/C_V_Characteristic_refined_nw25_50/biasing/analysis_correction",
        "Reference/E1/C_V_Characteristic_refined_nw30_50/biasing/analysis_correction",
        "Reference/E1/C_V_Characteristic_refined_dnw15_50/biasing/analysis_correction",
        "Reference/E1/C_V_Characteristic_refined_dnw20_50/biasing/analysis_correction",
        "Reference/E1/C_V_Characteristic_refined_dnw25_50/biasing/analysis_correction",
        "Reference/E1/C_V_Characteristic_refined_dnw30_50/biasing/analysis_correction",
    ]
    extract_sensors = [
        "X1",
        "R13",
        "X5",
        "X7",
        "X2",
        "R1",
        "X6",
        "E1_nw15_50",
        "E1_nw20_50",
        "E1_nw25_50",
        "E1_nw30_50",
        "E1_dnw15_50",
        "E1_dnw20_50",
        "E1_dnw25_50",
        "E1_dnw30_50",
    ]

    records = []
    print(np.asarray(extraction_files).dtype)
    for record in zip(
            extraction_files,
            extract_unbiased_full_groups,
            extract_unbiased_inter_groups,
            extract_biased_full_groups,
            extract_biased_inter_groups,
            extract_cv_groups,
            extract_sensors,
    ):
        records.append(record)

    rec_arrays = np.rec.array(records, dtype=group_provider_dtype)
    print(rec_arrays)
    generate_ba_summary(rec_arrays, SUMMARY_FILE)

    # we need to generate plots for all the different dependencies of sensors
    # but this implies to first
    sensor_primary_properties_type = np.dtype([
        ("sensor", 'S16'),
        ("pitch_x", np.float64),
        ("pitch_y", np.float64),
        ("implantation_size_x", np.float64),
        ("implantation_size_y", np.float64),
        ("implantation_depth", np.float64),
        ("sensor_depth", np.float64),
    ])
    sensor_properties_type = np.dtype([
        ("sensor", 'S16'),
        ("pitch_x", np.float64),
        ("pitch_y", np.float64),
        ("implantation_size_x", np.float64),
        ("implantation_size_y", np.float64),
        ("pixel_area", np.float64),
        ("implantation_area", np.float64),
        ("pixel_separation_x", np.float64),
        ("pixel_separation_y", np.float64),
        ("pixel_separation_area", np.float64),
        ("implantation_depth", np.float64),
        ("sensor_depth", np.float64),
        ("Perimeter", np.float64),
    ])

    sensor_primary_properties = np.rec.array([
        ("R1", 25, 100, 8, 81, 1, 150),
        ("R13", 50, 50, 30, 30, 1, 150),
        ("X1", 50, 50, 50, 50, 4, 150),
        ("X2", 50, 50, 50, 50, 4, 150),
        # ("X5", 50, 50, 50, 50, 150, 250),
        # ("X6", 50, 50, 50, 50, 150, 250),
        # ("X7", 50, 50, 50, 50, 150, 250),
        ("X5", 50, 50, 130, 10, 25, 150),
        ("X6", 50, 50, 130, 10, 25, 150),
        ("X7", 50, 50, 130, 10, 25, 150),
        ("E1_nw15_50", 50, 50, 15, 15, 1, 100),
        ("E1_nw20_50", 50, 50, 20, 20, 1, 100),
        ("E1_nw25_50", 50, 50, 25, 25, 1, 100),
        ("E1_nw30_50", 50, 50, 30, 30, 1, 100),
        ("E1_dnw15_50", 50, 50, 15, 15, 2, 100),
        ("E1_dnw20_50", 50, 50, 20, 20, 2, 100),
        ("E1_dnw25_50", 50, 50, 25, 25, 2, 100),
        ("E1_dnw30_50", 50, 50, 30, 30, 2, 100),
    ], dtype=sensor_primary_properties_type)

    sensors = sensor_primary_properties.sensor
    sensor_pitches_x = sensor_primary_properties.pitch_x
    sensor_pitches_y = sensor_primary_properties.pitch_y
    sensor_pixel_areas = sensor_pitches_y * sensor_pitches_x
    sensor_implant_sizes_x = sensor_primary_properties.implantation_size_x
    sensor_implant_sizes_y = sensor_primary_properties.implantation_size_y
    sensor_implant_areas = sensor_implant_sizes_y * sensor_implant_sizes_x
    sensor_pixel_separations_x = sensor_pitches_x - sensor_implant_sizes_x
    sensor_pixel_separations_y = sensor_pitches_y - sensor_implant_sizes_y
    sensor_pixel_separation_areas = sensor_pixel_areas - sensor_implant_areas

    zipping = zip(
        sensors,
        sensor_pitches_x,
        sensor_pitches_y,
        sensor_implant_sizes_x,
        sensor_implant_sizes_y,
        sensor_pixel_areas,
        sensor_implant_areas,
        sensor_pixel_separations_x,
        sensor_pixel_separations_y,
        sensor_pixel_separation_areas,
        sensor_primary_properties.implantation_depth,
        sensor_primary_properties.sensor_depth,
        2 * (sensor_primary_properties.implantation_size_x + sensor_primary_properties.implantation_size_y),
    )

    sensor_properties = np.rec.array([item for item in zipping], dtype=sensor_properties_type)

    # this will yield wrong results for the 3d sensors pixel separations
    for sensor_record in sensor_properties:
        assert isinstance(sensor_record, np.record)
        assert sensor_record.dtype == sensor_properties_type
        if sensor_record.sensor not in spatial_identifier:
            continue


        sensor_record.implantation_area = sensor_record.implantation_size_x * sensor_record.implantation_size_y * np.pi
        sensor_record.pixel_separation_x = sensor_record.pixel_separation_x = np.mean((sensor_record.pitch_x, sensor_record.pitch_y)) - sensor_record.implantation_size_y
        sensor_record.pixel_separation_area = sensor_record.pixel_area - np.pi * (sensor_record.implantation_size_y / 2) ** 2




    with tb.open_file(SUMMARY_FILE, mode='a') as h5_conslusion:
        if "SensorTypes" in h5_conslusion.root:
            h5_conslusion.root.SensorTypes.remove()
        table = h5_conslusion.create_table(where=h5_conslusion.root, name="SensorTypes", description=sensor_properties)
        table.flush()
        table.cols.sensor.create_csindex()
        table.flush()
