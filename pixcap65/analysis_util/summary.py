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
Analysis module to summarize the main/major results of the analysis for multiple sensors (at least one sensor) into a
single file, such that these results could be easily post processed.
"""
from os import PathLike

import logging
import numpy as np
import tables as tb
import time

try:
    # noinspection PyCompatibility
    from collections.abc import Iterable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Iterable

from examples.data_constants import R11_SCAN_FILE, R13_2_SCAN_FILE, E1_2_SCAN_FILE, X4_SCAN_FILE
from examples.data_constants import X1_SCAN_2_FILE, X2_SCAN_2_FILE
from examples.data_constants import X5_SCAN_FILE, X6_SCAN_FILE, X7_SCAN_FILE
from pixcap65.analysis import get_test_capacitance_data
from pixcap65.utility import synchronized_process_open_file
from pixcap65.utility.utils_2 import walk_to_node

SUMMARY_FILE = 'conclude_summary.h5'


logger = logging.getLogger(__name__)
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
test_design_values = {name.replace(" ", "_"): field_design_values[name] for name in [
    "0 w_o bump", "1 w_o bump", "2 w_o bump", "3 w_o bump", "4 w_o bump", "5",
    "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "17",
    "18", "19", "20", "21", "22", "23", "24", "25", "26", "27 w_ bump",
    "28 w_ bump", "29 w_ bump", "30 w_ bump", "31 w_ bump", "32 w_ bump", "33 w_ bump",
    "34 w_ bump", "35 w_o bump", "36 w_o bump", "37 w_o bump", "38 w_o bump", "39 w_o bump"]}
spatial_identifier = ["X{}".format(i) for i in range(3, 9)]


# but how to format such a table (to store a summary of all our measurement results)?
# we will need a master table 'to do' so.
# CHECK: should this here be moved to our general data structure definitions?
class MasterTableResultEntry(tb.IsDescription):
    """
    General structure of result entry in the summary table for all sensors.
    This is a general form of an entry consisting of the physical magnitude, a statistical uncertainty and two
    systematic uncertainty to tread the dispersion effects between different pixcap65 chips separately.

    The following class variable are to be understand as fields/entries of the table:
    :cvar magnitude: magnitude of the stored physical quantity.
    :cvar stat_error: statistical uncertainty of the stored physical quantity.
    :cvar systematic_general: systematic uncertainty of the physical quantity (general one: systematic effects from
        theory uncertainty, modelling, systematic of the analysis, failure to exactly reproduce conditions).
    :cvar systematic_dispersion: systematic uncertainty of the physical quantity by dispersion effects of systematical
        offsets between different setups/module/chips.

    The full data structure is given by :py:class:`pixcap65.analysis_util.summary.MasterExtractionTable`.
    """
    magnitude = tb.Float64Col(pos=0)
    stat_error = tb.Float64Col(pos=1)
    systematic_general = tb.Float64Col(pos=2)
    systematic_dispersion = tb.Float64Col(pos=3)


class MasterExtractionTable(tb.IsDescription):
    """
    Data structure of the general summary table for all sensors.
    It maps the name of a sensor against the individual averaged results like total-pixel capacitance,
    inter-pixel capacitance or depletion voltage of the sensor.

    The following class variables are to be understand as fields/entries of the table:
    :cvar sensor: name of the sensor for which this table row will summarize results.
    :cvar unbiased_capacitance: sensor average of the measured total-pixel capacitance when the sensor is not biased
        at all.
    :cvar unbiased_inter_capacitance: sensor average of the measured inter-pixel capacitance when the sensor is not
        biased at all. The single contributions are all summed up for the eight neighbouring pixels here.
    :cvar biased_capacitance: sensor average of the measured total-pixel capacitance when the sensor is biased.
    :cvar bias_voltage: applied reversed bias voltage to perform the biased measurement. This is not necessisarily
        the depletion voltage.
    :cvar biased_inter_capacitance: sensor average of the measured inter-pixel capacitance when the sensor is
        biased. The single contributions are all summed up for the eight neighbouring pixels here.
    :cvar biased_inter_capacitance_side: sensor average of the measured inter-pixel capacitance when the sensor is
        biased. The single contributions from the neighbouring pixels lighing to the sides are summed up here, all
        other neighbors are ignored.
    :cvar biased_inter_capacitance_top: sensor average of the measured inter-pixel capacitance when the sensor is
        biased. The single contributions from the neighbouring pixels lighing to the top and bottom are summed up
        here, all other neighbors are ignored.
    :cvar biased_back_capacitance: sensor average for the estimation of the in-pixel or backplane capacitance.
        The sensor is (completely) biased for this measurement.
    :cvar depletion_voltage: determined depletion voltage for this sensor. For some sensors it might not be the full
    depletion voltage but the surface depletion voltage.

    The individual result entries follow the scheme given by
    :py:class:`pixcap65.analysis_util.summary.MasterTableResultEntry`.
    """
    sensor = tb.StringCol(pos=0, itemsize=16)
    unbiased_capacitance = MasterTableResultEntry()
    unbiased_inter_capacitance = MasterTableResultEntry()
    biased_capacitance = MasterTableResultEntry()
    bias_voltage = tb.Float64Col(pos=1)
    biased_inter_capacitance = MasterTableResultEntry()
    biased_inter_capacitance_side = MasterTableResultEntry()
    biased_inter_capacitance_top = MasterTableResultEntry()
    biased_back_capacitance = MasterTableResultEntry()
    depletion_voltage = MasterTableResultEntry()


class GroupProviderScheme(tb.IsDescription):
    """
    Structure of how to provide where the data which should be summarised could be found.
    It contains one file per sensor and the path of the corresponding hdf hierarchy groups for the different entries
    within this file.

    Class variables are to be understand as the fields/entries of the tabel.

    :cvar file: path to the file on disk where the measurements for this particular sensor are stored.
    :cvar unbiased_group: path to hdf files group storing the measurements for the unbiased total-pixel capacitance
        estimation.
    :cvar unbiased_inter_pix_group: path to the hdf files group storing the measurements for the inter-pixel capacitance
        estimation of the unbiased sensor.
    :cvar biased_group: path to hdf files group storing the measurements for the biased total-pixel capacitance
        estimation.
    :cvar biased_inter_pix_group: path to the hdf files group storing the measurements for the inter-pixel capacitance
        estimation of the biased sensor.
    :cvar cv_group: path to the hdf files group storing the data for the cv characterization of the sensor.
    :cvar biased_inter_pix_group_sides: path to the hdf files group storing the measurements for the inter-pixel capacitance
        estimation of the biased sensor. Here, only the neighbours to the sides are considered.
    :cvar biased_inter_pix_group_tops: path to the hdf files group storing the measurements for the inter-pixel capacitance
        estimation of the biased sensor. Here only the neighbours above and below the measured pixel are considered.
    """
    file = tb.StringCol(itemsize=220, pos=0)
    unbiased_group = tb.StringCol(itemsize=100, pos=1)
    unbiased_inter_pix_group = tb.StringCol(itemsize=100, pos=2)
    biased_group = tb.StringCol(itemsize=100, pos=3)
    biased_inter_pix_group = tb.StringCol(itemsize=100, pos=4)
    cv_group = tb.StringCol(itemsize=100, pos=5)
    biased_inter_pix_group_sides = tb.StringCol(itemsize=100, pos=6)
    biased_inter_pix_group_tops = tb.StringCol(itemsize=100, pos=7)


# generate the table with summarieses all the data!
DEFAULT_ENTRY = (np.nan, np.nan, np.nan, np.nan)


# perhaps we should not nest function declarations?
def _generate_cap_entry(res_table):
    return (res_table.cap_corrected[0],
            res_table.cap_corrected_err[0],
            res_table.cap_systematic_error[0],
            res_table.cap_systematic_dispersion[0])


def _handle_group_input(record, h5: tb.File, row, key, bias_filter=None, require_correction=False):
    keys = np.atleast_1d(key)
    filters = np.atleast_1d(bias_filter)
    if record is not None and 'None' not in record:
        group = h5._get_or_create_path("/" + record, False)
        current_entry = DEFAULT_ENTRY

        for (tab_key, filter) in zip(keys, filters):
            if "DistResultfF" in group:
                if filter is None:
                    reading = group.DistResultfF.read()
                else:
                    reading = group.DistResultfF.read_where("""(bias == {})""".format(filter))
                result_table = np.rec.array(reading, dtype=group.DistResultfF.dtype)
                assert isinstance(result_table, np.recarray)
                correct_implant = 2 if require_correction else 1
                if result_table.shape[0] == 0 and filter is not None and require_correction:
                    result_table = np.rec.array(
                        group.DistResultfF.read_where("""(bias == {})""".format(14000)),
                        dtype=group.DistResultfF.dtype)
                    correct_implant = 8
                if result_table.shape[0] > 0:
                    if require_correction:
                        current_entry = tuple(
                            np.asarray(_generate_cap_entry(result_table), dtype=np.float64) / correct_implant)
                    else:
                        current_entry = _generate_cap_entry(result_table)

            logger.warning("Used the column %s and the filter %s for sensor %s and data %s:", tab_key, filter, row["sensor"], current_entry)
            row[tab_key] = current_entry
            current_entry = DEFAULT_ENTRY
    else:
        row[keys[-1]] = DEFAULT_ENTRY


def _handle_table_entry(entry: np.record, table_row: tb.tableextension.Row):
    with tb.open_file(str(entry.file)) as h5_file:
        current_entry = DEFAULT_ENTRY
        _handle_group_input(entry.unbiased_group, h5_file, table_row,'unbiased_capacitance')
        _handle_group_input(entry.biased_group, h5_file, table_row, 'biased_capacitance')
        _handle_group_input(entry.unbiased_inter_pix_group, h5_file, table_row, "unbiased_inter_capacitance", 14000)
        _handle_group_input(entry.biased_inter_pix_group, h5_file, table_row, [
            "biased_back_capacitance",
            "biased_inter_capacitance"
        ], [
            13000,
            14000
        ])

        # the implantation correction s
        _handle_group_input(entry.biased_inter_pix_group_sides, h5_file, table_row, "biased_inter_capacitance_side", 20000, True)
        _handle_group_input(entry.biased_inter_pix_group_tops, h5_file, table_row, "biased_inter_capacitance_top",
                            19000, True)

        # handle the C-V-Parameterisation
        # this works a bit different so it has its own single-use implementation
        if entry.cv_group is not None and 'None' not in entry.cv_group:
            group = h5_file._get_or_create_path("/" + entry.cv_group, False)
            if "SensorDepletionRaw" in group:
                result_table = np.rec.array(group.SensorDepletionRaw.read(),
                                            dtype=group.SensorDepletionRaw.dtype)
                assert isinstance(result_table, np.recarray)
                if result_table.shape[0] > 0:
                    idx = np.argmin(result_table.Ubi_corrected)
                    current_entry = (result_table.Ubi_corrected[idx],
                                     result_table.Ubi_corrected_error[idx],
                                     result_table.Ubi_systematic_corrected[idx],
                                     result_table.Ubi_systematic_dispersion_corrected_error[idx])

        table_row["depletion_voltage"] = current_entry


def generate_ba_summary(extraction: np.recarray, target_file):
    """
    Generate a summary table for the sensors specified by the `extraction` parameter.
    The newly created table will follow the structure given by
    :class:`pixcap65.analysis_util.summary.MasterExtractionTable` and will have the name `GeneralSummaryTable`.
    If such a table already exists in the `target_file`, it will be removed/deleted first.
    :param extraction:
    :param target_file: .h5-file (path to it) where the summary table should be saved.
    :return:
    """
    # here we need to access the distribution tables from the data sets
    # for the inter-pix: Which of the different distribution tables should be used in the end?
    # nevertheless we need to iterate the recarray
    with tb.open_file(target_file, 'a') as out_file:
        if "GeneralSummaryTable" in out_file.root:
            out_file.root.GeneralSummaryTable.remove()
        table = out_file.create_table(where=out_file.root, name="GeneralSummaryTable",
                                      description=MasterExtractionTable)
        current_row = table.row
        for entry in extraction:
            assert isinstance(entry, np.record)
            _handle_table_entry(entry, current_row)

            current_row['sensor'] = entry.sensor
            current_row['bias_voltage'] = entry.biasing
            current_row.append()

        table.flush()
        table.cols.sensor.create_csindex()
        table.flush()


def generate_test_summary(files: Iterable[PathLike], groups: Iterable[PathLike], sensors: Iterable[str],
                          summary_file: PathLike) -> None:
    """
    Collects the test capacitances from the different sensors and writes them into a single table, such that these could
    easily compared by a human.
    :param files: hdf-file from which to fetch the test capacitances
    :param groups: hdf files hierarchy groups where the measurements of the test capacitances are stored.
    :param sensors: iterable of sensor names such that the test capacitances could be mapped to a particular sensor.
    :param summary_file: path to the hdf (.h5) file where to store the summarized data of the test capacitances.
    """
    with synchronized_process_open_file(summary_file, mode='a') as summary_file:
        table_description = [("Sensor", 'S16'), ]
        for name in field_names:
            table_description.extend([("test_{}".format(name), np.float64), ])
            table_description.extend([("test_{}_error".format(name), np.float64), ])

        if "TestCap" in summary_file.root:
            summary_file.root.TestCap.remove()
            time.sleep(10)

        table_type = np.dtype(table_description)
        table = summary_file.create_table(where=summary_file.root, name="TestCap",
                                          title="Test Capacitances from the different sensors",
                                          description=table_type)
        for file, group_path, sensor in zip(files, groups, sensors):
            with synchronized_process_open_file(file, mode='r') as h5_file:
                group, _ = walk_to_node(h5_file.root, group_path, create=False, verify_create=True)
                test_cap, test_cap_err = get_test_capacitance_data(group, print_result=False)
                assert isinstance(sensor, str)
                result_data = [sensor.encode()]
                for cap, cap_err in zip(test_cap, test_cap_err):
                    result_data.append(cap)
                    result_data.append(cap_err)
                table.append([tuple(result_data), ])
        table.flush()
        table.cols.Sensor.create_csindex()
        table.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    summary_files = ["packaged/data/Bare_Sample_05_Extended_Scan.h5", R13_2_SCAN_FILE, R11_SCAN_FILE, X1_SCAN_2_FILE,
                     X2_SCAN_2_FILE, X5_SCAN_FILE, X6_SCAN_FILE,
                     X7_SCAN_FILE, E1_2_SCAN_FILE, X4_SCAN_FILE]
    summary_groups = [
        "Reference/Bare/unbiased_full_model/total_cap/analysis",
        "Reference/R13/unbiased_1_full_model/total_cap/analysis",
        "Reference/R1/unbiased_full_model/total_cap/analysis",
        "Thesis/ATLAS_ITk/X1/unbiased_61_full_model/total_cap/analysis",
        "Thesis/ATLAS_ITk/X2/unbiased_1_full_model/total_cap/analysis",
        "Thesis/ATLAS_ITk/X5/unbiased_full_model/total_cap/analysis",
        "Thesis/ATLAS_ITk/X6/unbiased_full_model/total_cap/analysis",
        "Thesis/ATLAS_ITk/X7/unbiased_full_model/total_cap/analysis",
        "Reference/E1/unbiased_full_model/total_cap/analysis",
        "Thesis/ATLAS_ITk/X4/unbiased_full_model/total_cap/analysis"
    ]
    summary_sensors = ["Bare", "R13", "R1", "X1", "X2", "X5", "X6", "X7", "E1", "X4"]

    logger.info("generate summary")
    generate_test_summary(summary_files, summary_groups, summary_sensors, SUMMARY_FILE)
    logger.info("Generated test.")

    with tb.open_file(R11_SCAN_FILE) as h5_file:
        get_test_capacitance_data(h5_file.root.Reference.R1.unbiased_full.total_cap.analysis)


    group_provider_dtype = np.dtype([('file', str, 220), ('unbiased_group', str, 100),
                                     ('unbiased_inter_pix_group', str, 100), ('biased_group', str, 100),
                                     ('biased_inter_pix_group', str, 100), ('cv_group', str, 100),
                                     ('sensor', str, 100), ('biasing', np.float64),
                                     ('biased_inter_pix_group_tops', str, 100),
                                     ('biased_inter_pix_group_sides', str, 100), ])
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
        X4_SCAN_FILE,
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
        "Thesis/ATLAS_ITk/X4/unbiased_full_model/total_cap/analysis_correction",
    ]
    extract_biased_full_groups = [
        "Thesis/ATLAS_ITk/X1/biased_80_V_full/total_cap/analysis_correction",
        "Reference/R13/biased_80_V_full_model/total_cap/analysis_correction",
        "Thesis/ATLAS_ITk/X5/biased_40_V_full_model/total_cap/analysis_correction",
        "Thesis/ATLAS_ITk/X7/biased_40.0_V_full_model/total_cap/analysis_correction",
        "Thesis/ATLAS_ITk/X2/biased_80_V_full/total_cap/analysis_correction",
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
        "Thesis/ATLAS_ITk/X4/biased_80_V_full_model/total_cap/analysis_correction",
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
        "Thesis/ATLAS_ITk/X4/inter_unbiased_full/inter_cap/analysis",
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
        "Thesis/ATLAS_ITk/X4/inter_biased_M_80_V_full/inter_cap/analysis",
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
        "Thesis/ATLAS_ITk/X4/C_V_Characteristic_refined_Extended/biasing/analysis_correction",
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
        "X4"
    ]
    extract_bias_voltages = [
        80,
        80,
        40,
        40,
        80,
        80,
        45,
        80,
        80,
        80,
        80,
        80,
        80,
        80,
        80,
        80,
    ]
    extract_biased_inter_groups_top = [
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
        "Thesis/ATLAS_ITk/X4/inter_biased_M_80_V_full/inter_cap/analysis",
    ]
    extract_biased_inter_groups_side = [
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
        "Thesis/ATLAS_ITk/X4/inter_biased_M_80_V_full/inter_cap/analysis",
    ]

    records = []
    logger.debug(np.asarray(extraction_files).dtype)
    for record in zip(
            extraction_files,
            extract_unbiased_full_groups,
            extract_unbiased_inter_groups,
            extract_biased_full_groups,
            extract_biased_inter_groups,
            extract_cv_groups,
            extract_sensors,
            extract_bias_voltages,
            extract_biased_inter_groups_top,
            extract_biased_inter_groups_side,
    ):
        records.append(record)

    rec_arrays = np.rec.array(records, dtype=group_provider_dtype)
    logger.debug(rec_arrays)
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
        ("R1", 25, 100, 6, 81, 3, 150),
        ("R13", 50, 50, 30, 30, 3, 150),
        ("X1", 50, 50, 45, 45, 3, 150),
        ("X2", 50, 50, 45, 45, 3, 150),
        # ("X5", 50, 50, 50, 50, 150, 250),
        # ("X6", 50, 50, 50, 50, 150, 250),
        # ("X7", 50, 50, 50, 50, 150, 250),
        ("X4", 50, 50, 130, 5, 25, 150),
        ("X5", 50, 50, 130, 5, 25, 150),
        ("X6", 50, 50, 130, 5, 25, 150),
        ("X7", 50, 50, 130, 5, 25, 150),
        ("E1_nw15_50", 50, 50, 15, 15, 3, 100),
        ("E1_nw20_50", 50, 50, 20, 20, 3, 100),
        ("E1_nw25_50", 50, 50, 25, 25, 3, 100),
        ("E1_nw30_50", 50, 50, 30, 30, 3, 100),
        ("E1_dnw15_50", 50, 50, 15, 15, 5, 100),
        ("E1_dnw20_50", 50, 50, 20, 20, 5, 100),
        ("E1_dnw25_50", 50, 50, 25, 25, 5, 100),
        ("E1_dnw30_50", 50, 50, 30, 30, 5, 100),
    ], dtype=sensor_primary_properties_type)

    sensors = sensor_primary_properties.sensor
    sensor_pitches_x = sensor_primary_properties.pitch_x
    sensor_pitches_y = sensor_primary_properties.pitch_y
    sensor_pixel_areas = sensor_pitches_y * sensor_pitches_x
    sensor_implant_sizes_x = sensor_primary_properties.implantation_size_x
    sensor_implant_sizes_y = sensor_primary_properties.implantation_size_y
    sensor_implant_areas = sensor_implant_sizes_y * sensor_implant_sizes_x
    sensor_pixel_separations_x = (sensor_pitches_x - sensor_implant_sizes_x - 4) / 2
    sensor_pixel_separations_y = (sensor_pitches_y - sensor_implant_sizes_y - 4) / 2
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
    for k, sensor_record in enumerate(sensor_properties):
        assert isinstance(sensor_record, np.record)
        assert sensor_record.dtype == sensor_properties_type
        if sensor_record.sensor.decode() not in spatial_identifier:
            continue

        logger.info("Cross check the spatial parameters!")
        logger.info(np.mean((sensor_record.pitch_x, sensor_record.pitch_y)))
        logger.info(sensor_record.implantation_size_x)
        logger.info(sensor_record.implantation_size_y)
        sensor_properties.implantation_area[
            k] = sensor_record.implantation_size_x * sensor_record.implantation_size_y * np.pi
        sensor_properties.pixel_separation_x[k] = np.mean(
            (sensor_record.pitch_x, sensor_record.pitch_y)) - sensor_record.implantation_size_y
        sensor_properties.pixel_separation_y[k] = np.mean(
            (sensor_record.pitch_x, sensor_record.pitch_y)) - sensor_record.implantation_size_y
        sensor_properties.pixel_separation_area[k] = sensor_record.pixel_area - np.pi * (
                sensor_record.implantation_size_y / 2) ** 2

    with tb.open_file(SUMMARY_FILE, mode='a') as h5_conclusion:
        if "SensorTypes" in h5_conclusion.root:
            h5_conclusion.root.SensorTypes.remove()
        table = h5_conclusion.create_table(where=h5_conclusion.root, name="SensorTypes", description=sensor_properties)
        table.flush()
        table.cols.sensor.create_csindex()
        table.flush()

    # correct all the test capacitance's for the parasitic capacitance of the measurement circuit!
    from examples.plot_dependencies import read_rec_array
    from pixcap65.pixcap.pixcap_structure import CAPACITANCE_CONVERSION_FACTOR

    with tb.open_file(SUMMARY_FILE, mode='a') as summary_file:
        test_data = read_rec_array(summary_file.root.TestCap)
        test_cap_intrinsic_names = ["test_{}_w__bump".format(it) for it in range(27, 35)]
        test_cap_intrinsic_error_names = ["test_{}_w__bump".format(it) for it in range(27, 35)]
        test_cap_intrinsic = np.concatenate([test_data[name] for name in test_cap_intrinsic_names])
        test_cap_intrinsic_error = np.concatenate([test_data[name] for name in test_cap_intrinsic_error_names])

        # make all this more accurately by using a fit and also determine the remaining capacitance's
        parasitic_mean_keep = np.average(test_cap_intrinsic, keepdims=True, weights=np.reciprocal(
            test_cap_intrinsic_error) ** 2) * CAPACITANCE_CONVERSION_FACTOR
        parasitic_mean = parasitic_mean_keep[0]
        parasitic_std = np.std(test_cap_intrinsic * CAPACITANCE_CONVERSION_FACTOR, mean=parasitic_mean_keep)
        logger.info("Some information from direct parasitic capacitance investigation.")
        logger.info("%f+-%f",parasitic_mean, parasitic_std)
        logger.info(14.901 - parasitic_mean)
        logger.info(np.sqrt(0.079 ** 2 + parasitic_std ** 2))

        # correct for some of the capacitances
        corrected_test_data = test_data.copy()
        assert isinstance(corrected_test_data, np.recarray)
        entry_fields = corrected_test_data.dtype.names
        assert entry_fields is not None
        for field in entry_fields:
            if field == "Sensor" or field in test_cap_intrinsic_names or field.endswith('_error'):
                continue
            logger.debug("The current field is %s.", field)
            corrected_test_data[field] = test_data[field] - parasitic_mean * 1.e-15
            corrected_test_data[field + '_error'] = np.sqrt(
                test_data[field + '_error'] ** 2 + parasitic_std * 1.e-15 ** 2)

        if "TestCapCorrected" in summary_file.root:
            summary_file.root.TestCapCorrected._f_remove()
        table = summary_file.create_table(where=summary_file.root, name='TestCapCorrected',
                                          title="Corrected Values of the test capacitances",
                                          description=corrected_test_data, )
        table.cols.Sensor.create_csindex()
        table.flush()
