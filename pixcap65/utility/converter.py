import logging
import numpy as np
import tables as tb
import time
import yaml

from pixcap65.analysis_util.utility import HIST_BIAS_MEAS_UNIT, HIST_CURRENT_MEAS_UNIT, GLOBAL_FILTERS
from pixcap65.configs.config_handler import extract_smu_voltage_error
from pixcap65.pixcap_65_test_total_cap import BiasTable, ScanConfigurationKeys
from pixcap65.plotting import X1_SCAN_2_FILE
from pixcap65.utility.tables_util import set_group_attribute, group_get_file, get_groups, list_group_attributes, \
    get_group_attribute
from pixcap65.utility.utils_2 import UNITS_ATTRIBUTE_KEY, create_carray, prevent_group_mix_up


def adjust_i_v_measurement(group, has_values=False):
    group.HistCurr.attrs[UNITS_ATTRIBUTE_KEY] = HIST_CURRENT_MEAS_UNIT
    if "HistCurrErr" in group:
        group.HistCurrErr.attrs[UNITS_ATTRIBUTE_KEY] = HIST_CURRENT_MEAS_UNIT
    else:
        print("WHY DOES NO ERROR HIST EXIST FOR THE I-V CURVE?????")
    if has_values:
        group.HistCurrValues.attrs[UNITS_ATTRIBUTE_KEY] = HIST_CURRENT_MEAS_UNIT

    set_group_attribute(group, "bias_current_unit", HIST_CURRENT_MEAS_UNIT)
    set_group_attribute(group, "bias_voltage_unit", HIST_BIAS_MEAS_UNIT)
    set_group_attribute(group, "bias_unit", HIST_BIAS_MEAS_UNIT)
    group.BiasVoltageHist.attrs[UNITS_ATTRIBUTE_KEY] = HIST_BIAS_MEAS_UNIT


def regenerate_i_v_errors(group):
    if "HistCurrValues" not in group and "HistCurr" in group:
        from pixcap65.configs.config_handler import extract_smu_current_error
        import yaml
        with open("pixcap65/configs/keithley_2410_range.yaml") as f:
            config = yaml.safe_load(f)
            data = group.HistCurr[:]
            errors = np.where(np.isfinite(data), extract_smu_current_error(config, data, 0.000001), np.nan)
            if "HistCurrErr" not in group:
                group_get_file(group).create_carray(where=group, name="HistCurrErr", obj=errors,
                                                    filters=tb.Filters(complib='blosc', fletcher32=False, complevel=5))
            else:
                group.HistCurrErr[:] = errors
    if "HistCurrValues" in group and "HistCurrErr" not in group:
        average = np.nanmean(group.HistCurrValues, axis=1, keepdims=True)
        uncertainty = np.nanstd(group.HistCurrValues, mean=average, axis=1)
        group.HistCurr[:] = average[:, 0]
        create_carray(group_get_file(group), where=group, name="HistCurrErr", obj=uncertainty,
                      filters=tb.Filters(complib='blosc', fletcher32=False, ), unit=HIST_CURRENT_MEAS_UNIT)


def regenerate_c_v_errors(group):
    regenerate_i_v_errors(group)
    for _, subgroup in get_groups(group):
        if "HistCurr" not in subgroup:
            print(subgroup)
            print("No entries?")
        else:
            regenerate_measurement_errors(subgroup)


def adjust_c_v_measurement(group, iv_values=False, cv_values=False):
    adjust_i_v_measurement(group, has_values=iv_values)

    assert isinstance(group, tb.Group)
    for _, subgroup in get_groups(group):
        set_group_attribute(subgroup, "bias_voltage_unit", HIST_BIAS_MEAS_UNIT)
        set_group_attribute(subgroup, "bias_unit", HIST_BIAS_MEAS_UNIT)
        set_group_attribute(subgroup, "freq_unit", "MHz")
        set_group_attribute(subgroup, "current_unit", HIST_CURRENT_MEAS_UNIT)
        if "HistCurr" not in subgroup:
            print(subgroup)
            print("No entries?")
        else:
            subgroup.HistCurr.attrs[UNITS_ATTRIBUTE_KEY] = HIST_CURRENT_MEAS_UNIT
            subgroup.HistCurrErr.attrs[UNITS_ATTRIBUTE_KEY] = HIST_CURRENT_MEAS_UNIT
            if cv_values:
                subgroup.HistCurrValues.attrs[UNITS_ATTRIBUTE_KEY] = HIST_CURRENT_MEAS_UNIT


def adjust_cap_measurement(group, has_values=False):
    group.HistCurr.attrs["Units"] = "A"
    if "HistCurrErr" not in group:
        from pixcap65.configs.config_handler import extract_smu_current_error
        import yaml
        with open("/configs/keithley_2602a_range.yaml") as f:
            config = yaml.safe_load(f)
            data = group.HistCurr[:]
            errors = extract_smu_current_error(config, data, 0.000001)
            group_get_file(group).create_carray(where=group, name="HistCurrErr", obj=errors,
                                                filters=tb.Filters(complib='blosc', fletcher32=False, complevel=5))
    group.HistCurrErr.attrs[UNITS_ATTRIBUTE_KEY] = HIST_CURRENT_MEAS_UNIT
    if has_values:
        group.HistCurrValues.attrs[UNITS_ATTRIBUTE_KEY] = HIST_CURRENT_MEAS_UNIT
    set_group_attribute(group, "current_unit", HIST_CURRENT_MEAS_UNIT)
    set_group_attribute(group, "freq_unit", "MHz")


def generate_pixel_dimensions(group):
    prevent_group_mix_up(group, "sensor")

    group_get_file(group).create_group(group, name="sensor")
    dimensions_array = np.full((40, 40, 2), fill_value=50)
    array = group_get_file(group).create_carray(where=group.sensor, name="PhysicalDimensions", obj=dimensions_array,
                                                filters=tb.Filters(complevel=5, complib='blosc', fletcher32=False))
    array.attrs[UNITS_ATTRIBUTE_KEY] = "um"
    array.flush()


def regenerate_measurement_errors(group):
    if "HistCurrValues" not in group:
        from pixcap65.configs.config_handler import extract_smu_current_error
        import yaml
        with open("pixcap65/configs/keithley_2602a_range.yaml") as f:
            config = yaml.safe_load(f)
            data = group.HistCurr[:]
            errors = np.where(np.isfinite(data), extract_smu_current_error(config, data, 0.000001), np.nan)
            if "HistCurrErr" not in group:
                group_get_file(group).create_carray(where=group, name="HistCurrErr", obj=errors,
                                                    filters=tb.Filters(complib='blosc', fletcher32=False, complevel=5))
            else:
                group.HistCurrErr[:] = errors

def regenerate_basi_table(group):
    assert "scan_params" in group
    out_file = group_get_file(group)
    table = out_file.create_table(group, name="BiasTable", description=BiasTable,
                                          filters=GLOBAL_FILTERS)
    entry = table.row

    # need to handle the actually measured voltages.
    try:
        internal_parameters = group.scan_parameters[:]
    except (KeyError, tb.exceptions.NoSuchNodeError):
        internal_parameters = group.scan_params[:]
    voltages = internal_parameters["hv_voltage"]
    try:
        with open("pixcap65/configs/keithley_2410_range.yaml", "r") as conf_file:
            scan_range_config = yaml.safe_load(conf_file)

        voltage_errors = extract_smu_voltage_error(scan_range_config, voltages, 1000)
    except:
        logging.warn("Could not extract voltage errors from keithley_2410_range.yaml", exc_info=True)
        voltage_errors = np.full_like(voltages, np.nan)

    scan_configuration = {}
    for key in list_group_attributes(group):
        if key.startswith("configuration_"):
            scan_configuration[key.removeprefix("configuration_")] = get_group_attribute(group, key)

    print(scan_configuration)
    scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] = -1 * np.arange(1, 350, 0.25)

    hist_bias_current = group.HistCurr[:]
    hist_bias_current_errors = group.HistCurrErr[:]
    for set_voltage, leak_current, current_error, meas_voltage, meas_voltage_error in zip(
            scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE], hist_bias_current,
            hist_bias_current_errors, voltages, voltage_errors):
        try:
            entry['Us'] = set_voltage
            entry['U'] = meas_voltage
            entry['I'] = leak_current
            entry['DI'] = current_error
            entry['DU'] = meas_voltage_error
            entry.append()
        except ValueError as e:
            logging.info("scan parameters")
            logging.info(set_voltage)
            logging.info("currents")
            logging.info(hist_bias_current.shape)
            logging.info(leak_current)
            logging.info("Errors")
            logging.info(hist_bias_current_errors.shape)
            logging.info(current_error)
            raise e


# FIXME: Why are there no error estimations for I-V curves?

if __name__ == "__main__":
    # with tb.open_file("packaged/3D_Sensor_H23_S24_Scan.h5", "a") as h5_file:
    #     generate_bias_table(h5_file.root.Thesis.ATLAS_ITk.X7.I_V_Characteristic.biasing.measurements)
    #     generate_bias_table(h5_file.root.Thesis.ATLAS_ITk.X7.C_V_Characteristic.biasing.measurements)
    #
    # with tb.open_file("packaged/3D_Sensor_I14_S24_Scan.h5", "a") as h5_file:
    #     generate_bias_table(h5_file.root.Thesis.ATLAS_ITk.X6.I_V_Characteristic.biasing.measurements)
    #     generate_bias_table(h5_file.root.Thesis.ATLAS_ITk.X6.C_V_Characteristic.biasing.measurements)

    with tb.open_file(X1_SCAN_2_FILE, "a") as h5_file:
        # adjust_cap_measurement(h5_file.root.ATLAS_Itk.X2.unbiased_1.total_cap.measurements, has_values=True)
        # adjust_cap_measurement(h5_file.root.ATLAS_Itk.X2.biased_80_V.total_cap.measurements, has_values=True)
        # adjust_i_v_measurement(h5_file.root.ATLAS_Itk.X2.I_V_Characteristic.biasing.measurements)
        # adjust_c_v_measurement(h5_file.root.ATLAS_Itk.X2.C_V_Characteristic.biasing.measurements)
        # adjust_cap_measurement(h5_file.root.total_cap.measurements, has_values=False)
        # regenerate_measurement_errors(h5_file.root.Reference.TESTS.unbiased_5_full.total_cap.measurements)
        # regenerate_measurement_errors(h5_file.root.Reference.TESTS.unbiased_4_full.total_cap.measurements)
        # regenerate_measurement_errors(h5_file.root["ATLAS ITk"].unbiased_3.total_cap.measurements)
        # regenerate_measurement_errors(h5_file.root["ATLAS ITk"].run_2.total_cap.measurements)
        # regenerate_measurement_errors(h5_file.root["ATLAS ITk"].run_1.total_cap.measurements)
        # regenerate_i_v_errors(h5_file.root["ATLAS ITk"].I_V_Characteristic.biasing.measurements)
        # regenerate_c_v_errors(h5_file.root["ATLAS ITk"].C_V_Characteristic.biasing.measurements)
        generate_pixel_dimensions(h5_file.root.ATLAS_ITk.X1)

        # group = h5_file.root.ATLAS_Itk.X2.unbiased_1.measurements
        # group.HistCurr.attrs["Units"] = "A"
        # group.HistCurrErr.attrs["Units"] = "A"
        # group.HistCurrValues.attrs["Units"] = "A"
        # group._f_setattr("current_unit", "A")
        # group._f_setattr("freq_unit", "MHz")
        # group._f_setattr("bias_current_unit", "A")
        # group._f_setattr("bias_voltage_unit", "V")
        # group._f_setattr("bias_unit", "V")
        # group.BiasVoltageHist.attrs["Units"] = "V"
        # assert isinstance(group, tb.Group)
        # for subgroup in group._v_groups.values():
        #     subgroup._f_setattr("bias_voltage_unit", "V")
        #     subgroup._f_setattr("bias_unit", "V")
        #     subgroup._f_setattr("freq_unit", "MHz")
        #     subgroup._f_setattr("current_unit", "A")
        #     if "HistCurr" not in subgroup:
        #         print(subgroup)
        #         print("No entries?")
        #     else:
        #         subgroup.HistCurr.attrs["Units"] = "A"
        #         subgroup.HistCurrErr.attrs["Units"] = "A"
        #         # subgroup.HistCurrValues.attrs["Units"] = "A"
        # regenerate_basi_table(h5_file.root.ATLAS_ITk.X1.I_V_Characteristic.biasing.measurements)
        generate_bias_table(h5_file.root.ATLAS_ITk.X1.I_V_Characteristic.biasing.measurements, transform_api=True)
        regenerate_c_v_errors(h5_file.root.ATLAS_ITk.X1.C_V_Characteristic_refined.biasing.measurements)
        regenerate_inter_pix_errors(h5_file.root.Thesis.ATLAS_ITk.X1.inter_unbiased_full.inter_cap.measurements)
        regenerate_inter_pix_errors(h5_file.root.Thesis.ATLAS_ITk.X1.inter_biased_M_80_V_full.inter_cap.measurements)

    with tb.open_file("packaged/X2_2_Scan.h5", "a") as h5_file:
        # generate_bias_table(h5_file.root.ATLAS_ITk.X2.I_V_Characteristic.biasing.measurements)
        generate_bias_table(h5_file.root.ATLAS_ITk.X2.C_V_Characteristic_refined.biasing.measurements)
        regenerate_c_v_errors(h5_file.root.ATLAS_ITk.X2.C_V_Characteristic_refined.biasing.measurements)
        generate_pixel_dimensions(h5_file.root.ATLAS_ITk.X2)
        h5_file.root.ATLAS_ITk.X2.C_V_Characteristic_refined.biasing.measurements.BiasVoltageHist.attrs["Units"] = "V"

    # with tb.open_file("packaged/R13_2_Scan.h5", "a") as h5_file:
    #     h5_file.copy_children(h5_file.root.ATLAS_ITk.X2, h5_file.root.Reference.R13, recursive=True, overwrite=True)
    #     h5_file.flush()
    #     with tb.open_file("packaged/R13_Renew_Scan.h5", "a") as second_file:
    #         h5_file.copy_children(h5_file.root, second_file.root, recursive=True, overwrite=True)
    #
    # with tb.open_file("packaged/R13_3_Scan.h5", "a") as h5_file:
    #     with tb.open_file("packaged/R13_Renew_Scan.h5", "a") as second_file:
    #         h5_file.copy_children(h5_file.root.Reference.R13, second_file.root.Reference.R13, recursive=True, overwrite=True)
    #
    # with tb.open_file("packaged/R13_Renew_Scan.h5", 'a') as h5_file:
    #     generate_pixel_dimensions(h5_file.root.Reference.R13)
    #     generate_bias_table(h5_file.root.Reference.R13.C_V_Characteristic_refined.biasing.measurements)
    #     regenerate_c_v_errors(h5_file.root.Reference.R13.C_V_Characteristic_refined.biasing.measurements)
    #     regenerate_inter_pix_errors(h5_file.root.Reference.R13.inter_unbiased_full.inter_cap.measurements)
    #     regenerate_inter_pix_errors(h5_file.root.Reference.R13.inter_biased_M_80_V_full.inter_cap.measurements)





    # When have I repaired all the implementations.
    # All the first try measurement series needs to consolidated and their entries needs to be adjusted for the new formats

