import numpy as np
import tables as tb

from analysis_util.utility import HIST_BIAS_MEAS_UNIT, HIST_CURRENT_MEAS_UNIT
from pixcap65.utility.tables_util import set_group_attribute, group_get_file, get_groups
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
        from configs.config_handler import extract_smu_current_error
        import yaml
        with open("/configs/keithley_2410_range.yaml") as f:
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
        from configs.config_handler import extract_smu_current_error
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
        from configs.config_handler import extract_smu_current_error
        import yaml
        with open("/configs/keithley_2602a_range.yaml") as f:
            config = yaml.safe_load(f)
            data = group.HistCurr[:]
            errors = np.where(np.isfinite(data), extract_smu_current_error(config, data, 0.000001), np.nan)
            if "HistCurrErr" not in group:
                group_get_file(group).create_carray(where=group, name="HistCurrErr", obj=errors,
                                                    filters=tb.Filters(complib='blosc', fletcher32=False, complevel=5))
            else:
                group.HistCurrErr[:] = errors


# FIXME: Why are there no error estimations for I-V curves?

if __name__ == "__main__":
    with tb.open_file('New_2_Scan.h5', "a") as h5_file:
        # adjust_cap_measurement(h5_file.root.ATLAS_Itk.X2.unbiased_1.total_cap.measurements, has_values=True)
        # adjust_cap_measurement(h5_file.root.ATLAS_Itk.X2.biased_80_V.total_cap.measurements, has_values=True)
        # adjust_i_v_measurement(h5_file.root.ATLAS_Itk.X2.I_V_Characteristic.biasing.measurements)
        # adjust_c_v_measurement(h5_file.root.ATLAS_Itk.X2.C_V_Characteristic.biasing.measurements)
        # adjust_cap_measurement(h5_file.root.total_cap.measurements, has_values=False)
        # regenerate_measurement_errors(h5_file.root["ATLAS ITk"].unbiased_4.total_cap.measurements)
        # regenerate_measurement_errors(h5_file.root["ATLAS ITk"].unbiased_3.total_cap.measurements)
        # regenerate_measurement_errors(h5_file.root["ATLAS ITk"].run_2.total_cap.measurements)
        # regenerate_measurement_errors(h5_file.root["ATLAS ITk"].run_1.total_cap.measurements)
        # regenerate_i_v_errors(h5_file.root["ATLAS ITk"].I_V_Characteristic.biasing.measurements)
        # regenerate_c_v_errors(h5_file.root["ATLAS ITk"].C_V_Characteristic.biasing.measurements)
        generate_pixel_dimensions(h5_file.root.ATLAS_Itk.X2)

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
