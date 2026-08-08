from importlib.resources import files

import logging
import numpy as np
import tables as tb
import time
import yaml

from pixcap65 import data_constants
from pixcap65.analysis_util.utility import HIST_BIAS_MEAS_UNIT, HIST_CURRENT_MEAS_UNIT, GLOBAL_FILTERS
from pixcap65.configs.config_handler import extract_smu_voltage_error
from pixcap65.data_constants import X2_SCAN_2_FILE
from pixcap65.pixcap.pixcap65_measurement import ScanConfigurationKeys
from pixcap65.pixcap_65_test_total_cap import BiasTable
from pixcap65.utility.tables_util import set_group_attribute, group_get_file, get_groups, list_group_attributes, \
    get_group_attribute
from pixcap65.utility.utils_2 import UNITS_ATTRIBUTE_KEY, create_carray, prevent_group_mix_up

logger = logging.getLogger(__name__)

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


def generate_pixel_dimensions(group, quad_length=50.0):
    prevent_group_mix_up(group, "sensor")

    group_get_file(group).create_group(group, name="sensor")
    dimensions_array = np.full((40, 40, 2), fill_value=quad_length)
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


def combine_cv_measurements(first_group: tb.Group, second_group: tb.Group):
    # parent group and file
    parent_group = first_group._v_parent
    h5_file = group_get_file(parent_group)

    if "C_V_Characteristic_refined_Extended_Combined" in parent_group:
        parent_group["C_V_Characteristic_refined_Extended_Combined"]._f_remove(recursive=True)
        time.sleep(10)
    target_group = h5_file.create_group(where=parent_group, name="C_V_Characteristic_refined_Extended_Combined")
    target_group = h5_file.create_group(where=target_group, name="biasing")
    target_group = h5_file.create_group(where=target_group, name="measurements")
    first_group = first_group.biasing.measurements
    second_group = second_group.biasing.measurements

    # load the basic data
    first_current_error = first_group.HistCurrErr[:]
    second_current_error = second_group.HistCurrErr[:]
    first_current_data = first_group.HistCurr[:]
    second_current_data = second_group.HistCurr[:]
    first_bias_table = np.rec.array(first_group.BiasTable[:], dtype=first_group.BiasTable.dtype)
    second_bias_table = np.rec.array(second_group.BiasTable[:], dtype=second_group.BiasTable.dtype)
    first_voltage_hist = first_group.BiasVoltageHist[:]
    second_voltage_hist = second_group.BiasVoltageHist[:]
    first_scan_parameters = np.rec.array(first_group.scan_params[:], dtype=first_group.scan_params.dtype)
    second_scan_parameters = np.rec.array(second_group.scan_params[:], dtype=second_group.scan_params.dtype)
    if len(first_voltage_hist.shape) <= 1:
        temp_array = np.full(shape=(first_voltage_hist.shape[0], 3), fill_value=np.nan)
        temp_array[:, 0] = first_voltage_hist
        if "U" in first_bias_table.dtype.fields:
            temp_array[:, 1] = first_bias_table.U
        elif "hv_voltage" in first_scan_parameters.dtype.fields:
            temp_array[:, 1] = first_bias_table.hv_voltage
        else:
            temp_array[:, 1] = first_voltage_hist

        if "DU" not in first_bias_table.dtype.fields or np.any(np.isnan(first_bias_table.DU)):
            try:
                with open("pixcap65/configs/keithley_2410_range.yaml", "r") as conf_file:
                    scan_range_config = yaml.safe_load(conf_file)

                temp_array[:, 2] = extract_smu_voltage_error(scan_range_config, first_voltage_hist, 1000)
            except:
                logging.warning("Could not extract voltage errors from keithley_2410_range.yaml", exc_info=True)
                temp_array[:, 2] = np.full_like(first_voltage_hist, np.nan)
        else:
            temp_array[:, 2] = first_bias_table.DU

        first_voltage_hist = temp_array

    if len(second_voltage_hist.shape) <= 1:
        temp_array = np.full(shape=(second_voltage_hist.shape[0], 3), fill_value=np.nan)
        temp_array[:, 0] = second_voltage_hist
        if "U" in second_bias_table.dtype.fields:
            temp_array[:, 1] = second_bias_table.U
        elif "hv_voltage" in second_scan_parameters.dtype.fields:
            temp_array[:, 1] = second_bias_table.hv_voltage
        else:
            temp_array[:, 1] = second_voltage_hist

        if "DU" not in second_bias_table.dtype.fields or np.any(np.isnan(second_bias_table.DU)):
            try:
                with open("pixcap65/configs/keithley_2410_range.yaml", "r") as conf_file:
                    scan_range_config = yaml.safe_load(conf_file)

                temp_array[:, 2] = extract_smu_voltage_error(scan_range_config, second_voltage_hist, 1000)
            except:
                logging.warning("Could not extract voltage errors from keithley_2410_range.yaml", exc_info=True)
                temp_array[:, 2] = np.full_like(second_voltage_hist, np.nan)
        else:
            temp_array[:, 2] = second_bias_table.DU

        second_voltage_hist = temp_array

    # first_unique_voltages = np.setdiff1d(first_scan_parameters.bias_voltage, second_scan_parameters.bias_voltage)
    # first_unique_voltages = np.setdiff1d(first_bias_table.Us, second_bias_table.Us)
    first_unique_voltages = np.setdiff1d(first_voltage_hist[:, 0], second_voltage_hist[:, 0])

    first_mask = np.abs(first_scan_parameters.bias_voltage) <= np.max(np.abs(first_unique_voltages))
    new_current_data = np.concat([first_current_data[first_mask], second_current_data])
    new_current_error = np.concat([first_current_error[first_mask], second_current_error])
    new_scan_parameters = np.concat([first_scan_parameters[first_mask], second_scan_parameters])
    new_bias_table = np.concat([first_bias_table[first_mask], second_bias_table])
    new_voltage_hist = np.concat([first_voltage_hist[first_mask], second_voltage_hist])


    # write the newly created data sets to the new group for the combined results.
    new_array = h5_file.create_carray(where=target_group, name="HistCurr", title=first_group.HistCurr.title, filters=first_group.HistCurr.filters, obj=new_current_data)
    for attribute in first_group.HistCurr.attrs._f_list():
        new_array.attrs[attribute] = first_group.HistCurr.attrs[attribute]
    new_array = h5_file.create_carray(where=target_group, name="HistCurrErr", title=first_group.HistCurrErr.title, filters=first_group.HistCurrErr.filters, obj=new_current_error)
    for attribute in first_group.HistCurrErr.attrs._f_list():
        new_array.attrs[attribute] = first_group.HistCurrErr.attrs[attribute]
    h5_file.create_table(where=target_group, name="scan_params", description=new_scan_parameters, filters=first_group.scan_params.filters)
    h5_file.create_table(where=target_group, name="BiasTable", description=new_bias_table, filters=first_group.BiasTable.filters)
    new_array = h5_file.create_carray(where=target_group, name="BiasVoltageHist", title=first_group.BiasVoltageHist.title, filters=first_group.BiasVoltageHist.filters, obj=new_voltage_hist)
    for attribute in first_group.BiasVoltageHist.attrs._f_list():
        new_array.attrs[attribute] = first_group.BiasVoltageHist.attrs[attribute]

    # get the data points (bias voltages) which are unique for both groups and extract the overlapping ones

    # we must copy all the sub measurement groups into the new group
    for voltage in first_unique_voltages:
        group_name = ("bias_{bias_voltage}_V"
                      .format(bias_voltage=voltage)
                      .replace('-', "M_")
                      .replace(".", "__"))
        h5_file.copy_node(where=first_group, name=group_name, new_name=group_name, newparent=target_group, overwrite=True, recursive=True)
    for voltage in second_voltage_hist[:, 0]:
        group_name = ("bias_{bias_voltage}_V"
                      .format(bias_voltage=voltage)
                      .replace('-', "M_")
                      .replace(".", "__"))
        h5_file.copy_node(where=second_group, name=group_name, new_name=group_name, newparent=target_group,
                          overwrite=True, recursive=True)

    for attribute in list_group_attributes(first_group):
        target_group._f_setattr(attribute, get_group_attribute(first_group, attribute))


def generate_bias_table(group, **kwargs):
    file = group_get_file(group)
    if "BiasTable" in group:
        group.BiasTable._f_remove()
        time.sleep(5)

    table = file.create_table(group, name="BiasTable", description=BiasTable,
                                          filters=tb.Filters(complib='blosc', fletcher32=False, complevel=5))
    entry = table.row

    # need to handle the actually measured voltages.
    internal_parameters = group.scan_params[:]
    try:
        hist_parameters = group.BiasVoltageHist[:]
    except (KeyError, tb.exceptions.NoSuchNodeError):
        hist_parameters = np.full(internal_parameters["hv_voltage"].shape[0], np.nan)
    bias_currents = group.HistCurr[:]
    bias_errors = group.HistCurrErr[:]
    new_voltage_hist_data = np.full(shape=(hist_parameters.shape[0], 3,), fill_value=np.nan)
    if np.any(np.isfinite(hist_parameters)):
        if len(hist_parameters.shape) > 1:
            voltages = hist_parameters[:, 1]
            voltage_errors = hist_parameters[:, 2]
            voltage_settings = hist_parameters[:, 0]
        else:
            voltages = hist_parameters
            voltage_errors = np.full_like(voltages, np.nan)
            voltage_settings = voltages.copy()
    else:
        # What 'to do' if this argument does not exist. We only started catching this lately.
        voltages = internal_parameters["hv_voltage"]
        voltage_errors = np.full_like(voltages, np.nan)
    if np.all(~np.isfinite(voltage_errors)):
        try:
            voltage_settings = internal_parameters["bias_voltage"]
            with open("pixcap65/configs/keithley_2410_range.yaml") as f:
                range_config = yaml.safe_load(f)
                voltage_errors = extract_smu_voltage_error(range_config, voltages, 1000)
        except:
            voltage_errors = np.full_like(voltages, np.nan)

    for set_voltage, leak_current, current_error, meas_voltage, meas_voltage_error in zip(
            voltage_settings, bias_currents,
            bias_errors, voltages, voltage_errors):
        try:
            entry['Us'] = set_voltage
            entry['U'] = meas_voltage
            entry['I'] = leak_current
            entry['DI'] = current_error
            entry['DU'] = meas_voltage_error
            entry.append()
        except ValueError as e:
            logger.info("scan parameters")
            logger.info(set_voltage)
            logger.info("currents")
            logger.info(bias_currents.shape)
            logger.info(leak_current)
            logger.info("Errors")
            logger.info(bias_errors.shape)
            logger.info(current_error)
            raise e

    # when the old api for the bias voltage hist is used we should transform it to the new one
    if len(hist_parameters) <= 1 and kwargs.pop("transform_api", False):
        new_voltage_hist_data[:, 0] = voltage_settings
        new_voltage_hist_data[:, 1] = voltages
        new_voltage_hist_data[:, 2] = voltage_errors
        if "BiasVoltageHist" in group:
            temp = group.BiasVoltageHist
            assert isinstance(temp, tb.CArray)
            temp._f_remove()
            time.sleep(30)
        file.create_carray(group, "BiasVoltageHist", obj=new_voltage_hist_data, filters=tb.Filters(complib='blosc', fletcher32=False, complevel=5))


def regenerate_inter_pix_errors(group):
    if "HistCurrValues" not in group:
        from pixcap65.configs.config_handler import extract_smu_current_error
        import yaml

        with files("pixcap65.configs").joinpath("keithley_2602a_range.yaml").open() as f:
            # with open("pixcap65/configs/keithley_2602a_range.yaml") as f:
            config = yaml.safe_load(f)
            print("Read the configuration")
            print(config)
            total_data = group.TotalHistCurr[:]
            total_errors = np.where(np.isfinite(total_data),
                                    extract_smu_current_error(config, total_data, 0.000001), np.nan)
            if "TotalHistCurrErr" not in group:
                group_get_file(group).create_carray(where=group, name="TotalHistCurrErr", obj=total_errors,
                                                    filters=tb.Filters(complib='blosc', fletcher32=False, complevel=5))
            else:
                group.TotalHistCurrErr[:] = total_errors
            inter_data_1 = group.InterHistCurrA[:]
            inter_errors_1 = np.where(np.isfinite(inter_data_1),
                                      extract_smu_current_error(config, inter_data_1, 0.000010), np.nan)
            if "InterHistCurrErrA" not in group:
                group_get_file(group).create_carray(where=group, name="InterHistCurrErrA", obj=inter_errors_1,
                                                    filters=tb.Filters(complib='blosc', fletcher32=False, complevel=5))
            else:
                group.InterHistCurrErrA[:] = inter_errors_1

            inter_data_2 = group.InterHistCurrB[:]
            inter_errors_2 = np.where(np.isfinite(inter_data_2),
                                      extract_smu_current_error(config, inter_data_2, 0.000001), np.nan)
            if "InterHistCurrErrB" not in group:
                group_get_file(group).create_carray(where=group, name="InterHistCurrErrB", obj=inter_errors_2,
                                                    filters=tb.Filters(complib='blosc', fletcher32=False, complevel=5))
            else:
                group.InterHistCurrErrB[:] = inter_errors_2


def split_sensor_group(group: tb.Group):
    h5_file = group_get_file(group)
    original_name = group._v_name
    parent_group = group._v_parent
    pixel_groups = data_constants.e1_pixel_groups

    def transfer_selected_data(fetched, target, **kwargs):
        for col_set, row_set in zip(kwargs["columns"], kwargs["rows"]):
            target[col_set[0]:col_set[1], row_set[0]:row_set[1]] = fetched[col_set[0]:col_set[1], row_set[0]:row_set[1]]

    def apply_changes(group: tb.Group, **kwargs):
        if "HistCurr" in group:
            fetch_data = group.HistCurr[:]
            if fetch_data.shape[:2] == (40, 40):
                new_data = np.full_like(fetch_data, np.nan)
                transfer_selected_data(fetch_data, new_data, **kwargs)
                if np.all(new_data == fetch_data):
                    print("Unexpectedly the new data has not changed at all.")

                group.HistCurr[:] = new_data
                group._g_flush_group()
                group_get_file(group).flush()

        if "InterHistCurrA" in group:
            fetch_data = group.InterHistCurrA[:]
            if fetch_data.shape[:2] == (40, 40):
                new_data = np.full_like(fetch_data, np.nan)
                transfer_selected_data(fetch_data, new_data, **kwargs)
                group.InterHistCurrA[:] = new_data
                group._g_flush_group()
                group_get_file(group).flush()

        if "InterHistCurrB" in group:
            fetch_data = group.InterHistCurrB[:]
            if fetch_data.shape[:2] == (40, 40):
                new_data = np.full_like(fetch_data, np.nan)
                transfer_selected_data(fetch_data, new_data, **kwargs)
                group.InterHistCurrB[:] = new_data
                group._g_flush_group()
                group_get_file(group).flush()

        if "TotalHistCurr" in group:
            fetch_data = group.TotalHistCurr[:]
            if fetch_data.shape[:2] == (40, 40):
                new_data = np.full_like(fetch_data, np.nan)
                transfer_selected_data(fetch_data, new_data, **kwargs)
                group.TotalHistCurr[:] = new_data
                group._g_flush_group()
                group_get_file(group).flush()

        if "HistCurrErr" in group:
            fetch_data = group.HistCurrErr[:]
            if fetch_data.shape[:2] == (40, 40):
                new_data = np.full_like(fetch_data, np.nan)
                transfer_selected_data(fetch_data, new_data, **kwargs)
                group.HistCurrErr[:] = new_data
                group._g_flush_group()
                group_get_file(group).flush()

        if "InterHistCurrErrA" in group:
            fetch_data = group.InterHistCurrErrA[:]
            if fetch_data.shape[:2] == (40, 40):
                new_data = np.full_like(fetch_data, np.nan)
                transfer_selected_data(fetch_data, new_data, **kwargs)
                group.InterHistCurrErrA[:] = new_data
                group._g_flush_group()
                group_get_file(group).flush()

        if "InterHistCurrErrB" in group:
            fetch_data = group.InterHistCurrErrB[:]
            if fetch_data.shape[:2] == (40, 40):
                new_data = np.full_like(fetch_data, np.nan)
                transfer_selected_data(fetch_data, new_data, **kwargs)
                group.InterHistCurrErrB[:] = new_data
                group._g_flush_group()
                group_get_file(group).flush()

        if "TotalHistCurrErr" in group:
            fetch_data = group.TotalHistCurrErr[:]
            if fetch_data.shape[:2] == (40, 40):
                new_data = np.full_like(fetch_data, np.nan)
                transfer_selected_data(fetch_data, new_data, **kwargs)
                group.TotalHistCurrErr[:] = new_data
                group._g_flush_group()
                group_get_file(group).flush()

        if "HistCurrValues" in group:
            fetch_data = group.HistCurrValues[:]
            if fetch_data.shape[:2] == (40, 40):
                new_data = np.full_like(fetch_data, np.nan)
                transfer_selected_data(fetch_data, new_data, **kwargs)
                group.HistCurrValues[:] = new_data
                group._g_flush_group()
                group_get_file(group).flush()

        # now it is still necessary to walkt the children
        for child in group._f_walk_groups():
            if child == group:
                continue
            if "analysis" in child._v_name:
                continue
            apply_changes(child, **kwargs)



    new_names = ["{}_{}".format(original_name, i) for i in pixel_groups.keys()]
    for (new_name, type_name) in zip(new_names, pixel_groups.keys()):
        new_group = h5_file.copy_node(where=parent_group, newparent=parent_group, name=original_name, newname=new_name, overwrite=True, recursive=True)
        # now it is necessary to update the data arrays to only feature the extracted data
        if isinstance(new_group, tb.Group):
            apply_changes(new_group, **pixel_groups[type_name])


# FIXME: Why are there no error estimations for I-V curves?

if __name__ == "__main__":
    # with tb.open_file(X1_SCAN_2_FILE, "a") as h5_file:
    #     generate_bias_table(h5_file.root.ATLAS_ITk.X1.I_V_Characteristic.biasing.measurements, transform_api=True)
    #     regenerate_c_v_errors(h5_file.root.ATLAS_ITk.X1.C_V_Characteristic_refined.biasing.measurements)
    #     regenerate_inter_pix_errors(h5_file.root.Thesis.ATLAS_ITk.X1.inter_unbiased_full.inter_cap.measurements)
    #     regenerate_inter_pix_errors(h5_file.root.Thesis.ATLAS_ITk.X1.inter_biased_M_80_V_full.inter_cap.measurements)
    #     combine_cv_measurements(h5_file.root.ATLAS_ITk.X1.C_V_Characteristic_refined,
    #                             h5_file.root.Thesis.ATLAS_ITk.X1.C_V_Characteristic_refined_Extended,)
    #     h5_file.copy_node(where=h5_file.root.ATLAS_ITk.X1, newparent=h5_file.root.Thesis.ATLAS_ITk.X1,
    #                       name="C_V_Characteristic_refined_Extended_Combined",
    #                       newname="C_V_Characteristic_refined_Extended_Combined",
    #                       overwrite=True, recursive=True)
    #     old_x1 = h5_file.root.ATLAS_ITk.X1
    #     for group in old_x1._f_iter_nodes():
    #         if not isinstance(group, tb.Group):
    #             continue
    #         if group == old_x1:
    #             continue
    #         print(group)
    #         h5_file.copy_node(where=old_x1, name=group._v_name,
    #                           newname=group._v_name,
    #                           newparent=h5_file.root.Thesis.ATLAS_ITk.X1, recursive=True, overwrite=True)
    #     generate_pixel_dimensions(h5_file.root.Thesis.ATLAS_ITk.X1)
    #
    with tb.open_file(X2_SCAN_2_FILE, "a") as h5_file:
        # generate_bias_table(h5_file.root.ATLAS_ITk.X2.I_V_Characteristic.biasing.measurements)
        generate_bias_table(h5_file.root.ATLAS_ITk.X2.C_V_Characteristic_refined.biasing.measurements)
        regenerate_c_v_errors(h5_file.root.ATLAS_ITk.X2.C_V_Characteristic_refined.biasing.measurements)
        generate_pixel_dimensions(h5_file.root.ATLAS_ITk.X2)
        h5_file.root.ATLAS_ITk.X2.C_V_Characteristic_refined.biasing.measurements.BiasVoltageHist.attrs["Units"] = "V"
        old_x2 = h5_file.root.ATLAS_ITk.X2
        assert isinstance(old_x2, tb.Group)
        for group in old_x2._f_iter_nodes():
            if not isinstance(group, tb.Group):
                continue
            if group == old_x2:
                continue
            print(group)
            h5_file.copy_node(where=old_x2, name=group._v_name,
                              newname=group._v_name,
                              newparent=h5_file.root.Thesis.ATLAS_ITk.X2, recursive=True, overwrite=True)

        with tb.open_file("packaged/data/X2_12_Renew_Scan.h5") as backing_file:
            backing_file.copy_children(backing_file.root.Thesis.ATLAS_ITk.X2, h5_file.root.Thesis.ATLAS_ITk.X2, recursive=True, overwrite=True)

    #
    # with tb.open_file(R11_SCAN_FILE, "a") as h5_file:
    #     generate_pixel_dimensions(h5_file.root.Reference.R1)
    #     old_sensor = h5_file.root.Reference.R11
    #     for group in old_sensor._f_iter_nodes():
    #         if not isinstance(group, tb.Group):
    #             continue
    #         if group == old_sensor:
    #             continue
    #         if group._v_name == "C_V_Characteristic_refined":
    #             continue
    #         if group._v_name == "I_V_Characteristic":
    #             continue
    #         print(group)
    #         h5_file.copy_node(where=old_sensor, name=group._v_name,
    #                           newname=group._v_name,
    #                           newparent=h5_file.root.Reference.R1, recursive=True, overwrite=True)
    #
    #     physical_dimensions = h5_file.root.Reference.R1.sensor.PhysicalDimensions[:]
    #     physical_dimensions[:, :] = np.asarray([6, 81], dtype=np.float64)
    #     physical_dimensions[:, 0] = np.nan
    #     h5_file.root.Reference.R1.sensor.PhysicalDimensions[:] = physical_dimensions
    #     h5_file.root.Reference.R1.sensor.PhysicalDimensions.flush()

    # with tb.open_file("packaged/data/R13_2_Scan.h5", "a") as h5_file:
    #     h5_file.copy_children(h5_file.root.ATLAS_ITk.X2, h5_file.root.Reference.R13, recursive=True, overwrite=True)
    #     h5_file.flush()
    #     with tb.open_file(R13_2_SCAN_FILE, "a") as second_file:
    #         h5_file.copy_children(h5_file.root, second_file.root, recursive=True, overwrite=True)
    #
    # with tb.open_file("packaged/data/R13_3_Scan.h5", "a") as h5_file:
    #     with tb.open_file(R13_2_SCAN_FILE, "a") as second_file:
    #         h5_file.copy_children(h5_file.root.Reference.R13, second_file.root.Reference.R13, recursive=True, overwrite=True)
    #
    # with tb.open_file(R13_2_SCAN_FILE, 'a') as h5_file:
    #     generate_pixel_dimensions(h5_file.root.Reference.R13, 30)
    #     generate_bias_table(h5_file.root.Reference.R13.C_V_Characteristic_refined.biasing.measurements)
    #     regenerate_c_v_errors(h5_file.root.Reference.R13.C_V_Characteristic_refined.biasing.measurements)
    #     regenerate_inter_pix_errors(h5_file.root.Reference.R13.inter_unbiased_full.inter_cap.measurements)
    #     regenerate_inter_pix_errors(h5_file.root.Reference.R13.inter_biased_M_80_V_full.inter_cap.measurements)
    #
    #     with tb.open_file(X2_SCAN_2_FILE, 'a') as old_file:
    #         old_file.copy_node(where=old_file.root.Reference.R13, newparent=h5_file.root.Reference.R13,
    #                            name="inter_unbiased_full_renew_Extended",
    #                            newname="inter_unbiased_renew_Extended_full", recursive=True, overwrite=True)
    #         old_file.copy_node(where=old_file.root.Reference.R13, newparent=h5_file.root.Reference.R13,
    #                            name="inter_biased_M_80_V_full_renew_Extended",
    #                            newname="inter_biased_M_80_V_renew_Extended_full", recursive=True, overwrite=True)

    # with tb.open_file(E1_2_SCAN_FILE, "a") as h5_file:
    #     generate_pixel_dimensions(h5_file.root.Reference.E1)
    #     physical_dimensions = h5_file.root.Reference.E1.sensor.PhysicalDimensions[:]
    #     physical_dimensions[:, 0] = np.nan
    #     for key, region in data_constants.e1_pixel_groups.items():
    #         for col_range, row_range in zip(region["columns"], region["rows"]):
    #             physical_dimensions[col_range[0]:col_range[1], row_range[0]:row_range[1]] = [data_constants.e1_pixel_dimensions[key],
    #                                                                                          data_constants.e1_pixel_dimensions[key]]
    #
    #     h5_file.root.Reference.E1.sensor.PhysicalDimensions[:] = physical_dimensions
    #     h5_file.root.Reference.E1.sensor.PhysicalDimensions.flush()
    #
    #     # still need to generate all the measurement errors
    #     regenerate_c_v_errors(h5_file.root.Reference.E1.C_V_Characteristic_refined.biasing.measurements)
    #     regenerate_inter_pix_errors(h5_file.root.Reference.E1.inter_unbiased_full.inter_cap.measurements)
    #     regenerate_inter_pix_errors(h5_file.root.Reference.E1.inter_biased_M_80_V_full.inter_cap.measurements)
    #
    #     split_sensor_group(h5_file.root.Reference.E1.inter_biased_M_80_V_full)
    #     split_sensor_group(h5_file.root.Reference.E1.C_V_Characteristic_refined)
    #     split_sensor_group(h5_file.root.Reference.E1.biased_80_V_full)
    #     split_sensor_group(h5_file.root.Reference.E1.unbiased_full)
    #     split_sensor_group(h5_file.root.Reference.E1.inter_unbiased_full)
    #
    # with tb.open_file(X4_SCAN_FILE, "a") as h5_file:
    #     generate_pixel_dimensions(h5_file.root.Thesis.ATLAS_ITk.X4)
    #     h5_file.copy_node(where=h5_file.root.Thesis.ATLAS_ITk.X4, name="inter_unbiased_full_Extended_Second",
    #                       newname="inter_unbiased_full", overwrite=True, recursive=True)
    #     h5_file.copy_node(where=h5_file.root.Thesis.ATLAS_ITk.X4, name="inter_biased_M_80_V_full_Extended",
    #                       newname="inter_biased_M_80_V_full", overwrite=True, recursive=True)

    # with tb.open_file(X5_SCAN_FILE, "a") as h5_file:
    #     generate_pixel_dimensions(h5_file.root.Thesis.ATLAS_ITk.X5)

    # with tb.open_file(X6_SCAN_FILE, "a") as h5_file:
    #     wrong_parent = h5_file.root.Thesis.ATLAS_ITk.X7
    #     right_parent = h5_file.root.Thesis.ATLAS_ITk.X6
    #     if "inter_unbiased_full" in wrong_parent:
    #         h5_file.move_node(where=wrong_parent, name="inter_unbiased_full", newparent=right_parent,)
    #         h5_file.move_node(where=wrong_parent, name="inter_biased_M_45.0_V_full", newparent=right_parent, newname="inter_biased_M_45_V_full")
    #         h5_file.move_node(where=wrong_parent, name="biased_45.0_V_full", newparent=right_parent, newname="biased_45_V_full")
    #         h5_file.move_node(where=wrong_parent, name="C_V_Characteristic_refined", newparent=right_parent, overwrite=True)
    #
    #     generate_pixel_dimensions(h5_file.root.Thesis.ATLAS_ITk.X6)
    #
    # with tb.open_file(X7_SCAN_FILE, 'a') as h5_file:
    #     generate_pixel_dimensions(h5_file.root.Thesis.ATLAS_ITk.X7)

    # When have I repaired all the implementations.
    # All the first try measurement series needs to consolidated and their entries needs to be adjusted for the new formats

