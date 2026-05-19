import numpy as np
import time

from pixcap65.pixcap_65_test_total_cap import ScanConfigurationKeys
from pixcap65.utils import PixCapSetup, PixcapMeasurements

scan_configuration = {
    'start_column': 5,
    'stop_column': 35,
    'start_row': 5,
    'stop_row': 35,

    'Vin': 1.0,  # input voltage in V
    'frequency_range': np.arange(1, 6.1, 0.5),  # frequency sweep in MHz
    # 'bias_range': -1 * np.geomspace(1, 80, 25),
    # 'bias': -80.0,   # bias voltage to apply in V

    'data_path': "Thesis/ATLAS_ITk/X1",
    "out_file_mode": "append",
}

if __name__ == "__main__":
    import shutil
    # one setup trial per measurement
    output_file = "X1_4_Renew_Scan.h5"

    # initial measurement sample
    scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS] = 30
    scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, 10.1, 1)
    # with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
    #     pix.pixcap.frequency_settling = 0.3
    #     with pix.binary_readout_mode() as binary_pix:
    #         binary_pix.scan(data_group_spec="unbiased_61_full")
    del scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS]
    try:
        time.sleep(5)
        # shutil.copyfile(output_file, output_file.replace(".h5", "_1.h5"))
    except:
        pass

    # I-V characterization!
    scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] = -1 * np.arange(1, 350, 0.25)
    scan_configuration[ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS] = 3
    # with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
    #     pix.bias_scan(data_group_spec="I_V_Characteristic")
    del scan_configuration[ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS]
    try:
        time.sleep(5)
        # shutil.copyfile(output_file, output_file.replace(".h5", "_2.h5"))
    except:
        pass
    # FIXME: there is an error with handle_cv_compaction


    # C-V characterization sample
    # coarse_bias_range = -1 * np.arange(1, 100.1, 0.75)
    fine_bias_range = -1 * np.concat((
        np.geomspace(0.1, 20, 20),
        np.arange(20, 60.1, 5),
        np.geomspace(60, 85, 20)
    ))
    # noqa: S125
    # scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] = coarse_bias_range
    scan_configuration.update(start_row=20, stop_row=25, start_column=20, stop_column=25)
    scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, 4.1, 0.85)
    if ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS in scan_configuration:
        del scan_configuration[ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS]
    if ScanConfigurationKeys.AVERAGE_MEASUREMENTS in scan_configuration:
        del scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS]

    # coarse scan
    # noqa: S125
    # assert np.all(scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] < 0)
    # with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
    #     pix.combined_bias_cv_scan(data_group_spec="C_V_Characteristic")
    try:
        time.sleep(5)
        # shutil.copyfile(output_file, output_file.replace(".h5", "_3.h5"))
    except:
        pass

    time.sleep(60)

    # fine scan
    scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] = fine_bias_range
    scan_configuration.update(start_row=10, stop_row=35, start_column=10, stop_column=35)
    scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, 4.1, 0.85)
    assert np.all(scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] < 0)
    # with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
    #     pix.frequency_settling = 0.3
    #     pix.combined_bias_cv_scan(data_group_spec="C_V_Characteristic_refined")
    try:
        time.sleep(5)
        # shutil.copyfile(output_file, output_file.replace(".h5", "_4.h5"))
    except:
        pass

    # completing scan
    if ScanConfigurationKeys.BIAS_VOLTAGE_RANGE in scan_configuration:
        del scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE]
    scan_configuration.update(start_row=1, stop_row=40, start_column=0, stop_column=40, bias=-80,
                              average_measurements=40)
    scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, 10.1, 1)
    scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS] = 30
    # with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
    #     pix.pixcap.frequency_settling = 0.3
    #     with pix.binary_readout_mode() as binary_pix:
    #         binary_pix.scan(data_group_spec="biased_80_V_full")
    try:
        time.sleep(5)
        # shutil.copyfile(output_file, output_file.replace(".h5", "_5.h5"))
    except:
        pass

    del scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS]
    scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1,6.1,0.5)
    scan_configuration.update(start_row=5, stop_row=35, start_column=5, stop_column=35)
    if ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE in scan_configuration:
        del scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE]
    with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.INTER_CAPACITANCE) as pix:
        pix.frequency_settling = 0.3
        pix.scan(data_group_spec="inter_unbiased_full")
    try:
        time.sleep(5)
        shutil.copyfile(output_file, output_file.replace(".h5", "_6.h5"))
    except:
        pass

    scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE] = -80
    with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.INTER_CAPACITANCE) as pix:
        pix.frequency_settling = 0.3
        pix.scan(data_group_spec="inter_biased_M_80_V_full")

    try:
        time.sleep(5)
        shutil.copyfile(output_file, output_file.replace(".h5", "_7.h5"))
    except:
        pass
