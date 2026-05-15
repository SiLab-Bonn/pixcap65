import numpy as np

from pixcap65.utils import PixCapSetup, PixcapMeasurements
from pixcap65.pixcap_65_test_total_cap import ScanConfigurationKeys, PixCap65Measurement

scan_configuration = {
    'start_column': 0,
    'stop_column': 40,
    'start_row': 0,
    'stop_row': 40,

    'Vin': 1.0,  # input voltage in V
    'frequency_range': np.arange(1, 4.1, 1),  # frequency sweep in MHz
    'bias_range': -1 * np.geomspace(1, 80, 25),
    # 'bias': -80.0,   # bias voltage to apply in V

    'data_path': "Reference/R13",
    "out_file_mode": "append",
}

if __name__ == "__main__":
    # one setup trial per measurement
    output_file = "R13_3_Scan.h5"

    # initial measurement sample
    # scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS] = 30
    # scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, 12.1, 1)
    # with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
    #     pix.pixcap.frequency_settling = 0.3
    #     print(type(pix))
    #     # pix.scan(data_group_spec="unbiased_1_full")
    #     with pix.binary_readout_mode() as binary_pix:
    #         binary_pix.scan(data_group_spec="unbiased_1_full")
    # del scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS]

    # I-V characterization!
    # scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] = -1 * np.arange(1, 100, 0.25)
    # scan_configuration[ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS] = 3
    # with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
    #     pix.bias_scan(data_group_spec="I_V_Characteristic")
    # del scan_configuration[ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS]

    # C-V characterization sample
    # coarse_bias_range = -1 * np.arange(1, 100.1, 0.75)
    # fine_bias_range = -1 * np.concat((
    #     np.geomspace(0.1, 20, 30),
    #     np.arange(20, 60.1, 5),
    #     np.geomspace(60, 85, 20)
    # ))
    # # scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] = coarse_bias_range
    # scan_configuration.update(start_row=20, stop_row=25, start_column=20, stop_column=25)
    # scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, 8.1, 1)
    # if ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS in scan_configuration:
    #     del scan_configuration[ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS]
    # if ScanConfigurationKeys.AVERAGE_MEASUREMENTS in scan_configuration:
    #     del scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS]

    # coarse scan
    # assert np.all(scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] < 0)
    # with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
    #     print(type(pix))
    #     pix.combined_bias_cv_scan(data_group_spec="C_V_Characteristic")
    #
    # time.sleep(60)

    # fine scan
    # scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] = fine_bias_range
    scan_configuration.update(start_row=10, stop_row=35, start_column=10, stop_column=35)
    scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, 4.1, 1)
    assert np.all(scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] < 0)
    with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
        from pixcap65.pixcap_65_test_total_cap import PixCap65Measurement
        pix.frequency_settling = 0.3
        pix.combined_bias_cv_scan(data_group_spec="C_V_Characteristic_refined")

    # completing scan
    # if ScanConfigurationKeys.BIAS_VOLTAGE_RANGE in scan_configuration:
    #     del scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE]
    # scan_configuration.update(start_row=1, stop_row=40, start_column=0, stop_column=40, bias=-80,
    #                           average_measurements=40)
    # scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, 12.1, 1)
    # scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS] = 30
    # with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
    #     pix.pixcap.frequency_settling = 0.3
    #     print(type(pix))
    #     print(issubclass(type(pix), PixCap65Measurement))
    #     print(isinstance(pix, PixCap65Measurement))
    #     # pix.scan(data_group_spec="biased_80_V_full")
    #     with pix.binary_readout_mode() as binary_pix:
    #         binary_pix.scan(data_group_spec="biased_80_V_full")

    del scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE]
    scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1,8.1,0.75)
    with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.INTER_CAPACITANCE) as pix:
        pix.frequency_settling = 0.3
        pix.scan(data_group_spec="inter_unbiased_full")

    scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE] = -80
    with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.INTER_CAPACITANCE) as pix:
        pix.frequency_settling = 0.3
        pix.scan(data_group_spec="inter_biased_M_80_V_full")
