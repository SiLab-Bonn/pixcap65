import logging

import numpy as np
import time
import multiprocessing

from pixcap65.pixcap_65_test_total_cap import ScanConfigurationKeys
from pixcap65.utils import PixCapSetup, PixcapMeasurements

MAXIMUM_CV_FREQ = 4.1
CV_FREQ_STEP = 1
HV_STEP = 0.3
MAXIMUM_HV_VOLTAGE = 60
MAXIMUM_FULL_SCAN_FREQ = 10.1
FULL_SCAN_FREQ_STEP = 0.75
MAXIMUM_INTER_PIX_FREQ = 6.1
INTER_PIX_FREQ_STEP = 0.5


lock = multiprocessing.RLock()

scan_configuration = {
    'start_column': 0,
    'stop_column': 40,
    'start_row': 0,
    'stop_row': 40,

    'Vin': 1.0,  # input voltage in V
    'frequency_range': np.arange(1, 6.1, 0.5),  # frequency sweep in MHz
    # 'bias_range': -1 * np.geomspace(1, 80, 25),
    # 'bias': -80.0,   # bias voltage to apply in V
    'bias_limit': 0.000008,
    'bias_sense_range': 0.00001,
    'bias_hv_limit': 0.000008,

    'data_path': "Thesis/ATLAS_ITk/X7",
    "out_file_mode": "append",
}


def perform_initial_total_measurement(cli_args):
    global scan_configuration
    with lock:
        scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS] = 30
        scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, MAXIMUM_FULL_SCAN_FREQ, FULL_SCAN_FREQ_STEP)

    if cli_args.total_unbiased:
        print("Will perform the unbiased total cap scan.")
        with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
            pix.pixcap.frequency_settling = cli_args.freq_settle
            with pix.binary_readout_mode() as binary_pix:
                binary_pix.scan(data_group_spec="unbiased_full")

    with lock:
        del scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS]
        try:
            time.sleep(cli_args.wait)
            if cli_args.total_unbiased:
                shutil.copyfile(output_file, output_file.replace(".h5", "_1.h5"))
        except:
            pass


def perform_iv_measurement(cli_args):
    global scan_configuration
    with lock:
        scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] = -1 * np.arange(0, MAXIMUM_HV_VOLTAGE, HV_STEP)
        scan_configuration[ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS] = 10
        assert np.all(0 >= scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE])
    if cli_args.iv_flag:
        print("Perform IV Scan.")
        with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
            pix.bias_scan(data_group_spec="I_V_Characteristic")

    with lock:
        del scan_configuration[ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS]
        try:
            time.sleep(cli_args.wait)
            if cli_args.iv_flag:
                shutil.copyfile(output_file, output_file.replace(".h5", "_2.h5"))
        except:
            pass


def perform_coarse_cv_scan(cli_args, hv_range):
    global scan_configuration

    with lock:
        assert np.all(scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] < 0)

    if cli_args.cv_general_flag and cli_args.cv_coarse_flag:
        print("Perform the coarse bias scan")
        with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
            pix.combined_bias_cv_scan(data_group_spec="C_V_Characteristic")

    with lock:
        try:
            time.sleep(cli_args.wait)
            if cli_args.cv_general_flag and cli_args.cv_coarse_flag:
                shutil.copyfile(output_file, output_file.replace(".h5", "_3.h5"))
        except:
            pass


def perform_fine_cv_scan(cli_args, hv_range):
    global scan_configuration

    with lock:
        scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] = hv_range
        scan_configuration.update(start_row=10, stop_row=35, start_column=10, stop_column=35)
        scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, MAXIMUM_CV_FREQ, CV_FREQ_STEP)
        assert np.all(scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] < 0)

    if cli_args.cv_general_flag and cli_args.cv_fine_flag:
        print("Perform the fine bias scan")
        with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
            pix.frequency_settling = cli_args.freq_settle
            pix.combined_bias_cv_scan(data_group_spec="C_V_Characteristic_refined")

    with lock:
        try:
            time.sleep(cli_args.wait)
            if cli_args.cv_general_flag and cli_args.cv_fine_flag:
                shutil.copyfile(output_file, output_file.replace(".h5", "_4.h5"))
        except:
            pass


def perform_biased_total_cap(cli_args):
    global scan_configuration
    with lock:
        if ScanConfigurationKeys.BIAS_VOLTAGE_RANGE in scan_configuration:
            del scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE]
        scan_configuration.update(start_row=1, stop_row=40, start_column=0, stop_column=40, bias=-1 * cli_args.hv_voltage,
                                  average_measurements=40)
        scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, MAXIMUM_FULL_SCAN_FREQ, FULL_SCAN_FREQ_STEP)
        scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS] = 30
        assert scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE] <= 0

    if cli_args.depleted_flag:
        print("Perform the full over-depleted scan.")
        with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
            pix.pixcap.frequency_settling = cli_args.freq_settle
            with pix.binary_readout_mode() as binary_pix:
                binary_pix.scan(data_group_spec="biased_{}_V_full".format(cli_args.hv_voltage))

    with lock:
        try:
            time.sleep(cli_args.wait)
            if cli_args.depleted_flag:
                shutil.copyfile(output_file, output_file.replace(".h5", "_5.h5"))
        except:
            pass
        finally:
            del scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS]


def perform_inter_pix_scan(cli_args):
    global scan_configuration

    with lock:
        scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, MAXIMUM_INTER_PIX_FREQ, INTER_PIX_FREQ_STEP)
        scan_configuration.update(start_row=10, stop_row=35, start_column=10, stop_column=35)
        if ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE in scan_configuration:
            del scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE]

    if cli_args.inter_pix_cap:
        with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.INTER_CAPACITANCE) as pix:
            pix.frequency_settling = cli_args.freq_settle
            pix.scan(data_group_spec="inter_unbiased_full")

    with lock:
        try:
            time.sleep(cli_args.wait)
            if cli_args.inter_pix_cap:
                shutil.copyfile(output_file, output_file.replace(".h5", "_6.h5"))
        except:
            pass
        finally:
            scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE] = -1 * cli_args.hv_voltage
            assert scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE] <= 0

    if cli_args.inter_pix_cap:
        with PixCapSetup(scan_configuration, output_file, measurement=PixcapMeasurements.INTER_CAPACITANCE) as pix:
            pix.frequency_settling = cli_args.freq_settle
            pix.scan(data_group_spec="inter_biased_M_{}_V_full".format(cli_args.hv_voltage))

    with lock:
        try:
            time.sleep(cli_args.wait)
            if cli_args.inter_pix_cap:
                shutil.copyfile(output_file, output_file.replace(".h5", "_7.h5"))
        except:
            pass


if __name__ == "__main__":
    import shutil
    import argparse
    # one setup trial per measurement
    output_file = "data/3D_Sensor_H23_S24_Full_Scan.h5"

    argument_parser = argparse.ArgumentParser("Pixcap65 Measurements")
    argument_parser.add_argument("-u", "--unbiased", action='store_true', dest='total_unbiased', help='Unbiased total capacitance measurement to perform')
    argument_parser.add_argument("-i", "--inter-pix", action='store_true', dest='inter_pix_cap', help='Measure the interpixel capacitances by both methods.')
    argument_parser.add_argument("-c", "--c-v-general", action='store_true', dest='cv_general_flag', help='Enable the measurement of the CV Characteristic')
    argument_parser.add_argument('--iv', action='store_true', dest='iv_flag', help='Enable the measurement of the IV Characteristic')
    argument_parser.add_argument("--wait", action='store', dest='wait', help='Wait before backing the measurement file up and stepping to the next measurement in the procedure.', default=5, type=float)
    argument_parser.add_argument("--hv_bias", action='store', dest='hv_voltage', default=80, type=float, help='HV bias voltage to use for biased measurements.')
    argument_parser.add_argument('--settling', action='store', dest='freq_settle', default=0.3, type=float, help='Settling time for the frequency before measurements.')
    argument_parser.add_argument('--depleted', action='store_true', dest='depleted_flag', help='Full sensor scan in over-depleted state.')
    argument_parser.add_argument('--c-v-fine', action='store_true', dest='cv_fine_flag', help='Enable the fine grid measurement of the CV Characteristic.')
    argument_parser.add_argument('--c-v-coarse', action='store_true', dest='cv_coarse_flag', help='Enable the coarse grid measurement of the CV Characteristic.')
    argument_parser.add_argument('-v', '--verbose', action='store_true', dest='verbose_flag', help='Enable verbose mode.')

    arguments = argument_parser.parse_args()

    print(type(arguments.wait))
    print(arguments.wait)
    print(type(arguments.hv_voltage))
    print(arguments.hv_voltage)
    print(type(arguments.freq_settle))
    print(arguments.freq_settle)
    print("biased_{}_V_full".format(arguments.hv_voltage))
    print("inter_biased_M_{}_V_full".format(arguments.hv_voltage))

    if arguments.verbose_flag:
        logging.root.setLevel(logging.DEBUG)

    # initial measurement sample
    perform_initial_total_measurement(arguments)
    time.sleep(600)

    # I-V characterization!
    perform_iv_measurement(arguments)
    time.sleep(600)

    # C-V characterization sample
    coarse_bias_range = -1 * np.arange(0.1, MAXIMUM_HV_VOLTAGE, HV_STEP)
    # fine grid for HPK sensors.
    # fine_bias_range = -1 * np.unique(np.concat((
    #     np.geomspace(0.1, 1, 8),
    #     np.geomspace(1, 15, 14),
    #     np.arange(15, 50.1, 5),
    #     np.geomspace(50,150, 26),
    #     np.arange(150, 350,10)
    # )))
    # fine grid sintef
    fine_bias_range = -1 * np.unique(np.concat((
        np.geomspace(0.1, 30, 30),
        np.geomspace(30, 58, 41)
    )))
    # fine grid R11/R13/R1
    # fine_bias_range = -1 * np.unique(np.concat((
    #     np.geomspace(0.1, 30, 20),
    #     np.arange(30, 160.1, 2)
    # )))
    # noqa: S125
    scan_configuration[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE] = coarse_bias_range
    scan_configuration.update(start_row=20, stop_row=25, start_column=20, stop_column=25)
    scan_configuration[ScanConfigurationKeys.FREQUENCY_RANGE] = np.arange(1, MAXIMUM_CV_FREQ, CV_FREQ_STEP)
    if ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS in scan_configuration:
        del scan_configuration[ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS]
    if ScanConfigurationKeys.AVERAGE_MEASUREMENTS in scan_configuration:
        del scan_configuration[ScanConfigurationKeys.AVERAGE_MEASUREMENTS]

    # coarse scan
    # noqa: S125
    perform_coarse_cv_scan(arguments, coarse_bias_range)
    time.sleep(600)

    # fine scan
    perform_fine_cv_scan(arguments, fine_bias_range)
    time.sleep(600)

    # completing scan
    perform_biased_total_cap(arguments)
    time.sleep(600)

    # perform the inter-pixel cap scans
    perform_inter_pix_scan(arguments)
    time.sleep(600)
