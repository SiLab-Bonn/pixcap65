"""
The latest version of the Pixcap65 test script for measuring the total pixel capacitance.

Changes compared to original script:
- Remote control of depletion voltage source
- Reading some current values before actual measurement to avoid incorrect currents due to initial oscillation
    effects of SMU
- Vary the order of column/row routing and switching frequency using the reversed arrays (uncomment corresponding
    lines in code)
- Fit also returns covariance matrix in order to extract the errors of the fit parameters if needed
- Output in txt file also includes offset (y-intercept) next to the slope

Changes compared to first/second modification:
- packaged the measurement of the total pixel capacitance into a class hierarchy (introduced a super class common
    to the different measurement procedures
- enabled the option to measure multiple currents and average over these to obtain an estimator for the currents
    standard error
- automatic error estimation by using information from the SMUs manual
"""

import logging
import os
import time
from collections import OrderedDict
from enum import StrEnum

import numpy as np
import tables as tb
import yaml
from bitarray import bitarray

import pixcap65_constants as c
from analysis import advanced_analysis_delegate
from configs.config_handler import extract_smu_current_error
from pixcap65 import Pixcap65, BasilConfigKeys
from plotting import plot_data_delegate
from utils_2 import walk_to_node

# constants for structuring of config readouts.
TOTAL_CAP_SEQ_SIZE = 4
TOTAL_CAP_SEQ_SIZE_KEY = "sequence_size"
NUMBER_AVERAGE_MEASUREMENTS_KEY = "average_measurements"
MEASURING_PIXEL_TEXT = 'Measuring pixel (%i, %i)...'

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_handler = logging.FileHandler('pixcap_65_test.log')
log_formater = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
log_handler.setFormatter(log_formater)
logger.addHandler(log_handler)


def store_scan_par_values(scan_parameters, scan_param_id, **kwargs):
    """
        Manually store the scan parameter values for the scan parameter id
        This allows to reconstruct the scan parameter values for a given parameter state vector
    """
    if scan_parameters.get(scan_param_id) and scan_parameters.get(scan_param_id) != kwargs:
        raise ValueError('You cannot change the scan parameter value of a scan parameter id')
    scan_parameters[scan_param_id] = kwargs


def _store_scan_par_values(h5_file, scan_parameters, group: tb.Group = None, **kwargs):
    """
        Create scan_params table after a scan
    """
    if group is None:
        group = h5_file.root
    # Create parameter description
    keys = set()  # find all keys to make the table column names
    for par_values in scan_parameters.values():
        keys.update(par_values.keys())
    fields = [('scan_param_id', np.uint32)]
    # FIXME only float32 supported so far
    # float64 should be available now
    fields.extend([(name, np.float64) for name in keys])
    # fields.extend([(name, np.float32) for name in keys])

    scan_par_table = h5_file.create_table(group, name='scan_params',
                                          title='Scan parameter values per scan parameter id',
                                          description=np.dtype(fields))
    for par_id, par_values in scan_parameters.items():
        a = np.full(shape=(1,), fill_value=np.nan).astype(np.dtype(fields))
        for key, val in par_values.items():
            a['scan_param_id'] = par_id
            a[key] = np.float32(val)
        scan_par_table.append(a)


class ScanConfigurationKeys(StrEnum):
    START_COLUMN = "start_column"
    STOP_COLUMN = "stop_column"
    START_ROW = "start_row"
    STOP_ROW = "stop_row"
    AVERAGE_MEASUREMENTS = NUMBER_AVERAGE_MEASUREMENTS_KEY
    VIN = "Vin"
    FREQUENCY_RANGE = "frequency_range"
    BIAS_VOLTAGE_RANGE = "bias_range"


scan_configuration = {
    'start_column': 20,
    'stop_column': 30,
    'start_row': 20,
    'stop_row': 30,
    # "average_measurements": 10,

    'Vin': 1.0,  # input voltage in V
    'frequency_range': np.arange(1, 12.1, 0.5),  # frequency sweep in MHz
    'bias_range': -1 * np.arange(1, 101.1, 0.5)
}

class BiasTable(tb.IsDescription):
    U = tb.Float32Col()
    I = tb.Float32Col()
    DI = tb.Float32Col()


class PixCap65Measurement(object):
    __n_measurements = 1
    smu_range_config = {}
    __group: tb.Group = None

    def __init__(self, scan_config, output_file, pix_config="pixcap65.yaml", **kwargs):
        self.dut = Pixcap65(pix_config)
        self.dut.init()

        # handle smu error configuration
        adjusted_config = self.dut._conf.copy()
        self._environ_config = OrderedDict()

        rl_mapping = {}
        tl_mapping = {}
        hl_mapping = {}

        if BasilConfigKeys.TRANSFER_LAYER in adjusted_config:
            for idx, layer in enumerate(adjusted_config[BasilConfigKeys.TRANSFER_LAYER]):
                if "name" in layer:
                    tl_mapping[layer["name"]] = idx
                else:
                    logger.info("Transfer layer at %i has no name. Will skip it.", idx)

        if BasilConfigKeys.HARDWARE_LAYER in adjusted_config:
            for idx, layer in enumerate(adjusted_config[BasilConfigKeys.HARDWARE_LAYER]):
                if "name" in layer:
                    hl_mapping[layer["name"]] = idx
                else:
                    logger.info("Hardware driver at %i has no name. Will skip it.", idx)

        if BasilConfigKeys.REGISTER_LAYER in adjusted_config:
            for idx, layer in enumerate(adjusted_config[BasilConfigKeys.REGISTER_LAYER]):
                if "name" in layer:
                    rl_mapping[layer["name"]] = idx
                else:
                    logger.info("Register at %i has no name. Will skip it.", idx)

        smu_keys = [self.dut.primary_smu_key, self.dut.bias_smu_key, self.dut.vm1_smu_key, self.dut.vm2_smu_key,
                    self.dut.vm3_smu_key]
        for smu in smu_keys:
            if smu in rl_mapping:
                hw_driver = self.dut._conf[BasilConfigKeys.REGISTER_LAYER][rl_mapping[smu]]['hw_driver']
                hl = self.dut._conf[BasilConfigKeys.HARDWARE_LAYER][hl_mapping[hw_driver]]
            elif smu in hl_mapping:
                hl = self.dut._conf[BasilConfigKeys.HARDWARE_LAYER][hl_mapping[smu]]
            else:
                raise AttributeError("The requested SMU {} does not exist.".format(smu))
            device_name = hl["init"]['device'].lower().replace(' ', '_')
            config_file = os.path.join(os.path.dirname(__file__), "configs",
                                       "{name}_range.yaml".format(name=device_name))
            assert os.path.exists(config_file) and os.path.isfile(config_file)
            print("Using the SMU range config file: ", config_file, " for the smu ", smu)
            with open(config_file, "r") as f:
                self.smu_range_config[smu] = yaml.safe_load(f)

        self.scan_config = scan_config

        self.output_file = output_file
        self.out_file_h5 = tb.open_file(self.output_file, mode='w')
        self.__group = self.out_file_h5.root

        self.scan_parameters = OrderedDict()

        self.n_frequencies = len(scan_config[ScanConfigurationKeys.FREQUENCY_RANGE])

    def update_config(self, scan_config):
        self.scan_config = scan_config
        self.n_frequencies = len(scan_config[ScanConfigurationKeys.FREQUENCY_RANGE])

    def configure(self):
        raise NotImplementedError

    def scan(self):
        raise NotImplementedError

    def analyze(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def close(self):
        self.out_file_h5.close()
        self.smu_off()
        self.set_bias_off()
        self.dut.close()

    @property
    def pixcap(self):
        return self.dut

    @property
    def seq_size(self):
        # granularity of the clock sequencer
        return self.pixcap.seq_size

    @property
    def filters(self):
        return tb.Filters(complib='blosc', complevel=5, fletcher32=False)

    def __enter__(self):
        try:
            self.configure()
        except:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        print("Exited from the pixcap chip!")
        return False

    @property
    def current_sense_range(self):
        return 0.000001

    # Handle the SMU!
    # these will now just forward the commands to the pixcap object
    def init_smu(self, voltage_range=1.5, current_limit=0.001, plc=10):
        self.pixcap.init_smu(self.scan_config[ScanConfigurationKeys.VIN], self.current_sense_range, voltage_range,
                             current_limit, plc)

    def smu_on(self):
        self.pixcap.smu_on()

    def get_source_current(self) -> float:
        return self.pixcap.get_source_current

    def get_source_current_multiple(self, n: int):
        return self.pixcap.get_source_current_multiple(n)

    def smu_off(self):
        self.pixcap.smu_off()

    # Handle the biasing supply
    def init_bias_voltage(self, voltage: float = -80.0):
        self.pixcap.init_bias(voltage)

    def set_bias_on(self):
        self.pixcap.bias_on()

    def set_bias_off(self):
        self.pixcap.bias_off()

    def set_bias_voltage(self, voltage: float):
        self.pixcap.bias_voltage = voltage

    @property
    def base_group(self):
        return self.__group

    @base_group.setter
    def base_group(self, value):
        return walk_to_node(self.out_file_h5.root, value, create=True)

    @property
    def smu_kwargs(self):
        return self.pixcap.smu_kwargs

    @property
    def analysis_group(self):
        return self.out_file_h5.root

    @property
    def measurement_group(self):
        return self.out_file_h5.root

    @property
    def n_measurements(self):
        return self.__n_measurements

    @n_measurements.setter
    def n_measurements(self, value):
        if value == -1:
            self.pixcap["SMU"].set_number_measurements(1, **self.pixcap.smu_kwargs)
        else:
            self.pixcap["SMU"].set_number_measurements(value, **self.pixcap.smu_kwargs)
        self.__n_measurements = value
        print("Set a new measure counter")
        print(value, self.__n_measurements)


class PixCap65TotalCap(PixCap65Measurement):
    current_smu_config = {}

    def __init__(self, scan_config, output_file, **kwargs):
        super(PixCap65TotalCap, self).__init__(scan_config, output_file, **kwargs)

        if "double_sweep" in scan_config and scan_config["double_sweep"]:
            self.n_frequencies *= 2
        self.hist_current = np.full(shape=(40, 40, self.n_frequencies),
                                    fill_value=np.nan)  # current value for each measured frequency per pixel
        self.n_measurements = scan_config.get(NUMBER_AVERAGE_MEASUREMENTS_KEY, 8)
        self.hist_individual_currents = np.full(shape=(40, 40, self.n_frequencies, self.n_measurements),
                                                fill_value=np.nan)
        if NUMBER_AVERAGE_MEASUREMENTS_KEY not in scan_config or scan_config[NUMBER_AVERAGE_MEASUREMENTS_KEY] < 1:
            self.n_measurements = -1
        self.pixcap.seq_size = self.scan_config.get(TOTAL_CAP_SEQ_SIZE_KEY, TOTAL_CAP_SEQ_SIZE)
        self.hist_current_errors = np.full(shape=(40, 40, self.n_frequencies), fill_value=np.nan)
        self.mode_logging_text = 'Scan pixel by single measurements.'
        self.handle_measurement = self._handle_single_measurement
        self.handle_bias_measurement = self._handle_single_measurement_bias
        self.bias_scan_parameters = OrderedDict()

    def configure(self):
        # settings for sensor depletion source
        self.pixcap.init_bias(voltage=-20, voltage_range=1000, current_range=0.000001, current_limit=0.000001)
        self.pixcap[self.pixcap.bias_smu_key].drain_error_queue()
        self.pixcap.bias_voltage = -1
        self.pixcap[self.pixcap.bias_smu_key].select_data_format()
        # self.pixcap.bias_on()
        # time.sleep(10)
        # self.pixcap[self.pixcap.bias_smu_key].set_number_triggers(1)
        # self.pixcap[self.pixcap.bias_smu_key].set_number_measurements(50)
        # self.pixcap[self.pixcap.bias_smu_key].set_number_triggers(50)
        # start = time.time()
        # self.pixcap[self.pixcap.bias_smu_key].multi_current_measurement()
        # first_2410 = self.pixcap[self.pixcap.bias_smu_key].get_multi_current()
        # print(f"It take {time.time() - start} seconds.")
        # print(self.pixcap[self.pixcap.bias_smu_key].get_trigger_delay())
        # print(len(first_2410.split(',')))
        # print(first_2410[1::2])
        # self.pixcap.bias_off()
        self.pixcap[self.pixcap.bias_smu_key].set_number_triggers(1)
        if self.n_measurements == -1:
            self.pixcap[self.pixcap.bias_smu_key].set_number_measurements(1)
        else:
            self.pixcap[self.pixcap.bias_smu_key].set_number_measurements(self.n_measurements)
        self.pixcap[self.pixcap.bias_smu_key].drain_error_queue()
        self.init_smu()
        self.pixcap["SMU_SETUP"].drain_error_queue()
        print("fetch some configurations from the smu!")
        print(self.pixcap[self.pixcap.primary_smu_key].get_sense_interval())
        print(self.pixcap["SMU_SETUP"].get_lan_config_method())
        print(self.pixcap["SMU_SETUP"].get_lan_ip_adress())

        self.dut['SEQ'].reset()
        self.dut['SEQ'].set_clk_divide(1)
        self.dut['SEQ'].set_repeat_start(0)
        self.dut['SEQ'].set_repeat(0)
        self.dut['SEQ'].set_size(self.seq_size)
        self.dut['SEQ']['CLK_0'][0:self.seq_size - 1] = bitarray('1000')
        self.dut['SEQ']['CLK_3'][0:self.seq_size - 1] = bitarray('0010')
        # self.dut['SEQ']['CLK_1'][0:self.seq_size - 1] =  bitarray('00000000000111111110')
        # self.dut['SEQ']['CLK_2'][0:self.seq_size - 1] =  bitarray('01111111110000000000')
        self.dut['SEQ'].write()
        self.dut['SEQ'].start()

        self.smu_on()

        # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
        logging.debug('Waiting for settling of SMU...')
        if self.n_measurements == -1:
            temp_measurements = 1
        else:
            temp_measurements = self.n_measurements
        self.pixcap[self.pixcap.bias_smu_key].set_number_measurements(1)
        self.pixcap.bias_on()
        for _ in range(0, 30):
            current = self.get_source_current()
            self.pixcap.bias_measure_current()
            logging.debug('Current: {}'.format(current))
            time.sleep(1)
        self.pixcap[self.pixcap.bias_smu_key].set_number_measurements(temp_measurements)
        self.pixcap.bias_off()

        # changed to simplify changes in the used SMU
        self.get_source_current()

        # update the scan config parameters
        if "data_path" in self.scan_config:
            self.base_group = self.scan_config["data_path"]

    def scan(self, data_group_spec=None):
        # select the group to write the analysis results to
        if data_group_spec is not None and isinstance(data_group_spec, str):
            data_group = walk_to_node(self.base_group, data_group_spec, create=True)
        else:
            data_group = self.out_file_h5.root

        if isinstance(data_group_spec, tb.Node):
            data_group = data_group_spec
        else:
            data_group = walk_to_node(data_group, "total_cap/measurements", create=True)
        data_group._f_setattr('frequencies', self.n_frequencies)

        row_range = range(self.scan_config[ScanConfigurationKeys.START_ROW],
                          self.scan_config[ScanConfigurationKeys.STOP_ROW])
        col_range = range(self.scan_config[ScanConfigurationKeys.START_COLUMN],
                          self.scan_config[ScanConfigurationKeys.STOP_COLUMN])
        frequency_range = self.scan_config[ScanConfigurationKeys.FREQUENCY_RANGE]

        # perform also a down sweep in frequency
        if "double_sweep" in self.scan_config and self.scan_config["double_sweep"]:
            frequency_range = np.concatenate((frequency_range, np.flip(frequency_range)))

        self.pre_scan_handler()
        logging.info(self.mode_logging_text)
        logger.info(self.mode_logging_text)
        continue_error = None
        try:
            for i_row in row_range:
                for i_col in col_range:
                    logging.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
                    logger.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
                    self.dut.disable_all_pixels()
                    self.dut.disable_all_columns()

                    self.dut.enable_column(i_col, c.EN_EOC_3)
                    self.dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)

                    for k, freq in enumerate(frequency_range):
                        self.pixcap.cvm_frequency = freq
                        # freq_conv = freq * self.seq_size
                        # self.dut['MIO_PLL'].setFrequency(freq_conv)
                        # time.sleep(1)

                        self.handle_measurement(i_col, i_row, k)
                        store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq)
        except KeyboardInterrupt as e:
            continue_error = e
            continue_saving_operation = True
        except:
            continue_saving_operation = False
            raise
        else:
            continue_saving_operation = True
        if continue_saving_operation:
            self.post_scan_handler(group=data_group)

            # if "average_measurements" in self.scan_config and self.scan_config["average_measurements"] > 1:
            #     logging.info(self.mode_logging_text)
            #     logger.info(self.mode_logging_text)
            #     for i_row in row_range:
            #         for i_col in col_range:
            #             logging.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
            #             logger.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
            #             self.dut.disable_all_pixels()
            #             self.dut.disable_all_columns()
            #
            #             self.dut.enable_column(i_col, c.EN_EOC_3)
            #             self.dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)
            #
            #             for k, freq in enumerate(frequency_range):
            #                 freq_conv = freq * self.seq_size
            #                 self.dut['MIO_PLL'].setFrequency(freq_conv)
            #                 time.sleep(1)
            #                 self.hist_individual_currents[i_col, i_row, k, :] = self.get_source_current_multiple(
            #                     self.n_measurements)[:]
            #                 store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq)
            #
            #     average_currents = np.nanmean(self.hist_individual_currents, axis=3, keepdims=True)
            #     self.hist_current = average_currents[:, :, :, 0]
            #     self.hist_current_errors = np.nanstd(self.hist_individual_currents, axis=3, mean=average_currents)
            # else:
            #     logging.info(self.mode_logging_text)
            #     logger.info(self.mode_logging_text)
            #     for i_row in row_range:
            #         for i_col in col_range:
            #             logging.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
            #             logger.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
            #             self.dut.disable_all_pixels()
            #             self.dut.disable_all_columns()
            ## select the group to write the analysis results to
            _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.scan_parameters, group=data_group)
            if np.all(np.isnan(self.hist_current)):
                raise ValueError("UNEXPECTED: All measurement entries are still NaN.")
            self.out_file_h5.create_carray(data_group,
                                           name='HistCurr',
                                           title='Current Histogram',
                                           obj=self.hist_current,
                                           filters=self.filters)

            # need the additional entries for the advanced averaging implementation
            if "average_measurements" in self.scan_config and self.scan_config["average_measurements"] > 1:
                self.out_file_h5.create_carray(data_group,
                                               name='HistCurrValues',
                                               title='Multiple Current Histogram',
                                               obj=self.hist_individual_currents,
                                               filters=self.filters,
                                               )

            if hasattr(self, "hist_current_errors") and not np.all(np.isnan(self.hist_current_errors)):
                self.out_file_h5.create_carray(data_group,
                                               name='HistCurrErr',
                                               title='Current Error Histogram',
                                               obj=self.hist_current_errors,
                                               filters=self.filters,
                                               )

        if continue_error is not None:
            self.out_file_h5.flush()
            raise continue_error

        # TODO: make it possible to directly export it also in a root tree.
        logging.info('Done')

    def bias_cv_scan(self, data_group_spec=None):
        assert "bias_range" in self.scan_config
        if data_group_spec is not None:
            data_group = walk_to_node(self.base_group, data_group_spec, create=True)
        else:
            data_group = self.out_file_h5.root

        data_group = walk_to_node(data_group, "biasing/measurements", create=True)
        bias_voltages = np.asarray(self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE])
        self.pixcap.bias_voltage = 0
        self.pixcap.bias_on()

        for bias_voltage in bias_voltages:
            self.pixcap.bias_voltage = bias_voltage
            bias_group_name = "bias_{bias_voltage}_V".format(bias_voltage=bias_voltage).replace('-', "M_").replace(".",
                                                                                                                   "__")
            if bias_group_name in data_group:
                # remove the biasing group
                scan_group = data_group[bias_group_name]
            else:
                scan_group = data_group._v_file.create_group(where=data_group, name=bias_group_name)
            logger.info("Perform sweep for bias voltage %f.", bias_voltage)
            self.scan_parameters = OrderedDict()
            self.scan(data_group_spec=scan_group)

        logger.info("Collect all the current information and package it together.")
        # save the bias voltages
        self.out_file_h5.create_carray(data_group, name="BiasVoltageHist", title="Histogram of chosen bias voltages",
                                       obj=bias_voltages, filters=self.filters)
        # TODO: implement this repackaging.
        # compacting will happen when analyzing the data!
        logger.info("Done bias measurements.")

    def bias_scan(self, data_group_spec=None):
        # select the group to write the analysis results to
        if data_group_spec is not None:
            data_group = walk_to_node(self.base_group, data_group_spec, create=True)
        else:
            data_group = self.out_file_h5.root

        data_group = walk_to_node(data_group, "biasing/measurements", create=True)

        # prepare the scan
        bias_voltages = np.asarray(self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE])
        self.n_voltages = bias_voltages.shape[0]
        data_group._f_setattr('voltages', self.n_voltages)
        self.hist_bias_current = np.full(shape=(self.n_voltages),
                                         fill_value=np.nan)  # current value for each measured frequency per pixel
        self.hist_bias_individual_currents = np.full(shape=(self.n_voltages, np.abs(self.n_measurements)),
                                                     fill_value=np.nan)
        self.hist_bias_current_errors = np.full(shape=(self.n_voltages), fill_value=np.nan)

        if "BiasVoltageHist" in data_group:
            data_group.remove("BiasVoltageHist")
            time.sleep(1)
        self.out_file_h5.create_carray(data_group, "BiasVoltageHist", obj=bias_voltages, filters=self.filters)

        self.pre_scan_handler(unit="bias")
        logging.info(self.mode_logging_text)
        logger.info(self.mode_logging_text)
        self.set_bias_voltage(0.)
        self.set_bias_on()
        self.dut.disable_all_pixels()
        self.dut.disable_all_columns()

        for k, bias_voltage in enumerate(bias_voltages):
            self.pixcap.bias_voltage = bias_voltage
            self.handle_bias_measurement(k)
            store_scan_par_values(scan_parameters=self.bias_scan_parameters, scan_param_id=k, bias_voltage=bias_voltage)

        self.post_scan_handler(unit="bias", group=data_group)
        _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.bias_scan_parameters, group=data_group)
        if np.all(np.isnan(self.hist_bias_current)):
            raise ValueError("UNEXPECTED: All measurement entries are still NaN.")
        self.out_file_h5.create_carray(data_group,
                                       name='HistCurr',
                                       title='Current Histogram',
                                       obj=self.hist_bias_current,
                                       filters=self.filters)

        # need the additional entries for the advanced averaging implementation
        if "average_measurements" in self.scan_config and self.scan_config["average_measurements"] > 1:
            self.out_file_h5.create_carray(data_group,
                                           name='HistCurrValues',
                                           title='Multiple Current Histogram',
                                           obj=self.hist_bias_individual_currents,
                                           filters=self.filters,
                                           )

        if hasattr(self, "hist_current_errors") and not np.all(np.isnan(self.hist_current_errors)):
            self.out_file_h5.create_carray(data_group,
                                           name='HistCurrErr',
                                           title='Current Error Histogram',
                                           obj=self.hist_bias_current_errors,
                                           filters=self.filters,
                                           )
        self.set_bias_off()

        logging.info('Done')

    def combined_bias_cv_scan(self, data_group_spec=None):
        # select the group to write the analysis results to
        if data_group_spec is not None:
            data_group = walk_to_node(self.base_group, data_group_spec, create=True)
        else:
            data_group = self.out_file_h5.root

        data_group = walk_to_node(data_group, "biasing/measurements", create=True)

        # prepare the scan
        bias_voltages = np.asarray(self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE])
        self.n_voltages = bias_voltages.shape[0]
        if "BiasVoltageHist" in data_group:
            data_group.remove("BiasVoltageHist")
            time.sleep(1)
        self.out_file_h5.create_carray(data_group, "BiasVoltageHist", obj=bias_voltages, filters=self.filters)
        data_group._f_setattr('voltages', self.n_voltages)
        self.hist_bias_current = np.full(shape=(self.n_voltages),
                                         fill_value=np.nan)  # current value for each measured frequency per pixel
        self.hist_bias_individual_currents = np.full(shape=(self.n_voltages, np.abs(self.n_measurements)),
                                                     fill_value=np.nan)
        self.hist_bias_current_errors = np.full(shape=(self.n_voltages), fill_value=np.nan)

        self.pre_scan_handler(unit="bias")
        logging.info(self.mode_logging_text)
        logger.info(self.mode_logging_text)
        self.set_bias_voltage(0.)
        self.set_bias_on()
        self.dut.disable_all_pixels()
        self.dut.disable_all_columns()

        for k, bias_voltage in enumerate(bias_voltages):
            self.pixcap.bias_voltage = bias_voltage
            self.scan_parameters = OrderedDict()
            self.handle_bias_measurement(k)
            store_scan_par_values(scan_parameters=self.bias_scan_parameters, scan_param_id=k, bias_voltage=bias_voltage)
            bias_group_name = "bias_{bias_voltage}_V".format(bias_voltage=bias_voltage).replace('-', "M_").replace(".",
                                                                                                                   "__")
            if bias_group_name in data_group:
                # remove the biasing group
                scan_group = data_group[bias_group_name]
            else:
                scan_group = data_group._v_file.create_group(where=data_group, name=bias_group_name)
            logger.info("Perform sweep for bias voltage %f.", bias_voltage)
            self.scan(data_group_spec=scan_group)

        self.post_scan_handler(unit="bias", group=data_group)
        _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.bias_scan_parameters, group=data_group)
        if np.all(np.isnan(self.hist_current)):
            raise ValueError("UNEXPECTED: All measurement entries are still NaN.")
        self.out_file_h5.create_carray(data_group,
                                       name='HistCurr',
                                       title='Current Histogram',
                                       obj=self.hist_bias_current,
                                       filters=self.filters)

        # need the additional entries for the advanced averaging implementation
        if "average_measurements" in self.scan_config and self.scan_config["average_measurements"] > 1:
            self.out_file_h5.create_carray(data_group,
                                           name='HistCurrValues',
                                           title='Multiple Current Histogram',
                                           obj=self.hist_bias_individual_currents,
                                           filters=self.filters,
                                           )

        if hasattr(self, "hist_current_errors") and not np.all(np.isnan(self.hist_current_errors)):
            self.out_file_h5.create_carray(data_group,
                                           name='HistCurrErr',
                                           title='Current Error Histogram',
                                           obj=self.hist_bias_current_errors,
                                           filters=self.filters,
                                           )
        self.set_bias_off()

        logging.info('Done')

    def plot(self):
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(self.output_file[:-3] + '.pdf') as output_pdf:
            plot_data_delegate(self.out_file_h5.root, self.out_file_h5.root, output_pdf)

    def analyze(self):
        advanced_analysis_delegate(self.out_file_h5, self.out_file_h5.root, self.out_file_h5.root)

    def post_scan_handler(self, unit=None, **kwargs):
        if "average_measurements" in self.scan_config and self.scan_config["average_measurements"] > 1:
            average_currents = np.nanmean(self.hist_individual_currents, axis=3, keepdims=True)
            self.hist_current = average_currents[:, :, :, 0]
            self.hist_current_errors = np.nanstd(self.hist_individual_currents, axis=3, mean=average_currents)
            if unit == "bias":
                average_currents = np.nanmean(self.hist_bias_individual_currents, axis=1, keepdims=True)
                self.hist_bias_current = average_currents[:, 0]
                self.hist_bias_current_errors = np.nanstd(self.hist_bias_individual_currents, axis=1,
                                                          mean=average_currents)
        else:
            if not np.all(np.isnan(self.hist_current)):
                try:
                    self.hist_current_errors = extract_smu_current_error(self.current_smu_config, self.hist_current,
                                                                         self.current_sense_range)
                except Exception as e:
                    logging.error(e.args)
                    logging.exception("Something went wrong during the estimation of the measurement errors.")
            if unit == "bias":
                try:
                    self.hist_bias_current_errors = extract_smu_current_error(
                        self.smu_range_config[self.pixcap.bias_smu_key], self.hist_bias_current,
                        self.current_sense_range)
                except Exception as e:
                    logging.error(e.args)
                    logging.exception("Something went wrong during the estimation of the measurement errors.")

        if unit == "bias" and "group" in kwargs:
            data_group = kwargs["group"]
            # fields = [("U", np.float64), ("I", np.float64), ("DI", np.float64)]
            table = self.out_file_h5.create_table(data_group, name="BiasTable", description=BiasTable,
                                                  filters=self.filters)
            entry = table.row
            for i in range(self.n_voltages):
                try:
                    entry['U'] = self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE][i]
                    entry['I'] = self.hist_bias_current[i]
                    entry['DI'] = self.hist_bias_current_errors[i]
                    # table.append([(self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE][i], self.hist_bias_current[i],
                    #                self.hist_bias_current_errors[i])])
                    entry.append()
                except ValueError as e:
                    print("currents")
                    print(self.hist_bias_current.shape)
                    print(self.hist_bias_current[i])
                    print("Errors")
                    print(self.hist_bias_current_errors.shape)
                    print(self.hist_bias_current_errors[i])
                    raise e

    def pre_scan_handler(self, unit=None):
        if NUMBER_AVERAGE_MEASUREMENTS_KEY in self.scan_config and self.scan_config[
            NUMBER_AVERAGE_MEASUREMENTS_KEY] > 1:
            self.n_measurements = self.scan_config[NUMBER_AVERAGE_MEASUREMENTS_KEY]
            # Added for convenience of the averaged measurements.
            # self.pixcap['SMU'].set_number_measurements(n_measurements, **self.smu_kwargs)
            individual_currents_shape = (40, 40, self.n_frequencies, self.n_measurements)
            if self.hist_individual_currents.shape != individual_currents_shape:
                self.hist_individual_currents = np.full(shape=(40, 40, self.n_frequencies, self.n_measurements),
                                                        fill_value=np.nan)
            if unit == "bias":
                individual_currents_shape = (self.n_voltages, self.n_measurements)
                if self.hist_bias_individual_currents.shape != individual_currents_shape:
                    self.hist_bias_individual_currents = np.full(shape=(self.n_voltages, self.n_measurements),
                                                                 fill_value=np.nan)

            self.mode_logging_text = "Average over multiple measurements!"
            self.handle_measurement = self._handle_averaged_measurement
            self.handle_bias_measurement = self._handle_averaged_measurement_bias
            self.current_smu_config = self.smu_range_config[self.pixcap.primary_smu_key]

        else:
            self.mode_logging_text = 'Scan pixel by single measurements.'
            self.handle_measurement = self._handle_single_measurement
            self.handle_bias_measurement = self._handle_single_measurement_bias
            self.current_smu_config = self.smu_range_config[self.pixcap.primary_smu_key]

    def _handle_single_measurement(self, col, row, k):
        current = self.get_source_current()
        self.hist_current[col, row, k] = current
        if k == 0:
            logging.debug('%f' % current)
            logger.debug('%f' % current)

    def _handle_averaged_measurement(self, col, row, k):
        self.hist_individual_currents[col, row, k, :] = self.pixcap.get_advanced_current_multiple(
            self.n_measurements)[:]

    def _handle_single_measurement_bias(self, k):
        current = self.pixcap.bias_measure_current()
        assert not np.isnan(current), "The bias measurement was not performed."
        self.hist_bias_current[k] = current
        if k == 0:
            logging.debug('%g' % current)
            logger.debug('%g' % current)

    def _handle_averaged_measurement_bias(self, k):
        measurement = self.pixcap.bias_advanced_current_multiple(
            self.n_measurements)[:]
        assert not np.all(np.isnan(measurement)), "The bias measurement was not performed."
        try:
            self.hist_bias_individual_currents[k, :] = measurement[1::2]
        except:
            print(k, self.n_measurements)
            print(measurement.shape)
            raise

    @property
    def current_sense_range(self):
        return 0.00001


if __name__ == '__main__':
    output_file = "./TEST.h5"
    # try:
    #     pix = PixCap65TotalCap(scan_configuration, output_file)
    #     pix.configure()
    #     pix.scan()
    # finally:
    #     pix.close()

    output_file_2 = "./R13_BIAS_CV_COMBI_6.h5"
    from plotting import plot_combined_data
    with PixCap65TotalCap(scan_configuration, output_file_2) as pix:
        pix.scan()
        pix.bias_scan()
        pix.bias_cv_scan()
        print(pix.out_file_h5)
        pix.combined_bias_cv_scan()

    # Analyse and plot data
    # analyze_data(output_file)
    # plot_data(output_file)

    # analyze_data(output_file_2)
    # plot_data(output_file_2)
    # plot_bias_data(output_file_2)
    # analyze_data(output_file_2, is_cv=True)
    # plot_cv_data(output_file_2)
    plot_combined_data(output_file_2, first_lower=-100, first_upper=-50, second_lower=-8, second_upper=0)
