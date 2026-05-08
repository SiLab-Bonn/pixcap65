"""
The latest version of the Pixcap65 test script for measuring the total pixel capacitance.

Changes compared to original script:
- Remote control of depletion voltage source
- Reading some current values before actual measurement to avoid incorrect currents due to initial
oscillation effects of SMU
- Vary the order of column/row routing and switching frequency using the reversed arrays
(uncomment corresponding lines in code)
- Fit also returns covariance matrix in order to extract the errors of the fit parameters if needed
- Output in txt file also includes offset (y-intercept) next to the slope

Changes compared to first/second modification:
- packaged the measurement of the total pixel capacitance into a class hierarchy (introduced a super class common
to the different measurement procedures
- enabled the option to measure multiple currents and average over these to obtain an estimator for the currents
standard error
- automatic error estimation by using information from the SMUs manual
"""

from __future__ import annotations

import logging
import os
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from enum import StrEnum
from typing import Iterable, Mapping, Any
from warnings import warn, deprecated

import gc
import numpy as np
import tables as tb
import time
import yaml
from numpy import ndarray
from tqdm import tqdm
# noinspection PyProtectedMember
from tqdm.contrib import DummyTqdmFile

from pixcap65.analysis import analysis_data_handle
from pixcap65.analysis_util.utility import HIST_CURRENT_MEAS_UNIT, HIST_BIAS_MEAS_UNIT
from pixcap65.configs.config_handler import extract_smu_current_error
from pixcap65.pixcap.pixcap65 import Pixcap65
from pixcap65.pixcap.pixcap_structure import BasilConfigKeys
from pixcap65.plotting import plot_data_delegate
from pixcap65.utility import pixcap65_constants as c
from pixcap65.utility.tables_util import get_group_attributes, set_group_attribute, \
    get_group_attribute, get_children, group_get_file, rename_node
from pixcap65.utility.tqdm_logging_utils import logging_redirect_tqdm
from pixcap65.utility.utils_2 import walk_to_node, prevent_group_mix_up

# constants for structuring of config readouts.
TOTAL_CAP_SEQ_SIZE = 4
TOTAL_CAP_SEQ_SIZE_KEY = "sequence_size"
NUMBER_AVERAGE_MEASUREMENTS_KEY = "average_measurements"
BIASING_NUMBER_AVERAGE_MEASUREMENTS_KEY = "bias_average_measurements"
MEASURING_PIXEL_TEXT = 'Measuring pixel (%i, %i)...'
UNCERT_ESTIMATION_ERROR_MSG = "Something went wrong during the estimation of the measurement errors."

BACKING_FORMAT_TEXT = "{old}_backing"
HV_WAIT = 0.01
HV_VOLTAGE_TOL = 1e-2
HV_CURRENT_LIMIT = 1e-7

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_handler = logging.FileHandler('pixcap_65_test.log')
log_formater = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
log_handler.setFormatter(log_formater)
logger.addHandler(log_handler)
logger.propagate = True


def store_scan_par_values(scan_parameters, scan_param_id, **kwargs):
    """
        Manually store the scan parameter values for the scan parameter id
        This allows to reconstruct the scan parameter values for a given parameter state vector
    """
    if scan_parameters.get(scan_param_id) and scan_parameters.get(scan_param_id) != kwargs:
        raise ValueError('You cannot change the scan parameter value of a scan parameter id')
    scan_parameters[scan_param_id] = kwargs


def _store_scan_par_values(h5_file, scan_parameters, group: tb.Group = None):
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
    fields.extend([(name, np.float64) for name in keys])

    if "scan_params" in group:
        logging.info("Storing scan parameter values but found an already existing table; will rename it")
        new_name = "scan_params"
        while new_name in group:
            new_name = BACKING_FORMAT_TEXT.format(old=new_name)
        group.scan_params.rename(new_name)
        # group.scan_params.rename("scan_params_old")
        time.sleep(5)

    try:
        scan_par_table = h5_file.create_table(group, name='scan_params',
                                              title='Scan parameter values per scan parameter id',
                                              description=np.dtype(fields))
        for par_id, par_values in scan_parameters.items():
            a = np.full(shape=(1,), fill_value=np.nan).astype(np.dtype(fields))
            for key, val in par_values.items():
                a['scan_param_id'] = par_id
                a[key] = np.float32(val)
            scan_par_table.append(a)
    except:
        logging.error("Failed to create and fill the scan parameters table. But will continue anyway.")


class ScanConfigurationKeys(StrEnum):
    START_COLUMN = "start_column"
    STOP_COLUMN = "stop_column"
    START_ROW = "start_row"
    STOP_ROW = "stop_row"
    AVERAGE_MEASUREMENTS = NUMBER_AVERAGE_MEASUREMENTS_KEY
    BIAS_AVERAGE_MEASUREMENTS = BIASING_NUMBER_AVERAGE_MEASUREMENTS_KEY
    VIN = "Vin"
    FREQUENCY_RANGE = "frequency_range"
    BIAS_VOLTAGE_RANGE = "bias_range"
    BIAS_VOLTAGE_SINGLE = "bias"


scan_configuration = {
    'start_column': 20,
    'stop_column': 22,
    'start_row': 20,
    'stop_row': 22,
    # "average_measurements": 30,
    # "bias_average_measurements": 3,

    'Vin': 1.0,  # input voltage in V
    'frequency_range': np.arange(1, 12.1, 0.5),  # frequency sweep in MHz
    # 'bias_range': -1 * np.arange(1, 100.1, 0.5),
    # 'bias': -80.0,   # bias voltage to apply in V

    'data_path': "Reference/TESTS",
    "out_file_mode": "append",
}


class BiasTable(tb.IsDescription):
    U = tb.Float32Col()
    I = tb.Float32Col()
    DI = tb.Float32Col()


class PixCap65Measurement(object, metaclass=ABCMeta):
    # instantiation
    def __init__(self, scan_config, output_file, pix_config="pixcap65.yaml", **kwargs):
        self.smu_range_config = {}
        self.__group = None

        self.dut = Pixcap65(pix_config)
        self.dut.init()

        self.bias_measurements = scan_config.get(BIASING_NUMBER_AVERAGE_MEASUREMENTS_KEY, 1)

        # handle smu error configuration
        adjusted_config = self.dut._conf.copy()
        self._environ_config = OrderedDict()
        import sys
        self.dummy_file = DummyTqdmFile(sys.stdout)
        self.dummy_error_file = DummyTqdmFile(sys.stderr)

        # TODO: extract the handling of basil!
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
        if (os.path.exists(self.output_file) and "out_file_mode" in self.scan_config and
                self.scan_config["out_file_mode"] == "append"):
            self.out_file_h5 = tb.open_file(self.output_file, mode='a')
        else:
            self.out_file_h5 = tb.open_file(self.output_file, mode='w')
        self.__group = self.out_file_h5.root

        self.scan_parameters = OrderedDict()

        self.n_frequencies = len(scan_config[ScanConfigurationKeys.FREQUENCY_RANGE])

        self.has_bias_supply = self.pixcap.has_bias_suppy

    # general interface
    def configure(self):
        """
        configure

        Handling the configuration of the pixcap measurement object and the physical setup
        All information additionally required will be fetched from the scan configuration.
        Some of the configuration needs to be done subclass implementations as the different measurement types could
        have different requirements onto the setup.
        """
        # change the output file if necessary
        if "output_file" in self.scan_config and os.path.exists(self.scan_config["output_file"]):
            self.out_file_h5.close()
            self.out_file_h5 = tb.open_file(self.scan_config["output_file"], mode='a')

        # settings for sensor depletion source
        # to exact: this should be done by the config_update handler as it relais on configuration options!
        # will be evaluated before the update of configuration is taken into account!
        if self.use_bias_supply and self.has_bias_supply:
            logging.warning("Bias supply is now active.")
            logger.warning("Bias supply is now active.")
            self.pixcap.init_bias(voltage=-0.1, voltage_range=1000, current_range=self.bias_sense_range,
                                  current_limit=0.000000050)

            self._smu_setup(self.pixcap.bias_smu_key).drain_error_queue()
            self.pixcap.bias_voltage = -0.1  # need to go to a save voltage for the setup

            # in principle these are all things which should be done by the pixcap dut object
            self.pixcap[self.pixcap.bias_smu_key].select_data_format()
            self.pixcap[self.pixcap.bias_smu_key].set_number_triggers(1)
            # Why it should have its own setups from the configuration!
            self.pixcap.n_bias_measurements = self.n_measurements
            self._smu_setup(self.pixcap.bias_smu_key).drain_error_queue()
        else:
            self.has_bias_supply = False

        # init the primary smu or VM3
        self.init_smu()
        self._smu_setup(self.pixcap.primary_smu_key).drain_error_queue()
        logger.info("fetch some configurations from the smu!")
        print(self.pixcap[self.pixcap.primary_smu_key].get_sense_interval())
        print(self._smu_setup(self.pixcap.primary_smu_key).get_lan_config_method())
        print(self._smu_setup(self.pixcap.primary_smu_key).get_lan_ip_adress())

        # update the scan config parameters
        self.update_config()

    def close(self):
        """
        close


        closes all open file handles used by the measurement as well as all connections to SMUs or other lab
        devices. The SMUs are switched off before closing the connection.
        :return:
        """
        self.out_file_h5.close()
        self.smu_off()
        if self.has_bias_supply:
            self.set_bias_off()

        # finally close the pixcap dut.
        self.dut.close()

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

    # measurement interface
    @abstractmethod
    def scan(self, data_group_spec=None):
        """
        scan

        Performs the scan over the pixels on the sensor and measures the requested quantities in dependence on some
        other quantities. The implementation will strongly depend on the particular measurement type. Thus, it is
        necessary to override this method.
        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        """
        raise NotImplementedError

    def perform_bias_scan(self, bias_voltages: np.ndarray, post_handler: Callable, parameters):
        try:
            with logging_redirect_tqdm():
                for k, bias_voltage in enumerate(bias_voltages):
                    self.pixcap.bias_voltage = bias_voltage
                    # check for the smu's settling here
                    with self.bias_without_averaging() as hv_less:
                        previous_measurement = hv_less.bias_measure_volts()
                        time.sleep(HV_WAIT)
                        current_measurement = hv_less.bias_measure_volts()
                        for _ in range(100):
                            if np.abs(current_measurement - previous_measurement) < HV_VOLTAGE_TOL * np.abs(current_measurement):
                                break
                            previous_measurement = current_measurement
                            current_measurement = hv_less.bias_measure_volts()
                        else:
                            logger.warning("Could not stabilize the HV voltage.")

                        # stabilize the currents
                        try:
                            back_nlpc = float(hv_less[hv_less.bias_smu_key].get_current_nlpc())
                            hv_less[hv_less.bias_smu_key].set_current_nlpc(1)
                            previous_measurement = hv_less.bias_measure_current()
                            time.sleep(1e-3)
                            current_measurement = hv_less.bias_measure_current()
                            for i in range(100):
                                if np.abs(
                                        current_measurement - previous_measurement) < HV_CURRENT_STABLE_TOL * np.abs(current_measurement) or np.abs(current_measurement) > self.hv_limit:
                                    break
                                previous_measurement = current_measurement
                                current_measurement = hv_less.bias_measure_current()
                            else:
                                logger.warning("Could not stabilize the current.")
                        finally:
                            hv_less[hv_less.bias_smu_key].set_current_nlpc(back_nlpc)

                        if np.abs(current_measurement) > self.hv_limit:
                            self.pixcap.bias_voltage = -0.1
                            logger.error("The measured current %f has exceeded the protection limit %f.", current_measurement, self.hv_limit)
                            break

                        logger.debug("Set the bias voltage to %f." % bias_voltage)

                    yield k, bias_voltage
                    with self.bias_without_averaging() as hv_less:
                        actual_bias_voltage = hv_less.bias_measure_volts()
                    store_scan_par_values(scan_parameters=parameters, scan_param_id=k,
                                          bias_voltage=bias_voltage, hv_voltage=actual_bias_voltage)

        finally:
            post_handler()

    @abstractmethod
    def analyze(self):
        """
        analyze

        Perform the analysis of the data measured by the particular procedure.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        """
        plot

        creates a graphical representation of the measurement and analysis results of the particular measurement type
        implement by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def handle_measurement_errors(self, unit):
        pass

    @abstractmethod
    def store_measurement_data(self, data_group, sequence_call, unit=None):
        pass

    # interface
    def update_config(self, new_config=None):
        """
        update_config

        Update the currently loaded scan configuration with the keys provided by the mapping.
        In addition, to updating the configuration mapping, some of the configurations are directly applied to make sure
        that the setup is consistent with actually loaded scan configuration.

        :param new_config: mapping of the new configuration items.
        """
        if new_config is not None:
            self.scan_config.update(new_config)
        self.n_frequencies = len(self.scan_config[ScanConfigurationKeys.FREQUENCY_RANGE])

        # update the scan config parameters
        if "data_path" in self.scan_config:
            self.base_group = self.scan_config["data_path"]

    def create_carray(self, where: tb.Group | str, name: str, unit=None, **kwargs) -> tb.CArray:
        """
        create_carray

        Utility function to create the array with the measurement data, to store the data into it.


        The array will always be created within the hdf file owned by the measurement object.
        :param where: group where to store the data in
        :param name: name of the data set; it should be unique and a valid python identifier
        :key input: specifies the device used for input
        :param unit: specifies the unit used for input
        :param kwargs: further arguments for :ref: `pytables` implementation
        :type unit: str
        :return: if the array/data set could be created, the created array, None otherwise.
        """
        from pixcap65.utility.utils_2 import create_carray
        return create_carray(self.out_file_h5, where, name, unit=unit, **kwargs)

    def _smu_setup(self, smu):
        """
        smu_setup

        Fetch the control of the actual smu device instead of a channel based implementation using registers
        :param smu: SMU to fetch top-level control for.
        :return: Fetched top-level control hardware layer.
        """
        return self.pixcap[self.pixcap.smu_setup_devices[smu]]

    def store_configuration(self, data_group: tb.Node | tb.Group):
        for config_key, config_setting in self.scan_config.items():
            if isinstance(config_setting, Iterable) or isinstance(config_setting, Mapping):
                continue
            assert isinstance(data_group, tb.Group)
            logger.warning("The present keys for config attributes are: %s", str(get_group_attributes(data_group)))
            attr_config_key = "configuration_{}".format(config_key)
            assert isinstance(data_group, tb.Group)
            if (attr_config_key in get_group_attributes(data_group) and
                    get_group_attribute(data_group, attr_config_key) != config_setting):
                # if attr_config_key in data_group._v_attrs and data_group._v_attrs[attr_config_key] != config_setting:
                logger.error("Unexpectedly the configuration key is already present.")
                temp_key = attr_config_key
                while temp_key in get_group_attributes(data_group):
                    # while temp_key in data_group._v_attrs:
                    temp_key = "{}_backing".format(temp_key)
                set_group_attribute(data_group, temp_key, get_group_attribute(data_group, attr_config_key))
                # data_group._f_setattr(temp_key, data_group._f_getattr(attr_config_key))
            try:
                set_group_attribute(data_group, attr_config_key, config_setting)
                # data_group._f_setattr(attr_config_key, config_setting)
            except:
                logger.error("Failed to write scan configuration option %s as attribute.", config_key, exc_info=True)

    def set_bias_measurement(self, data_group: tb.Group, sequence_call: bool):
        if ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE in self.scan_config and not sequence_call:
            # prevent compliance on power-on
            self.pixcap.bias_voltage = -0.1
            self.pixcap.bias_on()
            self.pixcap.bias_voltage = float(self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE])
            try:
                set_group_attribute(data_group, "bias_voltage", self.pixcap.bias_voltage)
                # data_group._f_setattr("bias_voltage", self.pixcap.bias_voltage)
            except:
                logger.error("Failed to save bias voltage as an attribute", exc_info=True)
                set_group_attribute(data_group, "bias_voltage",
                                    self.pixcap.get_smu_source_voltage(self.pixcap.bias_smu_key))
                # data_group._f_setattr("bias_voltage", self.pixcap.get_smu_source_voltage(self.pixcap.bias_smu_key))

            for i in range(40):
                print(self.pixcap.get_smu_source_voltage(self.pixcap.bias_smu_key))
                print(i, self.pixcap.bias_measure_volts(), self.pixcap.bias_measure_current())
                print(np.isclose(self.pixcap.get_smu_source_voltage(self.pixcap.bias_smu_key),
                                 self.pixcap.bias_measure_volts()))

    def get_data_group(self, data_group_spec, particular_group) -> tb.Group:
        if data_group_spec is not None and isinstance(data_group_spec, str):
            data_group, _ = walk_to_node(self.base_group, data_group_spec, create=True, verify_create=True)
        else:
            data_group = self.base_group

        if isinstance(data_group_spec, tb.Node):
            data_group = data_group_spec
        else:
            assert isinstance(data_group, tb.Group)
            data_group, verify_creation = walk_to_node(data_group, "{}/measurements".format(particular_group),
                                                       create=True,
                                                       verify_create=True)
            if not verify_creation:
                for key, value in get_children(data_group):
                    # for key, value in data_group._v_children.items():
                    new_name = key
                    while new_name in data_group:
                        new_name = BACKING_FORMAT_TEXT.format(old=new_name)
                    rename_node(data_group, new_name)

        assert isinstance(data_group, tb.Group)
        return data_group

    def store_iteration_parameters(self, freq, k: int):
        if 'bias' in self.scan_config and self.has_bias_supply:
            store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq,
                                  bias_voltage=self.pixcap["BIAS_SUPPLY"].get_source_voltage())
        else:
            store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq)

    def determine_measurement_uncertainty(self, smu: str, temp_data: ndarray):
        sense_range = self.bias_sense_range if smu == self.pixcap.bias_smu_key else self.current_sense_range
        if np.any(np.isfinite(temp_data)):
            try:
                return np.where(np.isfinite(temp_data), extract_smu_current_error(
                    self.smu_range_config[smu], temp_data, sense_range), np.nan)
            except Exception as e:
                logging.error(e.args)
                logging.exception(UNCERT_ESTIMATION_ERROR_MSG)
        return np.full_like(temp_data, fill_value=np.nan)

    def handle_cv_compaction(self, kwargs, unit):
        # Template method, formerly abstract
        pass

    def post_scan_handler(self, data_group, sequence_call=False, unit=None, saving_unit=None, **kwargs):
        """
        post_scan_handler

        Utility function to be called after a scan has been performed to clean up, perform the uncertainty estimation
        and prepare the measurement results to be written back to file.

        This handler also performs the averaging over the individual measurements if this is necessary.

        :param saving_unit:
        :param sequence_call:
        :param data_group:
        :param unit: Additional unit to activate for the post scan analysis
        :param data_group: for biasing measurements write to this group in the hdf file. (table to be written)
        """
        try:
            self.handle_measurement_errors(unit)
            self.handle_cv_compaction(kwargs, unit)
            if saving_unit is None:
                set_group_attribute(data_group, "freq_unit", "MHz")
                # data_group._f_setattr("freq_unit", "MHz")
                set_group_attribute(data_group, "current_unit", HIST_CURRENT_MEAS_UNIT)
                # data_group._f_setattr("current_unit", HIST_CURRENT_MEAS_UNIT)
            elif saving_unit == "bias":
                set_group_attribute(data_group, "bias_current_unit", HIST_CURRENT_MEAS_UNIT)
                set_group_attribute(data_group, "bias_voltage_unit", HIST_BIAS_MEAS_UNIT)
            self.handle_store_configuration(data_group, sequence_call)
            self.store_measurement_data(data_group, sequence_call, unit=saving_unit)
        finally:
            self.out_file_h5.flush()
            # do some clean-up for the performance
            gc.collect()

    def handle_store_configuration(self, data_group: tb.Group, sequence_call: bool):
        if not sequence_call:
            self.store_configuration(data_group)
        else:
            logger.debug(
                "Sequence call encountered when writing the configuration options as attribute. Skip this.")

    @contextmanager
    def binary_readout_mode(self):
        try:
            self.pixcap.binary_active = True
            self._smu_setup(self.pixcap.primary_smu_key).binary_format()
            if self.n_measurements > 5:
                self.pixcap[self.pixcap.primary_smu_key].set_current_nlpc(2)
            yield self
        finally:
            self.pixcap[self.pixcap.primary_smu_key].set_current_nlpc(10)
            self._smu_setup(self.pixcap.primary_smu_key).text_format()
            self.pixcap.binary_active = False

    # region Pixcap measurement properties
    @property
    def pixcap(self) -> Pixcap65:
        """Get the underlying pixcap object for handling the physical setup."""
        return self.dut

    @property
    def seq_size(self):
        """Get the granularity of the clock sequencer."""
        return self.pixcap.seq_size

    @seq_size.setter
    def seq_size(self, value):
        self.pixcap.seq_size = value

    @property
    def filters(self):
        """Get the measurement filters for storing data into hdf files."""
        return tb.Filters(complib='blosc', complevel=5, fletcher32=False)

    @property
    def current_sense_range(self):
        """Get the current sense range for capacitance measurement SMUs"""
        return 0.000001

    @property
    def bias_sense_range(self):
        """Get the current sense range for the HV supply."""
        return 0.000001

    @property
    def base_group(self) -> tb.Group:
        """Get the base group of the data structure to store the measurements."""
        assert self.__group is not None
        assert isinstance(self.__group, tb.Group)
        return self.__group

    @base_group.setter
    def base_group(self, value):
        temp_node, _ = walk_to_node(self.out_file_h5.root, value, create=True, verify_create=True)
        assert isinstance(temp_node, tb.Group)
        self.__group = temp_node

    @property
    def smu_kwargs(self):
        """Get the additional keyword arguments for the primary SMU"""
        return self.pixcap.smu_kwargs

    @property
    def analysis_group(self):
        """Get the analysis group of the data structure to store the results of the analysis."""
        warn("This function should not be used and could lead to undefined behaviour.")
        return self.base_group.analysis

    @property
    def measurement_group(self):
        """Get the measurement group of the data structure to store the results of the measurements."""
        warn("This function should not be used and could lead to undefined behaviour.")
        return self.base_group

    @property
    def n_measurements(self):
        """Get the number of measurements performed for capacitance estimation."""
        return self.pixcap.n_measurements

    @n_measurements.setter
    def n_measurements(self, value):
        self.pixcap.n_measurements = value

    @property
    def bias_measurements(self):
        return self.pixcap.n_bias_measurements

    @bias_measurements.setter
    def bias_measurements(self, value):
        self.pixcap.n_bias_measurements = value

    @property
    def col_start(self):
        """Get the first column of pixels"""
        return self.scan_config[ScanConfigurationKeys.START_COLUMN]

    @property
    def frequency_range(self):
        """Get the range of frequencies to scan the pixels for."""
        return self.scan_config[ScanConfigurationKeys.FREQUENCY_RANGE]

    @property
    def col_stop(self):
        """Get the last column of pixels"""
        return self.scan_config[ScanConfigurationKeys.STOP_COLUMN]

    @property
    def row_start(self):
        """Get the first row of pixels"""
        return self.scan_config[ScanConfigurationKeys.START_ROW]

    @property
    def row_stop(self):
        """Get the last row of pixels"""
        return self.scan_config[ScanConfigurationKeys.STOP_ROW]

    @property
    def averaging(self):
        return ScanConfigurationKeys.AVERAGE_MEASUREMENTS in self.scan_config and self.scan_config[
            ScanConfigurationKeys.AVERAGE_MEASUREMENTS] > 1

    @property
    def bias_averaging(self):
        return ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS in self.scan_config and self.scan_config[
            ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS] > 1

    @property
    def use_bias_supply(self):
        return self.bias_averaging or ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE in self.scan_config or \
            ScanConfigurationKeys.BIAS_VOLTAGE_RANGE in self.scan_config

    # endregion

    # region Handling of the Pixcap SMUs
    # TODO: these functions needs to be documented
    # Handle the SMU!
    # these will now just forward the commands to the pixcap object
    def init_smu(self, voltage_range=1.5, current_limit=0.0001, plc=None, **kwargs):
        if plc is None:
            plc = self.scan_config.get('plc_cycles', 10)
        current_range = kwargs.pop('current_range', self.current_sense_range)
        self.pixcap.init_smu(self.scan_config[ScanConfigurationKeys.VIN], current_range, voltage_range,
                             current_limit, plc, **kwargs)

    def smu_on(self):
        self.pixcap.smu_on()

    def smu_off(self):
        self.pixcap.smu_off()

    def get_source_current(self) -> float:
        return self.pixcap.get_source_current

    def get_source_current_multiple(self, n: int):
        return self.pixcap.get_source_current_multiple(n)

    # Handle the biasing supply
    def init_bias_voltage(self, voltage: float = -80.0):
        self.pixcap.init_bias(voltage, current_range=self.bias_sense_range)

    def set_bias_on(self):
        self.pixcap.bias_on()

    def set_bias_off(self):
        self.pixcap.bias_off()

    @deprecated("Use directly Pixcap65.bias_voltage attribute instead.")
    def set_bias_voltage(self, voltage: float):
        self.pixcap.bias_voltage = voltage
    # endregion
    @contextmanager
    def bias_without_averaging(self):
        if self.bias_averaging:
            # we must deactivate the averaging for the measurement
            back_n_bias_measurements = self.bias_measurements
            self.bias_measurements = 1
            try:
                yield self.pixcap
            finally:
                self.bias_measurements = back_n_bias_measurements
        else:
            try:
                yield self.pixcap
            finally:
                pass

    @contextmanager
    def without_averaging(self):
        if self.averaging:
            # we must deactivate the averaging for the measurement
            back_n_measurements = self.n_measurements
            self.n_measurements = 1
            try:
                yield self.pixcap
            finally:
                self.n_measurements = back_n_measurements
        else:
            try:
                yield self.pixcap
            finally:
                pass

    @property
    def hv_limit(self):
        return HV_CURRENT_LIMIT


class PixCap65TotalCap(PixCap65Measurement):
    # instantiation
    def __init__(self, scan_config, output_file, **kwargs):
        super(PixCap65TotalCap, self).__init__(scan_config, output_file, **kwargs)

        self.current_smu_config = {}
        if "double_sweep" in scan_config and scan_config["double_sweep"]:
            self.n_frequencies *= 2

        # the initialization of this could be moved to configuration?
        self.hist_current = np.full(shape=(40, 40, self.n_frequencies),
                                    fill_value=np.nan)  # current value for each measured frequency per pixel
        self.hist_current_errors = np.full(shape=(40, 40, self.n_frequencies), fill_value=np.nan)
        self.n_measurements = scan_config.get(NUMBER_AVERAGE_MEASUREMENTS_KEY, 1)
        self.hist_individual_currents = np.full(shape=(40, 40, self.n_frequencies, self.n_measurements),
                                                fill_value=np.nan)

        self.bias_measurements = scan_config.get(BIASING_NUMBER_AVERAGE_MEASUREMENTS_KEY, 1)
        self.hist_bias_current = np.full(shape=self.n_voltages, fill_value=np.nan)
        self.hist_bias_individual_currents = np.full(shape=(self.n_voltages, self.bias_measurements),
                                                     fill_value=np.nan)
        self.hist_bias_current_errors = np.full(shape=self.n_voltages, fill_value=np.nan)

        # this could safely be moved to the super class!
        self.pixcap.seq_size = self.scan_config.get(TOTAL_CAP_SEQ_SIZE_KEY, TOTAL_CAP_SEQ_SIZE)
        self.mode_logging_text = 'Scan pixel by single measurements.'
        self.handle_measurement = self._handle_single_measurement
        self.handle_bias_measurement = self._handle_single_measurement_bias
        self.bias_scan_parameters = OrderedDict()

    # general interface
    def configure(self):
        """
        configure

        Handling the configuration of the pixcap measurement object and the physical setup
        All information additionally required will be fetched from the scan configuration.
        For configuration only one SMU is needed.
        In Addition, also the sequence generator is configured for actual operation.
        """
        super(PixCap65TotalCap, self).configure()
        self.pixcap.seq_init(clk_0='1000', clk_3='0010')
        self.smu_on()

        # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
        self._smu_setup(self.pixcap.primary_smu_key).text_format()
        logging.debug('Waiting for settling of SMU...')
        if self.has_bias_supply:
            self.pixcap[self.pixcap.bias_smu_key].set_number_measurements(1)
            self.pixcap.bias_on()
        for _ in range(0, 30):
            current = self.get_source_current()
            if self.has_bias_supply:
                logging.debug('HV Current: {}'.format(self.pixcap.bias_measure_current()))
            logging.debug('Current: {}'.format(current))
            time.sleep(1)
        if self.has_bias_supply:
            self.pixcap[self.pixcap.bias_smu_key].set_number_measurements(self.bias_measurements)
            self.pixcap.bias_off()

        # changed to simplify changes in the used SMU
        self.get_source_current()

    # measurement interface
    def scan(self, data_group_spec=None, sequence_call=False):
        """
        scan

        Performs the scan over the pixels on the sensor and measures the requested quantities in dependence on some
        other quantities. Will scan the specified frequency range for each pixel specified by the scan configuration
        and measure the current to determine the total pixel capacitance.

        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        :param sequence_call: boolean, indicating whether the pixel-frequency scan is started from another measurement
            procedure.
        """
        # select the group to write the analysis results to
        data_group = self.get_data_group(data_group_spec, "total_cap")
        set_group_attribute(data_group, "frequencies", self.n_frequencies)

        # perform also a down sweep in frequency
        if "double_sweep" in self.scan_config and self.scan_config["double_sweep"]:
            frequency_range = np.concatenate((self.frequency_range, np.flip(self.frequency_range)))
        else:
            frequency_range = self.frequency_range

        self.pre_scan_handler()
        logging.info(self.mode_logging_text)
        logger.info(self.mode_logging_text)
        continue_error = None
        try:
            self.set_bias_measurement(data_group, sequence_call)
            with logging_redirect_tqdm():
                for i_row in tqdm(self.row_range, desc="Grid row Loop", leave=not sequence_call):
                    for i_col in tqdm(self.col_range, desc="Grid column Loop", leave=False):
                        logging.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
                        logger.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
                        self.dut.disable_all_pixels()
                        self.dut.disable_all_columns()

                        self.dut.enable_column(i_col, c.EN_EOC_3)
                        self.dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)

                        for k, freq in enumerate(frequency_range):
                            self.pixcap.cvm_frequency = freq
                            with self.without_averaging() as smu_less:
                                # stabilize the currents
                                try:
                                    back_nlpc = float(smu_less[smu_less.bias_smu_key].get_current_nlpc())
                                    smu_less[smu_less.bias_smu_key].set_current_nlpc(1)
                                    previous_measurement = self.get_source_current()
                                    time.sleep(1e-4)
                                    current_measurement = self.get_source_current()
                                    for _ in range(100):
                                        if np.abs(
                                                current_measurement - previous_measurement) < 0.05 * np.abs(
                                                current_measurement):
                                            break
                                        previous_measurement = current_measurement
                                        current_measurement = self.get_source_current()
                                    else:
                                        logger.warning("Could not stabilize the current.")
                                finally:
                                    smu_less[smu_less.bias_smu_key].set_current_nlpc(back_nlpc)
                            self.handle_measurement(i_col, i_row, k)
                            self.store_iteration_parameters(freq, k)
        except KeyboardInterrupt as e:
            logger.info("Caught KeyboardInterrupt. Will terminate the program softly.")
            continue_error = e
            continue_saving_operation = True
        else:
            continue_saving_operation = True
        if continue_saving_operation:
            self.post_scan_handler(data_group, sequence_call, group=data_group)

        if continue_error is not None:
            self.out_file_h5.flush()
            raise continue_error
        logging.info('Done')

    def bias_cv_scan(self, data_group_spec=None):
        """
        bias_cv_scan

        Scans over different bias voltages and therefore pixel depletion states and measures the current (to later
        obtain the pixels total capacitance in dependence on the bias voltage).

        The scans necessary for the estimation of the pixels capacitances are performed by the usual scan method.

        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        """
        assert "bias_range" in self.scan_config
        data_group = self.get_data_group(data_group_spec, "biasing")
        set_group_attribute(data_group, "bias_unit", HIST_BIAS_MEAS_UNIT)

        bias_voltages = np.asarray(self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE])
        self.pixcap.bias_voltage = 0
        self.pixcap.bias_on()
        try:
            with logging_redirect_tqdm():
                for bias_voltage in tqdm(bias_voltages, desc="Bias voltage scan"):
                    self.pixcap.bias_voltage = bias_voltage
                    bias_group_name = "bias_{bias_voltage}_V".format(bias_voltage=bias_voltage).replace('-',
                                                                                                        "M_").replace(
                        ".",
                        "__")
                    if bias_group_name in data_group:
                        # remove the biasing group or all of it's contents
                        scan_group = data_group[bias_group_name]
                        assert isinstance(scan_group, tb.Group)
                        for key, value in get_children(scan_group):
                            new_name = key
                            while new_name in data_group:
                                new_name = BACKING_FORMAT_TEXT.format(old=new_name)
                            rename_node(value, new_name)
                    else:
                        scan_group = group_get_file(data_group).create_group(where=data_group, name=bias_group_name)
                    logger.info("Perform sweep for bias voltage %f.", bias_voltage)
                    self.scan_parameters = OrderedDict()
                    self.pixcap.bias_voltage = bias_voltage
                    # check for the SMU's settling here
                    with self.bias_without_averaging() as hv_less:
                        previous_measurement = hv_less.bias_measure_volts()
                        time.sleep(HV_WAIT)
                        current_measurement = hv_less.bias_measure_volts()
                        for _ in range(100):
                            if np.abs(
                                    current_measurement - previous_measurement) < HV_VOLTAGE_TOL * np.abs(current_measurement):
                                break
                            previous_measurement = current_measurement
                            current_measurement = hv_less.bias_measure_volts()
                        else:
                            logger.warning("Could not stabilize the HV voltage.")

                        # stabilize the currents
                        try:
                            back_nlpc = float(hv_less[hv_less.bias_smu_key].get_current_nlpc())
                            hv_less[hv_less.bias_smu_key].set_current_nlpc(1)
                            previous_measurement = hv_less.bias_measure_current()
                            time.sleep(1e-3)
                            current_measurement = hv_less.bias_measure_current()
                            for _ in range(100):
                                if np.abs(
                                        current_measurement - previous_measurement) < HV_CURRENT_STABLE_TOL * np.abs(current_measurement) or np.abs(current_measurement) > self.hv_limit:
                                    break
                                previous_measurement = current_measurement
                                current_measurement = hv_less.bias_measure_current()
                            else:
                                logger.warning("Could not stabilize the current.")
                        finally:
                            hv_less[hv_less.bias_smu_key].set_current_nlpc(back_nlpc)

                        if np.abs(current_measurement) > self.hv_limit:
                            logger.error("The measured current %f exceed the protection limit %f.", current_measurement, self.hv_limit)
                            break

                        logger.debug("Set the bias voltage to %f." % bias_voltage)
                    self.scan(data_group_spec=scan_group, sequence_call=True)
        finally:
            logger.info("Collect all the current information and package it together.")
            # save the bias voltages
            self.create_carray(where=data_group, name="BiasVoltageHist",
                               title="Histogram of chosen bias voltages", obj=bias_voltages,
                               filters=self.filters, unit=HIST_BIAS_MEAS_UNIT)
            set_group_attribute(data_group, "bias_voltage_unit", HIST_BIAS_MEAS_UNIT)
            self.store_configuration(data_group)
            logger.info("Done bias measurements.")

    def second_bias_scan(self, data_group_spec=None):
        """
        second_bias_scan

        Scan different bias voltages and measure the detector leakage current.

        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        """
        assert self.has_bias_supply
        data_group = self.get_data_group(data_group_spec, "biasing")

        # prepare the scan
        # remains just for convenience.
        bias_voltages = self.bias_voltages
        set_group_attribute(data_group, 'voltages', self.n_voltages)
        self.hist_bias_current = np.full(shape=self.n_voltages,
                                         fill_value=np.nan)  # current value for each measured frequency per pixel
        self.hist_bias_individual_currents = np.full(shape=(self.n_voltages, self.bias_measurements),
                                                     fill_value=np.nan)
        self.hist_bias_current_errors = np.full(shape=self.n_voltages, fill_value=np.nan)

        prevent_group_mix_up(data_group, "BiasVoltageHist")
        self.create_carray(data_group, "BiasVoltageHist", obj=bias_voltages, filters=self.filters,
                           unit=HIST_BIAS_MEAS_UNIT)

        self.pre_scan_handler(unit="bias")
        logging.info(self.mode_logging_text)
        logger.info(self.mode_logging_text)
        self.pixcap.bias_voltage = 0.
        self.pixcap.bias_on()
        self.dut.disable_all_pixels()
        self.dut.disable_all_columns()

        def post_handling():
            self.post_scan_handler(data_group, False, unit="bias", saving_unit="bias", group=data_group)
            if self.has_bias_supply:
                self.pixcap.bias_off()
            logging.info('Done')

        for k, voltage in self.perform_bias_scan(bias_voltages, post_handling, self.bias_scan_parameters):
            self.handle_bias_measurement(k)

    def bias_scan(self, data_group_spec=None):
        """
        bias_scan

        Scan different bias voltages and measure the detector leakage current.

        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        """
        assert self.has_bias_supply
        data_group = self.get_data_group(data_group_spec, "biasing")

        # prepare the scan
        # remains just for convenience.
        bias_voltages = self.bias_voltages
        set_group_attribute(data_group, 'voltages', self.n_voltages)
        self.hist_bias_current = np.full(shape=self.n_voltages,
                                         fill_value=np.nan)  # current value for each measured frequency per pixel
        self.hist_bias_individual_currents = np.full(shape=(self.n_voltages, self.bias_measurements),
                                                     fill_value=np.nan)
        self.hist_bias_current_errors = np.full(shape=self.n_voltages, fill_value=np.nan)

        prevent_group_mix_up(data_group, "BiasVoltageHist")
        self.create_carray(data_group, "BiasVoltageHist", obj=bias_voltages, filters=self.filters,
                           unit=HIST_BIAS_MEAS_UNIT)

        self.pre_scan_handler(unit="bias")
        logging.info(self.mode_logging_text)
        logger.info(self.mode_logging_text)
        self.pixcap.bias_voltage = 0.
        self.pixcap.bias_on()
        self.dut.disable_all_pixels()
        self.dut.disable_all_columns()

        try:
            for k, bias_voltage in enumerate(bias_voltages):
                self.pixcap.bias_voltage = bias_voltage
                # check for the SMU's settling here
                with self.bias_without_averaging() as hv_less:
                    previous_measurement = hv_less.bias_measure_volts()
                    time.sleep(HV_WAIT)
                    current_measurement = hv_less.bias_measure_volts()
                    for _ in range(100):
                        if np.abs(current_measurement - previous_measurement) < HV_VOLTAGE_TOL * np.abs(current_measurement):
                            break
                        previous_measurement = current_measurement
                        current_measurement = hv_less.bias_measure_volts()
                    else:
                        logger.warning("Could not stabilize the HV voltage.")

                    # stabilize the currents
                    try:
                        back_nlpc = float(hv_less[hv_less.bias_smu_key].get_current_nlpc())
                        hv_less[hv_less.bias_smu_key].set_current_nlpc(1)
                        previous_measurement = hv_less.bias_measure_current()
                        time.sleep(1e-3)
                        current_measurement = hv_less.bias_measure_current()
                        for _ in range(100):
                            if np.abs(current_measurement - previous_measurement) < HV_CURRENT_STABLE_TOL * np.abs(current_measurement) or np.abs(current_measurement) > self.hv_limit:
                                break
                            previous_measurement = current_measurement
                            current_measurement = hv_less.bias_measure_current()
                        else:
                            logger.warning("Could not stabilize the current.")
                    finally:
                        hv_less[hv_less.bias_smu_key].set_current_nlpc(back_nlpc)

                    if np.abs(current_measurement) > self.hv_limit:
                        logger.error("The leakage current %f A measured for verification exceed the protection limit of %f", current_measurement, self.hv_limit)
                        break

                    logger.debug("Set the bias voltage to %f." % bias_voltage)
                # perform the actual measurement.
                self.handle_bias_measurement(k)
                with self.bias_without_averaging() as hv_less:
                    actual_bias_voltage = hv_less.bias_measure_volts()
                store_scan_par_values(scan_parameters=self.bias_scan_parameters, scan_param_id=k,
                                      bias_voltage=bias_voltage, hv_voltage=actual_bias_voltage)

        finally:
                self.post_scan_handler(data_group, False, unit="bias", saving_unit="bias", group=data_group)
                if self.has_bias_supply:
                    self.pixcap.bias_off()
                logging.info('Done')

    def combined_bias_cv_scan(self, data_group_spec=None):
        """
        combined_bias_cv_scan

        Combines the cv characterization scan over the pixels with the measurement of the detector leakage current in
        dependence on the applied external bias voltage.
        For each bias voltage to be scanned over, first the detector leakage current is measured and then the pixel
        matrix is scanned to later obtain the pixel capacitance's with this HV applied.

        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        """
        assert self.has_bias_supply
        data_group = self.get_data_group(data_group_spec, "biasing")

        # prepare the scan
        bias_voltages = self.bias_voltages
        prevent_group_mix_up(data_group, "BiasVoltageHist")
        self.create_carray(data_group, "BiasVoltageHist", obj=bias_voltages, filters=self.filters,
                           unit=HIST_CURRENT_MEAS_UNIT)
        set_group_attribute(data_group, "voltages", self.n_voltages)
        set_group_attribute(data_group, "bias_unit", HIST_CURRENT_MEAS_UNIT)
        self.hist_bias_current = np.full(shape=self.n_voltages,
                                         fill_value=np.nan)  # current value for each measured frequency per pixel
        self.hist_bias_individual_currents = np.full(shape=(self.n_voltages, self.bias_measurements),
                                                     fill_value=np.nan)
        self.hist_bias_current_errors = np.full(shape=self.n_voltages, fill_value=np.nan)

        self.pre_scan_handler(unit="bias")
        logging.info(self.mode_logging_text)
        logger.info(self.mode_logging_text)
        self.pixcap.bias_voltage = -0.1
        self.pixcap.bias_on()
        self.dut.disable_all_pixels()
        self.dut.disable_all_columns()

        try:
            from tqdm.contrib import tenumerate
            for k, bias_voltage in tenumerate(bias_voltages, desc="Bias Voltage Scan."):
                self.scan_parameters = OrderedDict()
                self.pixcap.bias_voltage = bias_voltage
                # check for the smu's settling here
                with self.bias_without_averaging() as hv_less:
                    previous_measurement = hv_less.bias_measure_volts()
                    time.sleep(HV_WAIT)
                    current_measurement = hv_less.bias_measure_volts()
                    for _ in range(100):
                        if np.abs(current_measurement - previous_measurement) < HV_VOLTAGE_TOL * np.abs(current_measurement):
                            break
                        previous_measurement = current_measurement
                        current_measurement = hv_less.bias_measure_volts()
                    else:
                        logger.warning("Could not stabilize the HV voltage.")

                    # stabilize the currents
                    try:
                        back_nlpc = float(hv_less[hv_less.bias_smu_key].get_current_nlpc())
                        hv_less[hv_less.bias_smu_key].set_current_nlpc(1)
                        previous_measurement = hv_less.bias_measure_current()
                        time.sleep(1e-3)
                        current_measurement = hv_less.bias_measure_current()
                        for _ in range(100):
                            if np.abs(
                                    current_measurement - previous_measurement) < HV_CURRENT_STABLE_TOL * np.abs(current_measurement) or np.abs(current_measurement) > self.hv_limit:
                                break
                            previous_measurement = current_measurement
                            current_measurement = hv_less.bias_measure_current()
                        else:
                            logger.warning("Could not stabilize the current.")
                    finally:
                        hv_less[hv_less.bias_smu_key].set_current_nlpc(back_nlpc)

                    if np.abs(current_measurement) > self.hv_limit:
                        self.pixcap.bias_voltage = -0.01
                        logger.error("The measured current %f exceed the protection limit %f.", current_measurement, self.hv_limit)
                        break

                    logger.debug("Set the bias voltage to %f." % bias_voltage)
                self.handle_bias_measurement(k)
                with self.bias_without_averaging() as hv_less:
                    actual_bias_voltage = hv_less.bias_measure_volts()
                store_scan_par_values(scan_parameters=self.bias_scan_parameters, scan_param_id=k,
                                      bias_voltage=bias_voltage, hv_voltage=actual_bias_voltage)
                bias_group_name = "bias_{bias_voltage}_V".format(bias_voltage=bias_voltage).replace('-', "M_").replace(
                    ".",
                    "__")
                if bias_group_name in data_group:
                    # remove the biasing group
                    scan_group = data_group[bias_group_name]
                    assert isinstance(scan_group, tb.Group)
                    for key, value in get_children(scan_group):
                        new_name = key
                        while new_name in data_group:
                            new_name = "{old}_backing".format(old=new_name)
                        rename_node(value, new_name)
                else:
                    scan_group = group_get_file(data_group).create_group(where=data_group, name=bias_group_name)
                logger.info("Performed sweep for bias voltage %f.", bias_voltage)
                self.scan(data_group_spec=scan_group, sequence_call=True)
        finally:
            self.post_scan_handler(data_group, unit="bias", saving_unit="bias", group=data_group)
            self.pixcap.bias_off()
            logging.info('Done')

    def plot(self):
        # TODO: requires rework for fetching the correct groups before plotting.
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(self.output_file[:-3] + '.pdf') as output_pdf:
            plot_data_delegate(self.out_file_h5.root, self.out_file_h5.root, output_pdf)

    def analyze(self):
        # TODO: requires rework for fetching the correct groups before analysing!
        analysis_data_handle(self.out_file_h5.root, self.base_group, self.base_group)

    def handle_measurement_errors(self, unit):
        if self.averaging:
            average_currents = np.nanmean(self.hist_individual_currents, axis=3, keepdims=True)
            self.hist_current = average_currents[:, :, :, 0]
            self.hist_current_errors = np.nanstd(self.hist_individual_currents, axis=3, mean=average_currents)

        else:
            self.hist_current_errors = self.determine_measurement_uncertainty(self.pixcap.primary_smu_key,
                                                                              self.hist_current)
        if unit == "bias":
            if self.bias_averaging:
                average_currents = np.nanmean(self.hist_bias_individual_currents, axis=1, keepdims=True)
                self.hist_bias_current = average_currents[:, 0]
                self.hist_bias_current_errors = np.nanstd(self.hist_bias_individual_currents, axis=1,
                                                          mean=average_currents)
            else:
                self.hist_bias_current_errors = self.determine_measurement_uncertainty(self.pixcap.bias_smu_key,
                                                                                       self.hist_bias_current)

    def store_measurement_data(self, data_group: tb.Group, sequence_call: bool, unit=None):
        try:
            if unit is None:
                # select the group to write the analysis results to
                assert isinstance(data_group, tb.Group)
                if np.all(np.isnan(self.hist_current)):
                    raise ValueError("UNEXPECTED: All measurement entries are still NaN.")
                self.create_carray(data_group, name='HistCurr', title='Current Histogram',
                                   obj=self.hist_current, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT)

                # need the additional entries for the advanced averaging implementation
                if self.averaging:
                    self.create_carray(data_group, name='HistCurrValues', title='Multiple Current Histogram',
                                       obj=self.hist_individual_currents, filters=self.filters,
                                       unit=HIST_CURRENT_MEAS_UNIT)

                if np.any(np.isfinite(self.hist_current_errors)):
                    self.create_carray(data_group, name='HistCurrErr', title='Current Error Histogram',
                                       obj=self.hist_current_errors, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT)
            elif unit == "bias":
                if np.all(np.isnan(self.hist_bias_current)):
                    raise ValueError("UNEXPECTED: All measurement entries are still NaN.")
                self.create_carray(data_group, name="HistCurr", title='Current Histogram', obj=self.hist_bias_current,
                                   filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT)
                # need the additional entries for the advanced averaging implementation
                if self.bias_averaging:
                    self.create_carray(data_group, name='HistCurrValues',
                                       title='Multiple Current Histogram',
                                       obj=self.hist_bias_individual_currents,
                                       filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT)

                if np.any(np.isfinite(self.hist_bias_current_errors)):
                    self.create_carray(data_group, name='HistCurrErr', title='Current Error Histogram',
                                       obj=self.hist_bias_current_errors,
                                       filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT)
        finally:
            assert isinstance(data_group, tb.Group)
            if unit is None:
                _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.scan_parameters, group=data_group)
            elif unit == "bias":
                _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.bias_scan_parameters,
                                       group=data_group)

    # interface
    def update_config(self, new_config=None):
        super(PixCap65TotalCap, self).update_config(new_config)
        self.n_measurements = self.scan_config.get(NUMBER_AVERAGE_MEASUREMENTS_KEY, 1)

        # FIXME: this here will double the expected frequencies on each configuration update, as well as on init!
        if "double_sweep" in self.scan_config and self.scan_config["double_sweep"]:
            self.n_frequencies *= 2

        self.bias_measurements = self.scan_config.get(BIASING_NUMBER_AVERAGE_MEASUREMENTS_KEY, 1)

    def handle_cv_compaction(self, kwargs: dict[str, Any], unit):
        if unit == "bias" and "group" in kwargs:
            data_group = kwargs["group"]
            table = self.out_file_h5.create_table(data_group, name="BiasTable", description=BiasTable,
                                                  filters=self.filters)
            entry = table.row
            for i in range(self.n_voltages):
                try:
                    entry['U'] = self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE][i]
                    entry['I'] = self.hist_bias_current[i]
                    entry['DI'] = self.hist_bias_current_errors[i]
                    entry.append()
                except ValueError as e:
                    print("scan parameters")
                    print(self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE][i])
                    print("currents")
                    print(self.hist_bias_current.shape)
                    print(self.hist_bias_current[i])
                    print("Errors")
                    print(self.hist_bias_current_errors.shape)
                    print(self.hist_bias_current_errors[i])
                    raise e

    def pre_scan_handler(self, unit=None):
        """
        pre_scan_handler

        Configures the data array such that there won't be any problem with temporarily saving the measurements
        performed during the scan.
        This should also catch any changes to the scan configuration not associated with an update call.
        Furthermore, the correct measurement routine for scan will be selected.

        :param unit: Additional unit to activate for the post scan analysis e.g. bias (it is the only implemented yet).
        """
        if self.averaging:
            self.n_measurements = self.scan_config[NUMBER_AVERAGE_MEASUREMENTS_KEY]
            individual_currents_shape = (40, 40, self.n_frequencies, self.n_measurements)
            if self.hist_individual_currents.shape != individual_currents_shape:
                self.hist_individual_currents = np.full(shape=individual_currents_shape,
                                                        fill_value=np.nan)

            self.mode_logging_text = "Average over multiple measurements!"
            self.handle_measurement = self._handle_averaged_measurement
            self.current_smu_config = self.smu_range_config[self.pixcap.primary_smu_key]
        else:
            self.mode_logging_text = 'Scan pixel by single measurements.'
            self.handle_measurement = self._handle_single_measurement

            # CHECK: where is it still used.
            self.current_smu_config = self.smu_range_config[self.pixcap.primary_smu_key]

        if self.bias_averaging:
            self.handle_bias_measurement = self._handle_averaged_measurement_bias
            if unit == "bias":
                individual_currents_shape = (self.n_voltages, self.bias_measurements)
                if self.hist_bias_individual_currents.shape != individual_currents_shape:
                    self.hist_bias_individual_currents = np.full(shape=individual_currents_shape,
                                                                 fill_value=np.nan)

        else:
            self.handle_bias_measurement = self._handle_single_measurement_bias

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
            self.bias_measurements)[:]
        assert not np.all(np.isnan(measurement)), "The bias measurement was not performed."
        try:
            self.hist_bias_individual_currents[k, :] = measurement[1::2]
        except:
            print(k, self.bias_measurements, file=self.dummy_file)
            print(measurement.shape, file=self.dummy_file)
            raise

    # region Pixcap Properties
    @property
    def col_range(self):
        """Get the range of columns to scan the pixels for."""
        return range(self.col_start, self.col_stop)

    @property
    def row_range(self):
        """Get the range of rows to scan the pixels for."""
        return range(self.row_start, self.row_stop)

    @property
    def bias_voltages(self):
        return np.asarray(self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE])

    @property
    def n_voltages(self):
        try:
            return self.bias_voltages.shape[0]
        except:
            return 1

    # endregion


if __name__ == '__main__':
    output_file_2 = "../Reference_Demo.h5"
    from pixcap65.utils import PixCapSetup, PixcapMeasurements
    with PixCapSetup(scan_configuration, output_file_2, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
        print(type(pix))
        # assert isinstance(pix, PixCap65TotalCap)
        pix.bias_scan(data_group_spec="simple_bias")
        pix.bias_scan_parameters = OrderedDict()
        pix.second_bias_scan(data_group_spec="improved_bias")

    del scan_configuration[BIASING_NUMBER_AVERAGE_MEASUREMENTS_KEY]
    with PixCapSetup(scan_configuration, output_file_2, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
        pix.bias_cv_scan(data_group_spec="cv_bias")



    # initial measurement sample
    with PixCapSetup(scan_configuration, output_file_2, measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as pix:
        # for larger averages
        pix.scan(data_group_spec="unbiased_4_full")

    # with PixCap65TotalCap(scan_configuration, output_file_2) as pix:
    #     try:
    #         pix.scan(data_group_spec="biased_80_V")
    #         pix.analyze()
    #         pix.plot()
    #     finally:
    #         pix.pixcap[pix.pixcap.smu_setup_devices[pix.pixcap.primary_smu_key]].text_format()
