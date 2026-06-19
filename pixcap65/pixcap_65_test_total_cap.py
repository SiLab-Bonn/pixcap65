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

from collections import OrderedDict
from importlib.resources import files

import gc
import logging
import numpy as np
import os
import sys
import tables as tb
import time
import warnings
import yaml
from abc import abstractmethod, ABCMeta
from collections.abc import Callable
from contextlib import contextmanager
from enum import StrEnum
from numpy import ndarray
from tqdm import tqdm
# noinspection PyProtectedMember
from tqdm.contrib import DummyTqdmFile
from typing import Iterable, Mapping, Any, Optional
from warnings import warn, deprecated

from pixcap65.analysis import analysis_data_handle
from pixcap65.analysis_util.utility import HIST_CURRENT_MEAS_UNIT, HIST_BIAS_MEAS_UNIT, handle_analysis_mix_up
from pixcap65.configs.config_handler import extract_smu_current_error, extract_smu_voltage_error
from pixcap65.pixcap.pixcap65 import Pixcap65
from pixcap65.pixcap.pixcap_structure import BasilConfigKeys
from pixcap65.plotting import plot_data_delegate
from pixcap65.utility import pixcap65_constants as c
from pixcap65.utility.basil_utils import extract_basil_layers
from pixcap65.utility.tables_util import get_group_attributes, set_group_attribute, \
    get_group_attribute, group_get_file, back_node
from pixcap65.utility.tqdm_logging_utils import advanced_tqdm_iterator
from pixcap65.utility.tqdm_logging_utils import logging_redirect_tqdm
from pixcap65.utility.utils_2 import prevent_group_mix_up
from pixcap65.utility.utils_2 import walk_to_node

START_HV_VOLTAGE = 0.

LOG_SET_BIAS = "Set the bias voltage to %f."

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
HV_CURRENT_STABLE_TOL = 1e-2
HV_CURRENT_LIMIT = 2.e-7

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_handler = logging.FileHandler('pixcap_65_test.log')
log_formater = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
log_handler.setFormatter(log_formater)
logger.addHandler(log_handler)
logger.propagate = True


def _get_enumerate(iterator, **kwargs) -> Iterable:
    use_tqdm = kwargs.pop("pbar", False)
    try:
        from tqdm.contrib import tenumerate
        if not use_tqdm:
            kwargs["disable"] = True
        return advanced_tqdm_iterator(iterator, tqdm_class=tenumerate, **kwargs)
    except ImportError:
        return enumerate(iterator)


# noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
def default_callback(group: tb.Group):
    # stub function for callback when no callback is required at all.
    pass


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
        logger.info("Storing scan parameter values but found an already existing table; will rename it")
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
        logger.error("Failed to create and fill the scan parameters table. But will continue anyway.")


class ScanConfigurationKeys(StrEnum):
    """
    Enumeration object for type-safe access to the names of the scan configuration keys.
    """
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
    BIAS_CURRENT_LIMIT = "bias_limit"
    BIAS_CURRENT_RANGE = "bias_sense_range"
    BIAS_HV_CURRENT_LIMIT = "bias_hv_limit"
    SMU_CURRENT_RANGE = "pixcap_range"
    SMU_CURRENT_LIMIT = "pixcap_limit"


scan_configuration = {
    'start_column': 0,
    'stop_column': 40,
    'start_row': 0,
    'stop_row': 40,
    "average_measurements": 25,
    # "bias_average_measurements": 3,

    'Vin': 1.0,  # input voltage in V
    'frequency_range': np.arange(1, 10.1, 1),  # frequency sweep in MHz
    # 'bias_range': -1 * np.geomspace(1, 80, 20),
    # 'bias': -80.0,   # bias voltage to apply in V

    'data_path': "Reference/Bare",
    "out_file_mode": "append",
}


class BiasTable(tb.IsDescription):
    Us = tb.Float32Col()
    I = tb.Float32Col()
    DI = tb.Float32Col()
    U = tb.Float32Col()
    DU = tb.Float32Col()


class MeasurementAbstract(object, metaclass=ABCMeta):
    """
    Abstract (base) class to define the most basic measurement class/procedure for working with Pixcap.
    """

    def __init__(self, *args, **kwargs):
        self.dummy_file = DummyTqdmFile(sys.stdout)
        self.dummy_error_file = DummyTqdmFile(sys.stderr)

    @abstractmethod
    def configure(self):
        """
        configure

        Handling the configuration of the pixcap measurement object and the physical setup
        All information additionally required will be fetched from the scan configuration.
        Some of the configuration needs to be done subclass implementations as the different measurement types could
        have different requirements onto the setup.
        """
        raise NotImplementedError("`configure` is abstract and therefore not implemented.")

    @abstractmethod
    def close(self):
        """
        close


        closes all open file handles used by the measurement as well as all connections to SMUs or other lab
        devices. The SMUs are switched off before closing the connection.
        """
        raise NotImplementedError("`close` is abstract and therefore not implemented.")

    @abstractmethod
    def scan(self, data_group_spec=None):
        """
        scan

        Performs the scan over the pixels on the sensor and measures the requested quantities in dependence on some
        other quantities. The implementation will strongly depend on the particular measurement type. Thus, it is
        necessary to override this method.
        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        """
        raise NotImplementedError("`scan` is abstract and therefore not implemented.")


class Pixcap65BaseMeasurement(MeasurementAbstract, metaclass=ABCMeta):
    def __init__(self, pix_config=None, **kwargs):
        super(Pixcap65BaseMeasurement, self).__init__(**kwargs)
        if pix_config is None or not os.path.exists(pix_config):
            logger.warning("The path to the pixcap firmware was recalculated from the package resources.")
            pix_config = files("pixcap65").joinpath("device", "ise", "pixcap65.bit")

        # init the dut
        self.dut = Pixcap65(pix_config)
        self.dut.init()


class PixCap65Measurement(Pixcap65BaseMeasurement, metaclass=ABCMeta):
    # instantiation
    def __init__(self, scan_config, output_file, pix_config="pixcap65.yaml", **kwargs):
        super(PixCap65Measurement, self).__init__(pix_config, **kwargs)
        self.mode_logging_text = 'Scan pixel by single measurements.'
        self.smu_range_config = {}
        self.__group = None

        # CHECK: whether this could be moved to the configure methods.
        # move only possible if the `__slots__` is used to fix the members to prevent warnings.
        self.bias_measurements = scan_config.get(BIASING_NUMBER_AVERAGE_MEASUREMENTS_KEY, 1)
        self.hist_bias_current = np.full(shape=self.n_voltages, fill_value=np.nan)
        self.hist_bias_current_errors = np.full(shape=self.n_voltages, fill_value=np.nan)
        self.hist_bias_individual_currents = np.full(shape=(self.n_voltages, self.bias_measurements),
                                                     fill_value=np.nan)

        # handle smu error configuration
        adjusted_config = self.dut._conf.copy()
        self._environ_config = OrderedDict()
        hl_mapping, _, rl_mapping = extract_basil_layers(adjusted_config)
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
            config_file = files("pixcap65.configs").joinpath(device_name).joinpath(
                "{name}_range.yaml".format(name=device_name))
            # config_file = os.path.join(os.path.dirname(__file__), "configs",
            #                            "{name}_range.yaml".format(name=device_name))
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

        self.bias_scan_parameters = OrderedDict()

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
        # to be exact: this should be done by the config_update handler as it relais on configuration options!
        # will be evaluated before the update of configuration is taken into account!
        if self.use_bias_supply and self.has_bias_supply:
            logging.warning("Bias supply is now active.")
            logger.warning("Bias supply is now active.")
            self.pixcap.init_bias(voltage=-0.1, voltage_range=1000, current_range=self.bias_sense_range,
                                  current_limit=self.bias_limit)

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

        # update the scan config parameters
        self.update_config()

    def close(self):
        """
        close


        closes all open file handles used by the measurement as well as all connections to SMUs or other lab
        devices. The SMUs are switched off before closing the connection.
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
        raise NotImplementedError("`scan` is abstract and therefore not implemented.")

    @property
    def buffered_bias_voltage(self):
        if self.has_bias_supply:
            return self.pixcap.get_smu_source_voltage(self.pixcap.bias_smu_key)
        return np.nan

    @buffered_bias_voltage.setter
    def buffered_bias_voltage(self, value):
        cached_voltage = self.buffered_bias_voltage
        while np.abs(cached_voltage - value) > 10:
            if cached_voltage > value:
                self.pixcap.bias_voltage = cached_voltage - 10
            else:
                self.pixcap.bias_voltage = self.buffered_bias_voltage + 10
            cached_voltage = self.buffered_bias_voltage

        self.pixcap.bias_voltage = value

    def perform_bias_scan(self, bias_voltages: np.ndarray, post_handler: Callable, parameters, data_group_spec,
                          handle_unit,
                          **kwargs):
        """
        perform_bias_scan

        :author: Dominik Fischer
        :date: 2026-05-14

        Contextmanager and iterable to consolidate all the common elements of the different scan over different
        HV bias voltages.
        First the general parameters are set up, and then we iterate over all the provided bias voltages.
        The iteration includes checks for stable HV voltages and leakage currents.

        After the voltage scan the (general) results are stored and some post handler could be performed
        if one is specified.

        The iterable object returned will be the current iteration index, the bias voltage and the data group
        for storage.


        :param bias_voltages: Iterable of the bias voltages to scan over.
        :param post_handler: callback to be executed after scan the is performed.
        :param parameters: mapping to store the scan parameters to.
        :param data_group_spec: specifier of the hdf files groups to store the results to.
        :param handle_unit: :ref: `unit` specifier/argument for measurement storage.
        :param kwargs: further keyword arguments to be propagated to the progress bar handler.
        :key pbar: boolean, whether to use tqdm for progress bars or not.
        """
        assert 'bias_range' in self.scan_config
        assert self.has_bias_supply
        allow_continuous_measurement = kwargs.pop('allow_continuous_measurement', False)
        data_group = self.get_data_group(data_group_spec, 'biasing')
        set_group_attribute(data_group, 'bias_unit', HIST_BIAS_MEAS_UNIT)

        # prepare the scan
        set_group_attribute(data_group, 'voltages', self.n_voltages)
        self.hist_bias_current = np.full(shape=self.n_voltages,
                                         fill_value=np.nan)  # current value for each measured frequency per pixel
        self.hist_bias_individual_currents = np.full(shape=(self.n_voltages, self.bias_measurements),
                                                     fill_value=np.nan)
        self.hist_bias_current_errors = np.full(shape=self.n_voltages, fill_value=np.nan)
        hist_bias_voltage = np.full(shape=self.n_voltages, fill_value=np.nan)
        hist_bias_voltage_error = np.full(shape=self.n_voltages, fill_value=np.nan)
        hist_bias_individual_voltages = np.full(shape=(self.n_voltages, 3), fill_value=np.nan)

        prevent_group_mix_up(data_group, 'BiasVoltageHist')
        self.pre_scan_handler(unit='bias')
        logging.info(self.mode_logging_text)
        logger.info(self.mode_logging_text)
        self.pixcap.bias_voltage = START_HV_VOLTAGE
        self.pixcap.bias_on()

        self.dut.disable_all_pixels()
        self.dut.disable_all_columns()
        try:
            for k, bias_voltage in _get_enumerate(bias_voltages, **kwargs):
                self.buffered_bias_voltage = bias_voltage
                # check for the SMU's settling here
                with self.bias_without_averaging() as hv_less:
                    current_measurement = self.verify_stable_hv(hv_less)
                    logger.debug("The hv voltage measurement is %f V", current_measurement)

                    # stabilize the currents
                    current_measurement = self.verify_hv_current(hv_less)

                    if np.abs(current_measurement) > self.hv_limit:
                        self.pixcap.bias_voltage = -0.1
                        logger.error("The measured current %f has exceeded the protection limit %f.",
                                     current_measurement, self.hv_limit)
                        break

                    logger.debug(LOG_SET_BIAS % bias_voltage)

                if allow_continuous_measurement:
                    self.pixcap.bias_initiate_multiple_voltage(3)
                    logger.warn("Tried to perform multiple voltage measurements.")

                yield k, bias_voltage, data_group
                if allow_continuous_measurement:
                    temp = self.pixcap.bias_read_multiple_voltage(3)
                    hist_bias_individual_voltages[k, :] = temp
                    actual_bias_voltage = np.mean(temp)
                    hist_bias_voltage[k] = actual_bias_voltage
                    hist_bias_voltage_error[k] = np.std(temp)
                else:
                    with self.bias_without_averaging() as hv_less:
                        actual_bias_voltage = hv_less.bias_measure_volts()
                        hist_bias_voltage_error = extract_smu_voltage_error(
                            self.smu_range_config[self.pixcap.bias_smu_key], bias_voltages, 1000)

                hist_bias_voltage[k] = actual_bias_voltage
                store_scan_par_values(scan_parameters=parameters, scan_param_id=k,
                                      bias_voltage=bias_voltage, hv_voltage=actual_bias_voltage)

        finally:
            # save the bias voltages
            if self.has_bias_supply:
                self.pixcap.bias_off()
            try:
                bias_result = np.full(shape=(self.n_voltages, 3), fill_value=np.nan)
                bias_result[:, 0] = bias_voltages
                bias_result[:, 1] = hist_bias_voltage
                bias_result[:, 2] = hist_bias_voltage_error
                self.create_carray(where=data_group, name='BiasVoltageHist', title='Histogram of chosen bias voltages',
                                   obj=bias_result, filters=self.filters, unit=HIST_BIAS_MEAS_UNIT)
            except:
                self.create_carray(where=data_group, name='BiasVoltageHist', title='Histogram of chosen bias voltages',
                                   obj=bias_voltages, filters=self.filters, unit=HIST_BIAS_MEAS_UNIT)
            post_handler(data_group)
            self.post_scan_handler(data_group, False, unit=handle_unit, saving_unit=handle_unit, group=data_group)
            set_group_attribute(data_group, 'bias_voltage_unit', HIST_BIAS_MEAS_UNIT)
            set_group_attribute(data_group, 'voltages', self.n_voltages)
            logging.info('Done bias measurements.')
            logger.info('Done bias measurements.')

    def verify_hv_current(self, hv_less: Pixcap65) -> float:
        """
        verify_hv_current

        :author: Dominik Fischer
        :date: 2026-05-14

        Verify that a stable working point w.r.t. the leakage current through the sensor is reached for the HV by
        performing repeated current measurement and waiting for a sufficiently small variation between the
        measurements. For these current measurements always single measurements are used without averaging.

        If the protection current is exceeded the waiting for a stable working point is aborted immediately.

        :param hv_less: :ref: `pixcap65.pixcap.Pixcap65` object to perform the measurements with.
        :return: float, leakage current from last measurement cycle.
        """
        back_nlpc = None
        try:
            back_nlpc = float(hv_less[hv_less.bias_smu_key].get_current_nlpc())
            hv_less[hv_less.bias_smu_key].set_current_nlpc(1)
            previous_measurement = hv_less.bias_measure_current()
            time.sleep(1e-3)
            current_measurement = hv_less.bias_measure_current()
            for _ in range(100):
                if np.abs(
                        current_measurement - previous_measurement) < HV_CURRENT_STABLE_TOL * np.abs(
                    current_measurement) or np.abs(current_measurement) > self.hv_limit:
                    break
                previous_measurement = current_measurement
                current_measurement = hv_less.bias_measure_current()
            else:
                logger.warning("Could not stabilize the current.")
        finally:
            if back_nlpc is not None:
                hv_less[hv_less.bias_smu_key].set_current_nlpc(back_nlpc)
        return current_measurement

    # noinspection PyMethodMayBeStatic
    def verify_stable_hv(self, hv_less: Pixcap65) -> float:
        """
        verify_stable_hv

        :author: Dominik Fischer
        :date: 2026-05-14

        Verify that a stable working point w.r.t. the applied voltage is reached for the HV by
        performing repeated voltage measurement and waiting for a sufficiently small variation between the
        measurements. For these voltage measurements always single measurements are used without averaging.

        :param hv_less: :ref: `pixcap65.pixcap.Pixcap65` object to perform the measurements with.
        :return: result from the last voltage measurement.
        """
        previous_measurement = hv_less.bias_measure_volts()
        time.sleep(HV_WAIT)
        current_measurement = hv_less.bias_measure_volts()
        for _ in range(100):
            if np.abs(current_measurement - previous_measurement) < HV_VOLTAGE_TOL * np.abs(
                    current_measurement):
                break
            previous_measurement = current_measurement
            current_measurement = hv_less.bias_measure_volts()
        else:
            logger.warning("Could not stabilize the HV voltage.")
        return current_measurement

    @abstractmethod
    def analyze(self):
        """
        analyze

        Perform the analysis of the data measured by the particular procedure.
        """
        raise NotImplementedError("`analyze` is abstract and therefore not implemented.")

    @abstractmethod
    def plot(self):
        """
        plot

        creates a graphical representation of the measurement and analysis results of the particular measurement type
        implement by a subclass.
        """
        raise NotImplementedError("`plot` is abstract and therefore not implemented.")

    @abstractmethod
    def handle_measurement_errors(self, unit):
        """
        handle_measurement_errors

        :author: Dominik Fischer
        :date: 2026-05-14

        Helper function to extract the measurement uncertainties either from multiple measurements under the same
        conditions or from a single measurement, the selected measurement range and the SMU's manual.


        :param unit: measurement mode used.
        """
        raise NotImplementedError("`handle_measurement_errors` is abstract and therefore not implemented.")

    @abstractmethod
    def store_measurement_data(self, data_group, sequence_call, unit=None):
        """
        store_measurement_data

        :author: Dominik Fischer
        :date: 2026-05-14

        Stores the measured data into a hdf file. The details will depend on the implementation.

        :param data_group: hdf file's group where the data should be stored.
        :param sequence_call: boolean, indicating this is called from another scan.
        :param unit: measurement mode used.
        """
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
        self.n_frequencies = self.frequency_range.shape[0]
        self.bias_measurements = self.scan_config.get(ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS, 1)
        self.n_measurements = self.scan_config.get(ScanConfigurationKeys.AVERAGE_MEASUREMENTS, 1)

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
        """
        store_configuration

        :author: Dominik Fischer
        :date: 2026-05-14

        Stores the configuration keys and values in the provided group used for saving the measurement results, to be
        able to extract certain parameters from the data files on later analysis stages.

        :param data_group: hdf files group where the data should be stored.
        """
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
        """
        set_bias_measurement

        :author: Dominik Fischer
        :date: 2026-05-14

        Controls the applied high voltage in measurement runs where the bias voltage is not scanned, e.g. the scan
        of a full sensor in the fully depleted state.

        It will also make sure that the high voltage and the leakage current are stable before starting the measurement.

        :param data_group: hdf files group where the data should be stored.
        :param sequence_call: boolean, indicating this is called from another scan.
        """
        if ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE in self.scan_config and not sequence_call:
            # prevent compliance on power-on
            self.pixcap.bias_voltage = -0.1
            self.pixcap.bias_on()
            self.pixcap.bias_voltage = float(self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE])
            try:
                set_group_attribute(data_group, "bias_voltage", self.pixcap.bias_voltage)
            except:
                logger.error("Failed to save bias voltage as an attribute", exc_info=True)
                set_group_attribute(data_group, "bias_voltage",
                                    self.pixcap.get_smu_source_voltage(self.pixcap.bias_smu_key))

            with self.bias_without_averaging() as hv_less:
                current_measurement = self.verify_stable_hv(hv_less)
                logger.debug("The hv voltage measurement is %f V", current_measurement)

                # stabilize the currents
                current_measurement = self.verify_hv_current(hv_less)

                if np.abs(current_measurement) > self.hv_limit:
                    self.pixcap.bias_voltage = -0.1
                    logger.error("The measured current %f has exceeded the protection limit %f.",
                                 current_measurement, self.hv_limit)
                    raise ValueError("The measured current %f has exceed the protection limit %f.", )

                logger.debug(LOG_SET_BIAS % float(self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE]))

    def get_data_group(self, data_group_spec, particular_group) -> tb.Group:
        """
        get_data_group

        :author: Dominik Fischer
        :date: 2026-05-14

        Get the hdf files group where the data should be stored from the additional path specific for this scan and the
        measurements objects base group extracted from the scan configuration on initialization.

        :param data_group_spec: specifier for the last part of the groups absolute path.
        :param particular_group: specifier for the particular kind of measurement which needs a group.
        :return: h5 group to store the measurements result to.
        """
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
                back_node(data_group, data_group)

        assert isinstance(data_group, tb.Group)
        return data_group

    def store_iteration_parameters(self, freq, k: int):
        """
        store_iteration_parameters

        :author: Dominik Fischer
        :date: 2026-05-14

        Stores the parameters of the current iteration into the scan parameters mapping to be written later to disk.
        It distinguishes between a bias scan and a 'regular' scan without varying the bias voltage.
        In the first case also the currently applied bias voltage is stored.

        :param freq: frequency set for this iteration
        :param k: index of the current iteration.
        """
        if 'bias' in self.scan_config and self.has_bias_supply:
            store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq,
                                  bias_voltage=self.pixcap["BIAS_SUPPLY"].get_source_voltage())
        else:
            store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq)

    def determine_measurement_uncertainty(self, smu: str, temp_data: ndarray, sense_range=None):
        """
        determine_measurement_uncertainty

        :author: Dominik Fischer
        :date: 2026-05-14

        Determines the measurement uncertainty from reading, the measurement range and the SMU's manual.

        :param smu: dut key of the SMU
        :param temp_data: data/data array for which the uncertainty should be determined.
        :return: measurement uncertainties.
        """
        if sense_range is None:
            sense_range = self.bias_sense_range if smu == self.pixcap.bias_smu_key else self.current_sense_range
        if np.any(np.isfinite(temp_data)):
            try:
                return np.where(np.isfinite(temp_data), extract_smu_current_error(
                    self.smu_range_config[smu], temp_data, sense_range), np.nan)
            except Exception as e:
                logging.error(e.args)
                logging.exception(UNCERT_ESTIMATION_ERROR_MSG)
                logger.error(e.args)
                logger.exception(UNCERT_ESTIMATION_ERROR_MSG)
        return np.full_like(temp_data, fill_value=np.nan)

    def handle_cv_compaction(self, kwargs, unit, data_group: tb.Group):
        """
        handle_cv_compaction

        :author: Dominik Fischer
        :date: 2026-05-14

        Fetch all the biasing information and the capacitance measurement data and combine them such that only one
        large data set will remain to simplify the upcoming analysis.
        Will also estimate the uncertainties of the applied bias voltage.

        :param kwargs: further keyword arguments to be submitted as a dictionary.
        :param unit: measurement kind (regular or bias or None)
        :param data_group: hdf files group where the measurement data is stored.
        """
        # Template method, formerly abstract
        pass

    def is_unit_averaging(self, unit=None):
        """
        is_unit_averaging

        Evaluates whether the provided measurement kind is in averaging measurement mode.
        :param unit: specifies the kind of measurement.
        :return: boolean, indicating whether the measurement kind is in averaging mode.
        """
        if unit == "regular":
            return self.averaging
        elif unit == "bias":
            return self.bias_averaging
        else:
            return False

    def post_scan_handler(self, data_group, sequence_call=False, unit=None, saving_unit: Optional[str] = "regular",
                          **kwargs):
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
            if saving_unit == "regular":
                set_group_attribute(data_group, "freq_unit", "MHz")
                set_group_attribute(data_group, "current_unit", HIST_CURRENT_MEAS_UNIT)
            elif saving_unit == "bias":
                set_group_attribute(data_group, "bias_current_unit", HIST_CURRENT_MEAS_UNIT)
                set_group_attribute(data_group, "bias_voltage_unit", HIST_BIAS_MEAS_UNIT)
            self.handle_store_configuration(data_group, sequence_call)
            self.store_measurement_data(data_group, sequence_call, unit=saving_unit)
            self.handle_cv_compaction(kwargs, unit, data_group)
        finally:
            self.out_file_h5.flush()
            # do some clean-up for the performance
            gc.collect()

    def handle_store_configuration(self, data_group: tb.Group, sequence_call: bool):
        """
        handle_store_configuration

        :author: Dominik Fischer
        :date: 2026-05-14

        If the current scan is not called from another scan procedure the scan configuration should be stored, where
        the data is stored, as well.

        :param data_group: hdf group where the data is stored at last.
        :param sequence_call: boolean, whether the current scan is called from another scan procedure.
        """
        if not sequence_call:
            self.store_configuration(data_group)
        else:
            logger.debug(
                "Sequence call encountered when writing the configuration options as attribute. Skip this.")

    @contextmanager
    def binary_readout_mode(self):
        """
        binary_readout_mode

        :author: Dominik Fischer
        :date: 2026-05-14

        Contextmanager to switch the Pixcap-Measurement into binary data acquisition mode of the SMU's
        (not the HV Supply)

        When the context is ending the changes will be reverted. In this mode a static plc value of 2 is used when
        more than 5 values should be acquired for each current measurement.
        """
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

    @contextmanager
    def enhanced_readout_mode(self):
        try:
            if self.n_measurements > 5:
                self.pixcap[self.pixcap.primary_smu_key].set_current_nlpc(2)
            yield self
        finally:
            self.pixcap[self.pixcap.primary_smu_key].set_current_nlpc(10)

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
        return self.scan_config.get(ScanConfigurationKeys.SMU_CURRENT_RANGE, 0.000001)

    @property
    def current_limit(self):
        return self.scan_config.get(ScanConfigurationKeys.SMU_CURRENT_LIMIT, 0.0001)

    @property
    def bias_sense_range(self):
        """Get the current sense range for the HV supply."""
        return self.scan_config.get(ScanConfigurationKeys.BIAS_CURRENT_RANGE, 0.000001)

    @property
    def bias_limit(self):
        return self.scan_config.get(ScanConfigurationKeys.BIAS_CURRENT_LIMIT, 0.00000005)

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
        """Get the number of bias measurements to be performed."""
        return self.pixcap.n_bias_measurements

    @bias_measurements.setter
    def bias_measurements(self, value):
        self.pixcap.n_bias_measurements = value

    @property
    def col_start(self):
        """Get the first column of pixels"""
        return self.scan_config[ScanConfigurationKeys.START_COLUMN]

    @property
    def frequency_range(self) -> np.ndarray:
        """Get the range of frequencies to scan the pixels for."""
        return np.asarray(self.scan_config[ScanConfigurationKeys.FREQUENCY_RANGE])

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
        """Get whether current measurement is in averaging mode."""
        return ScanConfigurationKeys.AVERAGE_MEASUREMENTS in self.scan_config and self.scan_config[
            ScanConfigurationKeys.AVERAGE_MEASUREMENTS] > 1

    @property
    def bias_averaging(self):
        """Get whether leakage current measurement is in averaging mode.
        (Meaning: Whether multiple values are read per current measurement)
        """
        return ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS in self.scan_config and self.scan_config[
            ScanConfigurationKeys.BIAS_AVERAGE_MEASUREMENTS] > 1

    @property
    def use_bias_supply(self):
        """Whether a HV supply is available and in use for the measurements."""
        return self.bias_averaging or ScanConfigurationKeys.BIAS_VOLTAGE_SINGLE in self.scan_config or \
            ScanConfigurationKeys.BIAS_VOLTAGE_RANGE in self.scan_config

    # endregion

    # region Handling of the Pixcap SMUs
    # Handle the SMU!
    # these will now just forward the commands to the pixcap object
    def init_smu(self, voltage_range=1.5, plc=None, **kwargs):
        """
        init_smu

        @author Dominik Fischer
        @date 2026-05-14

        Measurement handler to generally initialize a SMU to the settings/state required for the measurement.
        Should only be called on measurement run configuration.

        :param voltage_range: maximum voltage which should be sourced by the SMU.
        :key current_limit: maximum current which should be measured by the SMU. Set the current protection of the SMU.
        :param plc: number of power cycles to average the measured quantity over.
        :param kwargs: further keyword arguments to be propagated to the dut. (all not explicitly named keyword
        arguments are propagated to the dut SMU handler.)
        :key plc_cycles: :ref: `plc`
        :key current_range: maximum current to be measured by the SMU.
        """
        if plc is None:
            plc = self.scan_config.get('plc_cycles', 10)
        current_limit = kwargs.pop('current_limit', self.current_limit)
        current_range = kwargs.pop('current_range', self.current_sense_range)
        self.pixcap.init_smu(self.scan_config[ScanConfigurationKeys.VIN], current_range, voltage_range,
                             current_limit, plc, **kwargs)

    def smu_on(self):
        """
        Switches the SMU on by using the Pixcap Dut. This will only turn the primary SMU on which should be connected
        to the PCB port VM3.

        @author Dominik Fischer
        @date 2026-05-14
        """
        self.pixcap.smu_on()

    def smu_off(self):
        """
        Switches the SMU off by using the Pixcap Dut. This will only turn the primary SMU off which should be connected
        to the PCB port VM3.

        @author Dominik Fischer
        @date 2026-05-14
        """
        self.pixcap.smu_off()

    def get_source_current(self) -> float:
        """
        get_source_current

        @author Dominik Fischer
        @date 2026-05-14

        Measures the current by the primary SMU connected usually to PCB port VM3.
        Only to be used for acquiring a single measurement.

        :return: float, result of single current measurement.
        """
        return self.pixcap.get_source_current

    def get_source_current_multiple(self, n: int):
        """
        get_source_current_multiple

        Performs a current measurement by reading multiple current values from the primary SMU. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the slow implementation for this purpose consisting on single measurement calls to the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured currents in A; If the SMU is not active only NaN will be returned within the
            array.
        """
        return self.pixcap.get_source_current_multiple(n)

    # Handle the biasing supply
    def init_bias_voltage(self, voltage: float = -80.0):
        """
        init_bias

        Performs the initial setup for HV SMU.
        The setup could only be performed when the SMU is connected and active.
        This setup is designed only for sourcing voltage and measuring currents.
        So the current range, over-current protection and sourcing voltage will be set according to the provided
        arguments.
        On some SMUs it will also configure the measurement buffers for acquiring multiple readings at once.

        The voltage range will be set statically to 1.5 V, the current limit to 0.001 A and the plc to 10. The current
        range will be extracted from the measurement object's properties (:ref: `bias_sense_range`).

        For further information see :ref: `pixcap.pixcap65.Pixcap65.init_bias`.

        :param voltage: sourcing voltage for the SMU.
        """
        self.pixcap.init_bias(voltage, current_range=self.bias_sense_range)

    def set_bias_on(self):
        """
        bias_on

        Turns the output of the HV SMU on.
        The call to the SMU is only performed when the SMU is connected and active.

        For further information see :ref: `pixcap.pixcap65.Pixcap65.bias_on`.
        """
        self.pixcap.bias_on()

    def set_bias_off(self):
        """
        bias_off

        Turns the output of the HV SMU off.
        The call to the SMU is only performed when the SMU is connected and active.

        For further information see :ref: `pixcap.pixcap65.Pixcap65.bias_off`.
        """
        self.pixcap.bias_off()

    @deprecated("Use directly Pixcap65.bias_voltage attribute instead.")
    def set_bias_voltage(self, voltage: float):
        """
        See :ref: `pixcap.pixcap65.Pixcap65.bias_voltage`.
        """
        self.pixcap.bias_voltage = voltage

    # endregion

    @contextmanager
    def bias_without_averaging(self):
        """
        bias_without_averaging

        :author: Dominik Fischer
        :date: 2026-05-14

        Performs all further measurements/scans within an environment where averaging and therefore multiple current
        measurements are disabled for the HV supply SMU.
        """
        if self.bias_averaging:
            # we must deactivate the averaging for the measurement
            back_n_bias_measurements = self.bias_measurements
            self.bias_measurements = 1
            try:
                yield self.pixcap
            finally:
                self.bias_measurements = back_n_bias_measurements
        else:
            yield self.pixcap

    @contextmanager
    def without_averaging(self):
        """
        without_averaging

        :author: Dominik Fischer
        :date: 2026-05-14

        Performs all further measurements/scans within an environment where averaging and therefore multiple current
        measurements are disabled in general.
        :return:
        """
        if self.averaging:
            # we must deactivate the averaging for the measurement
            back_n_measurements = self.n_measurements
            self.n_measurements = 1
            try:
                yield self.pixcap
            finally:
                self.n_measurements = back_n_measurements
        else:
            yield self.pixcap

    @property
    def hv_limit(self):
        """Get the high voltage current limit to protect the pixel sensors from breakdown."""
        return self.scan_config.get(ScanConfigurationKeys.BIAS_HV_CURRENT_LIMIT, HV_CURRENT_LIMIT)

    @property
    def bias_voltages(self):
        """Get the bias voltages to be used for a scan over the HV supply."""
        return np.asarray(self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE])

    @property
    def n_voltages(self):
        """Get the number of voltages used for a scan over the HV supply."""
        try:
            return self.bias_voltages.shape[0]
        except:
            return 1

    @property
    @abstractmethod
    def col_range(self):
        raise NotImplementedError("`col_range` is abstract and therefore not implemented.")

    @property
    @abstractmethod
    def row_range(self):
        raise NotImplementedError("`row_range` is abstract and therefore not implemented.")

    def measurement_procedure(self, data_group, sequence_call, post_hook: Callable[tb.Group] = None,
                              reversed_order=False, internal_logger=logger):
        """
        measurement_procedure

        :author: Dominik Fischer
        :date: 2026-05-14

        Contextmanager for the general procedure when measuring capacitance's over the sensor connected.
        The setup of the HV and the necessary log redirection for usage of progress bars will be handled.
        Also, the general error handling is done at this point to prevent data loss by keyboard interrupt commands
        or similar aspects.
        At last the storage of the measurement data is triggered.

        :param data_group: hdf files group where the measurement data should be stored.
        :param sequence_call: whether the scan is called from another measurement procedure.
        :param post_hook: callback to be executed after performing the scan/iteration.
        """
        if post_hook is None:
            post_hook = default_callback

        continue_error = None
        voltage_reference_data = np.full((40, 40), fill_value=np.nan, dtype=np.float64)
        try:
            if reversed_order:
                row_range = self.col_range
                col_range = self.row_range
                row_desc = "Grid column Loop"
                col_desc = "Grid row Loop"
                row_unit = "row"
            else:
                row_range = self.row_range
                col_range = self.col_range
                row_desc = "Grid row Loop"
                col_desc = "Grid column Loop"
                row_unit = "column"
            self.set_bias_measurement(data_group, sequence_call)
            self.pre_scan_handler()
            logging.info(self.mode_logging_text)
            logger.info(self.mode_logging_text)
            with logging_redirect_tqdm():
                for i_row in advanced_tqdm_iterator(row_range, tqdm_class=tqdm, desc=row_desc,
                                                    leave=not sequence_call, unit=row_unit, logger=internal_logger,
                                                    colour='green'):
                    for i_col in advanced_tqdm_iterator(col_range, tqdm_class=tqdm, desc=col_desc,
                                                        leave=False, unit="pixel", logger=internal_logger,
                                                        colour='blue'):
                        if reversed_order:
                            yield i_row, i_col
                        else:
                            yield i_col, i_row

                        # last measure the voltage at the providing SMU
                        # voltage_reference_data[i_col, i_row] = self.pixcap.vm3_measure_volts()



        except KeyboardInterrupt as e:
            logger.info("Caught KeyboardInterrupt. Will terminate the program softly.")
            continue_error = e
            continue_saving_operation = True
        else:
            continue_saving_operation = True

        if continue_saving_operation:
            self.create_carray(data_group, "SMUVoltageHist", unit="V", obj=voltage_reference_data)
            self.post_scan_handler(data_group, sequence_call, group=data_group)

        post_hook(data_group)

        if continue_error is not None:
            self.out_file_h5.flush()
            raise continue_error
        logging.info('Done')
        logger.info('Done')

    def verify_stable_current(self, smu: str):
        """
        verify_stable_current

        :author: Dominik Fischer
        :date: 2026-05-14

        Verify that a stable working point w.r.t. the leakage current through the sensor is reached for the primary SMU
        of this measurement by performing repeated current measurement and waiting for a
        sufficiently small variation between the measurements. For these current measurements always
        single measurements with a plc of 1 are used without averaging.

        :param smu: str, dut key for the SMU working as the primary SMU.
        :return: float, leakage current from last measurement cycle.
        """
        with self.without_averaging() as smu_less:
            # stabilize the currents
            i = 0
            try:
                back_binary = self.pixcap.binary_active
                self.pixcap.binary_active = False
                try:
                    self._smu_setup(smu).text_format()
                except:
                    pass
                back_nlpc = float(smu_less[smu].get_current_nlpc())
                smu_less[smu].set_current_nlpc(1)
                previous_measurement = smu_less.smu_measure_current(smu)
                # logger.info("The stabilized first current is %f", previous_measurement * 1e9)
                time.sleep(1e-3)
                for i in range(100):
                    current_measurement = smu_less.smu_measure_current(smu)
                    # logger.info("The current %i measured for stabilization is %f.", i, current_measurement * 1e9)
                    if np.abs(
                            current_measurement - previous_measurement) < 0.005 * np.abs(
                        current_measurement):
                        break
                    previous_measurement = current_measurement
                else:
                    logger.warning("Could not stabilize the current.")
            finally:
                smu_less[smu].set_current_nlpc(back_nlpc)
                if back_binary:
                    self.pixcap.binary_active = True
                    self._smu_setup(smu).binary_format()
                logger.debug("Stabilized the measurement current and set the plc to %f after %i iterations.", back_nlpc,
                             i)

    @abstractmethod
    def pre_scan_handler(self, unit=None):
        """
        pre_scan_handler

        Configures the data array such that there won't be any problem with temporarily saving the measurements
        performed during the scan.
        This should also catch any changes to the scan configuration not associated with an update call.
        Furthermore, the correct measurement routine for scan will be selected.

        :param unit: Additional unit to activate for the post scan analysis e.g. bias (it is the only implemented yet).
        """
        pass

    def plot_bias(self, data_group_spec=None, **kwargs):
        """
        plot_bias

        :author: Dominik Fischer
        :date: 2026-05-14

        Actual implementation for presenting the results of the I-V characterization.
        It's just a simple plot with error bars for the different quantities.

        :param data_group_spec: specifier of the group containing the data to be plotted.
        :param kwargs: further keyword arguments to be propagated through
        :key suffix: additional suffix to use for naming the PDF containing the plots.
        :key use_group: boolean, whether to append the group name of the measurements to the PDF name.
        """
        from matplotlib.backends.backend_pdf import PdfPages
        from pixcap65.plotting import get_pdf_name, plot_bias_delegate

        suffix = kwargs.pop("suffix", "bias_data_intern")
        use_group = kwargs.pop("use_group", False)
        pdf_name = get_pdf_name(data_group_spec, self.output_file, suffix, use_group)
        with PdfPages(pdf_name) as output_pdf:
            base_group, _ = walk_to_node(self.base_group, data_group_spec, create=False, verify_create=True)
            plot_bias_delegate(base_group.biasing.measurements, output_pdf)

    @abstractmethod
    def _handle_single_measurement(self, col, row, k):
        pass

    @abstractmethod
    def _handle_averaged_measurement(self, col, row, k):
        pass


class PixCap65TotalCap(PixCap65Measurement):
    # instantiation
    def __init__(self, scan_config, output_file, **kwargs):
        super(PixCap65TotalCap, self).__init__(scan_config, output_file, **kwargs)
        if "double_sweep" in scan_config and scan_config["double_sweep"]:
            self.n_frequencies *= 2

        # the initialization of this could be moved to configuration?
        self.hist_current = np.full(shape=(40, 40, self.n_frequencies),
                                    fill_value=np.nan)  # current value for each measured frequency per pixel
        self.hist_current_errors = np.full(shape=(40, 40, self.n_frequencies), fill_value=np.nan)
        self.n_measurements = scan_config.get(NUMBER_AVERAGE_MEASUREMENTS_KEY, 1)
        self.hist_individual_currents = np.full(shape=(40, 40, self.n_frequencies, self.n_measurements),
                                                fill_value=np.nan)

        # this could safely be moved to the super class!
        self.seq_size = self.scan_config.get(TOTAL_CAP_SEQ_SIZE_KEY, TOTAL_CAP_SEQ_SIZE)
        self.handle_measurement = self._handle_single_measurement
        self.handle_bias_measurement = self._handle_single_measurement_bias

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
        logger.debug('Waiting for settling of SMU...')
        if self.has_bias_supply:
            self.pixcap[self.pixcap.bias_smu_key].set_number_measurements(1)
            self.pixcap.bias_on()
        for _ in range(0, 30):
            current = self.get_source_current()
            if self.has_bias_supply:
                logging.debug('HV Current: {}'.format(self.pixcap.bias_measure_current()))
                logger.debug('HV Current: {}'.format(self.pixcap.bias_measure_current()))
            logging.debug('Current: {}'.format(current))
            logger.debug('Current: {}'.format(current))
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
        for i_col, i_row in self.measurement_procedure(data_group, sequence_call):
            logging.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
            logger.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
            self.dut.disable_all_pixels()
            self.dut.disable_all_columns()

            self.dut.enable_column(i_col, c.EN_EOC_3)
            self.dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)

            for k, freq in enumerate(frequency_range):
                self.pixcap.cvm_frequency = freq
                self.verify_stable_current(self.pixcap.primary_smu_key)
                self.handle_measurement(i_col, i_row, k)
                self.store_iteration_parameters(freq, k)

    def bias_cv_scan(self, data_group_spec=None):
        """
        bias_cv_scan

        Scans over different bias voltages and therefore pixel depletion states and measures the current (to later
        obtain the pixels total capacitance in dependence on the bias voltage).

        The scans necessary for the estimation of the pixels capacitances are performed by the usual scan method.

        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        """

        def _post_handle(callback_group: tb.Group):
            self.store_configuration(callback_group)

        for k, bias_voltage, data_group in self.perform_bias_scan(self.bias_voltages, _post_handle,
                                                                  self.bias_scan_parameters,
                                                                  data_group_spec, None, pbar=True,
                                                                  desc="Bias voltage scan", logger=logger):
            scan_group = self.prepare_biased_regular_scan(bias_voltage, data_group)
            self.scan_parameters = OrderedDict()
            self.scan(data_group_spec=scan_group, sequence_call=True)

    # noinspection PyMethodMayBeStatic
    def prepare_biased_regular_scan(self, bias_voltage, data_group: tb.Group) -> tb.Node:
        """
        prepare_biased_regular_scan

        :author: Dominik Fischer
        :date: 2026-05-14

        Prepare the scan of the sensor and its capacitance when performing measurements of the C-V-Characteristics.

        :param bias_voltage: bias voltage for which to perform the next scan.
        :param data_group: hdf files group to store the results to.
        :return: scan group
        """
        bias_group_name = ("bias_{bias_voltage}_V"
                           .format(bias_voltage=bias_voltage)
                           .replace('-', "M_")
                           .replace(".", "__"))
        if bias_group_name in data_group:
            # remove the biasing group or all of its contents
            scan_group = data_group[bias_group_name]
            assert isinstance(scan_group, tb.Group)
            back_node(scan_group, data_group)
        else:
            scan_group = group_get_file(data_group).create_group(where=data_group, name=bias_group_name)
        logger.info("Perform sweep for bias voltage %f.", bias_voltage)
        return scan_group

    def bias_scan(self, data_group_spec=None):
        """
        second_bias_scan

        Scan different bias voltages and measure the detector leakage current.

        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        """

        # noinspection PyUnusedLocal
        def _post_handle(callback_group: tb.Group):
            # stub function to be provided by default as there are also applications where an additional handler is
            # required
            pass

        for k, voltage, data_group in self.perform_bias_scan(self.bias_voltages, _post_handle,
                                                             self.bias_scan_parameters,
                                                             data_group_spec, "bias", logger=logger):
            self.handle_bias_measurement(k)

    def combined_bias_cv_scan(self, data_group_spec=None):
        """
        combined_bias_cv_scan

        Combines the cv characterization scan over the pixels with the measurement of the detector leakage current in
        dependence on the applied external bias voltage.
        For each bias voltage to be scanned over, first the detector leakage current is measured and then the pixel
        matrix is scanned to later obtain the pixel capacitance's with this HV applied.

        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        """

        # noinspection PyUnusedLocal
        def _post_handle(callback_group: tb.Group):
            # stub function, required as there are applications where an additional handler is needed
            pass

        for k, bias_voltage, data_group in self.perform_bias_scan(self.bias_voltages, _post_handle,
                                                                  self.bias_scan_parameters,
                                                                  data_group_spec, "bias", pbar=True,
                                                                  desc="Bias Voltage Scan", logger=logger):
            scan_group = self.prepare_biased_regular_scan(bias_voltage, data_group)
            self.handle_bias_measurement(k)
            self.scan_parameters = OrderedDict()
            self.scan(data_group_spec=scan_group, sequence_call=True)

    def plot(self, data_group_spec=None, **kwargs):
        from matplotlib.backends.backend_pdf import PdfPages
        from pixcap65.plotting import get_pdf_name, get_analysis_group

        suffix = kwargs.pop("suffix", "general_data_intern")
        if kwargs.get("use_corrected", False):
            suffix = "{}_corrected".format(suffix)
        use_group = kwargs.pop("use_group", False)

        # determine the pdf file
        pdf_name = get_pdf_name(data_group_spec, self.output_file, suffix, use_group)
        with PdfPages(pdf_name) as output_pdf:
            plot_group, _ = walk_to_node(self.base_group, data_group_spec, create=False, verify_create=True)
            plot_data_delegate(plot_group.total_cap.measurements,
                               get_analysis_group(plot_group.total_cap, **kwargs),
                               output_pdf, **kwargs)

    def analyze(self, data_group_spec=None, **kwargs):
        # handle deprecated keyword arguments.
        correction_key_value = kwargs.pop("use_corrected", None)
        if correction_key_value is not None:
            msg = "keyword argument `use_corrected` is deprecated, use `apply_correction` instead. If `apply_correction` is also present this value will take precedence, otherwise the provided value will be used. This keyword argument will be removed in the future."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            kwargs.setdefault("apply_correction", correction_key_value)
        # handle the additional PDF file in case of plotting enabled
        fit_plot_pdf_name = kwargs.pop('fit_plot_pdf_name', None)
        if "plot" in kwargs and kwargs["plot"] and fit_plot_pdf_name is not None:
            from matplotlib.backends.backend_pdf import PdfPages
            assert "fit_plot_pdf_name" not in kwargs
            with PdfPages(fit_plot_pdf_name) as pdf:
                kwargs['fit_plot_pdf'] = pdf
                self.analyze(data_group_spec=data_group_spec, **kwargs)
                return

        kargs = kwargs.copy()
        kargs.pop("plot", None)
        kargs.pop("fit_plot_pdf", None)
        kargs.pop("use_kafe2", None)
        kargs.pop("output_pdf", None)
        kargs['no_plot'] = True

        # extract the hdf file groups to perform the analysis on.
        base_group, _ = walk_to_node(self.base_group, data_group_spec, create=False, verify_create=True)

        # special handling for C-V characterization.
        reference_group = base_group.total_cap

        handle_analysis_mix_up(reference_group)
        ana_group, _ = walk_to_node(reference_group, "analysis", create=True, verify_create=True)
        assert isinstance(ana_group, tb.Group)
        analysis_data_handle(self.out_file_h5, reference_group.measurements, ana_group, **kwargs)

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
            if unit == "regular":
                process_hist = self.hist_current
                process_hist_errors = self.hist_current_errors
                process_hist_m = self.hist_individual_currents
            elif unit == "bias":
                process_hist = self.hist_bias_current
                process_hist_m = self.hist_bias_individual_currents
                process_hist_errors = self.hist_bias_current_errors
            else:
                process_hist = np.full(5, np.nan)
                process_hist_errors = np.full(5, np.nan)
                process_hist_m = np.full(5, np.nan)

            if unit is not None:
                # select the group to write the analysis results to
                if np.all(np.isnan(process_hist)):
                    raise ValueError("UNEXPECTED: All measurement entries are still NaN.")
                self.create_carray(data_group, name='HistCurr', title='Current Histogram',
                                   obj=process_hist, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT)

                # need the additional entries for the advanced averaging implementation
                if self.is_unit_averaging(unit):
                    self.create_carray(data_group, name='HistCurrValues', title='Multiple Current Histogram',
                                       obj=process_hist_m, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT)

                if np.any(np.isfinite(process_hist_errors)):
                    self.create_carray(data_group, name='HistCurrErr', title='Current Error Histogram',
                                       obj=process_hist_errors, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT)
        finally:
            assert isinstance(data_group, tb.Group)
            if unit == "regular":
                _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.scan_parameters, group=data_group)
            elif unit == "bias":
                _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.bias_scan_parameters,
                                       group=data_group)

    # interface
    def update_config(self, new_config=None):
        super(PixCap65TotalCap, self).update_config(new_config)
        self.n_measurements = self.scan_config.get(NUMBER_AVERAGE_MEASUREMENTS_KEY, 1)

        self.n_frequencies = self.frequency_range.shape[0]
        if "double_sweep" in self.scan_config and self.scan_config["double_sweep"]:
            self.n_frequencies *= 2

        self.bias_measurements = self.scan_config.get(BIASING_NUMBER_AVERAGE_MEASUREMENTS_KEY, 1)

    def handle_cv_compaction(self, kwargs: dict[str, Any], unit, data_group: tb.Group):
        """
        handle_cv_compaction

        :author: Dominik Fischer
        :date: 2026-05-14

        Fetch all the biasing information and the capacitance measurement data and combine them such that only one
        large data set will remain to simplify the upcoming analysis.
        Will also estimate the uncertainties of the applied bias voltage.

        :param kwargs: further keyword arguments to be submitted as a dictionary.
        :param unit: measurement kind (regular or bias or None)
        :param data_group: hdf files group where the measurement data is stored.
        """
        if unit == "bias" and "group" in kwargs:
            data_group = kwargs["group"]
            table = self.out_file_h5.create_table(data_group, name="BiasTable", description=BiasTable,
                                                  filters=self.filters)
            entry = table.row

            # need to handle the actually measured voltages.
            internal_parameters = data_group.scan_params[:]
            hist_parameters = data_group.BiasVoltageHist[:]
            if np.any(np.isfinite(hist_parameters)):
                if len(hist_parameters) > 1:
                    voltages = hist_parameters[:, 1]
                    voltage_errors = hist_parameters[:, 2]
                else:
                    voltages = hist_parameters
                    voltage_errors = np.full_like(voltages, np.nan)
            else:
                voltages = internal_parameters["hv_voltage"]
                voltage_errors = np.full_like(voltages, np.nan)
            if np.all(~np.isfinite(voltage_errors)):
                try:
                    voltage_errors = extract_smu_voltage_error(self.smu_range_config[self.pixcap.bias_smu_key],
                                                               voltages, 1000)
                except:
                    voltage_errors = np.full_like(voltages, np.nan)

            for set_voltage, leak_current, current_error, meas_voltage, meas_voltage_error in zip(
                    self.scan_config[ScanConfigurationKeys.BIAS_VOLTAGE_RANGE], self.hist_bias_current,
                    self.hist_bias_current_errors, voltages, voltage_errors):
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
                    logger.info(self.hist_bias_current.shape)
                    logger.info(leak_current)
                    logger.info("Errors")
                    logger.info(self.hist_bias_current_errors.shape)
                    logger.info(current_error)
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
        else:
            self.mode_logging_text = 'Scan pixel by single measurements.'
            self.handle_measurement = self._handle_single_measurement

        if self.bias_averaging:
            self.handle_bias_measurement = self._handle_averaged_measurement_bias
            if unit == "bias":
                individual_currents_shape = (self.n_voltages, self.bias_measurements)
                if self.hist_bias_individual_currents.shape != individual_currents_shape:
                    self.hist_bias_individual_currents = np.full(shape=individual_currents_shape,
                                                                 fill_value=np.nan)

        else:
            self.handle_bias_measurement = self._handle_single_measurement_bias

    def _handle_single_measurement(self, col: int, row: int, k: int):
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
    # endregion


if __name__ == '__main__':
    output_file_2 = "../data/Reference_Demo.h5"
    from pixcap65.utils import PixCapSetup, PixcapMeasurements

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
