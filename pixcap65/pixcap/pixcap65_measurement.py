
# ----------------------------------------------------------
#  Copyright (c) .
#   All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------

from __future__ import annotations

try:
    # python 3.7 and above implementations for these operations
    from importlib.resources import files, as_file
except ImportError:
    # for python 2.x or python 3.x with x < 7 we need a backport here
    from importlib_resources import files, as_file


import logging
import numpy as np
import os
import sys
import tables as tb
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from enum import StrEnum
from tqdm.contrib import DummyTqdmFile
from typing import Mapping
from warnings import warn

from pixcap65.pixcap.pixcap65 import Pixcap65
from pixcap65.pixcap.pixcap65_measurements import NUMBER_AVERAGE_MEASUREMENTS_KEY, \
    BIASING_NUMBER_AVERAGE_MEASUREMENTS_KEY
from pixcap65.pixcap.pixcap_structure import BasilConfigKeys
from pixcap65.utility.utils_2 import walk_to_node

logger = logging.getLogger(__name__)


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

def load_firmware_main():
    """
        load_firmware

        @author: Dominik Fischer
        @date: 2026-08-13
        last update: 2026-08-13

        Module function to load the firmware from modules resources and write to a file (if this file does not already exist)
        to be loaded later on init of the measurement classes or the PixCap65 class itself.
        """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", action="store", default=None, help="configuration file or configuration mapping from which to determine the target location of the firmware.")

    arguments = parser.parse_args()
    load_firmware(arguments.config)

def load_configuration_main():
    """
        load_configuration

        @author: Dominik Fischer
        @date: 2026-08-13
        last update: 2026-08-13

        Module function to load the default configuration from the module/package resources and write to a file. (on disk)
        Such that it could be used for initializing the PixCap65 class or the corresponding measurement classes.
        """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-t", "--target", action="store", default=None, help="path where to write the default configuration to.")
    arguments = parser.parse_args()
    load_configuration(arguments.target)

# TODO: test implementations on the actual device
def load_firmware(config=None):
    """
    load_firmware

    @author: Dominik Fischer
    @date: 2026-08-13
    last update: 2026-08-13

    Module function to load the firmware from modules resources and write to a file (if this file does not already exist)
    to be loaded later on init of the measurement classes or the PixCap65 class itself.

    :param config: configuration file or configuration mapping from which to determine the target location of the firmware.
    """
    # when used as a script, we need to determine the parameters by some other means.
    print("loading firmware")
    with configuration_context(config=config) as safe_config:
        # from here we need to extract the firmware location!
        dut = Pixcap65(safe_config)
        for transfer in dut._conf[BasilConfigKeys.TRANSFER_LAYER]:
            print("new entry")
            print(type(transfer))
            if 'type' not in transfer or not transfer['type'] == 'SiUsb':
                continue
            if 'bit_file' not in transfer['init']:
                continue

            guess_path = transfer['init']['bit_file']
            print("What about the path to guess?", guess_path, os.path.abspath(guess_path))
            if not os.path.exists(guess_path):
                resource_firmware = files("pixcap65").joinpath('device', 'ise', 'pixcap65.bit')
                assert resource_firmware.is_file()
                if not os.path.exists(os.path.dirname(guess_path)):
                    os.makedirs(os.path.dirname(guess_path), exist_ok=True)
                with open(guess_path, 'wb') as bit_file:
                    with as_file(resource_firmware) as firmware:
                        print("Will write the firmware", firmware)
                        with open(firmware, 'rb') as guess_file:
                            bit_file.write(guess_file.read())

                break
        del dut


def load_configuration(target_path="pixcap65.yaml"):
    """
    load_configuration

    @author: Dominik Fischer
    @date: 2026-08-13
    last update: 2026-08-13

    Module function to load the default configuration from the module/package resources and write to a file. (on disk)
    Such that it could be used for initializing the PixCap65 class or the corresponding measurement classes.

    :param target_path: path where to write the default configuration to.
    """
    packaged_config = files("pixcap65").joinpath("pixcap", "pixcap65.yaml")
    assert packaged_config.is_file()
    with packaged_config.open('r') as pack:
        data = pack.read()
        with open(target_path, 'w') as target:
            target.write(data)


class Pixcap65BaseMeasurement(MeasurementAbstract, metaclass=ABCMeta):
    """
    Pixcap65BaseMeasurement

    Some measurements methods and handlers for initializing the setup and stopping connection to the setup
    appropriately when finished.
    In particular it makes sure that the firmware and the configuration files are available to the measurement classes
    when they are needed.
    This base class also provides a context manager implementation for the actually implementing measurement subclasses.
    """
    def __init__(self, pix_config=None, **kwargs):
        super(Pixcap65BaseMeasurement, self).__init__(**kwargs)
        with configuration_context(pix_config) as config:
            # init the dut
            self.dut = Pixcap65(config)

        # need to intercept firmware loading for MIO right here.
        for transfer in self.dut._conf[BasilConfigKeys.TRANSFER_LAYER]:
            if 'type' not in transfer or not transfer['type'] == 'SiUsb':
                continue
            if 'bit_file' not in transfer['init']:
                continue

            guess_path = transfer['init']['bit_file']
            if not os.path.exists(guess_path):
                resource_firmware = files("pixcap65").joinpath('device', 'ise', 'pixcap65.bit')
                assert resource_firmware.is_file()
                with as_file(resource_firmware) as firmware:
                    transfer['init']['bit_file'] = firmware
                    self.dut.init()
                    break
        else:
            self.dut.init()

        # prepare measurements and configuration
        self.mode_logging_text = 'Scan pixel by single measurements.'
        self.smu_range_config = {}
        self._group = None
        self.scan_config = None
        self.out_file_h5 = None

    def close(self):
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
        assert self._group is not None
        assert isinstance(self._group, tb.Group)
        return self._group

    @base_group.setter
    def base_group(self, value):
        temp_node, _ = walk_to_node(self.out_file_h5.root, value, create=True, verify_create=True)
        assert isinstance(temp_node, tb.Group)
        self._group = temp_node

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
    @property
    @abstractmethod
    def row_range(self):
        raise NotImplementedError("`row_range` is abstract and therefore not implemented.")

    @property
    @abstractmethod
    def col_range(self):
        raise NotImplementedError("`col_range` is abstract and therefore not implemented.")


@contextmanager
def configuration_context(config):
    """
    configuration_context

    @author: Dominik Fischer
    last update: 2026-08-13

    Utility to function to fetch and verify the correct configuration mapping for the setup to provide basil with correct information about the connected lab devices.
    Will yield the final configuration mapping.

    :param config: configuration Mapping or path to configuration file
    """
    try:
        # What about correct firmware entries here?
        # perhaps we should better read the mapping up-front?
        if isinstance(config, Mapping) or hasattr(config, "read"):
            yield config
        elif config is not None and not os.path.exists(config):
            # FIXME: these here will require that config is not None!
            if os.path.isdir("pixcap"):
                new_path = os.path.join("pixcap", config)
                assert os.path.exists(new_path)
                yield new_path
            elif os.path.dirname(config) == "pixcap":
                new_path = os.path.join("../..", config)
                new_path = os.path.normpath(new_path)
                assert os.path.exists(new_path)
                yield new_path
            else:
                packaged_config = files("pixcap65").joinpath("pixcap", "pixcap65.yaml")
                assert packaged_config.is_file()
                yield packaged_config.open('r')
        elif config is None:
            packaged_config = files("pixcap65").joinpath("pixcap", "pixcap65.yaml")
            assert packaged_config.is_file()
            yield packaged_config.open('r')
        else:
            yield config
    except AssertionError, FileNotFoundError:
        logger.error("Could not find neither the default configuration file nor a configuration file with the provided name.")
        raise


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
