"""
utils.py

"""
# ----------------------------------------------------------
#  Copyright (c) 2026. SiLab, Institute of Physics, University of Bonn.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ----------------------------------------------------------

import logging
import numpy as np
import time
# noinspection PyUnresolvedReferences
from basil.dut import Dut, Base
from enum import StrEnum
from typing import Optional, OrderedDict, Union

from pixcap65.pixcap_65_test_total_cap import PixCap65Measurement
from pixcap65.utility.basil_utils import extract_basil_layers

logger = logging.getLogger(__name__)


# FIXME: the location of the device directory for the firmware is difficult to handle when packaging!
# TODO: file is missing some docstrings

# general hierarchy to use:

# for general total cap measurements
# "base_path/.../total_cap/measurements/"
# "base_path/.../total_cap/analysis/"
# for cv-scan
# "base_path/.../biasing/measurements/bias_{bias_voltage}_V/"
# "base_path/.../biasing/analysis/bias_{bias_voltage}_V/"
# "base_path/.../biasing/measurements/BiasVoltageHist
# for bias scan
# "base_path/.../biasing/measurements/"

class PixcapMeasurements(StrEnum):
    """
    Enumeration of possible values to choose the measurement type/measurement class to perform measurements with using PixCap65.
    """
    TOTAL_CAPACITANCE = "total capacitance"
    INTER_CAPACITANCE = "inter-pixel capacitance"


def _remove_adjustment(adjusted, removes, sub_type):
    if sub_type in adjusted:
        for index in removes:
            del adjusted[sub_type][index]

def _mapping_handler(mapping, configuration, processed_keys, config_keys, sub_type, remove_indices, environment,
                     secondary_keys=[], hardware_keys=None):
    # some comments on the secondary keys argument
    # it should be hl_keys for registers
    # it should be tl_keys for hw
    if hardware_keys is None:
        hardware_keys = processed_keys
    for hardware in config_keys:
        if hardware in mapping and hardware not in processed_keys:
            driver = configuration[sub_type][mapping[hardware]]
            remove_indices.append(mapping[hardware])
            if "parent" in driver and sub_type != "registers":
                del driver["parent"]
            environment.append(driver)
            if sub_type != "registers" and "interface" in driver and driver["interface"] not in secondary_keys:
                secondary_keys.append(driver["interface"])
            elif "driver" in driver and driver["driver"] not in secondary_keys and sub_type == "registers":
                if not driver["driver"] or driver["driver"].lower() == "none":
                    pass
                elif driver["driver"] not in secondary_keys:
                    secondary_keys.append(driver["driver"])
            elif "hw_driver" in driver and driver["hw_driver"] not in hardware_keys:
                config_keys.append(driver["hw_driver"])
            processed_keys.append(hardware)

def default_lists(ls):
    """
    default_lists

    makes sure that these variable are not None but at least a empty list.

    :param ls: list variable (may be None)
    :return: list variable if it is not none else a empty list object
    """
    if ls is None:
        return []
    else:
        return ls


def extract_power_supply(adjusted_config: dict, environ_hl: list, hl_keys: list,
                         hl_mapping: dict, hl_remove_indices: list, tl_keys: list):
    """
    extract_power_supply

    Extract the controls of the power supply unit (PSU) from the pixcap65 configuration to control it independently
    by the PixCapSetup object.


    :param adjusted_config: configuration adjusted for use with pixcap65 after removing the power supply.
    :param environ_hl: hardware-layer of the environment configuration, which controls the psu.
    :param hl_keys: list of the hardware layer sub-keys used in the original pixcap65 configuration.
    :param hl_mapping: mapping of hardware layer keys to the index of their positioning within the config array.
    :param hl_remove_indices: hardware layer indices which corresponding elements should be removed later on.
    :param tl_keys: list of the transfer layer sub-keys used in the original pixcap65 configuration.
    """
    if "power" in hl_mapping:
        logger.info("Found the power supply in the list of devices in use.")
        power_driver = adjusted_config["hw_drivers"][hl_mapping["power"]]
        hl_remove_indices.append(hl_mapping["power"])
        if "parent" in power_driver:
            del power_driver["parent"]
            logger.info("removed some strange parent entries from power.")
        environ_hl.append(power_driver)
        if "interface" in power_driver and power_driver["interface"] not in tl_keys:
            tl_keys.append(power_driver["interface"])
        elif "hw_driver" in power_driver and power_driver["hw_driver"] not in hl_keys:
            hl_keys.append(power_driver["hw_driver"])
        # we should know that the power hardware layer is in use by this and not by the dut.!
        hl_keys.append("power")


class PixCapSetup(Dut):
    """
    Outer Wrapper class to separate power supply of the setup (and therefore preparing of the setup itself) from the
    actual measurements (scripts/classes).

    Will act like a context manager of the requested measurement type/class.
    """
    # TODO: remove this here in favour of an property.
    pixcap: Optional[PixCap65Measurement]

    def __init__(self, scan_config, output_file, config="pixcap65.yaml", name="PixcapSetup", measurement: Union[str, type, None] = None, hl_keys=None, tl_keys=None, rl_keys=None):
        if measurement is None:
            measurement = PixcapMeasurements.TOTAL_CAPACITANCE
        assert isinstance(measurement, PixcapMeasurements) or isinstance(measurement, str) or (
                isinstance(measurement, type) and issubclass(measurement,
                                                             PixCap65Measurement))

        # we need to catch the None case first here
        # FIXME: there parts of the data repositories packaged as well => these need to be removed.

        from pixcap65.pixcap.pixcap65_measurement import configuration_context
        with configuration_context(config) as context:
            # CHECK: should the configuration be updated right here for further usage?
            adjusted_config = self.init_environment(context, hl_keys, name, rl_keys, tl_keys)
        logger.debug("For the handling of the setup, we'll use the config:\n %s", str(self._environ_config))
        super(PixCapSetup, self).__init__(conf=self._environ_config)

        # get the correct pixcap measurement class
        pix_args = {
            "scan_config": scan_config,
            "output_file": output_file,
            "pix_config": adjusted_config,
        }
        if isinstance(measurement, str):
            self.measurement_arguments = pix_args
            match measurement:
                case PixcapMeasurements.TOTAL_CAPACITANCE:
                    from pixcap65.pixcap_65_test_total_cap import PixCap65TotalCap
                    self.measurement_class = PixCap65TotalCap
                case PixcapMeasurements.INTER_CAPACITANCE:
                    from pixcap65.pixcap_65_test_inter_cap import Pixcap65InterCap
                    self.measurement_class = Pixcap65InterCap
                case _:
                    raise ValueError("provided measurement class does not exist.")
            self.pixcap = None
        elif issubclass(measurement, PixCap65Measurement):
            self.measurement_class = measurement
            self.measurement_arguments = pix_args
            self.pixcap = None
        else:
            raise ValueError("The provided measurement object is not suitable. No measurement object provided.")

    def init_environment(self, context, hl_keys, name: str, rl_keys, tl_keys) -> dict:
        """
        init_environment

        Initialize the 'environment', prepare the configuration for switching the whole setup on or off.
        :param context: configuration to start, defines the configuration context
        :param hl_keys: list of the hardware layer sub-keys used in the original pixcap65 configuration.
        :param name: name of the setup.
        :param rl_keys: list of the register sub-keys used in the original pixcap65 configuration.
        :param tl_keys: list of the transfer layer sub-keys used in the original pixcap65 configuration.
        :return:
        """
        temp_dut = Base(context)
        adjusted_config = temp_dut._conf.copy()
        self._environ_config = OrderedDict()
        environ_registers = []
        environ_tl = []
        environ_hl = []

        rl_keys = default_lists(rl_keys)
        hl_keys = default_lists(hl_keys)
        tl_keys = default_lists(tl_keys)

        # map the names of the layer items to their index in the layer list
        hl_mapping, tl_mapping, rl_mapping = extract_basil_layers(adjusted_config)

        # remember which keys are already processed.
        processed_rl_keys = []
        processed_hl_keys = ["power"]
        processed_tl_keys = []

        # remember which indices should be removed from the layers for the start of pixcap.
        tl_remove_indices = []
        hl_remove_indices = []
        rl_remove_indices = []

        # process the provided config
        extract_power_supply(adjusted_config, environ_hl, hl_keys, hl_mapping, hl_remove_indices, tl_keys)

        # for register in rl_keys:
        #     if register in rl_mapping and register not in processed_rl_keys:
        #         driver = adjusted_config["registers"][rl_mapping[register]]
        #         rl_remove_indices.append(rl_mapping[register])
        #         environ_registers.append(driver)
        #         if "driver" in driver:
        #             if not driver["driver"] or driver["driver"].lower() == "none":
        #                 pass
        #             elif driver["driver"] not in hl_keys:
        #                 hl_keys.append(driver["driver"])
        #         elif "hw_driver" in driver and driver["hw_driver"] not in hl_keys:
        #             hl_keys.append(driver["hw_driver"])
        #         processed_rl_keys.append(register)
        _mapping_handler(rl_mapping, adjusted_config, processed_rl_keys, rl_keys, "registers", rl_remove_indices,
                         environ_registers, hl_keys, hl_keys)

        # we need to run this twice due to the dependence on the hardware drivers
        while np.any(np.array([entry not in processed_hl_keys for entry in hl_keys])):
            # for hardware in hl_keys:
            #     if hardware in hl_mapping and hardware not in processed_hl_keys:
            #         driver = adjusted_config["hw_drivers"][hl_mapping[hardware]]
            #         hl_remove_indices.append(hl_mapping[hardware])
            #         if "parent" in driver:
            #             del driver["parent"]
            #         environ_hl.append(driver)
            #         if "interface" in driver and driver["interface"] not in tl_keys:
            #             tl_keys.append(driver["interface"])
            #         elif "hw_driver" in driver and driver["hw_driver"] not in processed_hl_keys:
            #             hl_keys.append(driver["hw_driver"])
            #         processed_hl_keys.append(hardware)
            _mapping_handler(hl_mapping, adjusted_config, processed_hl_keys, hl_keys, "hw_drivers", hl_remove_indices,
                             environ_hl)

        _mapping_handler(tl_mapping, adjusted_config, processed_tl_keys, tl_keys, "transfer_layer", tl_remove_indices, environ_tl)
        # for connection in tl_keys:
        #     if connection in tl_mapping and connection not in processed_tl_keys:
        #         driver = adjusted_config["transfer_layer"][tl_mapping[connection]]
        #         tl_remove_indices.append(tl_mapping[connection])
        #         if "parent" in driver:
        #             del driver["parent"]
        #         environ_tl.append(driver)
        #         processed_tl_keys.append(connection)

        # check for conflicts between the setup delegation and the dut.
        for hardware, hw_idx in hl_mapping.items():
            if hardware not in hl_keys:
                if "interface" in adjusted_config["hw_drivers"][hw_idx] and adjusted_config["hw_drivers"][hw_idx][
                    "interface"] in tl_keys:
                    print(hardware)
                    print("upper")
                    print(hl_keys)
                    raise RuntimeError(
                        "Detected attempt to use a common transfer layer for controlling the setup and the dut.")
                elif "hw_driver" in adjusted_config["hw_drivers"][hw_idx] and adjusted_config["hw_drivers"][hw_idx][
                    "hw_driver"] in hl_keys:
                    print(hardware)
                    raise RuntimeError(
                        "Detected attempt to use a common hardware layer for controlling the setup and the dut.")

        tl_remove_indices.sort()
        hl_remove_indices.sort()
        rl_remove_indices.sort()
        _remove_adjustment(adjusted_config, rl_remove_indices, "registers")
        _remove_adjustment(adjusted_config, hl_remove_indices, "hw_drivers")
        _remove_adjustment(adjusted_config, tl_remove_indices, "transfer_layer")
        # if "registers" in adjusted_config:
        #     for index in rl_remove_indices:
        #         del adjusted_config["registers"][index]
        #
        # if "hw_drivers" in adjusted_config:
        #     for index in hl_remove_indices:
        #         del adjusted_config["hw_drivers"][index]
        #
        # if "transfer_layer" in adjusted_config:
        #     for index in tl_remove_indices:
        #         del adjusted_config["transfer_layer"][index]

        logger.debug("The adjusted pixcap config is:\n %s", str(adjusted_config))

        # build our own configuration and setup together
        self._environ_config["name"] = name
        self._environ_config["version"] = 0.01
        self._environ_config["transfer_layer"] = environ_tl
        self._environ_config["hw_drivers"] = environ_hl
        self._environ_config["registers"] = environ_registers
        return adjusted_config

    def close(self):
        """
        close

        Finish and shutdown the whole setup.
        """
        try:
            self["power"].set_enable(0, channel=1)
            self["power"].set_enable(0, channel=2)
            self["power"].set_enable(0, channel=3)
            import serial


        except:
            logger.error("Failed to clean up the setup handling.")
        super(PixCapSetup, self).close()

    def __enter__(self):
        Dut.init(self)
        from basil.HL.tti_ql355tp import ttiQl355tp
        assert isinstance(self["power"], ttiQl355tp)
        self["power"].identify_device()

        try:
            # RESET the power distribution and therefore the boards
            self["power"].set_enable(0, channel=1)
            self["power"].set_enable(0, channel=2)
            self["power"].set_enable(0, channel=3)
            self["power"].set_voltage(5.0, channel=1)
            self["power"].set_voltage(1.0, channel=2)
            self["power"].set_current_limit(0.800, channel=1)
            self["power"].set_current_limit(0.001, channel=2)
            time.sleep(5)
            self["power"].set_enable(1, channel=1)
            self["power"].set_enable(1, channel=2)
            self["power"].set_enable(1, channel=3)
            time.sleep(7)
            print(self["power"].get_current(channel=1))

            # init the pixcap system
            if "out_file" in self.measurement_arguments and "output_file" not in self.measurement_arguments:
                self.measurement_arguments["output_file"] = self.measurement_arguments["out_file"]
                logger.warning("added output_file argument")

            try:
                self.pixcap = self.measurement_class(**self.measurement_arguments)
            except TypeError:
                logger.error("Failed to instantiate the measurement object.")
                logger.error(self.measurement_arguments)
                logger.error(self.measurement_arguments.keys())
                raise
            assert self.pixcap is not None
            self.pixcap.configure()
            return self.pixcap
        except:
            logger.error("Failed to switch the Pixcap setup on.", exc_info=True)
            try:
                if self.pixcap is not None:
                    self.pixcap.close()
                    self.pixcap = None
            finally:
                self.close()
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.pixcap is not None
        self.pixcap.close()
        self.pixcap = None
        self.close()
        return False


if __name__ == "__main__":
    from pixcap_65_test_total_cap import scan_configuration

    with PixCapSetup(scan_configuration, "Setup_Demonstration.h5",
                     measurement=PixcapMeasurements.TOTAL_CAPACITANCE) as setup:
        pass
