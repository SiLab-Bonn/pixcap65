import logging
import time
from enum import StrEnum
from typing import Optional, OrderedDict

import numpy as np
from basil.HL.scpi import scpi
from basil.dut import Dut, Base

from pixcap_65_test_total_cap import PixCap65Measurement

logger = logging.getLogger(__name__)


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
    TOTAL_CAPACITANCE = "total capacitance"
    INTER_CAPACITANCE = "inter-pixel capacitance"

class PixCapSetup(Dut):
    pixcap: Optional[PixCap65Measurement]

    def __init__(self, scan_config, output_file, config="", name="PixcapSetup", measurement=None, hl_keys=None, tl_keys=None, rl_keys=None):
        if measurement is None:
            measurement = PixcapMeasurements.TOTAL_CAPACITANCE
        assert isinstance(measurement, PixcapMeasurements) or issubclass(measurement, PixCap65Measurement)
        temp_dut = Base(config)
        adjusted_config = temp_dut._conf.copy()
        self._environ_config = OrderedDict()
        environ_registers = []
        environ_tl = []
        environ_hl = []

        if rl_keys is None:
            rl_keys = []
        
        if hl_keys is None:
            hl_keys = []

        if tl_keys is None:
            tl_keys = []

        rl_mapping = {}
        tl_mapping = {}
        hl_mapping = {}

        processed_rl_keys = []
        processed_hl_keys = ["power"]
        processed_tl_keys = []

        tl_remove_indices = []
        hl_remove_indices = []
        rl_remove_indices = []

        if "transfer_layer" in adjusted_config:
            for idx, layer in enumerate(adjusted_config["transfer_layer"]):
                if "name" in layer:
                    tl_mapping[layer["name"]] = idx
                else:
                    logger.info("Transfer layer at %i has no name. Will skip it.", idx)

        if "hw_drivers" in adjusted_config:
            for idx, layer in enumerate(adjusted_config["hw_drivers"]):
                if "name" in layer:
                    hl_mapping[layer["name"]] = idxh in rl_
                else:
                    logger.info("Hardware driver at %i has no name. Will skip it.", idx)

        if "registers" in adjusted_config:
            for idx, layer in enumerate(adjusted_config["registers"]):
                if "name" in layer:
                    rl_mapping[layer["name"]] = idx
                else:
                    logger.info("Register at %i has no name. Will skip it.", idx)

        # process the provided config
        if "power" in hl_mapping:
            power_driver = adjusted_config["hw_drivers"][hl_mapping["power"]]
            hl_remove_indices.append(hl_mapping["power"])
            environ_hl.append(power_driver)
            if "interface" in power_driver and power_driver["interface"] not in tl_keys:
                tl_keys.append(power_driver["interface"])
            elif "hw_driver" in power_driver and power_driver["hw_driver"] not in hl_keys:
                hl_keys.append(power_driver["hw_driver"])

        for register in rl_keys:
            if register in rl_mapping and register not in processed_rl_keys:
                driver = adjusted_config["registers"][rl_mapping[register]]
                rl_remove_indices.append(rl_mapping[register])
                environ_registers.append(driver)
                if "driver" in driver:
                    if not driver["driver"] or driver["driver"].lower() == "none":
                        pass
                    elif driver["driver"] not in hl_keys:
                        hl_keys.append(driver["driver"])
                elif "hw_driver" in driver and driver["hw_driver"] not in hl_keys:
                    hl_keys.append(driver["hw_driver"])
                processed_rl_keys.append(register)

        # we need to run this twice due to the dependence on the hardware drivers
        while np.any(np.array([entry not in processed_hl_keys for entry in hl_keys])):
            for hardware in hl_keys:
                if hardware in hl_mapping and hardware not in processed_hl_keys:
                    driver = adjusted_config["hw_drivers"][hl_mapping[hardware]]
                    hl_remove_indices.append(hl_mapping[hardware])
                    environ_hl.append(driver)
                    if "interface" in driver and driver["interface"] not in tl_keys:
                        tl_keys.append(driver["interface"])
                    elif "hw_driver" in driver and driver["hw_driver"] not in processed_hl_keys:
                        hl_keys.append(driver["hw_driver"])
                    processed_hl_keys.append(hardware)

        for connection in tl_keys:
            if connection in tl_mapping and connection not in processed_tl_keys:
                driver = adjusted_config["transfer_layer"][tl_mapping[connection]]
                tl_remove_indices.append(tl_mapping[connection])
                environ_tl.append(driver)
                processed_tl_keys.append(connection)

        # check for conflicts betweent the setup delegation and the dut.
        for hardware, hw_idx in hl_mapping.items():
            if hardware not in hl_keys:
                if "interface" in adjusted_config["hw_drivers"][hw_idx] and adjusted_config["hw_drivers"][hw_idx]["interface"] in tl_keys:
                    raise RuntimeError("Detected attempt to use a common transfer layer for controlling the setup and the dut.")
                elif "hw_driver" in adjusted_config["hw_drivers"][hw_idx] and adjusted_config["hw_drivers"][hw_idx]["hw_driver"] in hl_keys:
                    raise RuntimeError("Detected attempt to use a common hardware layer for controlling the setup and the dut.")
                

        tl_remove_indices.sort()
        hl_remove_indices.sort()
        rl_remove_indices.sort()
        if "registers" in adjusted_config:
            for index in rl_remove_indices:
                del adjusted_config["registers"][index]

        if "hw_drivers" in adjusted_config:
            for index in hl_remove_indices:
                del adjusted_config["hw_drivers"][index]

        if "transfer_layer" in adjusted_config:
            for index in tl_remove_indices:
                del adjusted_config["transfer_layer"][index]

        print(adjusted_config)

        # build our own configuration and setup together
        self._environ_config["name"] = name
        self._environ_config["version"] = 0.01
        self._environ_config["transfer_layer"] = environ_tl
        self._environ_config["hw_drivers"] = environ_hl
        self._environ_config["registers"] = environ_registers
        print(self._environ_config)
        super(PixCapSetup, self).__init__(config=self._environ_config)





        # get the correct pixcap measurement class
        assert False, "Just a enforced exit from this class to check the dictionary modification behaviour."
        pix_args = {
            "scan_config"   :   scan_config,
            "output_file"   :   output_file,
            "pix_config"    :   adjusted_config,
        }
        if issubclass(measurement, PixcapMeasurement):
            self.pixcap = measurement(**pix_args)
        match (measurement):
            case PixcapMeasurements.TOTAL_CAPACITANCE: 
                from pixcap_65_test_total_cap import PixCap65TotalCap
                self.pixcap = PixCap65TotalCap(**pix_args)
            case PixcapMeasurements.INTER_CAPACITANCE:
                from pixcap_65_test_inter_cap import Pixcap65TestInterCap
                self.pixcap = Pixcap65TestInterCap(**pix_args)
            case _:
                raise ValueError(f"provided measurement class does not exist.")

    def __enter__(self):
        Dut.init()
        self["power"].set_enable(0, channel=1)
        self["power"].set_enable(0, channel=2)
        time.sleep(5)
        self["power"].set_enable(1, channel=1)
        self["power"].set_enable(1, channel=2)
        time.sleep(5)
        self.pixcap.configure()
        return self.pixcap

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pixcap.close()
        self.pixcap = None
        self.close()
        return False


if __name__ == "__main__":
    dut = Dut("demo.yaml")
    dut.init()
    smu = dut["SMU"]
    assert isinstance(smu, scpi)
    src_u = 1.0
    current_range = 0.00001
    voltage_range=1.5
    current_limit=0.001
    plc=10
    smu.off(channel=1)
    smu.clear_buffer1(channel=1)
    smu.clear_buffer2(channel=1)
    smu.set_buffer1_mode(0, channel=1)
    smu.set_buffer2_mode(0, channel=1)
    smu.source_volt(channel=1)
    smu.set_voltage_range(voltage_range, channel=1)
    smu.set_current_nlpc(plc, channel=1)
    smu.set_voltage(src_u, channel=1)
    smu.set_current_limit(current_limit, channel=1)
    smu.set_current_sense_range(current_range, channel=1)
    dut["SMU"].on(channel=1)
    time.sleep(5)
    print(dut["SMU"].get_current(channel=1))
    dut["SMU"].off(channel=1)
    dut.close()