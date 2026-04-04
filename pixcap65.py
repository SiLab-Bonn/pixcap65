#
# ------------------------------------------------------------
# Copyright (c) SILAB , Physics Institute of Bonn University
# ------------------------------------------------------------
#
# SVN revision information:
#  $Rev:: 418                   $:
#  $Author:: HK    $:
#  $Date:: 2015-01-04 10:56:36 #$:
#

import logging
import time

import numpy as np
from basil.dut import Dut

import pixcap65_constants as c

# perhaps add the channel information to the pixcap config file and extract it from here!
float_initialiser = np.float32


class Pixcap65(Dut):
    __smu_kwargs = {}
    __bias_kwargs = {}
    __bias_smu_key = 'BIAS_SUPPLY'

    def init(self, init_conf=None, **kwargs):
        Dut.init(self, init_conf=init_conf, **kwargs)

        # extract additional SMU configuration from the config file
        # its a bit dirty but it should work for now.
        # if 'hw_drivers' in self._conf:
        #     for driver in self._conf['hw_drivers']:
        #         if 'name' in driver and driver['name'] == "SMU":
        #             smu_config = driver
        #             break
        #     else:
        #         logging.error("No SMU device found in the hardware configuration file.")
        #         smu_config = {}
        #
        #     if 'pixcap_init' in smu_config:
        #         if 'single_channel' in smu_config['pixcap_init']:
        #             if smu_config['pixcap_init']['single_channel']:
        #                 common_smu_possible = False
        #             else:
        #                 self.__smu_kwargs['channel'] = smu_config['pixcap_init']['channel']
        #         elif 'channel' in smu_config['pixcap_init']:
        #             self.__smu_kwargs['channel'] = smu_config['pixcap_init']['channel']
        #         else:
        #             common_smu_possible = False
        #
        #     # extract the additional configuration for the biasing supply
        #     # if no further configuration could be found, use the default values
        #     if 'transfer_layer' not in self._conf:
        #         tl_config = {}
        #         primary_interface = smu_config['interface']
        #         for entry in self._conf['transfer_layer']:
        #             if 'type' in entry and entry['type'] == 'Serial':
        #                 serial_interface = entry['name']
        #                 interface_drivers = []
        #                 for driver in self._conf['hw_drivers']:
        #                     if 'interface' not in driver:
        #                         continue
        #                     if driver['interface'] != serial_interface:
        #                         continue
        #                     if 'type' not in driver:
        #                         continue
        #                     if driver['type'] != 'scpi':
        #                         continue
        #                     interface_drivers.append(driver)
        #                 tl_config[serial_interface] = interface_drivers.copy()
        #
        #         # check the additional devices on the primary interface
        #         suitable_devices = []
        #         for device in tl_config[primary_interface]:
        #             if device['name'] == "SMU":
        #                 continue
        #             suitable_devices.append(device)
        #         for serial in tl_config.keys():
        #             if serial == primary_interface:
        #                 continue
        #             suitable_devices.extend(tl_config[serial])
        #
        #         if len(suitable_devices) == 0:
        #             assert common_smu_possible, "No suitable SMU device found on any serial interface."
        #             self.__bias_smu_key = 'SMU'
        #             bias_config = smu_config
        #         elif len(suitable_devices) == 1:
        #             self.__bias_smu_key = suitable_devices[0]['name']
        #             bias_config = suitable_devices[0]
        #         else:
        #             logging.warning("More than one SMU device found on the serial interfaces. Will choose the first beginning with SMU.")
        #             for device in suitable_devices:
        #                 if device['name'].startswith('SMU'):
        #                     self.__bias_smu_key = device['name']
        #                     bias_config = device
        #                     break
        #             else:
        #                 raise Exception("More than one SMU device found on the serial interfaces.")
        #
        #         # extract the additional configuration for the biasing supply
        #         if 'pixcap_init' in bias_config:
        #             if 'single_channel' in bias_config['pixcap_init'] and not bias_config['pixcap_init'][
        #                 'single_channel']:
        #                 self.__bias_kwargs['channel'] = bias_config['pixcap_init']['channel']
        #             elif 'channel' in smu_config['pixcap_init']:
        #                 self.__bias_kwargs['channel'] = bias_config['pixcap_init']['channel']
        #     else:
        #         logging.error("The hardware configuration file does not contain the transfer layer configuration.")
        #
        # else:
        #     logging.error("The hardware configuration file does not contain the hardware drivers.")

        # setup the chip
        self.switch_on_power_supply_voltages(1)
        self.init_config()
        # self['SPI'].set_size(9960)
        # self.reset_chip()

    def close(self):
        self.switch_on_power_supply_voltages(0)
        self['SEQ'].clear()
        Dut.close(self)

    def switch_on_power_supply_voltages(self, pwr_en):
        """
        Switches on default supply voltages
        """
        if (pwr_en):
            self['VDD'].set_current_limit(100, unit='mA')
            # Power
            self['VDD'].set_voltage(1.2, unit='V')
            self['VDD'].set_enable(pwr_en)

            time.sleep(1.0)
            print('')
            print('VDD:\t', format(self['VDD'].get_voltage(unit='V'), '.3f'), 'V\t',
                  format(self['VDD'].get_current(unit='mA'), '.3f'), 'mA')
            print('')
        else:
            self['VDD'].set_enable(pwr_en)

    def get_status(self):
        status = {}
        status['Time'] = time.strftime("%d %M %Y %H:%M:%S")
        status['VDD'] = {'voltage(V)': format(self['VDD'].get_voltage(unit='V'), '.3f'),
                         'current(mA)': format(self['VDD'].get_current(unit='mA'), '.3f')}
        return status

    def enable_pixel_clk(self, i_col, i_pix, clk_mask):
        self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['CLK_EN'] = clk_mask
        self.update_spi()

    def enable_pixel_bias(self, i_col, i_pix, bias_mask):
        self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['SEL'] = bias_mask
        self.update_spi()

    def enable_column(self, i_col, EOC_MASK):
        self['SPI']['COL'][c.COL_NMAX - i_col]['EOC'] = EOC_MASK
        self.update_spi()

    def disable_all_pixels(self):
        for i_col in range(0, c.COL_NMAX + 1):
            for i_pix in range(0, c.PIX_NMAX + 1):
                self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['CLK_EN'] = 0
                self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['SEL'] = 0
        self.update_spi()

    def update_spi(self):
        self['SPI'].write()
        self['SPI'].start()

    def disable_all_columns(self):
        for i_col in range(0, c.COL_NMAX + 1):
            self['SPI']['COL'][c.COL_NMAX - i_col]['EOC'] = 0
        self.update_spi()

    def reset_chip(self):
        # reset shift register
        self['GPIO']['RST_B'] = 0
        self['GPIO'].write()
        time.sleep(0.1)
        self['GPIO']['RST_B'] = 1
        self['GPIO'].write()

    def init_config(self):
        self['SPI'].set_size(9960)
        self.reset_chip()

    # Handle the SMU!
    def init_smu(self, src_u, current_range, voltage_range=1.5, current_limit=0.001, plc=10):
        self['SMU'].off(**self.smu_kwargs)
        self['SMU'].clear_buffer1(**self.smu_kwargs)
        self['SMU'].clear_buffer2(**self.smu_kwargs)
        self['SMU'].set_buffer1_mode(0, **self.smu_kwargs)
        self['SMU'].set_buffer2_mode(0, **self.smu_kwargs)
        self['SMU'].source_volt(**self.smu_kwargs)
        self['SMU'].set_voltage_range(voltage_range, **self.smu_kwargs)
        self['SMU'].set_current_nlpc(plc, **self.smu_kwargs)
        self['SMU'].set_voltage(src_u, **self.smu_kwargs)
        self['SMU'].set_current_limit(current_limit, **self.smu_kwargs)
        self['SMU'].set_current_sense_range(current_range, **self.smu_kwargs)

    def smu_on(self):
        self['SMU'].on(**self.smu_kwargs)

    @property
    def get_source_current(self) -> float:
        result = self['SMU'].get_current(**self.smu_kwargs)
        if not (isinstance(result, float)
                or isinstance(result, int)
                or isinstance(result, np.float32)
                or isinstance(result, np.float64)
                or isinstance(result, str)):
            print(type(result), result)
            raise Exception(f"The current returned {result} which was not recognised as a format.")
        if isinstance(result, str) and ',' in result:
            current = float_initialiser(result.split(',')[1])
        else:
            current = float_initialiser(result)
        if np.isnan(current):
            logging.warning("It was a NaN value measured by the SMU.")
            raise Exception(f"The current returned {current} was not recognised as a number.")
        return current

    def averaged_current(self, n: int = 10):
        self['SMU'].set_number_measurements(n, **self.smu_kwargs)
        self['SMU'].multi_current_measurement(**self.smu_kwargs)
        result = self['SMU'].get_averaged_current(**self.smu_kwargs)
        print(result)
        print("Will now exit for convenience!")
        import sys
        sys.exit(0)

    def get_source_current_multiple(self, n: int):
        def measurement_step():
            current = self.get_source_current
            time.sleep(1e-6)
            return current

        return np.array([measurement_step() for _ in range(n)])

        # alternative but potentially faster implementation
        # self['SMU'].set_number_measurements(n, **self.smu_kwargs)
        # self['SMU'].multi_current_measurement(**self.smu_kwargs)
        # result = self['SMU'].get_multi_current(**self.smu_kwargs)
        # return np.array(result.split(','), dtype=float_initialiser)

    def smu_off(self):
        self['SMU'].off(**self.smu_kwargs)

    # Handle the biasing supply
    def init_bias_voltage(self, voltage: float = -80.0, voltage_range=1.5, current_limit=0.001, current_range=0.00001):
        # Refactor this according to the actual setup
        # settings for sensor depletion source
        self[self.bias_smu_key].off(**self.smu_bias_kwargs)
        self[self.bias_smu_key].source_volt(**self.smu_bias_kwargs)
        self[self.bias_smu_key].set_voltage_range(voltage_range, **self.smu_bias_kwargs)
        self[self.bias_smu_key].set_current_nlpc(self.smu_plc, **self.smu_bias_kwargs)
        self[self.bias_smu_key].set_voltage(voltage, **self.smu_bias_kwargs)
        self[self.bias_smu_key].set_current_limit(current_limit, **self.smu_bias_kwargs)
        self[self.bias_smu_key].set_current_sense_range(current_range, **self.smu_bias_kwargs)

    def set_bias_on(self):
        self[self.bias_smu_key].on(**self.smu_bias_kwargs)

    def set_bias_off(self):
        self[self.bias_smu_key].off(**self.smu_bias_kwargs)

    def set_bias_voltage(self, voltage: float):
        self[self.bias_smu_key].set_voltage(voltage, **self.smu_bias_kwargs)
        time.sleep(1)

    @property
    def bias_voltage(self):
        return self[self.bias_smu_key].get_voltage(**self.smu_bias_kwargs)

    @bias_voltage.setter
    def bias_voltage(self, voltage):
        self.set_bias_voltage(voltage)

    @property
    def smu_plc(self):
        return 10

    # Handle the implementation of the SMU config
    @property
    def smu_kwargs(self):
        return self.__smu_kwargs

    @property
    def smu_bias_kwargs(self):
        return self.__bias_kwargs

    @property
    def bias_smu_key(self):
        return self.__bias_smu_key
