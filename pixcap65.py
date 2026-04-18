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
from enum import StrEnum
from typing import Any

import numpy as np
from basil.dut import Dut
from numpy import ndarray

import pixcap65_constants as c

# perhaps add the channel information to the pixcap config file and extract it from here!
float_initialiser = np.float32

logger = logging.getLogger(__name__)

class BasilConfigKeys(StrEnum):
    TRANSFER_LAYER = 'transfer_layer'
    HARDWARE_LAYER = 'hw_drivers'
    REGISTER_LAYER = 'registers'


class Pixcap65(Dut):
    __smu_kwargs = {}
    __bias_kwargs = {}
    __bias_smu_key = 'BIAS_SUPPLY'
    __primary_smu_key = 'SMU'
    __vm1_smu_key = 'VM1'
    __vm2_smu_key = 'VM2'
    __vm3_smu_key = 'VM3'
    __mio_pll_key = 'MIO_PLL'

    __smu_keys = [__primary_smu_key, __vm1_smu_key, __vm2_smu_key, __vm3_smu_key, __bias_smu_key]
    smu_setup_devices = {} # hold the keys for the access to parts of a smu which are not channel depend for each of the smu registers.

    __seq_size = 1
    __current_cvm_frequency = 0
    # __source_settling_time = 1
    # __frequency_settling = 1
    __source_settling_time = 0.2
    __frequency_settling = 0.2
    __n_measurements = {
                        __primary_smu_key: 1,
                        __vm1_smu_key: 1,
                        __vm2_smu_key: 1,
                        __vm3_smu_key: 1,
                        __bias_smu_key: 1,
                        }

    binary_active = False

    @property
    def frequency_settling(self):
        return self.__frequency_settling

    @frequency_settling.setter
    def frequency_settling(self, value):
        self.__frequency_settling = value

    @property
    def primary_smu_key(self):
        return self.__primary_smu_key

    @property
    def vm1_smu_key(self):
        return self.__vm1_smu_key

    @property
    def vm2_smu_key(self):
        return self.__vm2_smu_key

    @property
    def vm3_smu_key(self):
        return self.__vm3_smu_key

    def init(self, init_conf=None, **kwargs):
        from usb.core import USBTimeoutError
        try:
            Dut.init(self, init_conf=init_conf, **kwargs)
        except USBTimeoutError:
            # perform a power cycle if possible
            logger.error("An USB error occured try to solve the issue by a power cycle.")
            if "power" in list(self._hardware_layer.keys()):
                power_driver = self._hardware_layer["power"]
                from basil.dut import Base
                assert isinstance(power_driver, Base)
                if not power_driver._intf.is_initialized:
                    power_driver._intf.init()
                if power_driver.is_initialized:
                    power_driver.init()

                power_driver.set_enable(0, channel=1)
                power_driver.set_enable(0, channel=2)
                time.sleep(5)
                power_driver.set_enable(1, channel=1)
                power_driver.set_enable(1, channel=2)

                try:
                    power_driver.close()
                except:
                    power_driver.is_initialized = True

                if power_driver._intf.is_initialized:
                    try:
                        power_driver._intf.close()
                    except:
                        power_driver._intf.is_initialized = True

                # now try the configuration of the board again.
                time.sleep(10)
                logger.info("Power cycle for FPGA completed")
                Dut.init(self, init_conf=init_conf, **kwargs)
            else:
                raise

        # setup the chip
        self.switch_on_power_supply_voltages(1)
        self.init_config()

        # make sure that we have a setup mapping for the different SMUs
        from basil.RL.FunctionalRegister import FunctionalRegister
        for smu_key in self.__smu_keys:
            if isinstance(self[smu_key], FunctionalRegister):
                # adjust the register
                try:
                    self.smu_setup_devices[smu_key] = self[smu_key]._drv.name
                except:
                    self.smu_setup_devices[smu_key] = smu_key
            else:
                self.smu_setup_devices[smu_key] = smu_key

        print(self.smu_setup_devices)


    def close(self):
        self.switch_on_power_supply_voltages(0)
        try:
            self[self.smu_setup_devices[self.primary_smu_key]].text_format()
        except ValueError:
            pass
        self['SEQ'].clear()
        Dut.close(self)

    def switch_on_power_supply_voltages(self, pwr_en):
        """
        Switches on default supply voltages
        """
        if pwr_en:
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

    def set_frequency(self, freq: float):
        self.cvm_frequency = freq

    @property
    def cvm_frequency(self):
        return self.__current_cvm_frequency

    @cvm_frequency.setter
    def cvm_frequency(self, freq):
        prev_freq = self.__current_cvm_frequency
        self.__current_cvm_frequency = freq
        try:
            logger.debug("Set the frequency to %f MHz with seq size %i.", freq, self.seq_size)
            eff_freq = freq * self.seq_size
            res = self[self.__mio_pll_key].setFrequency(eff_freq)
            if not res:
                logger.warning("Could not set the MIO PLL frequency to %d Hz.", freq)
        except:
            # error occured => revert the whole thing!
            self.__current_cvm_frequency = prev_freq
            raise
        else:
            time.sleep(self.frequency_settling)

    @property
    def source_settling_time(self):
        return self.__source_settling_time

    @source_settling_time.setter
    def source_settling_time(self, time):
        self.__source_settling_time = time

    # Handle the SMU!
    # region Accesor methods for a general SMU
    # MARK: perhaps this functions should be shifted direct to a SMU type in the basil framework?
    def smu_init(self, smu: str, current_limit: float, current_range, plc: int, src_u, voltage_range: float, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self[smu].off(**kwargs)
        try:
            if self[smu].get_buffer2_mode(**kwargs).startswith("NEXT"):
                print("Buffer 1 is still in data taking mode. Need to clear it.")
                self[smu].disable_buffer(**kwargs)
                time.sleep(10)
            if self[smu].get_buffer2_mode(**kwargs).startswith("NEXT"):
                print("Buffer 2 is still in data taking mode. Need to clear it.")
                self[smu].disable_buffer(**kwargs)
                time.sleep(10)
            self[smu].set_number_trigger_points(1)
        except ValueError:
            pass

        self[smu].clear_buffer1(**kwargs)
        self[smu].clear_buffer2(**kwargs)
        try:
            self[smu].set_buffer1_mode(0, **kwargs)
            self[smu].set_buffer2_mode(0, **kwargs)
        except ValueError:
            logger.warning("No buffer modes for the selected SMU. Will skip buffer mode configuration.")
        try:
            self[smu].disable_filter()
        except ValueError:
            logger.exception("No filter settings for the selected SMU. Will skip filter configuration.")
        self[smu].source_volt(**kwargs)
        self[smu].set_voltage_range(voltage_range, **kwargs)
        self[smu].set_current_nlpc(plc, **kwargs)
        self[smu].set_voltage(src_u, **kwargs)
        self[smu].set_current_limit(current_limit, **kwargs)
        self[smu].set_current_sense_range(current_range, **kwargs)

    def smu_source_volt(self, smu:str, **kwargs):
        if kwargs is None:
            kwargs = {}
        self[smu].source_volt(**kwargs)

    def smu_source_current(self, smu: str, **kwargs):
        if kwargs is None:
            kwargs = {}
        self[smu].source_current(**kwargs)

    def set_smu_source_voltage(self, smu: str, voltage: float, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self[smu].set_voltage(voltage, **kwargs)
        time.sleep(self.source_settling_time)

    def set_smu_source_current(self, smu: str, current: float, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self[smu].set_current(current, **kwargs)
        time.sleep(self.source_settling_time)

    def get_smu_source_voltage(self, smu: str, **kwargs):
        if kwargs is None:
            kwargs = {}
        return self[smu].get_source_voltage(**kwargs)

    def get_smu_source_current(self, smu: str, **kwargs):
        if kwargs is None:
            kwargs = {}
        return self[smu].get_source_current(**kwargs)

    def smu_measure_current(self, smu: str, kwargs=None) -> Any:
        if kwargs is None:
            kwargs = {}
        result = self[smu].get_current(**kwargs)
        if not (isinstance(result, float)
                or isinstance(result, int)
                or isinstance(result, np.float32)
                or isinstance(result, np.float64)
                or isinstance(result, str)):
            print(type(result), result)
            raise TypeError("The current returned {result} which was not recognised as a format.".format(result=result))
        if isinstance(result, str) and ',' in result:
            current = float_initialiser(result.split(',')[1])
        else:
            current = float_initialiser(result)
        if np.isnan(current):
            logging.warning("It was a NaN value measured by the SMU.")
            raise ValueError("The current returned {current} was not recognised as a number.".format(current=current))
        return current

    def smu_measure_voltage(self, smu: str, kwargs=None) -> Any:
        if kwargs is None:
            kwargs = {}
        result = self[smu].get_voltage(**kwargs)
        if not (isinstance(result, float)
                or isinstance(result, int)
                or isinstance(result, np.float32)
                or isinstance(result, np.float64)
                or isinstance(result, str)):
            print(type(result), result)
            raise TypeError("The current returned {result} which was not recognised as a format.".format(result=result))
        if isinstance(result, str) and ',' in result:
            voltage = float_initialiser(result.split(',')[1])
        else:
            voltage = float_initialiser(result)
        if np.isnan(voltage):
            logging.warning("It was a NaN value measured by the SMU.")
            raise ValueError("The current returned {current} was not recognised as a number.".format(current=voltage))
        return voltage

    def smu_averaged_current(self, n: int, smu: str, kwargs=None) -> tuple[float, float]:
        if kwargs is None:
            kwargs = {}
        try:
            self[smu].disable_filter()
        except ValueError:
            pass
        self[smu].set_number_measurements(n, **kwargs)
        self[smu].multi_current_measurement(**kwargs)
        result = self[smu].get_averaged_current(**kwargs)
        return tuple([float(elem) for elem in result.split(',')[:2]])

    def smu_averaged_voltage(self, n: int, smu: str, kwargs=None) -> tuple[float, float]:
        if kwargs is None:
            kwargs = {}
        try:
            self[smu].disable_filter()
        except ValueError:
            pass
        self[smu].set_number_measurements(n, **kwargs)
        self[smu].multi_voltage_measurement(**kwargs)
        result = self[smu].get_averaged_voltage(**kwargs)
        return tuple([float(elem) for elem in result.split(',')[:2]])

    def smu_advanced_current_multiple(self, n: int | None, smu: str, kwargs=None) -> ndarray:
        if kwargs is None:
            kwargs = {}
        try:
            # this function might not be implemented.
            self[smu].disable_filter()
        except ValueError:
            pass
        if n is not None and self.__n_measurements[smu] != n:
            self.set_smu_measurements(smu, n, kwargs)
        result = self[smu].get_advanced_current(**kwargs)
        if "binary_enabled" in kwargs and kwargs["binary_enabled"]:
            n = self.__n_measurements[smu]
            if result.shape[0] > n:
                offset = int(result.shape[0] % n)
                shift = int(result.shape[0] // n)
                return result[offset::shift]
            return result
        return np.array(result.split(','), dtype=float_initialiser)

    def smu_advanced_voltage_multiple(self, n: int, smu: str, kwargs=None) -> ndarray:
        if kwargs is None:
            kwargs = {}
        try:
            # this function might not be implemented.
            self[smu].disable_filter()
        except ValueError:
            pass
        if n is not None and self.__n_measurements[smu] != n:
            self.set_smu_measurements(smu, n, kwargs)
        result = self[smu].get_advanced_voltage(**kwargs)
        if "binary_enabled" in kwargs and kwargs["binary_enabled"]:
            n = self.__n_measurements[smu]
            if result.shape[0] > n:
                offset = int(result.shape[0] % n)
                shift = int(result.shape[0] // n)
                return result[offset::shift]
            return result
        return np.array(result.split(','), dtype=float_initialiser)

    def general_smu_current_multiple(self, smu: str, n: int, kwargs=None) -> ndarray:
        if kwargs is None:
            kwargs = {}
        try:
            self[smu].disable_filter()
        except ValueError:
            pass
        def measurement_step():
            current = self.smu_measure_current(smu, kwargs=kwargs)
            time.sleep(1e-6)
            return current

        return np.array([measurement_step() for _ in range(n)])

    def general_smu_voltage_multiple(self, smu: str, n: int, kwargs=None) -> ndarray:
        if kwargs is None:
            kwargs = {}

        try:
            self[smu].disable_filter()
        except ValueError:
            pass
        def measurement_step():
            current = self.smu_measure_voltage(smu, kwargs=kwargs)
            time.sleep(1e-6)
            return current

        return np.array([measurement_step() for _ in range(n)])

    def smu_output_on(self, smu: str, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self[smu].on(**kwargs)

    def smu_output_off(self, smu: str, kwargs=None):
        if kwargs is None:
            kwargs = {}
        self[smu].off(**kwargs)

    def set_smu_measurements(self, smu: str, value: int, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if value == -1:
            self[smu].set_number_measurements(1, **kwargs)
            try:
                self[smu].set_number_triggers(1, **kwargs)
            except ValueError:
                pass
        else:
            self[smu].set_number_measurements(value, **kwargs)
            try:
                self[smu].set_number_triggers(value, **kwargs)
            except ValueError:
                pass
        self.__n_measurements[smu] = value
    # endregion

    # implementations for the different SMU's in use with pixcap
    # region Primary SMU used for VM 3
    # primary smu used for VM 3
    def init_smu(self, src_u, current_range, voltage_range=1.5, current_limit=0.001, plc=10):
        self.smu_init(self.__primary_smu_key, current_limit, current_range, plc, src_u, voltage_range, kwargs=self.smu_kwargs)

    def smu_on(self):
        self.smu_output_on(self.__primary_smu_key, kwargs=self.smu_kwargs)

    def smu_off(self):
        self.smu_output_off(self.__primary_smu_key, kwargs=self.smu_kwargs)

    def smu_set_current_source(self):
        self.smu_source_current(self.__primary_smu_key, kwargs=self.smu_kwargs)

    def smu_set_voltage_source(self):
        self.smu_source_volt(self.__primary_smu_key, kwargs=self.smu_kwargs)

    @property
    def source_voltage_smu(self):
        return self.get_smu_source_voltage(self.__primary_smu_key, kwargs=self.smu_kwargs)

    @source_voltage_smu.setter
    def source_voltage_smu(self, value):
        self.set_smu_source_voltage(self.__primary_smu_key, value, kwargs=self.smu_kwargs)

    @property
    def source_current_smu(self):
        return self.get_smu_source_current(self.__primary_smu_key, kwargs=self.smu_kwargs)

    @source_current_smu.setter
    def source_current_smu(self, value):
        self.set_smu_source_current(self.__primary_smu_key, value, kwargs=self.smu_kwargs)

    @property
    def get_source_current(self) -> float:
        return self.smu_measure_current(self.__primary_smu_key, kwargs=self.smu_kwargs)

    def smu_measure_volts(self):
        return self.smu_measure_voltage(self.__primary_smu_key, kwargs=self.smu_kwargs)

    def averaged_current(self, n: int = 10):
        return self.smu_averaged_current(n, self.__primary_smu_key, kwargs=self.smu_kwargs)

    def averaged_voltage(self, n: int = 10):
        return self.smu_averaged_voltage(n, self.__primary_smu_key, kwargs=self.smu_kwargs)

    def get_source_current_multiple(self, n: int):
        return self.general_smu_current_multiple(self.__primary_smu_key, n, kwargs=self.smu_kwargs)

    def get_source_voltage_multiple(self, n: int):
        return self.general_smu_voltage_multiple(self.__primary_smu_key, n, kwargs=self.smu_kwargs)

    def get_advanced_current_multiple(self, n: int):
        kargs = self.smu_kwargs.copy()
        kargs['binary_enabled'] = self.binary_active
        kargs['data_points'] = self.n_measurements
        return self.smu_advanced_current_multiple(n, self.__primary_smu_key, kwargs=kargs)

    def get_advanced_voltage_multiple(self, n: int):
        kargs = self.smu_kwargs.copy()
        kargs['binary_enabled'] = self.binary_active
        kargs['data_points'] = self.n_measurements
        return self.smu_advanced_voltage_multiple(n, self.__primary_smu_key, kwargs=kargs)
    # endregion

    # region Handle the biasing supply.
    # Handle the biasing supply
    def init_bias(self, voltage, current_range, voltage_range=1.5, current_limit=0.001, plc=10):
        self.smu_init(self.__bias_smu_key, current_limit, current_range, plc, voltage, voltage_range,
                      kwargs=self.smu_bias_kwargs)

    def bias_on(self):
        self.smu_output_on(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    def bias_off(self):
        self.smu_output_off(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    def bias_current_source(self):
        self.smu_source_current(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    def bias_voltage_source(self):
        self.smu_source_volt(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    @property
    def bias_voltage(self):
        return self.get_smu_source_voltage(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    @bias_voltage.setter
    def bias_voltage(self, value):
        self.set_smu_source_voltage(self.__bias_smu_key, value, kwargs=self.smu_bias_kwargs)

    @property
    def bias_current(self):
        return self.get_smu_source_current(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    @bias_current.setter
    def bias_current(self, value):
        self.set_smu_source_current(self.__bias_smu_key, value, kwargs=self.smu_bias_kwargs)

    def bias_measure_current(self) -> float:
        return self.smu_measure_current(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    def bias_measure_volts(self):
        return self.smu_measure_voltage(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    def bias_averaged_current(self, n: int = 10):
        return self.smu_averaged_current(n, self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    def bias_averaged_voltage(self, n: int = 10):
        return self.smu_averaged_voltage(n, self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    def bias_current_multiple(self, n: int):
        return self.general_smu_current_multiple(self.__bias_smu_key, n, kwargs=self.smu_bias_kwargs)

    def bias_voltage_multiple(self, n: int):
        return self.general_smu_voltage_multiple(self.__bias_smu_key, n, kwargs=self.smu_bias_kwargs)

    def bias_advanced_current_multiple(self, n: int):
        return self.smu_advanced_current_multiple(n, self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    def bias_advanced_voltage_multiple(self, n: int):
        return self.smu_advanced_voltage_multiple(n, self.__bias_smu_key, kwargs=self.smu_bias_kwargs)
    # endregion

    # region Handle the VM1 Connector SMU
    def init_vm1(self, src_u, current_range, voltage_range=1.5, current_limit=0.001, plc=10):
        self.smu_init(self.__vm1_smu_key, current_limit, current_range, plc, src_u, voltage_range,
                      kwargs=self.smu_vm1_kwargs)

    def vm1_on(self):
        self.smu_output_on(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_off(self):
        self.smu_output_off(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_current_source(self):
        self.smu_source_current(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_voltage_source(self):
        self.smu_source_volt(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    @property
    def vm1_voltage(self):
        return self.get_smu_source_voltage(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    @vm1_voltage.setter
    def vm1_voltage(self, value):
        self.set_smu_source_voltage(self.__vm1_smu_key, value, kwargs=self.smu_vm1_kwargs)

    @property
    def vm1_current(self):
        return self.get_smu_source_current(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    @vm1_current.setter
    def vm1_current(self, value):
        self.set_smu_source_current(self.__vm1_smu_key, value, kwargs=self.smu_vm1_kwargs)

    def vm1_measure_current(self) -> float:
        return self.smu_measure_current(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_measure_volts(self):
        return self.smu_measure_voltage(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_averaged_current(self, n: int = 10):
        return self.smu_averaged_current(n, self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_averaged_voltage(self, n: int = 10):
        return self.smu_averaged_voltage(n, self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_current_multiple(self, n: int):
        return self.general_smu_current_multiple(self.__vm1_smu_key, n, kwargs=self.smu_vm1_kwargs)

    def vm1_voltage_multiple(self, n: int):
        return self.general_smu_voltage_multiple(self.__vm1_smu_key, n, kwargs=self.smu_vm1_kwargs)

    def vm1_advanced_current_multiple(self, n: int):
        return self.smu_advanced_current_multiple(n, self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_advanced_voltage_multiple(self, n: int):
        return self.smu_advanced_voltage_multiple(n, self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)
    # endregion

    # region Handle the VM2 Connector SMU
    def init_vm2(self, src_u, current_range, voltage_range=1.5, current_limit=0.001, plc=10):
        self.smu_init(self.__vm2_smu_key, current_limit, current_range, plc, src_u, voltage_range,
                      kwargs=self.smu_vm2_kwargs)

    def vm2_on(self):
        self.smu_output_on(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_off(self):
        self.smu_output_off(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_current_source(self):
        self.smu_source_current(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_voltage_source(self):
        self.smu_source_volt(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    @property
    def vm2_voltage(self):
        return self.get_smu_source_voltage(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    @vm2_voltage.setter
    def vm2_voltage(self, value):
        self.set_smu_source_voltage(self.__vm2_smu_key, value, kwargs=self.smu_vm2_kwargs)

    @property
    def vm2_current(self):
        return self.get_smu_source_current(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    @vm2_current.setter
    def vm2_current(self, value):
        self.set_smu_source_current(self.__vm2_smu_key, value, kwargs=self.smu_vm2_kwargs)

    def vm2_measure_current(self) -> float:
        return self.smu_measure_current(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_measure_volts(self):
        return self.smu_measure_voltage(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_averaged_current(self, n: int = 10):
        return self.smu_averaged_current(n, self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_averaged_voltage(self, n: int = 10):
        return self.smu_averaged_voltage(n, self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_current_multiple(self, n: int):
        return self.general_smu_current_multiple(self.__vm2_smu_key, n, kwargs=self.smu_vm2_kwargs)

    def vm2_voltage_multiple(self, n: int):
        return self.general_smu_voltage_multiple(self.__vm2_smu_key, n, kwargs=self.smu_vm2_kwargs)

    def vm2_advanced_current_multiple(self, n: int):
        return self.smu_advanced_current_multiple(n, self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_advanced_voltage_multiple(self, n: int):
        return self.smu_advanced_voltage_multiple(n, self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)
    # endregion

    # region Handle the VM3 Connector SMU
    def init_vm3(self, src_u, current_range, voltage_range=1.5, current_limit=0.001, plc=10):
        self.smu_init(self.__vm3_smu_key, current_limit, current_range, plc, src_u, voltage_range,
                      kwargs=self.smu_vm3_kwargs)

    def vm3_on(self):
        self.smu_output_on(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_off(self):
        self.smu_output_off(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_current_source(self):
        self.smu_source_current(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_voltage_source(self):
        self.smu_source_volt(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    @property
    def vm3_voltage(self):
        return self.get_smu_source_voltage(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    @vm3_voltage.setter
    def vm3_voltage(self, value):
        self.set_smu_source_voltage(self.__vm3_smu_key, value, kwargs=self.smu_vm3_kwargs)

    @property
    def vm3_current(self):
        return self.get_smu_source_current(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    @vm3_current.setter
    def vm3_current(self, value):
        self.set_smu_source_current(self.__vm3_smu_key, value, kwargs=self.smu_vm3_kwargs)

    def vm3_measure_current(self) -> float:
        return self.smu_measure_current(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_measure_volts(self):
        return self.smu_measure_voltage(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_averaged_current(self, n: int = 10):
        return self.smu_averaged_current(n, self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_averaged_voltage(self, n: int = 10):
        return self.smu_averaged_voltage(n, self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_current_multiple(self, n: int):
        return self.general_smu_current_multiple(self.__vm3_smu_key, n, kwargs=self.smu_vm3_kwargs)

    def vm3_voltage_multiple(self, n: int):
        return self.general_smu_voltage_multiple(self.__vm3_smu_key, n, kwargs=self.smu_vm3_kwargs)

    def vm3_advanced_current_multiple(self, n: int):
        return self.smu_advanced_current_multiple(n, self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_advanced_voltage_multiple(self, n: int):
        return self.smu_advanced_voltage_multiple(n, self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)
    # endregion

    # general properties!
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
    def smu_vm1_kwargs(self):
        return {}

    @property
    def smu_vm2_kwargs(self):
        return {}

    @property
    def smu_vm3_kwargs(self):
        return {}

    @property
    def bias_smu_key(self):
        return self.__bias_smu_key

    @property
    def seq_size(self):
        return self.__seq_size

    @seq_size.setter
    def seq_size(self, size):
        logger.debug("Setting seq size to %i.", size)
        self.__seq_size = size

    @property
    def n_measurements(self):
        return self.__n_measurements[self.primary_smu_key]

    @n_measurements.setter
    def n_measurements(self, value):
        if value == -1:
            self[self.primary_smu_key].set_number_measurements(1, **self.smu_kwargs)
        else:
            self[self.primary_smu_key].set_number_measurements(value, **self.smu_kwargs)
        self.__n_measurements[self.primary_smu_key] = value
