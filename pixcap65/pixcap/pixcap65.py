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
import numpy as np
import time
from basil.RL.FunctionalRegister import FunctionalRegister
from basil.dut import Dut
from numpy import ndarray
from typing import Any

from pixcap65.utility import pixcap65_constants as c

SMU_DISABLED_MSG = "The voltage could only be measured for an active smu but '%s' is inactive."
SMU_DISABLED_CURRENT_MSG = "The current could only be measured for an active smu but '%s' is inactive."

# perhaps add the channel information to the pixcap config file and extract it from here!
float_initialiser = np.float32

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Pixcap65(Dut):
    __slots__ = ["binary_active"]

    def __init__(self, conf):
        Dut.__init__(self, conf)

        self.__smu_kwargs = {}
        self.__bias_kwargs = {}
        self.__bias_smu_key = 'BIAS_SUPPLY'
        self.__primary_smu_key = 'SMU'
        self.__vm1_smu_key = 'VM1'
        self.__vm2_smu_key = 'VM2'
        self.__vm3_smu_key = 'VM3'
        self.__mio_pll_key = 'MIO_PLL'
        self.__has_smu = {}

        self.__smu_keys = [
            self.__primary_smu_key, self.__vm1_smu_key, self.__vm2_smu_key, self.__vm3_smu_key, self.__bias_smu_key
        ]
        self.smu_setup_devices = {}  # hold the keys for the access to parts of a smu which are not channel depend
        # for each of the smu registers.

        self.__seq_size = 1
        self.__current_cvm_frequency = 0
        self.__source_settling_time = 0.2
        self.__frequency_settling = 0.2
        self.__n_measurements = {
            self.__primary_smu_key: 1,
            self.__vm1_smu_key: 1,
            self.__vm2_smu_key: 1,
            self.__vm3_smu_key: 1,
            self.__bias_smu_key: 1,
        }
        self.binary_active = False

        # MARK: perhaps this guard should be generalized for every smu!
        if "active" in self[self.__bias_smu_key]._init and self[self.__bias_smu_key]._init["active"]:
            self.__has_bias_suppy = True
        else:
            self.__has_bias_suppy = False
        if isinstance(self[self.__bias_smu_key], FunctionalRegister) and self[self.__bias_smu_key]._drv is None:
            self.__has_bias_suppy = False

    # region Pixcap65 PCB/Chip Properties
    @property
    def has_bias_suppy(self):
        """Gets whether the HV SMU is active and connected."""
        return self.__has_bias_suppy

    @property
    def frequency_settling(self):
        """Get the settling time for the clock's frequency."""
        return self.__frequency_settling

    @frequency_settling.setter
    def frequency_settling(self, value):
        self.__frequency_settling = value

    @property
    def primary_smu_key(self):
        """Get the dut device key for the primary smu."""
        return self.__primary_smu_key

    @property
    def vm1_smu_key(self):
        """Get the dut device key for the SMU supplying the VM1 connector."""
        return self.__vm1_smu_key

    @property
    def vm2_smu_key(self):
        """Get the dut device key for the SMU supplying the VM2 connector."""
        return self.__vm2_smu_key

    @property
    def vm3_smu_key(self):
        """Get the dut device key for the SMU supplying the VM3 connector."""
        return self.__vm3_smu_key

    @property
    def cvm_frequency(self):
        """
        Get the frequency set at the MIO PLL clock.
        It retrieves the frequency set for capacitance measurement not the higher one for the MIO PLL which accounts for
        the size of the pattern and applies the pattern switching frequency.
        :return: current MIO frequency in MHz.
        """
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
            # error occurred => revert the whole thing!
            self.__current_cvm_frequency = prev_freq
            raise
        else:
            time.sleep(self.frequency_settling)

    @property
    def source_settling_time(self):
        """Get the settling time for the voltage supplies in case of changes at the source."""
        return self.__source_settling_time

    @source_settling_time.setter
    def source_settling_time(self, s_time):
        self.__source_settling_time = s_time

    @property
    def has_smu(self):
        """Get the mapping of the sums to the state whether they are active and connected by the configuration file."""
        return self.__has_smu

    # general properties!
    @property
    def smu_plc(self):
        """Gets number of power cycles to average the measurements over."""
        return 10

    # Handle the implementation of the SMU config
    @property
    def smu_kwargs(self):
        """Gets the primary SMUs additional keyword arguments."""
        return self.__smu_kwargs

    @property
    def smu_bias_kwargs(self):
        """Gets the HV SMUs additional keyword arguments."""
        return self.__bias_kwargs

    @property
    def smu_vm1_kwargs(self):
        """Get the additional keyword arguments for the SMU connected to the PCBs VM1 port."""
        return {}

    @property
    def smu_vm2_kwargs(self):
        """Get the additional keyword arguments for the SMU connected to the PCBs VM2 port."""
        return {}

    @property
    def smu_vm3_kwargs(self):
        """Get the additional keyword arguments for the SMU connected to the PCBs VM3 port."""
        return {}

    @property
    def bias_smu_key(self):
        """Gets the dut key for the HV SMU."""
        return self.__bias_smu_key

    @property
    def seq_size(self):
        """Gets the current size of the sequence generators pattern."""
        return self.__seq_size

    @seq_size.setter
    def seq_size(self, size):
        logger.debug("Setting seq size to %i.", size)
        self.__seq_size = size

    @property
    def n_measurements(self):
        """Gets the number of measurements to be performed by the primary SMU."""
        return self.__n_measurements[self.primary_smu_key]

    @n_measurements.setter
    def n_measurements(self, value):
        if value == -1:
            self[self.primary_smu_key].set_number_measurements(1, **self.smu_kwargs)
        else:
            self[self.primary_smu_key].set_number_measurements(value, **self.smu_kwargs)
        self.__n_measurements[self.primary_smu_key] = value

    @property
    def n_bias_measurements(self):
        """Gets the number of measurements to be performed by the primary SMU."""
        return self.__n_measurements[self.bias_smu_key]

    @n_bias_measurements.setter
    def n_bias_measurements(self, value):
        if value == -1:
            self[self.bias_smu_key].set_number_measurements(1, **self.__bias_kwargs)
        else:
            self[self.bias_smu_key].set_number_measurements(value, **self.__bias_kwargs)
        self.__n_measurements[self.bias_smu_key] = value

    # endregion

    def init(self, init_conf=None, **kwargs):
        # fallback handler in case there are issues with siusb.
        from usb.core import USBTimeoutError
        try:
            Dut.init(self, init_conf=init_conf, **kwargs)
        except USBTimeoutError:
            # perform a power cycle if possible
            # ocured
            logger.error("A USB error occurred try to solve the issue by a power cycle.")
            if "power" in list(self._hardware_layer.keys()):
                # FIXME: type error (very static for the used psu)
                power_driver = self._hardware_layer["power"]
                from basil.HL.tti_ql355tp import ttiQl355tp
                assert isinstance(power_driver, ttiQl355tp)
                if not power_driver._intf.is_initialized:
                    power_driver._intf.init()
                if power_driver.is_initialized:
                    power_driver.init()

                power_driver.set_enable(0, channel=1)
                # power_driver.set_enable(0, channel=2)
                power_driver.set_enable(0, channel=3)
                time.sleep(5)
                power_driver.set_enable(1, channel=1)
                # power_driver.set_enable(1, channel=2)
                power_driver.set_enable(1, channel=3)

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
                time.sleep(7)
                logger.info("Power cycle for FPGA completed")
                Dut.init(self, init_conf=init_conf, **kwargs)
            else:
                raise

        # set up the chip
        self.switch_on_power_supply_voltages(1)
        self.init_config()

        # make sure that we have a setup mapping for the different SMUs
        from basil.RL.FunctionalRegister import FunctionalRegister
        for smu_key in self.__smu_keys:
            if isinstance(self[smu_key], FunctionalRegister):
                # adjust the register
                try:
                    self.smu_setup_devices[smu_key] = self[smu_key]._drv.name
                    self.__has_smu[smu_key] = self[smu_key]._init.get("active", False)
                except:
                    self.smu_setup_devices[smu_key] = smu_key
                    self.__has_smu[smu_key] = False
            else:
                self.smu_setup_devices[smu_key] = smu_key

        print(self.smu_setup_devices)
        assert self.has_bias_suppy == self.__has_smu[self.bias_smu_key]

    # region Control and setup handling of the Pixcap Chip.
    def close(self):
        """
        close

        Close the pixcap test device and turn off any still active supply voltages.
        """
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

        :param pwr_en: bit code for the output state (0 - off, 1 - on)
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
        """
        get_status

        Fetch the current state of the pixcap chips power supply
        :return: current time and information about the power supply of the pixcap chips logic.
        """
        # noinspection PyDictCreation
        status = {}
        status['Time'] = time.strftime("%d %M %Y %H:%M:%S")
        status['VDD'] = {'voltage(V)': format(self['VDD'].get_voltage(unit='V'), '.3f'),
                         'current(mA)': format(self['VDD'].get_current(unit='mA'), '.3f')}
        return status

    def enable_pixel_clk(self, i_col, i_pix, clk_mask):
        """
        enable_pixel_clk

        Enable a specific pixel clk (multiple of these) for a specific pixel.
        Note: this won't have any effect if the clock is not active for column containing the pixel.

        :param i_col: number of the column in which a pixel should be activated
        :param i_pix: row number of the pixel to be activated
        :param clk_mask: bit_wise OR of all clk nets to be activated for the specified pixel
        """
        self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['CLK_EN'] = clk_mask
        self.update_spi()

    def enable_pixel_bias(self, i_col, i_pix, bias_mask):
        """
        enable_pixel_bias

        Enable a specific static bias over a specific pixel instead of using clock nets for pulsed bias.

        :param i_col: number of the column in which a pixel should be activated
        :param i_pix: row number of the pixel to be activated
        :param bias_mask: bit_wise OR of all bias nets to be activated for the specified pixel
        """
        self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['SEL'] = bias_mask
        self.update_spi()

    # noinspection PyPep8Naming
    def enable_column(self, i_col, eoc_mask):
        """
        enable_column

        Enable functions of a column for the specified column at the end of column of block of the pixcap chip for the
        specified column.

        :param i_col: column for which to activate a 'feature'/function !!no bit mask for specifying
            multiple columns at once.!!
        :param eoc_mask: Bit mask (multiply by bitwise OR) of the functions to be activated.
        """
        self['SPI']['COL'][c.COL_NMAX - i_col]['EOC'] = eoc_mask
        self.update_spi()

    def disable_all_pixels(self):
        """
        Disable all pixels.

        This will deactivate all static bias connections and clock net connection for all the pixels on the sensor.
        This won't have an impact onto the End-Of-Column functions/controller. Specific functions enabled for columns
        there will remain active for these columns also the pixels could no longer use them.
        """
        for i_col in range(0, c.COL_NMAX + 1):
            for i_pix in range(0, c.PIX_NMAX + 1):
                self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['CLK_EN'] = 0
                self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['SEL'] = 0
        self.update_spi()

    def update_spi(self):
        """
        Write the new data to the SPI to change the configuration of the pixcap chip and the MIO
        """
        self['SPI'].write()
        self['SPI'].start()

    def disable_all_columns(self):
        """
        disable_all_columns

        Disable all active functions for all columns at the End-Of-Column controller.
        The functions will be effectively deactivated for the pixel in the columns as well.
        """
        for i_col in range(0, c.COL_NMAX + 1):
            self['SPI']['COL'][c.COL_NMAX - i_col]['EOC'] = 0
        self.update_spi()

    def reset_chip(self):
        """
        reset_chip

        Resets the whole pixcap chip back to default by 'pushing' the reset button which will also erase all the
        SPI registers such that no previous data will be present any more.
        """
        # reset shift register
        self['GPIO']['RST_B'] = 0
        self['GPIO'].write()
        time.sleep(0.1)
        self['GPIO']['RST_B'] = 1
        self['GPIO'].write()

    def init_config(self):
        """
        init_config

        (re-)init the pixcap chip. This will first configure the spi (more precisely it's size) and afterwards perform a
        reset if the pixcap chip to get a well-defined state of the hardware.
        """
        self['SPI'].set_size(9960)
        self.reset_chip()

    def set_frequency(self, freq: float):
        """
        set_frequency

        Sets the frequency of the clock nets at the MIO.
        This will take into account the size of the sequence generators patterns as these will require a higher
        frequency to be set at the MIO.
        The effectively set frequency at the MIO will be the switching frequency within the pattern.
        :param freq: frequency in MHz to be set for the clocks.
        """
        self.cvm_frequency = freq

    def seq_init(self, clk_0=None, clk_1=None, clk_2=None, clk_3=None):
        """
        seq_init

        Perform the initial setup of the sequence generator.
        It may be necessary to set the patterns for all the clocks.
        At last the sequence generators patter creation is started.

        :param clk_0: bitarray or string, None, bit pattern for the first clock net (CLK 0).
        :param clk_1: bitarray or string, None, bit pattern for the second clock net (CLK 1).
        :param clk_2: bitarray or string, None, bit pattern for the third clock net (CLK 2).
        :param clk_3: bitarray or string, None, bit pattern for the third clock net (CLK 3).
        """
        assert self.seq_size >= 4

        # prepare the sequence generator
        self['SEQ'].reset()
        self['SEQ'].set_clk_divide(1)
        self['SEQ'].set_repeat_start(0)
        self['SEQ'].set_repeat(0)
        self['SEQ'].set_size(self.seq_size)

        # configure the switching patterns
        self.configure_seq(0, clk_0)
        self.configure_seq(1, clk_1)
        self.configure_seq(2, clk_2)
        self.configure_seq(3, clk_3)

        # start the sequence generator
        self['SEQ'].write()
        self['SEQ'].start()

    def configure_seq(self, clock: int, configuration=None):
        """
        configure_seq

        Configuration handler for the clock nets.
        Will set the specified bit pattern as the sequence for the sequence generator of the given clock net.

        :param clock: number/id of the clock need for which the pattern should be configured
        :param configuration: str or bitarray, defines the pattern to configure for the sequence generator
        """
        from bitarray import bitarray
        clock_id = "CLK_{}".format(clock)
        logger.debug("Want to configure the clock '%s'.", clock_id)
        if configuration is not None and isinstance(configuration, str):
            self['SEQ'][clock_id][0:self.seq_size - 1] = bitarray(configuration)
        elif configuration is not None and isinstance(configuration, bitarray):
            self['SEQ'][clock_id][0:self.seq_size - 1] = configuration

    def has_configured_smu(self, smu: str):
        """
        has_configured_smu

        Fetch the active state of the specified smu and whether it is connected or not.
        :param smu: smu dut key for which the state should be checked
        :return: boolean, whether the smu is connected and active or not
        """
        assert smu in self.__smu_keys
        return self.has_smu[smu]

    # endregion

    # Handle the SMU!
    # region Accessor methods for a general SMU
    # MARK: perhaps this functions should be shifted direct to a SMU type in the basil framework?
    def smu_init(self, smu: str, current_limit: float, current_range, plc: int, src_u, voltage_range: float,
                 kwargs=None):
        """
        smu_init

        Performs the initial setup for the specified smu.
        The setup could only be performed when the SMU is connected and active.
        This setup is designed only for sourcing voltage and measuring currents.
        So the current range, over-current protection and sourcing voltage will be set according to the provided
        arguments.
        On some SMUs it will also configure the measurement buffers for acquiring multiple readings at once.

        :param smu: SMU dut key of the SMU to be initially set up.
        :param current_limit: current compliance limit to be set in A.
        :param current_range: current measurement range/maximum current intended to be measured in A.
        :param plc: number of power supply cycles to be averaged over when measuring.
        :param src_u: sourcing voltage for the SMU.
        :param voltage_range:
        :param kwargs: further keyword arguments to be forwarded to the call of the lab device by basil
        """

        if kwargs is None:
            kwargs = {}
        if not self.has_configured_smu(smu):
            logger.debug("SMU initialization is only possible for active smu but '%s' is inactive.", smu)
            return
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

    def smu_source_volt(self, smu: str, **kwargs):
        """
        smu_source_volt

        Sets the mode of the SMUs source to sourcing voltage.
        A call to the SMU is only performed when the SMU is connected and active.

        :param smu: smu dut key of the SMU to configure
        :param kwargs: further keyword arguments to be forwarded to the call of the lab device by basil
        """
        if not self.has_configured_smu(smu):
            logger.debug("The sourcing mode could only be set for an active smu but '%s' is inactive.", smu)
            return
        if kwargs is None:
            kwargs = {}
        self[smu].source_volt(**kwargs)

    def smu_source_current(self, smu: str, **kwargs):
        """
        smu_source_current

        Sets the mode of the SMUs source to sourcing current.
        A call to the SMU is only performed when the SMU is connected and active.

        :param smu: smu dut key of the SMU to configure
        :param kwargs: further keyword arguments to be forwarded to the call of the lab device by basil
        """
        if not self.has_configured_smu(smu):
            logger.debug("The sourcing mode could only be set for an active smu but '%s' is inactive.", smu)
            return
        if kwargs is None:
            kwargs = {}
        self[smu].source_current(**kwargs)

    def set_smu_source_voltage(self, smu: str, voltage: float, kwargs=None):
        """
        set_smu_source_voltage

        Sets the voltage to be sourced by the SMU (constant voltage mode of the SMU).
        The call to the SMU is only performed when the SMU is connected and active.
        After changing the configuration, the settling of the smu will be taken into account to make sure that no
        measurement is performed within the settling interval of the voltage source.

        :param smu: smu dut key of the SMU to configure
        :param voltage: float, voltage to be sourced by the SMU
        :param kwargs: further keyword arguments to be forwarded to the call of the lab device by basil
        """
        if not self.has_configured_smu(smu):
            logger.debug("The source voltage could only be set for an active SMU but '%s' is inactive.", smu)
            return
        if kwargs is None:
            kwargs = {}
        logger.warning("Requested the source voltage settings for smu %s to %f", smu, voltage)
        self[smu].set_voltage(voltage, **kwargs)
        time.sleep(self.source_settling_time)

    def set_smu_source_current(self, smu: str, current: float, kwargs=None):
        """
            set_smu_source_current

            Sets the current to be sourced by the SMU (constant current mode of the SMU).
            The call to the SMU is only performed when the SMU is connected and active.
            After changing the configuration, the settling of the smu will be taken into account to make sure that no
            measurement is performed within the settling interval of the current source.

            :param smu: smu dut key of the SMU to configure
            :param current: float, current to be sourced by the SMU
            :param kwargs: further keyword arguments to be forwarded to the call of the lab device by basil
            """
        if not self.has_configured_smu(smu):
            logger.debug("The source current could only be set for an active SMU but '%s' is inactive.", smu)
            return
        if kwargs is None:
            kwargs = {}
        self[smu].set_current(current, **kwargs)
        time.sleep(self.source_settling_time)

    def get_smu_source_voltage(self, smu: str, **kwargs):
        """
        get_smu_source_voltage

        Fetches the source voltage set for the SMUs voltage source (applies to constant voltage mode of the SMU)
        This is not the actually applied voltage by the SMU.
        The call to the SMU is only performed when the SMU is connected and active.

        :param smu: smu dut key of the SMU to configure
        :param kwargs: further keyword arguments to be forwarded to the call of the lab device by basil
        :return: source voltage set in V
        """
        if not self.has_configured_smu(smu):
            logger.debug("The source voltage setting could only be read for an active SMU but '%s' is inactive.", smu)
            return np.nan
        if kwargs is None:
            kwargs = {}
        try:
            name = self[self.smu_setup_devices[smu]].get_name()
        except ValueError:
            name = "UNIDENTIFIED"
        logger.info("Attempting to get source voltage for smu %s (%s)" % (smu, name))
        return float(self[smu].get_source_voltage(**kwargs))

    def get_smu_source_current(self, smu: str, **kwargs):
        """
            get_smu_source_current

            Fetches the source current set for the SMUs current source (applies to constant current mode of the SMU)
            This is not the actually applied current by the SMU.
            The call to the SMU is only performed when the SMU is connected and active.

            :param smu: smu dut key of the SMU to configure
            :param kwargs: further keyword arguments to be forwarded to the call of the lab device by basil
            :return: source current set in A
            """
        if not self.has_configured_smu(smu):
            logger.debug("The source current setting could only be read for an active SMU but '%s' is inactive.", smu)
            return np.nan
        if kwargs is None:
            kwargs = {}
        return self[smu].get_source_current(**kwargs)

    def smu_measure_current(self, smu: str, kwargs=None) -> Any:
        """
        smu_measure_current

        Read the current through the SMU. (Performs a current measurement in constant voltage mode.)
        The call to the SMU is only performed when the SMU is connected and active.

        :param smu: smu dut key of the SMU to configure.
        :param kwargs: further keyword arguments to be forwarded to the call to the lab device by basil.
        :return: measured current in A or nan if the measurement fails or the SMU is not connected/active.
        """
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_MSG, smu)
            return np.nan
        if kwargs is None:
            kwargs = {}
        result = self[smu].get_current(**kwargs)
        if not (isinstance(result, float)
                or isinstance(result, int)
                or isinstance(result, np.float32)
                or isinstance(result, np.float64)
                or isinstance(result, str)):
            print(type(result), result)
            raise TypeError("The current returned {result} which was not recognized as a format.".format(result=result))
        if isinstance(result, str) and ',' in result:
            current = float_initialiser(result.split(',')[1])
        else:
            current = float_initialiser(result)
        if np.isnan(current):
            logging.warning("It was a NaN value measured by the SMU.")
            raise ValueError("The current returned {current} was not recognized as a number.".format(current=current))

        logger.debug("The measured current is %g A.", current)
        return current

    def smu_measure_voltage(self, smu: str, kwargs=None) -> float:
        """
        smu_measure_voltage

        Read the voltage over the SMU contacts. (Performs a voltage measurement in constant current mode.)
        The call to the SMU is only performed when the SMU is connected and active.

        :param smu: smu dut key of the SMU to configure.
        :param kwargs: further keyword arguments to be forwarded to the call to the lab device by basil.
        :return: measured voltage in V or nan if the measurement fails or the SMU is not connected/active.
        """
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_MSG, smu)
            return np.nan
        if kwargs is None:
            kwargs = {}
        result = self[smu].get_voltage(**kwargs)
        if not (isinstance(result, float)
                or isinstance(result, int)
                or isinstance(result, np.float32)
                or isinstance(result, np.float64)
                or isinstance(result, str)):
            print(type(result), result)
            raise TypeError("The current returned {result} which was not recognized as a format.".format(result=result))
        if isinstance(result, str) and ',' in result:
            voltage = float_initialiser(result.split(',')[0])
        else:
            voltage = float_initialiser(result)
        if np.isnan(voltage):
            logging.warning("It was a NaN value measured by the SMU.")
            raise ValueError("The current returned {current} was not recognized as a number.".format(current=voltage))
        return voltage

    def smu_initiate_multiple_current(self, n: int, smu: str, kwargs=None):
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_CURRENT_MSG, smu)
            return

        if kwargs is None:
            kwargs = {}
        try:
            self[smu].disable_filter()
        except ValueError:
            pass

        if n is not None and self.__n_measurements[smu] != n:
            self.set_smu_measurements(smu, n, kwargs)

        self[smu].multi_current_measurement(**kwargs)

    def smu_initiate_multiple_voltage(self, n: int, smu: str, kwargs=None):
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_MSG, smu)
            return

        if kwargs is None:
            kwargs = {}
        try:
            self[smu].disable_filter()
        except ValueError:
            pass

        if n is not None and self.__n_measurements[smu] != n:
            self.set_smu_measurements(smu, n, kwargs)

        self[smu].multi_voltage_measurement(**kwargs)

    def smu_read_multiple_current(self, n: int, smu: str, kwargs=None) -> np.ndarray:
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_CURRENT_MSG, smu)
            if n is not None and self.__n_measurements[smu] != n:
                return np.full(n, np.nan)
            else:
                return np.full(self.__n_measurements[smu], np.nan)

        if kwargs is None:
            kwargs = {}
        result = self[smu].get_multi_current(**kwargs)
        if "binary_enabled" in kwargs and kwargs["binary_enabled"]:
            n = self.__n_measurements[smu]
            assert isinstance(n, int)
            if result.shape[0] > n:
                offset = int(result.shape[0] % n)
                shift = int(result.shape[0] // n)
                return result[offset::shift]
            return result
        return np.array(result.split(','), dtype=float_initialiser)

    def smu_read_multiple_voltage(self, n: int, smu: str, kwargs = None) -> np.ndarray:
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_CURRENT_MSG, smu)
            if n is not None and self.__n_measurements[smu] != n:
                return np.full(n, np.nan)
            else:
                return np.full(self.__n_measurements[smu], np.nan)

        if kwargs is None:
            kwargs = {}
        result = self[smu].get_multi_voltage(**kwargs)
        if "binary_enabled" in kwargs and kwargs["binary_enabled"]:
            n = self.__n_measurements[smu]
            assert isinstance(n, int)
            if result.shape[0] > n:
                offset = int(result.shape[0] % n)
                shift = int(result.shape[0] // n)
                return result[offset::shift]
            return result
        return np.array(result.split(','), dtype=float_initialiser)

    def smu_averaged_current(self, n: int, smu: str, kwargs=None) -> tuple[float, ...]:
        """
        smu_averaged_current

        Will perform a current measurement by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the specified
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :param smu: smu dut key of the SMU to configure.
        :param kwargs: further keyword arguments to be forwarded to the call to the lab device by basil.
        :return: (average current reading, uncertainty of the current reading) in A.
        """
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_CURRENT_MSG, smu)
            if n is not None and self.__n_measurements[smu] != n:
                return np.full(n, np.nan)
            else:
                return np.full(self.__n_measurements[smu], np.nan)
        if kwargs is None:
            kwargs = {}
        try:
            self[smu].disable_filter()
        except ValueError:
            pass
        self[smu].set_number_measurements(n, **kwargs)
        self[smu].multi_current_measurement(**kwargs)
        self[smu].create_average_current(**kwargs)
        result = self[smu].get_averaged_current(**kwargs)
        split = result.split(',', 1)
        return float(split[0]), float(split[1])

    def smu_averaged_voltage(self, n: int, smu: str, kwargs=None) -> tuple[float, float]:
        """
        smu_averaged_voltage

        Will perform a voltage measurement by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the specified
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :param smu: smu dut key of the SMU to configure.
        :param kwargs: further keyword arguments to be forwarded to the call to the lab device by basil.
        :return: (average voltage reading, uncertainty of the voltage reading) in V.
        """
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_MSG, smu)
            if n is not None and self.__n_measurements[smu] != n:
                return np.full(n, np.nan)
            else:
                return np.full(self.__n_measurements[smu], np.nan)
        if kwargs is None:
            kwargs = {}
        try:
            self[smu].disable_filter()
        except ValueError:
            pass
        self[smu].set_number_measurements(n, **kwargs)
        self[smu].multi_voltage_measurement(**kwargs)
        self[smu].create_average_voltage(**kwargs)
        result = self[smu].get_averaged_voltage(**kwargs)
        split = result.split(',', 1)
        return float(split[0]), float(split[1])

    def smu_advanced_current_multiple(self, n: int | None, smu: str, kwargs=None) -> ndarray:
        """
        smu_advanced_current_multiple

        Performs a current measurement by reading multiple current values from the SMU. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the specified
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :param smu: smu dut key of the SMU to configure.
        :param kwargs: further keyword arguments to be forwarded to the call to the lab device by basil or additionally
            'binary_enabled' in the case that the special binary readout mode should be used.
        :return: array of the measured currents in A; If the SMU is not active only NaN will be returned within the
            array.
        """
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_CURRENT_MSG, smu)
            if n is not None and self.__n_measurements[smu] != n:
                return np.full(n, np.nan)
            else:
                return np.full(self.__n_measurements[smu], np.nan)
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
            assert isinstance(n, int)
            if result.shape[0] > n:
                offset = int(result.shape[0] % n)
                shift = int(result.shape[0] // n)
                return result[offset::shift]
            return result
        return np.array(result.split(','), dtype=float_initialiser)

    def smu_advanced_voltage_multiple(self, n: int, smu: str, kwargs=None) -> ndarray:
        """
        smu_advanced_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the SMU. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the specified
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :param smu: smu dut key of the SMU to configure.
        :param kwargs: further keyword arguments to be forwarded to the call to the lab device by basil or additionally
            'binary_enabled' in the case that the special binary readout mode should be used.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_MSG, smu)
            if n is not None and self.__n_measurements[smu] != n:
                return np.full(n, np.nan)
            else:
                return np.full(self.__n_measurements[smu], np.nan)
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
        """
        general_smu_current_multiple

        Performs a current measurement by reading multiple current values from the SMU. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the specified
        smu is used.


        This is the slow implementation for this purpose consisting on single measurement calls to the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :param smu: smu dut key of the SMU to configure.
        :param kwargs: further keyword arguments to be forwarded to the call to the lab device by basil or additionally
            'binary_enabled' in the case that the special binary readout mode should be used.
        :return: array of the measured currents in A; If the SMU is not active only NaN will be returned within the
            array.
        """
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_CURRENT_MSG, smu)
            if n is not None and self.__n_measurements[smu] != n:
                return np.full(n, np.nan)
            else:
                return np.full(self.__n_measurements[smu], np.nan)
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
        """
        general_smu_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the SMU. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the specified
        smu is used.


        This is the slow implementation for this purpose consisting of single measurement calls to the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :param smu: smu dut key of the SMU to configure.
        :param kwargs: further keyword arguments to be forwarded to the call to the lab device by basil or additionally
            'binary_enabled' in the case that the special binary readout mode should be used.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        if not self.has_configured_smu(smu):
            logger.debug(SMU_DISABLED_MSG, smu)
            if n is not None and self.__n_measurements[smu] != n:
                return np.full(n, np.nan)
            else:
                return np.full(self.__n_measurements[smu], np.nan)
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
        """
        smu_output_on

        Turns the output of the specified SMU on.
        The call to the SMU is only performed when the SMU is connected and active.

        :param smu: smu dut key of the SMU to configure
        :param kwargs: further keyword arguments to be forwarded to the call to the lab device by basil.
        """
        if not self.has_configured_smu(smu):
            logger.debug("The output could only be turned on for an active smu but '%s' is inactive.", smu)
            return
        if kwargs is None:
            kwargs = {}
        self[smu].on(**kwargs)

    def smu_output_off(self, smu: str, kwargs=None):
        """
        smu_output_off

        Turns the output of the specified SMU off.
        The call to the SMU is only performed when the SMU is connected and active.

        :param smu: smu dut key of the SMU to configure
        :param kwargs: further keyword arguments to be forwarded to the call to the lab device by basil.
        """
        if not self.has_configured_smu(smu):
            logger.debug("The output could only be turned off for an active smu but '%s' is inactive.", smu)
            return
        if kwargs is None:
            kwargs = {}
        self[smu].off(**kwargs)

    def set_smu_measurements(self, smu: str, value: int, kwargs=None):
        """
        set_smu_measurements

        Configures the number of measurements to be performed on reading (using reading buffer) for the specific SMU.
        The call to the SMU is only performed when the SMU is connected and active.
        If the requested number of measurements is negative, it will be replaced by 1 before transmission.
        Additionally, the number of measurements property for the SMU is updated with the new value.

        :param smu: smu dut key of the SMU to configure
        :param value: number of measurements to be set for readings.
        :param kwargs: further keyword arguments to be forwarded to the call to the lab device by basil.
        """
        if not self.has_configured_smu(smu):
            logger.debug(
                "The number of measurements to be done could only be set for an active smu but '%s' is inactive.", smu)
            return
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
    def init_smu(self, src_u, current_range, voltage_range=1.5, current_limit=0.001, plc=10, **kwargs):
        """
        init_smu

        Performs the initial setup for primary SMU usually supplying the VM3 connector
        The setup could only be performed when the SMU is connected and active.
        This setup is designed only for sourcing voltage and measuring currents.
        So the current range, over-current protection and sourcing voltage will be set according to the provided
        arguments.
        On some SMUs it will also configure the measurement buffers for acquiring multiple readings at once.

        :param current_limit: current compliance limit to be set in A
        :param current_range: current measurement range/maximum current intended to be measured in A
        :param plc: number of power supply cycles to be averaged over when measuring
        :param src_u: sourcing voltage for the SMU
        :param voltage_range:
        :param kwargs: further keyword arguments to be forwarded to the call of the lab device by basil
        :keyword smu: smu dut key of the SMU to configure
        """
        active_smu_key = kwargs.get("smu", self.__primary_smu_key)
        self.smu_init(active_smu_key, current_limit, current_range, plc, src_u, voltage_range, kwargs=self.smu_kwargs)

    def smu_on(self):
        """
        smu_on

        Turns the output of the primary SMU on.
        The call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_output_on(self.__primary_smu_key, kwargs=self.smu_kwargs)

    def smu_off(self):
        """
        smu_off

        Turns the output of the primary SMU off.
        The call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_output_off(self.__primary_smu_key, kwargs=self.smu_kwargs)

    def smu_set_current_source(self):
        """
        smu_set_current_source

        Sets the mode of the primary (VM3) SMUs source to sourcing current.
        A call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_source_current(self.__primary_smu_key, kwargs=self.smu_kwargs)

    def smu_set_voltage_source(self):
        """
        smu_set_voltage_source

        Sets the mode of the primary SMUs source to sourcing voltage.
        A call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_source_volt(self.__primary_smu_key, kwargs=self.smu_kwargs)

    @property
    def source_voltage_smu(self):
        """
        Gets the voltage to be sourced by the primary SMU (constant voltage mode of the SMU).
        The call to the SMU is only performed when the SMU is connected and active.
        After changing the configuration, the settling of the smu will be taken into account to make sure that no
        measurement is performed within the settling interval of the voltage source.
        """
        return self.get_smu_source_voltage(self.__primary_smu_key, kwargs=self.smu_kwargs)

    @source_voltage_smu.setter
    def source_voltage_smu(self, value):
        self.set_smu_source_voltage(self.__primary_smu_key, value, kwargs=self.smu_kwargs)

    @property
    def source_current_smu(self):
        """
        Gets the current to be sourced by the primary SMU (constant current mode of the SMU).
        The call to the SMU is only performed when the SMU is connected and active.
        After changing the configuration, the settling of the smu will be taken into account to make sure that no
        measurement is performed within the settling interval of the current source.
        """
        return self.get_smu_source_current(self.__primary_smu_key, kwargs=self.smu_kwargs)

    @source_current_smu.setter
    def source_current_smu(self, value):
        self.set_smu_source_current(self.__primary_smu_key, value, kwargs=self.smu_kwargs)

    @property
    def get_source_current(self) -> float:
        """
        Read the current through the primary SMU. (Performs a current measurement in constant voltage mode.)
        The call to the SMU is only performed when the SMU is connected and active.
        """
        return self.smu_measure_current(self.__primary_smu_key, kwargs=self.smu_kwargs)

    def smu_measure_volts(self):
        """
        smu_measure_volts

        Read the voltage over the primary SMU contacts. (Performs a voltage measurement in constant current mode.)
        The call to the primary SMU is only performed when the SMU is connected and active.
        """
        return self.smu_measure_voltage(self.__primary_smu_key, kwargs=self.smu_kwargs)

    def averaged_current(self, n: int = 10):
        """
        averaged_current

        Will perform a current measurement at the primary SMU by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform
        :return: (average current reading, uncertainty of the current reading) in A
        """
        return self.smu_averaged_current(n, self.__primary_smu_key, kwargs=self.smu_kwargs)

    def averaged_voltage(self, n: int = 10):
        """
        averaged_voltage

        Will perform a voltage measurement at the primary SMU by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: (average voltage reading, uncertainty of the voltage reading) in V
        """
        return self.smu_averaged_voltage(n, self.__primary_smu_key, kwargs=self.smu_kwargs)

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
        return self.general_smu_current_multiple(self.__primary_smu_key, n, kwargs=self.smu_kwargs)

    def get_source_voltage_multiple(self, n: int):
        """
        get_source_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the primary SMU. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the slow implementation for this purpose consisting of single measurement calls to the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        return self.general_smu_voltage_multiple(self.__primary_smu_key, n, kwargs=self.smu_kwargs)

    def get_advanced_current_multiple(self, n: int):
        """
        get_advanced_current_multiple

        Performs a current measurement by reading multiple current values from the primary SMU. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform
        :return: array of the measured currents in A; If the SMU is not active only NaN will be returned within the
            array.
        """
        keyword_arguments = self.smu_kwargs.copy()
        keyword_arguments['binary_enabled'] = self.binary_active
        keyword_arguments['data_points'] = self.n_measurements
        return self.smu_advanced_current_multiple(n, self.__primary_smu_key, kwargs=keyword_arguments)

    def get_advanced_voltage_multiple(self, n: int):
        """
        get_advanced_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the primary SMU. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        kargs = self.smu_kwargs.copy()
        kargs['binary_enabled'] = self.binary_active
        kargs['data_points'] = self.n_measurements
        return self.smu_advanced_voltage_multiple(n, self.__primary_smu_key, kwargs=kargs)

    def initiate_multiple_current(self, n: int, kwargs=None):
        self.smu_initiate_multiple_current(n, self.primary_smu_key, kwargs=kwargs)

    def initiate_multiple_voltage(self, n: int, kwargs=None):
        self.smu_initiate_multiple_voltage(n, self.primary_smu_key, kwargs)

    def get_read_multiple_current(self, n: int, kwargs=None) -> np.ndarray:
        return self.smu_read_multiple_current(n, self.primary_smu_key, kwargs=kwargs)

    def get_read_multiple_voltage(self, n: int, kwargs=None) -> np.ndarray:
        return self.smu_read_multiple_voltage(n, self.primary_smu_key, kwargs)

    # endregion

    # region Handle the biasing supply.
    # Handle the biasing supply
    def init_bias(self, voltage, current_range, voltage_range=1.5, current_limit=0.001, plc=10):
        """
        init_bias

        Performs the initial setup for HV SMU.
        The setup could only be performed when the SMU is connected and active.
        This setup is designed only for sourcing voltage and measuring currents.
        So the current range, over-current protection and sourcing voltage will be set according to the provided
        arguments.
        On some SMUs it will also configure the measurement buffers for acquiring multiple readings at once.

        :param current_limit: current compliance limit to be set in A.
        :param current_range: current measurement range/maximum current intended to be measured in A.
        :param plc: number of power supply cycles to be averaged over when measuring.
        :param voltage: sourcing voltage for the SMU.
        :param voltage_range: voltage range of the SMUs source.
        """
        if self.has_bias_suppy:
            self.smu_init(self.__bias_smu_key, current_limit, current_range, plc, voltage, voltage_range,
                          kwargs=self.smu_bias_kwargs)

    def bias_on(self):
        """
        bias_on

        Turns the output of the HV SMU on.
        The call to the SMU is only performed when the SMU is connected and active.
        """
        if self.has_bias_suppy:
            self.smu_output_on(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    def bias_off(self):
        """
        bias_off

        Turns the output of the HV SMU off.
        The call to the SMU is only performed when the SMU is connected and active.
        """
        if self.has_bias_suppy:
            self.smu_output_off(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    def bias_current_source(self):
        """
        bias_current_source

        Sets the mode of the HV SMUs source to sourcing current.
        A call to the SMU is only performed when the SMU is connected and active.
        """
        if self.has_bias_suppy:
            self.smu_source_current(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    def bias_voltage_source(self):
        """
        bias_voltage_source

        Sets the mode of the HV SMUs source to sourcing voltage.
        A call to the SMU is only performed when the SMU is connected and active.
        """
        if self.has_bias_suppy:
            self.smu_source_volt(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)

    @property
    def bias_voltage(self):
        """
        Gets the voltage to be sourced by the HV SMU (constant voltage mode of the SMU).
        The call to the SMU is only performed when the SMU is connected and active.
        After changing the configuration, the settling of the smu will be taken into account to make sure that no
        measurement is performed within the settling interval of the voltage source.
        """
        if self.has_bias_suppy:
            # FIXME: fails with command not found. But interestingly it is immune to modifications of the scpi class.
            logger.info("Attempted to read the bias voltage.")
            return self.get_smu_source_voltage(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)
        return 0.0

    @bias_voltage.setter
    def bias_voltage(self, value):
        if self.has_bias_suppy:
            logger.info("Attempted to set the bias voltage to %f", value)
            self.set_smu_source_voltage(self.__bias_smu_key, value, kwargs=self.smu_bias_kwargs)

    @property
    def bias_current(self):
        """
        Gets the current to be sourced by the HV SMU (constant current mode of the SMU).
        The call to the SMU is only performed when the SMU is connected and active.
        After changing the configuration, the settling of the smu will be taken into account to make sure that no
        measurement is performed within the settling interval of the current source.
        """
        if self.has_bias_suppy:
            return self.get_smu_source_current(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)
        return np.nan

    @bias_current.setter
    def bias_current(self, value):
        if self.has_bias_suppy:
            self.set_smu_source_current(self.__bias_smu_key, value, kwargs=self.smu_bias_kwargs)

    def bias_measure_current(self) -> float:
        """
        bias_measure_current

        Read the current through the HV SMU. (Performs a current measurement in constant voltage mode.)
        The call to the HV SMU is only performed when the SMU is connected and active.
        """
        if self.has_bias_suppy:
            return self.smu_measure_current(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)
        return np.nan

    def bias_measure_volts(self):
        """
        bias_measure_volts

        Read the voltage over the HV SMU contacts. (Performs a voltage measurement in constant current mode.)
        The call to the HV SMU is only performed when the SMU is connected and active.
        """
        if self.has_bias_suppy:
            return self.smu_measure_voltage(self.__bias_smu_key, kwargs=self.smu_bias_kwargs)
        return np.nan

    def bias_averaged_current(self, n: int = 10):
        """
        bias_averaged_current

        Will perform a current measurement at the HV SMU by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform
        :return: (average current reading, uncertainty of the current reading) in A
        """
        if self.has_bias_suppy:
            return self.smu_averaged_current(n, self.__bias_smu_key, kwargs=self.smu_bias_kwargs)
        elif n is None:
            return np.full(10, fill_value=np.nan)
        return np.full(n, fill_value=np.nan)

    def bias_averaged_voltage(self, n: int = 10):
        """
        bias_averaged_voltage

        Will perform a voltage measurement at the HV SMU by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: (average voltage reading, uncertainty of the voltage reading) in V
        """
        if self.has_bias_suppy:
            return self.smu_averaged_voltage(n, self.__bias_smu_key, kwargs=self.smu_bias_kwargs)
        elif n is None:
            return np.full(10, fill_value=np.nan)
        return np.full(n, fill_value=np.nan)

    def bias_current_multiple(self, n: int):
        """
        bias_current_multiple

        Performs a current measurement by reading multiple current values from the HV SMU. The full dataset of
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
        if self.has_bias_suppy:
            return self.general_smu_current_multiple(self.__bias_smu_key, n, kwargs=self.smu_bias_kwargs)
        elif n is None:
            return np.full(10, fill_value=np.nan)
        return np.full(n, fill_value=np.nan)

    def bias_voltage_multiple(self, n: int):
        """
        bias_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the HV SMU. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the slow implementation for this purpose consisting of single measurement calls to the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        if self.has_bias_suppy:
            return self.general_smu_voltage_multiple(self.__bias_smu_key, n, kwargs=self.smu_bias_kwargs)
        elif n is None:
            return np.full(10, fill_value=np.nan)
        return np.full(n, fill_value=np.nan)

    def bias_advanced_current_multiple(self, n: int):
        """
        bias_advanced_current_multiple

        Performs a current measurement by reading multiple current values from the HV SMU. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform
        :return: array of the measured currents in A; If the SMU is not active only NaN will be returned within the
            array.
        """
        if self.has_bias_suppy:
            results = self.smu_advanced_current_multiple(n, self.__bias_smu_key, kwargs=self.smu_bias_kwargs)
            if self[self.bias_smu_key].get_buffer2_mode().startswith("NEXT"):
                logger.info("Still in data taking mode. 2")
                self[self.bias_smu_key].disable_buffer()
            return results
        elif n is None:
            return np.full(10, fill_value=np.nan)
        return np.full(n, fill_value=np.nan)

    def bias_advanced_voltage_multiple(self, n: int):
        """
        bias_advanced_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the HV SMU. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        if self.has_bias_suppy:
            return self.smu_advanced_voltage_multiple(n, self.__bias_smu_key, kwargs=self.smu_bias_kwargs)
        elif n is None:
            return np.full(10, fill_value=np.nan)
        return np.full(n, fill_value=np.nan)

    def bias_initiate_multiple_current(self, n: int, kwargs=None):
        self.smu_initiate_multiple_current(n, self.bias_smu_key, kwargs=kwargs)

    def bias_initiate_multiple_voltage(self, n: int, kwargs=None):
        self.smu_initiate_multiple_voltage(n, self.bias_smu_key, kwargs=kwargs)

    def bias_read_multiple_current(self, n: int, kwargs=None) -> np.ndarray:
        return self.smu_read_multiple_current(n, self.bias_smu_key, kwargs=kwargs)

    def bias_read_multiple_voltage(self, n: int, kwargs=None) -> np.ndarray:
        return self.smu_read_multiple_voltage(n, self.bias_smu_key, kwargs=kwargs)

    # endregion

    # region Handle the VM1 Connector SMU
    def init_vm1(self, src_u, current_range, voltage_range=1.5, current_limit=0.001, plc=10):
        """
        init_vm1

        Performs the initial setup for SMU connected to the VM1 PCB port.
        The setup could only be performed when the SMU is connected and active.
        This setup is designed only for sourcing voltage and measuring currents.
        So the current range, over-current protection and sourcing voltage will be set according to the provided
        arguments.
        On some SMUs it will also configure the measurement buffers for acquiring multiple readings at once.

        :param current_limit: current compliance limit to be set in A.
        :param current_range: current measurement range/maximum current intended to be measured in A.
        :param plc: number of power supply cycles to be averaged over when measuring.
        :param src_u: sourcing voltage for the SMU.
        :param voltage_range: voltage range of the SMUs source.
        """
        self.smu_init(self.__vm1_smu_key, current_limit, current_range, plc, src_u, voltage_range,
                      kwargs=self.smu_vm1_kwargs)

    def vm1_on(self):
        """
        vm1_on

        Turns the output of the SMU connected to the PCBs VM1 port on.
        The call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_output_on(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_off(self):
        """
        vm1_off

        Turns the output of the SMU connected to the PCBS VM1 port off.
        The call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_output_off(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_current_source(self):
        """
        vm1_current_source

        Sets the mode of the SMUs source connected to the PCBs VM1 port to sourcing current.
        A call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_source_current(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_voltage_source(self):
        """
        vm1_voltage_source

        Sets the mode of the SMUs source connected to the PCBs VM1 port to sourcing voltage.
        A call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_source_volt(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    @property
    def vm1_voltage(self):
        """
        Gets the voltage to be sourced by the SMU connected to the PCBs VM1 port (constant voltage mode of the SMU).
        The call to the SMU is only performed when the SMU is connected and active.
        After changing the configuration, the settling of the smu will be taken into account to make sure that no
        measurement is performed within the settling interval of the voltage source.
        """
        return self.get_smu_source_voltage(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    @vm1_voltage.setter
    def vm1_voltage(self, value):
        self.set_smu_source_voltage(self.__vm1_smu_key, value, kwargs=self.smu_vm1_kwargs)

    @property
    def vm1_current(self):
        """
        Gets the current to be sourced by the SMU connected to the PCBs VM1 port (constant current mode of the SMU).
        The call to the SMU is only performed when the SMU is connected and active.
        After changing the configuration, the settling of the smu will be taken into account to make sure that no
        measurement is performed within the settling interval of the current source.
        """
        return self.get_smu_source_current(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    @vm1_current.setter
    def vm1_current(self, value):
        self.set_smu_source_current(self.__vm1_smu_key, value, kwargs=self.smu_vm1_kwargs)

    def vm1_measure_current(self) -> float:
        """
        vm1_measure_current

        Read the current through the SMU connected to the PCBs VM1 port. (Performs a current measurement in constant voltage mode.)
        The call to the SMU is only performed when the SMU is connected and active.
        """
        return self.smu_measure_current(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_measure_volts(self):
        """
        vm1_measure_volts

        Read the voltage over the SMU contacts connected to the PCBs VM1 port. (Performs a voltage measurement in constant current mode.)
        The call to the SMU is only performed when the SMU is connected and active.
        """
        return self.smu_measure_voltage(self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_averaged_current(self, n: int = 10):
        """
        vm1_averaged_current

        Will perform a current measurement at the SMU connected to the PCBs VM1 port by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform
        :return: (average current reading, uncertainty of the current reading) in A
        """
        return self.smu_averaged_current(n, self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_averaged_voltage(self, n: int = 10):
        """
        vm1_averaged_voltage

        Will perform a voltage measurement at the SMU connected to the PCBs VM1 port by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: (average voltage reading, uncertainty of the voltage reading) in V
        """
        return self.smu_averaged_voltage(n, self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_current_multiple(self, n: int):
        """
        vm1_current_multiple

        Performs a current measurement by reading multiple current values from the SMU connected to the PCBs VM1 port. The full dataset of
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
        return self.general_smu_current_multiple(self.__vm1_smu_key, n, kwargs=self.smu_vm1_kwargs)

    def vm1_voltage_multiple(self, n: int):
        """
        vm1_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the SMU connected to the PCBs VM1 port. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the slow implementation for this purpose consisting of single measurement calls to the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        return self.general_smu_voltage_multiple(self.__vm1_smu_key, n, kwargs=self.smu_vm1_kwargs)

    def vm1_advanced_current_multiple(self, n: int):
        """
        vm1_advanced_current_multiple

        Performs a current measurement by reading multiple current values from the SMU connected to the PCBs VM1 port. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured currents in A; If the SMU is not active only NaN will be returned within the
            array.
        """
        return self.smu_advanced_current_multiple(n, self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_advanced_voltage_multiple(self, n: int):
        """
        vm1_advanced_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the SMU connected to the PCBs VM1 port. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        return self.smu_advanced_voltage_multiple(n, self.__vm1_smu_key, kwargs=self.smu_vm1_kwargs)

    def vm1_initiate_multiple_current(self, n: int, kwargs=None):
        self.smu_initiate_multiple_current(n, self.vm1_smu_key, kwargs=kwargs)

    def vm1_initiate_multiple_voltage(self, n: int, kwargs=None):
        self.smu_initiate_multiple_voltage(n, self.vm1_smu_key, kwargs=kwargs)

    def vm1_read_multiple_current(self, n: int, kwargs=None) -> np.ndarray:
        return self.smu_read_multiple_current(n, self.vm1_smu_key, kwargs=kwargs)

    def vm1_read_multiple_voltage(self, n: int, kwargs=None) -> np.ndarray:
        return self.smu_read_multiple_voltage(n, self.vm1_smu_key, kwargs=kwargs)

    # endregion

    # region Handle the VM2 Connector SMU
    def init_vm2(self, src_u, current_range, voltage_range=1.5, current_limit=0.001, plc=10):
        """
        init_vm2

        Performs the initial setup for SMU connected to the VM2 PCB port.
        The setup could only be performed when the SMU is connected and active.
        This setup is designed only for sourcing voltage and measuring currents.
        So the current range, over-current protection and sourcing voltage will be set according to the provided
        arguments.
        On some SMUs it will also configure the measurement buffers for acquiring multiple readings at once.

        :param current_limit: current compliance limit to be set in A.
        :param current_range: current measurement range/maximum current intended to be measured in A.
        :param plc: number of power supply cycles to be averaged over when measuring.
        :param src_u: sourcing voltage for the SMU.
        :param voltage_range: voltage range of the SMUs source.
        """
        self.smu_init(self.__vm2_smu_key, current_limit, current_range, plc, src_u, voltage_range,
                      kwargs=self.smu_vm2_kwargs)

    def vm2_on(self):
        """
        vm2_on

        Turns the output of the SMU connected to the PCBs VM2 port on.
        The call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_output_on(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_off(self):
        """
        vm2_off

        Turns the output of the SMU connected to the PCBS VM2 port off.
        The call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_output_off(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_current_source(self):
        """
        vm2_current_source

        Sets the mode of the SMUs source connected to the PCBs VM2 port to sourcing current.
        A call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_source_current(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_voltage_source(self):
        """
        vm2_voltage_source

        Sets the mode of the SMUs source connected to the PCBs VM2 port to sourcing voltage.
        A call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_source_volt(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    @property
    def vm2_voltage(self):
        """
        Gets the voltage to be sourced by the SMU connected to the PCBs VM2 port (constant voltage mode of the SMU).
        The call to the SMU is only performed when the SMU is connected and active.
        After changing the configuration, the settling of the smu will be taken into account to make sure that no
        measurement is performed within the settling interval of the voltage source.
        """
        return self.get_smu_source_voltage(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    @vm2_voltage.setter
    def vm2_voltage(self, value):
        self.set_smu_source_voltage(self.__vm2_smu_key, value, kwargs=self.smu_vm2_kwargs)

    @property
    def vm2_current(self):
        """
        Gets the current to be sourced by the SMU connected to the PCBs VM2 port (constant current mode of the SMU).
        The call to the SMU is only performed when the SMU is connected and active.
        After changing the configuration, the settling of the smu will be taken into account to make sure that no
        measurement is performed within the settling interval of the current source.
        """
        return self.get_smu_source_current(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    @vm2_current.setter
    def vm2_current(self, value):
        self.set_smu_source_current(self.__vm2_smu_key, value, kwargs=self.smu_vm2_kwargs)

    def vm2_measure_current(self) -> float:
        """
        vm2_measure_current

        Read the current through the SMU connected to the PCBs VM2 port. (Performs a current measurement in constant voltage mode.)
        The call to the SMU is only performed when the SMU is connected and active.
        """
        return self.smu_measure_current(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_measure_volts(self):
        """
        vm2_measure_volts

        Read the voltage over the SMU contacts connected to the PCBs VM2 port. (Performs a voltage measurement in constant current mode.)
        The call to the SMU is only performed when the SMU is connected and active.
        """
        return self.smu_measure_voltage(self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_averaged_current(self, n: int = 10):
        """
        vm2_averaged_current

        Will perform a current measurement at the SMU connected to the PCBs VM2 port by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform
        :return: (average current reading, uncertainty of the current reading) in A
        """
        return self.smu_averaged_current(n, self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_averaged_voltage(self, n: int = 10):
        """
        vm2_averaged_voltage

        Will perform a voltage measurement at the SMU connected to the PCBs VM2 port by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: (average voltage reading, uncertainty of the voltage reading) in V
        """
        return self.smu_averaged_voltage(n, self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_current_multiple(self, n: int):
        """
        vm2_current_multiple

        Performs a current measurement by reading multiple current values from the SMU connected to the PCBs VM2 port. The full dataset of
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
        return self.general_smu_current_multiple(self.__vm2_smu_key, n, kwargs=self.smu_vm2_kwargs)

    def vm2_voltage_multiple(self, n: int):
        """
        vm2_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the SMU connected to the PCBs VM2 port. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the slow implementation for this purpose consisting of single measurement calls to the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        return self.general_smu_voltage_multiple(self.__vm2_smu_key, n, kwargs=self.smu_vm2_kwargs)

    def vm2_advanced_current_multiple(self, n: int):
        """
        vm2_advanced_current_multiple

        Performs a current measurement by reading multiple current values from the SMU connected to the PCBs VM2 port. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured currents in A; If the SMU is not active only NaN will be returned within the
            array.
        """
        return self.smu_advanced_current_multiple(n, self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_advanced_voltage_multiple(self, n: int):
        """
        vm2_advanced_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the SMU connected to the PCBs VM2 port. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        return self.smu_advanced_voltage_multiple(n, self.__vm2_smu_key, kwargs=self.smu_vm2_kwargs)

    def vm2_initiate_multiple_current(self, n: int, kwargs=None):
        self.smu_initiate_multiple_current(n, self.vm2_smu_key, kwargs=kwargs)

    def vm2_initiate_multiple_voltage(self, n: int, kwargs=None):
        self.smu_initiate_multiple_voltage(n, self.vm2_smu_key, kwargs=kwargs)

    def vm2_read_multiple_current(self, n: int, kwargs=None) -> np.ndarray:
        return self.smu_read_multiple_current(n, self.vm2_smu_key, kwargs=kwargs)

    def vm2_read_multiple_voltage(self, n: int, kwargs=None) -> np.ndarray:
        return self.smu_read_multiple_voltage(n, self.vm2_smu_key, kwargs=kwargs)
    # endregion

    # region Handle the VM3 Connector SMU
    def init_vm3(self, src_u, current_range, voltage_range=1.5, current_limit=0.001, plc=10):
        """
        init_vm3

        Performs the initial setup for the SMU connected to the VM3 SMU port.
        The setup could only be performed when the SMU is connected and active.
        This setup is designed only for sourcing voltage and measuring currents.
        So the current range, over-current protection and sourcing voltage will be set according to the provided
        arguments.
        On some SMUs it will also configure the measurement buffers for acquiring multiple readings at once.

        :param current_limit: current compliance limit to be set in A.
        :param current_range: current measurement range/maximum current intended to be measured in A.
        :param plc: number of power supply cycles to be averaged over when measuring.
        :param src_u: sourcing voltage for the SMU.
        :param voltage_range: voltage range of the SMUs source.
        """
        self.smu_init(self.__vm3_smu_key, current_limit, current_range, plc, src_u, voltage_range,
                      kwargs=self.smu_vm3_kwargs)

    def vm3_on(self):
        """
        vm3_on

        Turns the output of the SMU connected to the PCBs VM3 port on.
        The call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_output_on(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_off(self):
        """
        vm3_off

        Turns the output of the SMU connected to the PCBS VM3 port off.
        The call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_output_off(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_current_source(self):
        """
        vm3_current_source

        Sets the mode of the SMUs source connected to the PCBs VM3 port to sourcing current.
        A call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_source_current(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_voltage_source(self):
        """
        vm3_voltage_source

        Sets the mode of the SMUs source connect to the PCBs VM3 port to sourcing voltage.
        A call to the SMU is only performed when the SMU is connected and active.
        """
        self.smu_source_volt(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    @property
    def vm3_voltage(self):
        """
        Gets the voltage to be sourced by the SMU connected to the PCBs VM3 port (constant voltage mode of the SMU).
        The call to the SMU is only performed when the SMU is connected and active.
        After changing the configuration, the settling of the smu will be taken into account to make sure that no
        measurement is performed within the settling interval of the voltage source.
        """
        return self.get_smu_source_voltage(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    @vm3_voltage.setter
    def vm3_voltage(self, value):
        self.set_smu_source_voltage(self.__vm3_smu_key, value, kwargs=self.smu_vm3_kwargs)

    @property
    def vm3_current(self):
        """
        Gets the current to be sourced by the SMU connected to the PCBs VM3 port (constant current mode of the SMU).
        The call to the SMU is only performed when the SMU is connected and active.
        After changing the configuration, the settling of the smu will be taken into account to make sure that no
        measurement is performed within the settling interval of the current source.
        """
        return self.get_smu_source_current(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    @vm3_current.setter
    def vm3_current(self, value):
        self.set_smu_source_current(self.__vm3_smu_key, value, kwargs=self.smu_vm3_kwargs)

    def vm3_measure_current(self) -> float:
        """
        vm3_measure_current

        Read the current through the SMU connected to the PCBs VM3 port. (Performs a current measurement in constant voltage mode.)
        The call to the SMU is only performed when the SMU is connected and active.
        """
        return self.smu_measure_current(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_measure_volts(self):
        """
        vm3_measure_volts

        Read the voltage over the SMU contacts connected to the PCBs VM3 port. (Performs a voltage measurement in constant current mode.)
        The call to the SMU is only performed when the SMU is connected and active.
        """
        return self.smu_measure_voltage(self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_averaged_current(self, n: int = 10):
        """
        vm3_averaged_current

        Will perform a current measurement at the SMU connected to the PCBs VM3 port by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform
        :return: (average current reading, uncertainty of the current reading) in A
        """
        return self.smu_averaged_current(n, self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_averaged_voltage(self, n: int = 10):
        """
        vm3_averaged_voltage

        Will perform a voltage measurement at the SMU connected to the PCBs VM3 port by measuring multiple times and read only the averaged value.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.
        The call to the SMU is only performed when the SMU is connected and active.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: (average voltage reading, uncertainty of the voltage reading) in V
        """
        return self.smu_averaged_voltage(n, self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_current_multiple(self, n: int):
        """
        vm3_current_multiple

        Performs a current measurement by reading multiple current values from the SMU connected to the PCBs VM3 port. The full dataset of
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
        return self.general_smu_current_multiple(self.__vm3_smu_key, n, kwargs=self.smu_vm3_kwargs)

    def vm3_voltage_multiple(self, n: int):
        """
        vm3_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the SMU connected to the PCBs VM3 port. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the slow implementation for this purpose consisting of single measurement calls to the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        return self.general_smu_voltage_multiple(self.__vm3_smu_key, n, kwargs=self.smu_vm3_kwargs)

    def vm3_advanced_current_multiple(self, n: int):
        """
        vm3_advanced_current_multiple

        Performs a current measurement by reading multiple current values from the SMU connected to the PCBs VM3 port. The full dataset of
        measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured currents in A; If the SMU is not active only NaN will be returned within the
            array.
        """
        return self.smu_advanced_current_multiple(n, self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_advanced_voltage_multiple(self, n: int):
        """
        vm3_advanced_voltage_multiple

        Performs a voltage measurement by reading multiple voltage values from the SMU connected to the PCBs VM3 port.
        The full dataset of measurements will be returned.
        The call to the SMU is only performed when the SMU is connected and active.
        If no number of measurements is explicitly specified the number of measurements properties for the
        smu is used.


        This is the fast implementation for this purpose using directly dedicated functions of the SMU.

        :param n: number of measurements to be performed or None when the property should be used to determine the
            number of measurements to perform.
        :return: array of the measured voltages in V; If the SMU is not active only NaN will be returned within the
            array.
        """
        return self.smu_advanced_voltage_multiple(n, self.__vm3_smu_key, kwargs=self.smu_vm3_kwargs)

    def vm3_initiate_multiple_current(self, n: int, kwargs=None):
        self.smu_initiate_multiple_current(n, self.vm3_smu_key, kwargs=kwargs)

    def vm3_initiate_multiple_voltage(self, n: int, kwargs=None):
        self.smu_initiate_multiple_voltage(n, self.vm3_smu_key, kwargs=kwargs)

    def vm3_read_multiple_current(self, n: int, kwargs=None) -> np.ndarray:
        return self.smu_read_multiple_current(n, self.vm3_smu_key, kwargs=kwargs)

    def vm3_read_multiple_voltage(self, n: int, kwargs=None) -> np.ndarray:
        return self.smu_read_multiple_voltage(n, self.vm3_smu_key, kwargs=kwargs)
    # endregion
