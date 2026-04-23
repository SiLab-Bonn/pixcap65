"""
Script for measuring Inter Pixel Capacitance 
"""

import gc
import logging
import time
from collections.abc import Iterable, Mapping

import numpy as np
import pylab as pl
import tables as tb
from bitarray import bitarray
from tqdm import tqdm

import pixcap65_constants as c
from configs.config_handler import extract_smu_current_error
from pixcap_65_test_total_cap import PixCap65Measurement, ScanConfigurationKeys, MEASURING_PIXEL_TEXT, \
    store_scan_par_values, _store_scan_par_values
from tqdm_logging_utils import logging_redirect_tqdm

UNCERT_ESTIMATION_ERROR_MSG = "Something went wrong during the estimation of the measurement errors."

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

scan_configuration = {
    'start_column': 15,
    'stop_column': 35,
    'start_row': 15,
    'stop_row': 35,

    'Vin': 1.0,  # input voltage in V
    'frequency_range': np.arange(1, 12.1, 1),  # .astype(np.float) # [MHz]
    'bias': -80,

    'data_path': "Reference/R13",
    "out_file_mode": "append",
}


class InterCap(tb.IsDescription):
    col = tb.Int32Col(pos=0)
    row = tb.Int32Col(pos=1)
    inter_cap_a = tb.Float64Col(pos=2)
    inter_cap_b = tb.Float64Col(pos=3)
    total_cap_b = tb.Float64Col(pos=4)
    leakage_a = tb.Float64Col(pos=5)
    leakage_b = tb.Float64Col(pos=6)
    leakage_total = tb.Float64Col(pos=7)


class Pixcap65InterCap(PixCap65Measurement):
    def __init__(self, scan_config, output_file, **kwargs):
        super(Pixcap65InterCap, self).__init__(scan_config, output_file, **kwargs)

        # prepare the data fields for the measurement
        self.inter_hist_current_1 = np.full(shape=(40, 40, self.n_frequencies),
                                            fill_value=np.nan)  # current value for each measured frequency per pixel
        self.inter_hist_current_2 = np.full(shape=(40, 40, self.n_frequencies),
                                            fill_value=np.nan)  # current value for each measured frequency per pixel
        self.total_hist_current = np.full(shape=(40, 40, self.n_frequencies),
                                          fill_value=np.nan)  # current value for each measured frequency per pixel

        self.inter_hist_current_1_error = np.full(shape=(40, 40, self.n_frequencies),
                                            fill_value=np.nan)  # current value for each measured frequency per pixel
        self.inter_hist_current_2_error = np.full(shape=(40, 40, self.n_frequencies),
                                            fill_value=np.nan)  # current value for each measured frequency per pixel
        self.total_hist_current_error = np.full(shape=(40, 40, self.n_frequencies),
                                          fill_value=np.nan)  # current value for each measured frequency per pixel
        # this kind of setup is somewhat misplaced.
        self.pixcap.seq_size = 4

    def configure(self):
        # already done by super-class
        super(Pixcap65InterCap, self).configure()
        # self.pixcap['SMU3'].off()
        # self.pixcap['SMU3'].source_volt()
        # self.pixcap['SMU3'].set_voltage_range(1.5)
        # self.pixcap['SMU3'].set_current_nlpc(10)
        # self.pixcap['SMU3'].set_voltage(1.0)
        # self.pixcap['SMU3'].set_current_limit(0.001)
        # self.pixcap['SMU3'].set_current_sense_range(0.00001)

        self.init_smu(smu=self.pixcap.vm2_smu_key)
        # self.pixcap['SMU2'].off()
        # self.pixcap['SMU2'].source_volt()
        # self.pixcap['SMU2'].set_voltage_range(1.5)
        # self.pixcap['SMU2'].set_current_nlpc(10)
        # self.pixcap['SMU2'].set_voltage(1.0)
        # self.pixcap['SMU2'].set_current_limit(0.001)
        # self.pixcap['SMU2'].set_current_sense_range(0.00001)

        # settings for sensor depletion source
        # self.init_smu(smu=self.pixcap.vm1_smu_key)
        # self.pixcap['SMU1'].off()
        # self.pixcap['SMU1'].source_volt()
        # self.pixcap['SMU1'].set_voltage_range(1.5)
        # self.pixcap['SMU1'].set_current_nlpc(10)
        # self.pixcap['SMU1'].set_voltage(-80.0)
        # self.pixcap['SMU1'].set_current_limit(0.001)
        # self.pixcap['SMU1'].set_current_sense_range(0.00001)

        self.pixcap['SEQ'].reset()
        self.pixcap['SEQ'].set_clk_divide(1)
        self.pixcap['SEQ'].set_repeat_start(0)
        self.pixcap['SEQ'].set_repeat(0)
        self.pixcap['SEQ'].set_size(self.seq_size)

        self.pixcap['SEQ']['CLK_0'][0:self.seq_size - 1] = bitarray('0100')
        self.pixcap['SEQ']['CLK_1'][0:self.seq_size - 1] = bitarray('0100')
        self.pixcap['SEQ']['CLK_2'][0:self.seq_size - 1] = bitarray('0001')
        self.pixcap['SEQ']['CLK_3'][0:self.seq_size - 1] = bitarray('0001')

        self.pixcap['SEQ'].write()
        self.pixcap['SEQ'].start()
        # self.pixcap.frequency_settling = 1

        self.pixcap.vm3_on()
        self.pixcap.vm2_on()
        # self.pixcap.vm1_on()
        # self.pixcap['SMU3'].on()
        # self.pixcap['SMU2'].on()
        # self.pixcap['SMU1'].on()

        # Wherefore is this long sleep statement?
        # time.sleep(15)

        # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
        logging.debug('Waiting for settling of SMU...')
        for _ in range(0, 30):
            current = self.get_source_current()
            logging.debug('Current: {}'.format(current))
            time.sleep(1)

        # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
        for _ in range(0, 20):
            c3 = self.pixcap.vm3_measure_current()
            c2 = self.pixcap.vm2_measure_current()
            # c1 = self.pixcap.vm1_measure_current()
            # c3 = self.pixcap['SMU3'].get_reading()
            # c2 = self.pixcap['SMU2'].get_reading()
            logging.debug('C3: {}'.format(c3))
            logging.debug('C2: {}'.format(c2))
            # logging.debug('C1: {}'.format(c1))
            # print('c3:', c3)
            # print('c2:', c2)
            time.sleep(1)

    def scan(self, data_group_spec=None):
        # some further setup to be done right before the measurement
        from utils_2 import walk_to_node
        if data_group_spec is not None and isinstance(data_group_spec, str):
            data_group = walk_to_node(self.base_group, data_group_spec, create=True)
        else:
            data_group = self.base_group

        if isinstance(data_group_spec, tb.Node):
            data_group = data_group_spec
        else:
            data_group, verify_creation = walk_to_node(data_group, "inter_cap/measurements", create=True,
                                                       verify_create=True)
            if not verify_creation:
                for key, value in data_group._v_children.items():
                    new_name = key
                    while new_name in data_group:
                        new_name = "{old}_backing".format(old=new_name)
                    value.rename(new_name)

        data_group._f_setattr('frequencies', self.n_frequencies)

        if 'bias' in self.scan_config:
            self.pixcap.bias_voltage = -0.1
            self.pixcap.bias_on()
            self.pixcap.bias_voltage = float(self.scan_config["bias"])
            # self.pixcap.bias_on()
            # time.sleep(10)
            data_group._f_setattr("bias_voltage", self.pixcap.get_smu_source_voltage(self.pixcap.bias_smu_key))
            for i in range(40):
                print(self.pixcap.get_smu_source_voltage(self.pixcap.bias_smu_key))
                print(i, self.pixcap.bias_measure_volts(), self.pixcap.bias_measure_current())
            # while not np.isclose(self.pixcap.bias_measure_volts(), self.pixcap.bias_voltage):
            #     logger.info("Wait for settling of the bias supply.")
            #     time.sleep(1)

        try:
            with logging_redirect_tqdm():
                for i_row in tqdm(self.row_range, desc="Scanning column"):
                    for i_col in tqdm(self.col_range, desc="Scanning column", leave=False):
                        logging.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
                        logger.info(MEASURING_PIXEL_TEXT % (i_col, i_row))


                        # current_array1 = []
                        # current_array2 = []
                        self.pixcap.disable_all_pixels()
                        self.pixcap.disable_all_columns()

                        # enable columns of pixel under test and surrounding pixels
                        # self.pixcap.enable_column(i_col, c.EN_EOC_2 | c.EN_EOC_1 | c.EN_EOC_3)
                        # self.pixcap.enable_column(i_col + 1 | i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)
                        self.pixcap.enable_column(i_col, c.EN_EOC_2 | c.EN_EOC_3)
                        self.pixcap.enable_column(i_col + 1 | i_col - 1, c.EN_EOC_3)
                        # Should the order have an inpact?
                        # self.pixcap.enable_column(i_col + 1, c.EN_EOC_1 | c.EN_EOC_3)
                        # self.pixcap.enable_column(i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)

                        # time.sleep(1)

                        # enable pixel under test
                        self.pixcap.enable_pixel_clk(i_col, i_row, c.EN_CLK_2 | c.EN_CLK_0)

                        # enable pixels surrounding pixel under test
                        # Should see whether the active clock 1 has any impact on the measurement.
                        # self.pixcap.enable_pixel_clk(i_col, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
                        # self.pixcap.enable_pixel_clk(i_col + 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
                        # self.pixcap.enable_pixel_clk(i_col + 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
                        # self.pixcap.enable_pixel_clk(i_col + 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
                        # self.pixcap.enable_pixel_clk(i_col, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
                        # self.pixcap.enable_pixel_clk(i_col - 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
                        # self.pixcap.enable_pixel_clk(i_col - 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
                        # self.pixcap.enable_pixel_clk(i_col - 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col, i_row + 1, c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col + 1, i_row + 1, c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col + 1, i_row, c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col + 1, i_row - 1, c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col, i_row - 1, c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col - 1, i_row - 1, c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col - 1, i_row, c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col - 1, i_row + 1, c.EN_CLK_3)

                        # Which SMU takes which role here?
                        for k, freq in enumerate(self.freq_sweep_array):
                            logging.info("Set the frequency to %f MHz for the measurement.", freq)
                            self.pixcap.cvm_frequency = freq
                            # temp = freq * self.seq_size
                            # self.pixcap['MIO_PLL'].setFrequency(temp)

                            # TODO: Refactor the reading process of the SMU!
                            # Why read the value two times?
                            # result1 = self.pixcap['SMU3'].get_reading()
                            self.inter_hist_current_1[i_col, i_row, k] = self.pixcap.vm3_measure_current()
                            # self.inter_hist_current_1[i_col, i_row, k] = self.pixcap['SMU3'].get_reading().split(',')[1]
                            # current_array1.append(float(result1.split(',')[1]))

                            # result2 = self.pixcap['SMU2'].get_reading()
                            # perhaps the wrong capacitance!
                            # self.inter_hist_current_2[i_col, i_row, k] = self.pixcap.vm_2_measure_current()
                            # self.inter_hist_current_2[i_col, i_row, k] = self.pixcap['SMU2'].get_reading().split(',')[1]
                            # current_array2.append(float(result2.split(',')[1]))
                            self.total_hist_current[i_col, i_row, k] = self.pixcap.vm2_measure_current()

                            # extract the inter pix current
                            # self.inter_hist_current_2[i_col, i_row, k] = self.pixcap.vm_3_measure_current()
                            if 'bias' in self.scan_config:
                                store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq, bias_voltage=
                                self.pixcap["BIAS_SUPPLY"].get_source_voltage())
                            else:
                                store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq)
        finally:
            # make sure that every possible measurement taken is also saved
            try:
                # make sure the measurement points will have uncertainties.
                if np.all(np.isfinite(self.inter_hist_current_1)):
                    try:
                        self.inter_hist_current_1_error = extract_smu_current_error(
                            self.smu_range_config[self.pixcap.vm3_smu_key], self.inter_hist_current_1,
                                                                             self.current_sense_range)
                    except Exception as e:
                        logging.error(e.args)
                        logging.exception(UNCERT_ESTIMATION_ERROR_MSG)
                if np.all(np.isfinite(self.inter_hist_current_2)):
                    try:
                        self.inter_hist_current_2_error = extract_smu_current_error(
                            self.smu_range_config[self.pixcap.vm1_smu_key], self.inter_hist_current_2,
                            self.current_sense_range)
                    except Exception as e:
                        logging.error(e.args)
                        logging.exception(UNCERT_ESTIMATION_ERROR_MSG)
                if np.all(np.isfinite(self.total_hist_current)):
                    try:
                        self.total_hist_current_error = extract_smu_current_error(
                            self.smu_range_config[self.pixcap.vm2_smu_key], self.total_hist_current,
                            self.current_sense_range)
                    except Exception as e:
                        logging.error(e.args)
                        logging.exception(UNCERT_ESTIMATION_ERROR_MSG)



                data_group._f_setattr("freq_unit", "MHz")
                data_group._f_setattr("current_unit", "A")
                assert isinstance(data_group, tb.Group)
                _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.scan_parameters, group=data_group)
                temp_array = self.create_carray(data_group,
                                               name='TotalHistCurr',
                                               title='Current Histogram for the total capacitance measurement',
                                               obj=self.total_hist_current,
                                               filters=self.filters)
                temp_array.attrs["Input"] = "VM2"
                temp_array.attrs["Units"] = "A"
                temp_array.flush()
                temp_array = self.create_carray(data_group,
                                                name='TotalHistCurrErr',
                                                title='Error Histogram of the current for the total capacitance measurement',
                                                obj=self.total_hist_current_error,
                                                filters=self.filters)
                temp_array.attrs["Input"] = "VM2"
                temp_array.attrs["Units"] = "A"
                temp_array.flush()
                temp_array = self.create_carray(data_group,
                                               name='InterHistCurrA',
                                               title='Current Histogram for the inter capacitance measurement',
                                               obj=self.inter_hist_current_1,
                                               filters=self.filters)
                temp_array.attrs["Input"] = "VM3"
                temp_array.attrs["Units"] = "A"
                temp_array.flush()
                temp_array = self.create_carray(data_group,
                                                name='InterHistCurrErrA',
                                                title='Error Histogram of inter current A for the inter capacitance measurement',
                                                obj=self.inter_hist_current_1_error,
                                                filters=self.filters)
                temp_array.attrs["Input"] = "VM3"
                temp_array.attrs["Units"] = "A"
                temp_array.flush()
                temp_array = self.create_carray(data_group,
                                               name='InterHistCurrB',
                                               title='Current Histogram for the inter capacitance measurement',
                                               obj=self.inter_hist_current_2,
                                               filters=self.filters)
                temp_array.attrs["Input"] = "NOT IN USED"
                temp_array.attrs["Units"] = "A"
                temp_array.flush()
                temp_array = self.create_carray(data_group,
                                                name='InterHistCurrErrB',
                                                title='Error Histogram of inter current B for the inter capacitance measurement',
                                                obj=self.inter_hist_current_2_error,
                                                filters=self.filters)
                temp_array.attrs["Input"] = "NOT USED"
                temp_array.attrs["Units"] = "A"
                temp_array.flush()

                for config_key, config_setting in self.scan_config.items():
                    if isinstance(config_setting, Iterable) or isinstance(config_setting, Mapping):
                        continue
                    logger.warning("The present keys for config attributes are: %s", str(data_group._v_attrs))
                    attr_config_key = "configuration_{}".format(config_key)
                    if attr_config_key in data_group._v_attrs and data_group._v_attrs[attr_config_key] != config_setting:
                        logger.error("Unexpectedly the configuration key is already present.")
                        temp_key = attr_config_key
                        while temp_key in data_group._v_attrs:
                            temp_key = "{}_backing".format(temp_key)
                        data_group._f_setattr(temp_key, data_group._f_getattr(attr_config_key))
                    try:
                        data_group._f_setattr(attr_config_key, config_setting)
                    except:
                        logger.error("Failed to write scan configuration option %s as attribute.", config_key,
                                     exc_info=True)
            finally:
                self.out_file_h5.flush()
                # do some cleanup for the performance
                gc.collect()

    @property
    def current_array1(self):
        return self.measurement_group.InterHistCurrA

    @property
    def current_array2(self):
        return self.measurement_group.InterHistCurrB

    @property
    def current_array3(self):
        return self.measurement_group.TotalHistCurr

    @property
    def freq_sweep_array_plot(self):
        return np.arange(0, 5.1, 1)

    def analyze(self):
        result_table = self.out_file_h5.create_table(self.analysis_group, name='result_table', description=InterCap,
                                                     title='result_table')
        inter_capacitance_a = self.out_file_h5.create_carray(self.analysis_group, name='InterHistCapA',
                                                             atom=tb.Float64Atom(), shape=(40, 40),
                                                             title='Inter Capacitance A', filters=self.filters)
        inter_capacitance_b = self.out_file_h5.create_carray(self.analysis_group, name='InterHistCapB',
                                                             atom=tb.Float64Atom(), shape=(40, 40),
                                                             title='Inter Capacitance B', filters=self.filters)
        total_capacitance = self.out_file_h5.create_carray(self.analysis_group, name='TotalHistCap',
                                                           atom=tb.Float64Atom(), shape=(40, 40),
                                                           title='Total Pixel Capacitance Histogram',
                                                           filters=self.filters)
        inter_leakage_a = self.out_file_h5.create_carray(self.analysis_group, name='InterHistLeakA',
                                                         atom=tb.Float64Atom(), shape=(40, 40),
                                                         title='Inter Leakage Current Histogram A',
                                                         filters=self.filters)
        inter_leakage_b = self.out_file_h5.create_carray(self.analysis_group, name='InterLeakCapB',
                                                         atom=tb.Float64Atom(), shape=(40, 40),
                                                         title='Inter Leakage Current Histogram B',
                                                         filters=self.filters)
        total_leakage = self.out_file_h5.create_carray(self.analysis_group, name='TotalHistLeak', atom=tb.Float64Atom(),
                                                       shape=(40, 40), title='Total Leakage Current Histogram',
                                                       filters=self.filters)
        for i_row in self.row_range:
            for i_col in self.col_range:
                # I'm not quite sure whether this association of the SMUs to the different capacitance's ist correct.
                # apply linear fit to measured current values; also returns covariance matrix
                matrix1 = np.polyfit(self.freq_sweep_array, self.current_array1[i_col, i_row, :], 1, cov=True)
                matrix2 = np.polyfit(self.freq_sweep_array, self.current_array2[i_col, i_row, :], 1, cov=True)
                matrix3 = np.polyfit(self.freq_sweep_array, self.current_array3[i_col, i_row, :], 1, cov=True)

                a, b = matrix1[0][0], matrix1[0][1]  # fit parameters of reference pixel
                inter_capacitance_a[i_col, i_row] = a
                inter_leakage_a[i_col, i_row] = b

                e, d = matrix2[0][0], matrix2[0][1]  # fit parameters of neighbouring pixels
                inter_capacitance_b[i_col, i_row] = e
                inter_leakage_b[i_col, i_row] = d

                f, g = matrix3[0][0], matrix3[0][1]
                total_capacitance[i_col, i_row] = f
                total_leakage[i_col, i_row] = g

                logging.debug('{} {} {} {}'.format(i_col, i_row, a, e))
                result_table.append([{i_col, i_row, a, e, f, b, d, g}])

    def plot(self):
        # TODO: prepare some general statistical analysis of the data here!
        # TODO: additionally combine all of these plots into one figure or pdf document.
        for i_row in self.row_range:
            for i_col in self.col_range:
                a = self.analysis_group.InterHistCapA[i_col, i_row]
                b = self.analysis_group.InterHistLeakA[i_col, i_row]
                d = self.analysis_group.InterHistCapB[i_col, i_row]
                e = self.analysis_group.InterLeakCapB[i_col, i_row]
                f = self.analysis_group.TotalHistCap[i_col, i_row]
                g = self.analysis_group.TotalHistLeak[i_col, i_row]

                fit_fn = a * self.freq_sweep_array_plot + b
                pl.plot(self.freq_sweep_array, self.current_array1, 'o',
                        label='COL({i_col})PIX({i_row}), I3'.format(i_col=i_col, i_row=i_row))
                pl.plot(self.freq_sweep_array_plot, fit_fn, label='a={a:.3E}, b={b:.3E}'.format(a=a, b=b))

                fit_fn = e * self.freq_sweep_array_plot + d
                pl.plot(self.freq_sweep_array, self.current_array2, 'o',
                        label='COL({i_col})PIX({i_row}), I2'.format(i_col=i_col, i_row=i_row))
                pl.plot(self.freq_sweep_array_plot, fit_fn, label='c={a:.3E}, d={b:.3E}'.format(a=e, b=d))

                fit_fn = f * self.freq_sweep_array_plot + g
                pl.plot(self.freq_sweep_array, self.current_array3, 'o',
                        label='COL({i_col})PIX({i_row}), I2'.format(i_col=i_col, i_row=i_row))
                pl.plot(self.freq_sweep_array_plot, fit_fn, label='c={a:.3E}, d={b:.3E}'.format(a=e, b=d))

    def close(self):
        # time.sleep(300)

        # self.pixcap.vm1_off()
        self.pixcap.vm2_off()
        self.pixcap.vm3_off()
        # self.pixcap['SMU2'].off()
        # self.pixcap['SMU3'].off()
        super(Pixcap65InterCap, self).close()

        # pl.legend(loc='best')
        # pl.xlabel('Freq [MHz]')
        # pl.ylabel('I [A]')
        # pl.show()

    @property
    def pixcap(self):
        return self.dut

    # route through sensor matrix without edges
    @property
    def row_start(self):
        return self.scan_config[ScanConfigurationKeys.START_ROW]

    @property
    def row_stop(self):
        return self.scan_config[ScanConfigurationKeys.STOP_ROW]

    @property
    def col_start(self):
        return self.scan_config[ScanConfigurationKeys.START_COLUMN]

    @property
    def col_stop(self):
        return self.scan_config[ScanConfigurationKeys.STOP_COLUMN]

    # specialized for the inter capacitance measurement.
    @property
    def col_range(self):
        return range(self.col_start, self.col_stop + 1)

    @property
    def row_range(self):
        return range(self.row_stop, self.row_start - 1, -1)

    @property
    def freq_sweep_array(self):
        return np.asarray(self.scan_config[ScanConfigurationKeys.FREQUENCY_RANGE])


# Commented out this strange top level code instead of good scripting practice!

# dut = Pixcap65("pixcap65.yaml")
# dut.init()
#
# seq_size = 4  # granularity of the clock sequencer
#
# dut['SMU3'].off()
# dut['SMU3'].source_volt()
# dut['SMU3'].set_voltage_range(1.5)
# dut['SMU3'].set_current_nlpc(10)
# dut['SMU3'].set_voltage(1.0)
# dut['SMU3'].set_current_limit(0.001)
# dut['SMU3'].set_current_sense_range(0.00001)
#
# dut['SMU2'].off()
# dut['SMU2'].source_volt()
# dut['SMU2'].set_voltage_range(1.5)
# dut['SMU2'].set_current_nlpc(10)
# dut['SMU2'].set_voltage(1.0)
# dut['SMU2'].set_current_limit(0.001)
# dut['SMU2'].set_current_sense_range(0.00001)
#
# # settings for sensor depletion source
# # dut['SMU1'].off()
# # dut['SMU1'].source_volt()
# # dut['SMU1'].set_voltage_range(1.5)
# # dut['SMU1'].set_current_nlpc(10)
# # dut['SMU1'].set_voltage(-80.0)
# # dut['SMU1'].set_current_limit(0.001)
# # dut['SMU1'].set_current_sense_range(0.00001)
#
# dut['SEQ'].reset()
# dut['SEQ'].set_clk_divide(1)
# dut['SEQ'].set_repeat_start(0)
# dut['SEQ'].set_repeat(0)
# dut['SEQ'].set_size(seq_size)
#
# dut['SEQ']['CLK_0'][0:seq_size - 1] = bitarray('0100')
# dut['SEQ']['CLK_1'][0:seq_size - 1] = bitarray('0100')
# dut['SEQ']['CLK_2'][0:seq_size - 1] = bitarray('0001')
# dut['SEQ']['CLK_3'][0:seq_size - 1] = bitarray('0001')
#
# dut['SEQ'].write()
# dut['SEQ'].start()
#
# dut['SMU3'].on()
# dut['SMU2'].on()
# # dut['SMU1'].on()
#
# time.sleep(15)
#
# # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
# for i in range(0, 20):
#     c3 = dut['SMU3'].get_reading()
#     c2 = dut['SMU2'].get_reading()
#     print('c3:', c3)
#     print('c2:', c2)
#     time.sleep(1)
#
# # route through sensor matrix without edges
# row_start = 2
# row_stop = 39
# col_start = 1
# col_stop = 38
#
# row_range = range(row_stop, row_start - 1, -1)
# col_range = range(col_start, col_stop + 1)
#
# freq_sweep_array = np.arange(1, 4.1, 1)  # .astype(np.float) # [MHz]
# table_first_row = ["row\col"]
# table_first_row.extend(col_range)
# table_row = []
# table_storage = []  # some additional list to store results during measurement
#
# data_file = open("./pixcap_full_data_image1.txt", "w")
# # data_file.write(','.join(map(str,table_first_row))+'\n')
#
# for i_row in row_range:
#     table_row = []
#     for i_col in col_range:
#         current_array1 = []
#         current_array2 = []
#         fit_params = []
#         table_storage = []
#         dut.disable_all_pixels()
#         dut.disable_all_columns()
#
#         # enable columns of pixel under test and surrounding pixels
#         dut.enable_column(i_col, c.EN_EOC_2 | c.EN_EOC_1 | c.EN_EOC_3)
#         dut.enable_column(i_col + 1 | i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)
#         # dut.enable_column(i_col + 1, c.EN_EOC_1 | c.EN_EOC_3)
#         # dut.enable_column(i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)
#
#         time.sleep(1)
#
#         # enable pixel under test
#         dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_2 | c.EN_CLK_0)
#
#         # enable pixels surrounding pixel under test
#         dut.enable_pixel_clk(i_col, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
#         dut.enable_pixel_clk(i_col + 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
#         dut.enable_pixel_clk(i_col + 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
#         dut.enable_pixel_clk(i_col + 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
#         dut.enable_pixel_clk(i_col, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
#         dut.enable_pixel_clk(i_col - 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
#         dut.enable_pixel_clk(i_col - 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
#         dut.enable_pixel_clk(i_col - 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
#
#         for freq in freq_sweep_array:
#             temp = freq * seq_size
#             dut['MIO_PLL'].setFrequency(temp)
#
#             result1 = dut['SMU3'].get_reading()
#             current_array1.append(float(result1.split(',')[1]))
#
#             result2 = dut['SMU2'].get_reading()
#             current_array2.append(float(result2.split(',')[1]))
#
#         # apply linear fit to measured current values; also returns covariance matrix
#         matrix1 = np.polyfit(freq_sweep_array, current_array1, 1, cov=True)
#         matrix2 = np.polyfit(freq_sweep_array, current_array2, 1, cov=True)
#
#         a, b = matrix1[0][0], matrix1[0][1]  # fit parameters of reference pixel
#         fit_params.append(a)
#         fit_params.append(b)
#
#         e, d = matrix2[0][0], matrix2[0][1]  # fit parameters of neighbouring pixels
#         fit_params.append(e)
#         fit_params.append(d)
#
#         print(i_col, i_row, a, e)
#
#         # data structure in txt file: "slope, offset (y-intercept), slope, offset (y-intercept)"
#         table_storage.append(fit_params)
#         table_row.append(table_storage)
#
#         freq_sweep_array_plot = np.arange(0, 5.1, 1)
#
#         fit_fn = a * freq_sweep_array_plot + b
#         pl.plot(freq_sweep_array, current_array1, 'o',
#                 label='COL({i_col})PIX({i_row}), I3'.format(i_col=i_col, i_row=i_row))
#         pl.plot(freq_sweep_array_plot, fit_fn, label='a={a:.3E}, b={b:.3E}'.format(a=a, b=b))
#
#         fit_fn = e * freq_sweep_array_plot + d
#         pl.plot(freq_sweep_array, current_array2, 'o',
#                 label='COL({i_col})PIX({i_row}), I2'.format(i_col=i_col, i_row=i_row))
#         pl.plot(freq_sweep_array_plot, fit_fn, label='c={a:.3E}, d={b:.3E}'.format(a=e, b=d))
#
#     # print(','.join(map(str, table_row)))
#     # data_file.write(','.join(map(str, table_row)) + '\n')
#     data_file.write('\n'.join(map(str, table_row)) + '\n')  # write data in new lines
#
# data_file.close()
#
# # time.sleep(300)
#
# # dut['SMU1'].off()
# dut['SMU2'].off()
# dut['SMU3'].off()
#
# pl.legend(loc='best')
# pl.xlabel('Freq [MHz]')
# pl.ylabel('I [A]')
# pl.show()
# dut.close()

if __name__ == "__main__":
    output_file = "./R13-Interpixel_Scan.h5"
    # with Pixcap65InterCap(scan_configuration, output_file) as pix:
        # pix.scan(data_group_spec="demo_measurement_1_80_V")
    from utils import PixCapSetup
    with PixCapSetup(scan_configuration, output_file, measurement=Pixcap65InterCap) as pix:
        pix.scan(data_group_spec="demo_measurement_5_80_V")


