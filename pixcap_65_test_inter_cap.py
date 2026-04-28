"""
Script for measuring Inter Pixel Capacitance 
"""

import logging

import numpy as np
import tables as tb
import time
from tqdm import tqdm

from analysis_util.utility import HIST_CURRENT_MEAS_UNIT
from pixcap_65_test_total_cap import PixCap65Measurement, MEASURING_PIXEL_TEXT, \
    _store_scan_par_values
from utility import pixcap65_constants as c
from utility.tqdm_logging_utils import logging_redirect_tqdm

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

scan_configuration = {
    'start_column': 15,
    'stop_column': 35,
    'start_row': 15,
    'stop_row': 35,

    'Vin': 1.0,  # input voltage in V
    'frequency_range': np.arange(1, 8.1, 1),  # .astype(np.float) # [MHz]
    # 'bias': -80,

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

        # current value for each measured frequency per pixel
        self.inter_hist_current_1_error = np.full(shape=(40, 40, self.n_frequencies),
                                                    fill_value=np.nan)
        # current value for each measured frequency per pixel
        self.inter_hist_current_2_error = np.full(shape=(40, 40, self.n_frequencies),
                                                    fill_value=np.nan)
        # current value for each measured frequency per pixel
        self.total_hist_current_error = np.full(shape=(40, 40, self.n_frequencies),
                                                fill_value=np.nan)

        # this kind of setup is somewhat misplaced.
        self.pixcap.seq_size = 4

    def configure(self):
        # already done by super-class
        super(Pixcap65InterCap, self).configure()
        self.init_smu(smu=self.pixcap.vm2_smu_key)
        self.init_smu(smu=self.pixcap.vm1_smu_key)

        self.pixcap.seq_init(clk_0='0100', clk_1='0100', clk_2='0001', clk_3='0001')

        self.pixcap.vm3_on()
        self.pixcap.vm2_on()
        self.pixcap.vm1_on()

        # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
        logging.debug('Waiting for settling of SMU...')
        for _ in range(0, 20):
            c3 = self.pixcap.vm3_measure_current()
            c2 = self.pixcap.vm2_measure_current()
            c1 = self.pixcap.vm1_measure_current()
            logging.debug('C3: {}'.format(c3))
            logging.debug('C2: {}'.format(c2))
            logging.debug('C1: {}'.format(c1))
            time.sleep(1)

    def scan(self, data_group_spec=None, sequence_call: bool = False):
        """
        scan
        Performs the scan over the pixels on the sensor and measures the requested quantities in dependence on some
        other quantities.
        Will scan the specified frequency range for each pixel specified by the scan configuration and measure the
        current at all three SMU channels to obtain information about the pixel capacitance and the inter-pixel
        capacitance.

        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        """
        # some further setup to be done right before the measurement
        data_group = self.get_data_group(data_group_spec, "inter_cap")
        data_group._f_setattr('frequencies', self.n_frequencies)

        self.set_bias_measurement(data_group, False)
        try:
            with logging_redirect_tqdm():
                for i_row in tqdm(self.row_range, desc="Grid row Loop"):
                    for i_col in tqdm(self.col_range, desc="Grid column Loop", leave=False):
                        logging.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
                        logger.info(MEASURING_PIXEL_TEXT % (i_col, i_row))

                        self.pixcap.disable_all_pixels()
                        self.pixcap.disable_all_columns()

                        # enable columns of pixel under test and surrounding pixels
                        self.pixcap.enable_column(i_col, c.EN_EOC_1 | c.EN_EOC_2 | c.EN_EOC_3)
                        self.pixcap.enable_column(i_col + 1, c.EN_EOC_1 | c.EN_EOC_3)
                        self.pixcap.enable_column(i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)

                        # enable pixel under test
                        self.pixcap.enable_pixel_clk(i_col, i_row, c.EN_CLK_2 | c.EN_CLK_0)

                        # enable pixels surrounding pixel under test
                        self.pixcap.enable_pixel_clk(i_col, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col + 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col + 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col + 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col - 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col - 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
                        self.pixcap.enable_pixel_clk(i_col - 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)

                        # Which SMU takes which role here?
                        for k, freq in enumerate(self.frequency_range):
                            self.pixcap.cvm_frequency = freq

                            self.inter_hist_current_1[i_col, i_row, k] = self.pixcap.vm3_measure_current()

                            # perhaps the wrong capacitance!
                            self.total_hist_current[i_col, i_row, k] = self.pixcap.vm2_measure_current()

                            # extract the inter pix current
                            self.inter_hist_current_2[i_col, i_row, k] = self.pixcap.vm1_measure_current()

                            self.store_iteration_parameters(freq, k)
        finally:
            self.post_scan_handler(data_group, sequence_call)
            logger.info("Done")

    def store_measurement_data(self, data_group: tb.Group, sequence_call: bool, unit=None):
        assert isinstance(data_group, tb.Group)
        _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.scan_parameters, group=data_group)
        self.create_carray(data_group, name='TotalHistCurr', title='Current Histogram for the total capacitance measurement', obj=self.total_hist_current, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT, input="VM2")
        self.create_carray(data_group, name='TotalHistCurrErr', title='Error Histogram of the current for the total capacitance measurement', obj=self.total_hist_current_error, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT, input="VM2")
        self.create_carray(data_group, name='InterHistCurrA', title='Current Histogram for the inter capacitance measurement', obj=self.inter_hist_current_1, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT, input="VM3")
        self.create_carray(data_group, name='InterHistCurrErrA', title='Error Histogram of inter current A for the inter capacitance measurement', obj=self.inter_hist_current_1_error, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT, input="VM3")
        self.create_carray(data_group, name='InterHistCurrB', title='Current Histogram for the inter capacitance measurement', obj=self.inter_hist_current_2, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT, input="VM1")
        self.create_carray(data_group, name='InterHistCurrErrB', title='Error Histogram of inter current B for the inter capacitance measurement', obj=self.inter_hist_current_2_error, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT, input="VM1")

    def handle_measurement_errors(self, unit = None):
        # make sure the measurement points will have uncertainties.
        self.inter_hist_current_1_error = self.determine_measurement_uncertainty(self.pixcap.vm3_smu_key,
                                                                                 self.inter_hist_current_1)
        self.inter_hist_current_2_error = self.determine_measurement_uncertainty(self.pixcap.vm1_smu_key,
                                                                                 self.inter_hist_current_2)
        self.total_hist_current_error = self.determine_measurement_uncertainty(self.pixcap.vm2_smu_key,
                                                                               self.total_hist_current)

    def analyze(self):
        pass

    def plot(self):
        pass

    def close(self):
        self.pixcap.vm1_off()
        self.pixcap.vm2_off()
        self.pixcap.vm3_off()
        super(Pixcap65InterCap, self).close()

    # region Pixcap Properties
    # specialized for the inter capacitance measurement.
    @property
    def col_range(self):
        """Get the range of columns to scan the pixels for. (specialised for inter-pix)"""
        return range(self.col_start, self.col_stop + 1)

    @property
    def row_range(self):
        """Get the range of rows to scan the pixels for. (specialised for inter-pix)"""
        return range(self.row_stop, self.row_start - 1, -1)
    # endregion


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
        pix.scan(data_group_spec="demo_measurement_18_unbiased_V")


