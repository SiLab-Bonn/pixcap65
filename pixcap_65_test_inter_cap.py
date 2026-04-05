"""
Script for measuring Inter Pixel Capacitance 
"""

import gc
import logging
import time

import numpy as np
import pylab as pl
import tables as tb
from bitarray import bitarray

import pixcap65_constants as c
from pixcap_65_test_total_cap import PixCap65Measurement
from pixcap_65_test_total_cap import store_scan_par_values

logging.getLogger().setLevel(logging.INFO)

scan_configuration = {
    'start_column': 1,
    'stop_column': 38,
    'start_row': 2,
    'stop_row': 39,

    'frequency_range': np.arange(1, 4.1, 1)  # .astype(np.float) # [MHz]
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
    def __init__(self, scan_config, output_file):
        super(Pixcap65InterCap).__init__(scan_config, output_file)

        # prepare the data fields for the measurement
        self.inter_hist_current_1 = np.full(shape=(40, 40, self.n_frequencies),
                                            fill_value=np.nan)  # current value for each measured frequency per pixel
        self.inter_hist_current_2 = np.full(shape=(40, 40, self.n_frequencies),
                                            fill_value=np.nan)  # current value for each measured frequency per pixel
        self.total_hist_current_2 = np.full(shape=(40, 40, self.n_frequencies),
                                            fill_value=np.nan)  # current value for each measured frequency per pixel

    def configure(self):
        self.pixcap['SMU3'].off()
        self.pixcap['SMU3'].source_volt()
        self.pixcap['SMU3'].set_voltage_range(1.5)
        self.pixcap['SMU3'].set_current_nlpc(10)
        self.pixcap['SMU3'].set_voltage(1.0)
        self.pixcap['SMU3'].set_current_limit(0.001)
        self.pixcap['SMU3'].set_current_sense_range(0.00001)

        self.pixcap['SMU2'].off()
        self.pixcap['SMU2'].source_volt()
        self.pixcap['SMU2'].set_voltage_range(1.5)
        self.pixcap['SMU2'].set_current_nlpc(10)
        self.pixcap['SMU2'].set_voltage(1.0)
        self.pixcap['SMU2'].set_current_limit(0.001)
        self.pixcap['SMU2'].set_current_sense_range(0.00001)

        # settings for sensor depletion source
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

        self.pixcap['SMU3'].on()
        self.pixcap['SMU2'].on()
        # self.pixcap['SMU1'].on()

        # Wherefore is this long sleep statement?
        time.sleep(15)

        # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
        logging.debug('Waiting for settling of SMU...')
        for _ in range(0, 30):
            current = self.get_source_current()
            logging.debug('Current: {}'.format(current))
            time.sleep(1)

        # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
        for _ in range(0, 20):
            # I'm not sure whether this will work at all?
            c3 = self.pixcap['SMU3'].get_reading()
            c2 = self.pixcap['SMU2'].get_reading()
            print('c3:', c3)
            print('c2:', c2)
            time.sleep(1)

    def scan(self):
        for i_row in self.row_range:
            for i_col in self.col_range:
                current_array1 = []
                current_array2 = []
                self.pixcap.disable_all_pixels()
                self.pixcap.disable_all_columns()

                # enable columns of pixel under test and surrounding pixels
                self.pixcap.enable_column(i_col, c.EN_EOC_2 | c.EN_EOC_1 | c.EN_EOC_3)
                self.pixcap.enable_column(i_col + 1 | i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)
                # self.pixcap.enable_column(i_col + 1, c.EN_EOC_1 | c.EN_EOC_3)
                # self.pixcap.enable_column(i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)

                time.sleep(1)

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

                for k, freq in enumerate(self.freq_sweep_array):
                    temp = freq * self.seq_size
                    self.pixcap['MIO_PLL'].setFrequency(temp)

                    # TODO: Refactor the reading process of the SMU!
                    result1 = self.pixcap['SMU3'].get_reading()
                    self.inter_hist_current_1[i_col, i_row, k] = self.pixcap['SMU3'].get_reading().split(',')[1]
                    current_array1.append(float(result1.split(',')[1]))

                    result2 = self.pixcap['SMU2'].get_reading()
                    self.inter_hist_current_2[i_col, i_row, k] = self.pixcap['SMU2'].get_reading().split(',')[1]
                    current_array2.append(float(result2.split(',')[1]))
                    store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq)

            self.out_file_h5.create_carray(self.measurement_group,
                                           name='TotalHistCurr',
                                           title='Current Histogram for the total capacitance measurement',
                                           obj=self.total_hist_current_2,
                                           filters=self.filters).flush()
            self.out_file_h5.create_carray(self.measurement_group,
                                           name='InterHistCurrA',
                                           title='Current Histogram for the inter capacitance measurement',
                                           obj=self.inter_hist_current_1,
                                           filters=self.filters).flush()
            self.out_file_h5.create_carray(self.measurement_group,
                                           name='InterHistCurrB',
                                           title='Current Histogram for the inter capacitance measurement',
                                           obj=self.inter_hist_current_2,
                                           filters=self.filters).flush()
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
        time.sleep(300)

        super(Pixcap65InterCap, self).close()

        self.pixcap['SMU2'].off()
        self.pixcap['SMU3'].off()

        pl.legend(loc='best')
        pl.xlabel('Freq [MHz]')
        pl.ylabel('I [A]')
        pl.show()

    @property
    def pixcap(self):
        return self.dut

    # route through sensor matrix without edges
    @property
    def row_start(self):
        return self.scan_config['start_row']

    @property
    def row_stop(self):
        return self.scan_config['stop_row']

    @property
    def col_start(self):
        return self.scan_config['start_column']

    @property
    def col_stop(self):
        return self.scan_config['stop_column']

    @property
    def col_range(self):
        return range(self.col_start, self.col_stop + 1)

    @property
    def row_range(self):
        return range(self.row_stop, self.row_start - 1, -1)

    @property
    def freq_sweep_array(self):
        return self.scan_config['frequency_range']


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
    print("Currently the main script for this kind of measurement is missing!")
    output_file = "./Test.h5"
    with Pixcap65InterCap(scan_configuration, output_file) as pix:
        pix.scan()
        pix.analyze()
        pix.plot()
