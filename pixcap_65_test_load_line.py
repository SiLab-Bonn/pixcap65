"""
Script for measuring the load line of a test capacitor
Outputs in txt file the number of bits m that were set to 1 and the corresponding measured current 
Calculate t_charge with t_charge = m/seq_size * 1/f_rep
"""

import logging
import time

import numpy as np
from bitarray import bitarray

import pixcap65_constants as c
from pixcap_65_test_total_cap import PixCap65Measurement

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# noinspection SpellCheckingInspection
scan_configuration = {
    'start_column': 12,
    'stop_column': 12,
    'start_row': 0,
    'stop_row': 0,

    'Vin': 1.0,  # input voltage in V
    # 'frequency_range': np.arange(1, 4.1, 1)  # .astype(np.float) # [MHz]
    'frequency_range': [1.0]
}


class Pixcap65LoadLine(PixCap65Measurement):
    def __init__(self, scan_config, output_file):
        super(Pixcap65LoadLine).__init__(scan_config, output_file)

        # prepare the measurement fields
        self.hist_current = np.full(shape=(self.seq_size / 2 - 1, 40, 40, self.n_frequencies + 1), fill_value=np.nan)

    def configure(self):
        # TODO: Refactor the smu access methods for better readability!
        # TODO: Refactor this into a init method for the SMUs!
        self.dut['SMU3'].off()
        self.dut['SMU3'].source_volt()
        self.dut['SMU3'].set_voltage_range(1.5)
        self.dut['SMU3'].set_current_nlpc(10)
        self.dut['SMU3'].set_voltage(1.0)
        self.dut['SMU3'].set_current_limit(0.001)
        self.dut['SMU3'].set_current_sense_range(0.00001)

    def scan(self):
        m = self.seq_size / 2 - 1  # define index of the last 1 in order to create a non-overlapping clock sequence
        assert isinstance(m, int), "m has to be an integer!"
        # I do not see the point of a second variable of the same value?
        cnt = self.seq_size / 2 - 1  # counter for numbering in output file

        # create initial bit arrays for a given sequencer size
        bit_array_clk_3 = bitarray(self.seq_size)
        bit_array_clk_3.setall(0)
        bit_array_clk_3[1:m] = 1

        bit_array_clk_0 = bitarray(self.seq_size)
        bit_array_clk_0.setall(0)
        bit_array_clk_0[m + 1:-1] = 1

        # vary charging time by looping over the number of bits in bit_array_clk_3 that are set to 1;
        # number is reduced by one in every step
        for i in range(m, 0, -1):
            self.pixcap['SEQ'].reset()
            self.pixcap['SEQ'].set_clk_divide(1)
            self.pixcap['SEQ'].set_repeat_start(0)
            self.pixcap['SEQ'].set_repeat(0)
            self.pixcap['SEQ'].set_size(self.seq_size)
            self.pixcap['SEQ']['CLK_0'][0:self.seq_size - 1] = bit_array_clk_0
            self.pixcap['SEQ']['CLK_3'][0:self.seq_size - 1] = bit_array_clk_3
            self.pixcap['SEQ'].write()
            self.pixcap['SEQ'].start()

            # maybe this step does not have to be within the loop over m
            # did not know if SMU has to be set on after changing the clock sequencer
            self.pixcap['SMU3'].on()
            self.pixcap['SMU3'].get_reading()

            if 'invert_slicing' in self.scan_config and self.scan_config['invert_slicing']:
                col_range = self.row_range
                row_range = self.col_range
            else:
                col_range = self.col_range
                row_range = self.row_range

            for i_row in row_range:
                for i_col in col_range:
                    if 'invert_slicing' in self.scan_config and self.scan_config['invert_slicing']:
                        temp_row, temp_col = i_row, i_col
                        i_row, i_col = temp_col, temp_row
                    self.pixcap.disable_all_pixels()
                    self.pixcap.disable_all_columns()
                    self.pixcap.enable_column(i_col, c.EN_EOC_3)
                    time.sleep(1)
                    self.pixcap.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)

                    for k, freq in enumerate(self.freq_sweep_array):
                        self.pixcap.seq_size = self.seq_size
                        self.pixcap.set_frequency(freq)
                        result = self.pixcap['SMU3'].get_reading()
                        self.hist_current[cnt - 1, i_col, i_row, 0] = cnt
                        self.hist_current[cnt - 1, i_col, i_row, k + 1] = float(result.split(',')[1])

                logger.info(bit_array_clk_3)
                # logger.info(self.hist_current[cnt-1, i_col, i_row, :])
                logger.info(self.hist_current[cnt - 1, col_range[-1], i_row, :])

            # reduce charging time with every iteration by setting last bit 1 -> 0 and decrement counter
            bit_array_clk_3[i] = 0
            cnt = cnt - 1
        self.out_file_h5.create_carray(where=self.out_file_h5.root, name="HistCurr", title="Current Histogram",
                                       obj=self.hist_current, filters=self.filters)
        self.out_file_h5.flush()

    def close(self):
        self.dut['SMU3'].off()
        super(Pixcap65LoadLine, self).close()
        logger.debug("Done and closed the pixcap system.")

    def analyze(self):
        logger.info("There is nothing to analyze for the load line test.")

    def plot(self):
        logger.info("There is nothing to plot for the load line test.")

    # properties of the measurement class
    @property
    def seq_size(self):
        return 128  # granularity of the clock sequencer

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
    def row_range(self):
        return range(self.row_stop, self.row_start - 1, -1)

    @property
    def col_range(self):
        return range(self.col_start, self.col_stop + 1)

    @property
    def freq_sweep_array(self):
        return np.asarray(self.scan_config['frequency_range'], dtype=np.float64)


# # schedule this strange top-level script for removal! (for now it is uncommented)
# dut = Pixcap65("pixcap65.yaml")
# dut.init()
#
# seq_size  = 128 # granularity of the clock sequencer
#
# dut['SMU3'].off()
# dut['SMU3'].source_volt()
# dut['SMU3'].set_voltage_range(1.5)
# dut['SMU3'].set_current_nlpc(10)
# dut['SMU3'].set_voltage(1.0)
# dut['SMU3'].set_current_limit(0.001)
# dut['SMU3'].set_current_sense_range(0.00001)
#
# m = seq_size/2 - 1 # define index of the last 1 in order to create a non-overlapping clock sequence
# cnt = seq_size/2 - 1 # counter for numbering in output file
# table_row = []
#
# # create initial bit arrays for a given sequencer size
# bit_array_CLK_3 = bitarray(seq_size)
# bit_array_CLK_3.setall(0)
# bit_array_CLK_3[1:m] = 1
#
# bit_array_CLK_0 = bitarray(seq_size)
# bit_array_CLK_0.setall(0)
# bit_array_CLK_0[m+1:-1] = 1
#
# # vary charging time by looping over the number of bits in bit_array_CLK_3 that are set to 1; number is reduced by one in every step
# for i in range(m , 0 , -1):
#
#     dut['SEQ'].reset()
#     dut['SEQ'].set_clk_divide(1)
#     dut['SEQ'].set_repeat_start(0)
#     dut['SEQ'].set_repeat(0)
#     dut['SEQ'].set_size(seq_size)
#     dut['SEQ']['CLK_0'][0:seq_size-1] =  bit_array_CLK_0
#     dut['SEQ']['CLK_3'][0:seq_size-1] =  bit_array_CLK_3
#     dut['SEQ'].write()
#     dut['SEQ'].start()
#
#     # maybe this step does not have to be within the loop over m
#     # did not know if SMU has to be set on after changing the clock sequencer
#     dut['SMU3'].on()
#     dut['SMU3'].get_reading()
#
#     row_start =  0
#     row_stop  =  0
#     col_start =  12
#     col_stop  =  12
#
#     row_range = range(row_stop, row_start-1,-1)
#     col_range = range(col_start, col_stop+1)
#
#     # freq_sweep_array = np.arange(1, 4.1, 1)#.astype(np.float) # [MHz]
#     freq_sweep_array = [1.0] # can also uncomment loop over frequencies
#     # Why defining this never used fields?
#     table_first_row = ["row\col"]
#     table_first_row.extend(col_range)
#
#     data_file = open("./pixcap_full_data_image1.txt", "w")
#
#     # how to distinguish the different rows and columns in the table?
#     for i_row in row_range:
#         for i_col in col_range:
#             current_array = []
#             dut.disable_all_pixels()
#             dut.disable_all_columns()
#             dut.enable_column(i_col, c.EN_EOC_3)
#             time.sleep(1)
#             dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)
#
#             for freq in freq_sweep_array:
#                 temp = freq * seq_size
#                 dut['MIO_PLL'].setFrequency(temp)
#                 time.sleep(1)
#                 result = dut['SMU3'].get_reading()
#                 current_array.append(cnt)
#                 current_array.append(float(result.split(',')[1]))
#
#             table_row.append(current_array)
#
#         print(bit_array_CLK_3)
#         # this print statement seems to be senseless as it could only show the currents for the last column processed
#         print(current_array)
#
#     # reduce charging time with every iteration by setting last bit 1 -> 0 and decrement counter
#     bit_array_CLK_3[i] = 0
#     cnt = cnt - 1
# data_file.write('\n'.join(map(str, table_row)) + '\n')
#
# data_file.close()
# dut['SMU3'].off()
# dut.close()

if __name__ == "__main__":
    output_file = "./pixcap_full_data_image1.h5"
    with Pixcap65LoadLine(scan_configuration, output_file) as pix:
        pix.scan()
        pix.analyze()
        pix.plot()
