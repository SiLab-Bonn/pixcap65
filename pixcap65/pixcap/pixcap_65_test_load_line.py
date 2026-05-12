"""
Script for measuring the load line of a test capacitor
Outputs in txt file the number of bits m that were set to 1 and the corresponding measured current 
Calculate t_charge with t_charge = m/seq_size * 1/f_rep
"""

import logging

import numpy as np
import time
from bitarray import bitarray

from pixcap65.pixcap_65_test_total_cap import PixCap65Measurement
from pixcap65.utility import pixcap65_constants as c

LOAD_LINE_SEQ_SIZE = 128

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
    def handle_measurement_errors(self, unit):
        raise NotImplementedError("Pixcap65LoadLine.handle_measurement_errors")

    def store_measurement_data(self, data_group, sequence_call, unit=None):
        raise NotImplementedError("Pixcap65LoadLine.store_measurement_data")

    def __init__(self, scan_config, out_file):
        self.seq_size = LOAD_LINE_SEQ_SIZE
        super(Pixcap65LoadLine).__init__(scan_config, out_file)

        # prepare the measurement fields
        self.hist_current = np.full(shape=(int(self.seq_size / 2 - 1), 40, 40, self.n_frequencies + 1),
                                    fill_value=np.nan)

    def scan(self, data_group_spec=None, sequence_call=False):
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
            self.pixcap.seq_init(clk_0=bit_array_clk_0, clk_3=bit_array_clk_3)
            # self.pixcap['SEQ'].reset()
            # self.pixcap['SEQ'].set_clk_divide(1)
            # self.pixcap['SEQ'].set_repeat_start(0)
            # self.pixcap['SEQ'].set_repeat(0)
            # self.pixcap['SEQ'].set_size(self.seq_size)
            # self.pixcap['SEQ']['CLK_0'][0:self.seq_size - 1] = bit_array_clk_0
            # self.pixcap['SEQ']['CLK_3'][0:self.seq_size - 1] = bit_array_clk_3
            # self.pixcap['SEQ'].write()
            # self.pixcap['SEQ'].start()

            # maybe this step does not have to be within the loop over m
            # did not know if SMU has to be set on after changing the clock sequencer
            self.smu_on()
            self.get_source_current()

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
                        self.pixcap.cvm_frequency = freq
                        result = self.get_source_current()
                        self.hist_current[cnt - 1, i_col, i_row, 0] = cnt
                        if isinstance(result, float):
                            self.hist_current[cnt - 1, i_col, i_row, k + 1] = result
                        elif isinstance(result, str):
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
        self.smu_off()
        super(Pixcap65LoadLine, self).close()
        logger.debug("Done and closed the pixcap system.")

    def analyze(self):
        logger.info("There is nothing to analyze for the load line test.")

    def plot(self):
        logger.info("There is nothing to plot for the load line test.")

    # properties of the measurement class
    @property
    def row_range(self):
        return range(self.row_stop, self.row_start - 1, -1)

    @property
    def col_range(self):
        return range(self.col_start, self.col_stop + 1)

    @property
    def freq_sweep_array(self):
        return np.asarray(self.scan_config['frequency_range'], dtype=np.float64)


if __name__ == "__main__":
    output_file = "./pixcap_full_data_image1.h5"
    with Pixcap65LoadLine(scan_configuration, output_file) as pix:
        pix.scan()
        pix.analyze()
        pix.plot()
