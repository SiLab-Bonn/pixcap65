"""
Script for small tests.
This is a updated version for usage of the new integrated api of the :py:class:`pixcap65.pixcap.Pixcap65` dut wrapper.
"""

import logging
import numpy as np
import os
import time
from bitarray import bitarray
from matplotlib import pyplot as plt
from typing import Any

from pixcap65.pixcap.pixcap65 import Pixcap65
from pixcap65.utility import pixcap65_constants as c

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':
    dut = Pixcap65("pixcap65.yaml")
    dut.init()

    dut.seq_size = 4  # granularity of the clock sequencer

    # should be used with port VM3 of the SMU.
    dut.init_smu(1.0, 0.000001, 1.5, 0.001)
    # will propagte current_limit, current_range, plc, src_u, voltage_range
    dut[dut.smu_setup_devices(dut.primary_smu_key)].drain_error_queue()

    dut.seq_init(clk_0=bitarray('1000'), clk_3=bitarray('0010'))
    # or alternatively (lines are also commented-out in the original script)
    # dut.seq_init(clk_0=bitarray('1000'),
    #              clk_1=bitarray('00000000000111111110'),
    #              clk_2=bitarray('01111111110000000000'),
    #              clk_3=bitarray('0010'))

    dut.smu_on()
    time.sleep(1)
    _ = dut.get_source_current

    # noinspection DuplicatedCode
    row_start = 0
    row_stop = 0
    col_start = 5
    col_stop = 9

    row_range = range(row_stop, row_start - 1, -1)
    col_range = range(col_start, col_stop + 1)

    freq_sweep_array = np.arange(1, 4.1, 1)  # .astype(np.float) # [MHz]
    # I will not interfere with the legacy handling of the data store.
    table_first_row = list[Any]()
    table_first_row.append("row\\col")
    table_first_row.extend(col_range)
    table_row = []

    data_file = open(os.path.expanduser('~/work/RD53/PixCap65/Measurements/pixcap_full_data_image1.txt'), "w")
    # data_file.write(','.join(map(str,table_first_row))+'\n')

    for i_row in row_range:
        table_row = []
        for i_col in col_range:
            current_array = []
            dut.disable_all_pixels()
            dut.disable_all_columns()
            dut.enable_column(i_col, c.EN_EOC_3)
            dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)

            for freq in freq_sweep_array:
                dut.cvm_frequency = freq
                result = dut.get_source_current
                if isinstance(result, str):
                    # on present to contain the form of the original script
                    # the measurement routine is not capable of returning a string value.
                    current_array.append(float(result.split(',')[1]))
                else:
                    current_array.append(result)

            # noinspection DuplicatedCode
            plt.plot(freq_sweep_array, current_array, label="COL[" + str(i_col) + ']ROW[' + str(i_row) + ']')
            a, b = np.polyfit(freq_sweep_array, current_array, 1)
            print(i_col, i_row, a)
            table_row.append(a)
            fit_fn = a * freq_sweep_array + b
            plt.plot(freq_sweep_array, current_array, 'o', label='COL({i_col})PIX(0)'.format(i_col=i_col))
            plt.plot(freq_sweep_array, fit_fn, label='a={a:.3E}, b={b:.3E}'.format(a=a, b=b))

        print(','.join(map(str, table_row)))
        data_file.write(','.join(map(str, table_row)) + '\n')

    data_file.close()
    dut.smu_off()
    plt.legend(loc='best')
    plt.xlabel('Freq [MHz]')
    plt.ylabel('I [A]')
    plt.show()
    dut.close()
