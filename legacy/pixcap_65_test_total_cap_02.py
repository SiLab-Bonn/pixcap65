"""
The latest version of the Pixcap65 test script for measuring the total pixel capacitance.

Changes compared to original script:
- Remote control of depletion voltage source
- Reading some current values before actual measurement to avoid incorrect currents due to initial oscillation effects of SMU
- Vary the order of column/row routing and switching frequency using the reversed arrays (uncomment corresponding lines in code)
- Fit also returns covariance matrix in order to extract the errors of the fit parameters if needed
- Output in txt file also includes offset (y-intercept) next to the slope

This script is slightly modified to use the newly implemented api of the dut.
The data structure written to disk or the analysis and plotting routines were not modified.
Therefore, some parts of this script are exact duplicates to :py:mod:`examples.pixcap_65_test_total_cap`.
"""
import logging
import numpy as np
import time
from bitarray import bitarray
from matplotlib import pyplot as plt
from typing import Any

import pixcap65.utility.pixcap65_constants as c
from pixcap65.pixcap.pixcap65 import Pixcap65

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == "__main__":
    dut = Pixcap65("pixcap65.yaml")
    dut.init()

    dut.seq_size = 4  # granularity of the clock sequencer

    # settings for sensor depletion source
    # dut.init_bias(-80.0, 0.00001, 1.5, 0.001)

    dut.init_smu(1.0, 0.00001, 1.5, 0.001)

    dut.seq_init(clk_0=bitarray('1000'), clk_3=bitarray('0010'))
    # or alternative approach, which is commented-out in the original script
    # dut.seq_init(clk_0=bitarray('1000'),
    #              clk_1=bitarray('00000000000111111110'),
    #              clk_2=bitarray('01111111110000000000'),
    #              clk_3=bitarray('0010'))

    dut.smu_on()
    # time.sleep(1)
    # dut.bias_on()

    # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
    for i in range(0, 20):
        c3 = dut.get_source_current
        print('c3:', c3)
        time.sleep(1)

    _ = dut.get_source_current

    # noinspection DuplicatedCode
    row_start = 0
    row_stop = 40
    col_start = 0
    col_stop = 39

    row_range = range(row_stop, row_start - 1, -1)
    col_range = range(col_start, col_stop + 1)
    # col_range = col_range[::-1] #reversed array
    # row_range = row_range[::-1]

    freq_sweep_array = np.arange(1, 4.1, 1)  # .astype(np.float) # [MHz]
    # freq_sweep_array = freq_sweep_array[::-1] #reversed array

    # will not interfere with legacy handling of data store.
    table_first_row = list[Any]()
    table_first_row.append("row\\col")
    table_first_row.extend(col_range)
    table_row = []
    table_storage = []  # some additional list to store results during measurement

    data_file = open("./pixcap_full_data_image1.txt", "w")
    # data_file.write(','.join(map(str,table_first_row))+'\n')

    for i_row in row_range:
        table_row = []
        for i_col in col_range:
            current_array = []
            table_storage = []
            dut.disable_all_pixels()
            dut.disable_all_columns()

            dut.enable_column(i_col, c.EN_EOC_3)
            dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)

            for freq in freq_sweep_array:
                dut.cvm_frequency = freq

                result = dut.get_source_current
                if isinstance(result, str):
                    # the actual implementation of the measurement/reading routine is not capable of return string
                    # values
                    current_array.append(float(result.split(',')[1]))
                else:
                    current_array.append(result)

            # apply linear fit to measured current values; also returns covariance matrix
            # noinspection DuplicatedCode
            matrix = np.polyfit(freq_sweep_array, current_array, 1, cov=True)

            a, b = matrix[0][0], matrix[0][1]
            da = matrix[1][0][0]  # squared fit error of a
            db = matrix[1][1][1]  # squared fit error of b
            print(i_col, i_row, a)
            table_storage.append(a)
            table_storage.append(b)

            # data structure in txt file: "slope, offset (y-intercept)"
            table_row.append(table_storage)

            fit_fn = a * freq_sweep_array + b
            plt.plot(freq_sweep_array, current_array, 'o', label='COL({i_col})PIX(0)'.format(i_col=i_col))
            plt.plot(freq_sweep_array, fit_fn, label='a={a:.3E}, b={b:.3E}'.format(a=a, b=b))

        print(','.join(map(str, table_row)))
        # data_file.write(','.join(map(str, table_row)) + '\n')
        data_file.write('\n'.join(map(str, table_row)) + '\n')  # write data in new lines

    data_file.close()
    print('done')
    # time.sleep(180)
    # dut.bias_off()
    dut.smu_off()
    plt.legend(loc='best')
    plt.xlabel('Freq [MHz]')
    plt.ylabel('I [A]')
    plt.show()
    dut.close()
