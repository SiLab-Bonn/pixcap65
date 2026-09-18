"""
Legacy script for using the :py:class:`pixcap65.pixcap.Pixcap65` dut to measure the inter-pixel capacitances' of
a connected sensor.
In the current state of the script biasing of the sensor during the measurement is explicitly disabled.
To enable it, un-comment the corresponding lines.
This script is slightly modified to use the newly implemented api of the dut.
The data structure written to disk or the analysis and plotting routines were not modified.
Therefore, some parts of this script are exact duplicates to :py:mod:`legacy.pixcap_65_test_inter_cap`.
"""
import numpy as np
import time
from bitarray import bitarray
from matplotlib import pyplot as plt
from typing import Any

from pixcap65.pixcap.pixcap65 import Pixcap65
from pixcap65.utility import pixcap65_constants as c

if __name__ == "__main__":
    dut = Pixcap65("pixcap65.yaml")
    dut.init()

    dut.seq_size = 4  # granularity of the clock sequencer

    dut.init_smu(1.0, 0.00001, 1.5, 0.001)

    dut.init_vm2(1.0, 0.00001, 1.5, 0.001)

    # settings for sensor depletion source
    # dut.init_bias(-80.0, 0.00001, 1.5, 0.001)

    dut.seq_init(clk_0=bitarray('0100'),
                 clk_1=bitarray('0100'),
                 clk_2=bitarray('0001'),
                 clk_3=bitarray('0001'),)

    dut.vm3_on()
    dut.vm2_on()
    # dut.bias_on()

    time.sleep(15)

    # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
    for i in range(0, 20):
        c3 = dut.vm3_measure_current()
        c2 = dut.vm2_measure_current()
        print('c3:', c3)
        print('c2:', c2)
        time.sleep(1)

    # route through sensor matrix without edges
    # noinspection DuplicatedCode
    row_start = 2
    row_stop = 39
    col_start = 1
    col_stop = 38

    row_range = range(row_stop, row_start - 1, -1)
    col_range = range(col_start, col_stop + 1)

    freq_sweep_array = np.arange(1, 4.1, 1)  # .astype(np.float) # [MHz]

    # will not interfere with the legacy handling of storing data.
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
            current_array1 = []
            current_array2 = []
            fit_params = []
            table_storage = []
            dut.disable_all_pixels()
            dut.disable_all_columns()

            # enable columns of pixel under test and surrounding pixels
            dut.enable_column(i_col, c.EN_EOC_2 | c.EN_EOC_1 | c.EN_EOC_3)
            # the original statement would not have worked at all.
            dut.enable_column(i_col + 1, c.EN_EOC_1 | c.EN_EOC_3)
            dut.enable_column(i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)

            # noinspection DuplicatedCode
            time.sleep(1)

            # enable pixel under test
            dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_2 | c.EN_CLK_0)

            # enable pixels surrounding pixel under test
            dut.enable_pixel_clk(i_col, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
            dut.enable_pixel_clk(i_col + 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
            dut.enable_pixel_clk(i_col + 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
            dut.enable_pixel_clk(i_col + 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
            dut.enable_pixel_clk(i_col, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
            dut.enable_pixel_clk(i_col - 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
            dut.enable_pixel_clk(i_col - 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
            dut.enable_pixel_clk(i_col - 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)

            for freq in freq_sweep_array:
                dut.cvm_frequency = freq

                result1 = dut.vm3_measure_current()
                if isinstance(result1, str):
                    # the actually used measurement/reading routine is not capable of
                    # returning a string value
                    current_array1.append(float(result1.split(',')[1]))
                else:
                    current_array1.append(float(result1))

                result2 = dut.vm2_measure_current()
                if isinstance(result2, str):
                    # the actually used measurement/reading routine is not capable of returning a string value
                    current_array2.append(float(result2.split(',')[1]))
                else:
                    current_array2.append(float(result2))

            # apply linear fit to measured current values; also returns covariance matrix
            # noinspection DuplicatedCode
            matrix1 = np.polyfit(freq_sweep_array, current_array1, 1, cov=True)
            matrix2 = np.polyfit(freq_sweep_array, current_array2, 1, cov=True)

            a, b = matrix1[0][0], matrix1[0][1]  # fit parameters of reference pixel
            fit_params.append(a)
            fit_params.append(b)

            e, d = matrix2[0][0], matrix2[0][1]  # fit parameters of neighbouring pixels
            fit_params.append(e)
            fit_params.append(d)

            print(i_col, i_row, a, e)

            # data structure in txt file: "slope, offset (y-intercept), slope, offset (y-intercept)"
            table_storage.append(fit_params)
            table_row.append(table_storage)

            freq_sweep_array_plot = np.arange(0, 5.1, 1)

            fit_fn = a * freq_sweep_array_plot + b
            plt.plot(freq_sweep_array, current_array1, 'o',
                     label='COL({i_col})PIX({i_row}), I3'.format(i_col=i_col, i_row=i_row))
            plt.plot(freq_sweep_array_plot, fit_fn, label='a={a:.3E}, b={b:.3E}'.format(a=a, b=b))

            fit_fn = e * freq_sweep_array_plot + d
            plt.plot(freq_sweep_array, current_array2, 'o',
                     label='COL({i_col})PIX({i_row}), I2'.format(i_col=i_col, i_row=i_row))
            plt.plot(freq_sweep_array_plot, fit_fn, label='c={a:.3E}, d={b:.3E}'.format(a=e, b=d))

        # print(','.join(map(str, table_row)))
        # data_file.write(','.join(map(str, table_row)) + '\n')
        data_file.write('\n'.join(map(str, table_row)) + '\n')  # write data in new lines

    data_file.close()

    # time.sleep(300)

    # dut.bias_off()
    dut.vm2_off()
    dut.vm3_off()

    plt.legend(loc='best')
    plt.xlabel('Freq [MHz]')
    plt.ylabel('I [A]')
    plt.show()
    dut.close()
