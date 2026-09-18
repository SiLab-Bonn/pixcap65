"""
Legacy script for PixCap65 to determine the charging behaviour of the capacitance for different
charge-pulse lengths (and also their frequency dependency).
This script (or its results) could be used to determine the optimal frequency range manually.
This script is slightly modified to use the newly implemented api of the dut.
The data structure written to disk or the analysis and plotting routines were not modified.
Therefore, some parts of this script are exact duplicates to :py:mod:`examples.pixcap_65_test_load_line`.

:author: Hans Krüger
:version: 0.2

last modification by Dominik Fischer on 2026-09-18
"""
import time
from bitarray import bitarray
from typing import Any

import pixcap65.utility.pixcap65_constants as c
from pixcap65.pixcap.pixcap65 import Pixcap65

if __name__ == "__main__":
    dut = Pixcap65("pixcap65.yaml")
    dut.init()

    dut.seq_size = 128  # granularity of the clock sequencer
    dut.frequency_settling = 1

    dut.init_smu(1.0, 0.00001, 1.5, 0.001)

    m = dut.seq_size // 2 - 1  # define index of the last 1 in order to create a non-overlapping clock sequence
    cnt = dut.seq_size / 2 - 1  # counter for numbering in output file
    table_row = []

    # create initial bit arrays for a given sequencer size
    # noinspection DuplicatedCode
    bit_array_CLK_3 = bitarray(dut.seq_size)
    bit_array_CLK_3.setall(0)
    bit_array_CLK_3[1:m] = 1

    bit_array_CLK_0 = bitarray(dut.seq_size)
    bit_array_CLK_0.setall(0)
    bit_array_CLK_0[m + 1:-1] = 1

    data_file = open("./pixcap_full_data_image1.txt", "w")

    # vary charging time by looping over the number of bits in bit_array_CLK_3 that are set to 1;
    # number is reduced by one in every step
    for i in range(m, 0, -1):

        dut.seq_init(clk_0=bit_array_CLK_0, clk_3=bit_array_CLK_3)

        # maybe this step does not have to be within the loop over m
        # did not know if SMU has to be set on after changing the clock sequencer
        dut.smu_on()
        _ = dut.get_source_current

        # noinspection DuplicatedCode
        row_start = 0
        row_stop = 0
        col_start = 12
        col_stop = 12

        row_range = range(row_stop, row_start - 1, -1)
        col_range = range(col_start, col_stop + 1)

        # freq_sweep_array = np.arange(1, 4.1, 1)#.astype(np.float) # [MHz]
        freq_sweep_array = [1.0]  # can also uncomment loop over frequencies

        # Why defining this never used fields?
        # will not interfere with legacy handling of data store.
        table_first_row = list[Any]()
        table_first_row.append("row\\col")
        table_first_row.extend(col_range)
        current_array = []

        # how to distinguish the different rows and columns in the table?
        for i_row in row_range:
            for i_col in col_range:
                current_array = []
                dut.disable_all_pixels()
                dut.disable_all_columns()
                dut.enable_column(i_col, c.EN_EOC_3)
                time.sleep(1)
                dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)

                for freq in freq_sweep_array:
                    dut.cvm_frequency = freq
                    result = dut.get_source_current
                    current_array.append(cnt)
                    if isinstance(result, str):
                        # the actual implementation is not capable of returning a string value
                        current_array.append(float(result.split(',')[1]))
                    else:
                        current_array.append(result)

                table_row.append(current_array)

            print(bit_array_CLK_3)
            # this print statement seems to be senseless as it could only show the currents for the last column
            # processed
            print(current_array)

        # reduce charging time with every iteration by setting last bit 1 -> 0 and decrement counter
        bit_array_CLK_3[i] = 0
        cnt = cnt - 1
    data_file.write('\n'.join(map(str, table_row)) + '\n')

    data_file.close()
    dut.smu_off()
    dut.close()
