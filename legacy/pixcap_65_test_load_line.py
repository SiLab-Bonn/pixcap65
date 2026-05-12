from typing import Any

import time
from bitarray import bitarray

import pixcap65.utility.pixcap65_constants as c
from pixcap65.pixcap.pixcap65 import Pixcap65

if __name__ == "__main__":
    # schedule this strange top-level script for removal! (for now it is uncommented)
    dut = Pixcap65("pixcap65.yaml")
    dut.init()

    seq_size = 128  # granularity of the clock sequencer

    dut['SMU3'].off()
    dut['SMU3'].source_volt()
    dut['SMU3'].set_voltage_range(1.5)
    dut['SMU3'].set_current_nlpc(10)
    dut['SMU3'].set_voltage(1.0)
    dut['SMU3'].set_current_limit(0.001)
    dut['SMU3'].set_current_sense_range(0.00001)

    m = seq_size // 2 - 1  # define index of the last 1 in order to create a non-overlapping clock sequence
    cnt = seq_size / 2 - 1  # counter for numbering in output file
    table_row = []

    # create initial bit arrays for a given sequencer size
    bit_array_CLK_3 = bitarray(seq_size)
    bit_array_CLK_3.setall(0)
    bit_array_CLK_3[1:m] = 1

    bit_array_CLK_0 = bitarray(seq_size)
    bit_array_CLK_0.setall(0)
    bit_array_CLK_0[m + 1:-1] = 1

    data_file = open("./pixcap_full_data_image1.txt", "w")

    # vary charging time by looping over the number of bits in bit_array_CLK_3 that are set to 1;
    # number is reduced by one in every step
    for i in range(m, 0, -1):

        dut['SEQ'].reset()
        dut['SEQ'].set_clk_divide(1)
        dut['SEQ'].set_repeat_start(0)
        dut['SEQ'].set_repeat(0)
        dut['SEQ'].set_size(seq_size)
        dut['SEQ']['CLK_0'][0:seq_size - 1] = bit_array_CLK_0
        dut['SEQ']['CLK_3'][0:seq_size - 1] = bit_array_CLK_3
        dut['SEQ'].write()
        dut['SEQ'].start()

        # maybe this step does not have to be within the loop over m
        # did not know if SMU has to be set on after changing the clock sequencer
        dut['SMU3'].on()
        dut['SMU3'].get_reading()

        row_start = 0
        row_stop = 0
        col_start = 12
        col_stop = 12

        row_range = range(row_stop, row_start - 1, -1)
        col_range = range(col_start, col_stop + 1)

        # freq_sweep_array = np.arange(1, 4.1, 1)#.astype(np.float) # [MHz]
        freq_sweep_array = [1.0]  # can also uncomment loop over frequencies
        # Why defining this never used fields?
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
                    temp = freq * seq_size
                    dut['MIO_PLL'].setFrequency(temp)
                    time.sleep(1)
                    result = dut['SMU3'].get_reading()
                    current_array.append(cnt)
                    current_array.append(float(result.split(',')[1]))

                table_row.append(current_array)

            print(bit_array_CLK_3)
            # this print statement seems to be senseless as it could only show the currents for the last column processed
            print(current_array)

        # reduce charging time with every iteration by setting last bit 1 -> 0 and decrement counter
        bit_array_CLK_3[i] = 0
        cnt = cnt - 1
    data_file.write('\n'.join(map(str, table_row)) + '\n')

    data_file.close()
    dut['SMU3'].off()
    dut.close()
