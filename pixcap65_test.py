"""
Script for small tests
"""

#import array
#import itertools
#import operator
#import time
import _thread
import yaml

from pixcap65 import pixcap65
import pixcap65_constants as c
from basil.dut import Dut

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


import time
from bitarray import bitarray
import logging
logging.getLogger().setLevel(logging.DEBUG)


dut = pixcap65("pixcap65.yaml")
dut.init()

seq_size  = 4 # granularity of the clock sequencer


dut['SMU'].off()
dut['SMU'].source_volt()
dut['SMU'].set_voltage_range(1.5)
dut['SMU'].set_current_nlpc(10)
dut['SMU'].set_voltage(1.0)
dut['SMU'].set_current_limit(0.001)
dut['SMU'].set_current_sense_range(0.000001)


dut['SEQ'].reset()
dut['SEQ'].set_clk_divide(1)
dut['SEQ'].set_repeat_start(0) 
dut['SEQ'].set_repeat(0) 
dut['SEQ'].set_size(seq_size)
dut['SEQ']['CLK_0'][0:seq_size-1] =  bitarray('1000') 
#dut['SEQ']['CLK_1'][0:seq_size-1] =  bitarray('00000000000111111110') 
#dut['SEQ']['CLK_2'][0:seq_size-1] =  bitarray('01111111110000000000') 
dut['SEQ']['CLK_3'][0:seq_size-1] =  bitarray('0010') 
dut['SEQ'].write()
dut['SEQ'].start()


dut['SMU'].on()
time.sleep(1)
dut['SMU'].get_reading()


row_start =  0
row_stop  =  0
col_start =  5
col_stop  =  9

row_range = range(row_stop, row_start-1,-1)
col_range = range(col_start, col_stop+1)


freq_sweep_array = np.arange(1, 4.1, 1)#.astype(np.float) # [MHz]
table_first_row = ["row\col"]
table_first_row.extend(col_range)
table_row = []


data_file = open("D:/Users/Hans/work/RD53/PixCap65/Measurements/pixcap_full_data_image1.txt", "w")
#data_file.write(','.join(map(str,table_first_row))+'\n')

for i_row in row_range:
    table_row=[]
    for i_col in col_range:
        current_array = []
        dut.disable_all_pixels()
        dut.disable_all_columns()
        dut.enable_column(i_col, c.EN_EOC_3)
        dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)
        
        for freq in freq_sweep_array:
            temp = freq * seq_size
            dut['MIO_PLL'].setFrequency(temp)
            #time.sleep(1)
            result = dut['SMU'].get_reading()
            current_array.append(float(result.split(',')[1]))
        
        pl.plot(freq_sweep_array, current_array, label = "COL["+ str(i_col) + ']ROW[' + str(i_row) + ']')
        a,b = np.polyfit(freq_sweep_array, current_array, 1) 
        print(i_col, i_row, a)
        table_row.append(a)
        fit_fn = a*freq_sweep_array + b
        pl.plot(freq_sweep_array, current_array, 'o', label = 'COL({i_col})PIX(0)'.format(i_col=i_col))
        pl.plot(freq_sweep_array, fit_fn, label = 'a={a:.3E}, b={b:.3E}'.format(a=a, b=b))
    
    print(','.join(map(str, table_row)))
    data_file.write( ','.join(map(str, table_row)) + '\n')    
    
data_file.close()
dut['SMU'].off()
pl.legend(loc='best')
pl.xlabel('Freq [MHz]')
pl.ylabel('I [A]')
pl.show()
dut.close()


