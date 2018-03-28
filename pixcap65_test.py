"""
Script for small tests
"""

#import array
#import itertools
#import operator
#import time
import thread
import yaml

from pixcap65 import pixcap65
import pixcap65_constants as c
from basil.dut import Dut

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

import SiLibUSB    
import time
from bitarray import bitarray
import logging
logging.getLogger().setLevel(logging.DEBUG)


dut = pixcap65("pixcap65.yaml")
dut.init()

seq_size  = 4 # granularity of the clock sequencer


dut['SMU'].off()
dut['SMU'].source_volt()
dut['SMU']._intf.write('SOUR:VOLT:RANGE 2.2')
dut['SMU']._intf.write('SENS:CURR:NPLC 10')
dut['SMU'].set_voltage(1.2)
dut['SMU'].set_current_limit(0.0001)

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
dut['SMU'].get_current()

# time.sleep(2)
# full_string = dut['SMU'].get_current()
# print 'Voltage:' +  full_string.split(',')[0]
# print 'Current: ' + full_string.split(',')[1]


freq_sweep_array = np.arange(1, 10, 1)#.astype(np.float) # [MHz]

#freq_sweep_array = [1.0, 2.0, 5.0, 10.0, 20.0]


for i_col in range(17, 18):
    current_array = []
    
    COL_DUT = i_col
    PIX_DUT = 0
    dut.disbale_all_pixels()
    dut.disable_all_columns()
    dut.enable_column(COL_DUT, c.EN_EOC_3)
    dut.enable_pixel_clk(COL_DUT, PIX_DUT, c.EN_CLK_0 | c.EN_CLK_3)
    
    for freq in freq_sweep_array:
        temp = freq * seq_size
        dut['MIO_PLL'].setFrequency(temp)
        #time.sleep(1)
        result = dut['SMU'].get_current()
        current_array.append(float(result.split(',')[1]))
    
#    pl.plot(freq_sweep_array, current_array, c = colors[color_index] , label = "COL["+ str(COL_DUT) + ']PIX[' + str(PIX_DUT) + ']') 
#    pl.plot(freq_sweep_array, current_array, label = "COL["+ str(COL_DUT) + ']PIX[' + str(PIX_DUT) + ']')
    a,b = np.polyfit(freq_sweep_array, current_array, 1) 
    print  a 
    fit_fn = a*freq_sweep_array + b
    pl.plot(freq_sweep_array, current_array, 'o', label = 'COL({i_col})PIX(0)'.format(i_col=i_col))
    pl.plot(freq_sweep_array, fit_fn, label = 'a={a:.3E}, b={b:.3E}'.format(a=a, b=b))
    


pl.legend(loc='best')
#ylim(0,2)
pl.xlabel('Freq [MHz]')
pl.ylabel('I [A]')
pl.show()
 


dut['SMU'].off()
dut.close()


