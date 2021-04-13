"""
Script for measuring Inter Pixel Capacitance 
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
import coloredlogs
logging.getLogger().setLevel(logging.INFO)


dut = pixcap65("pixcap65.yaml")
dut.init()

seq_size  = 4 #granularity of the clock sequencer

dut['SMU3'].off()
dut['SMU3'].source_volt()
dut['SMU3'].set_voltage_range(1.5)
dut['SMU3'].set_current_nlpc(10)
dut['SMU3'].set_voltage(1.0)
dut['SMU3'].set_current_limit(0.001)
dut['SMU3'].set_current_sense_range(0.00001)

dut['SMU2'].off()
dut['SMU2'].source_volt()
dut['SMU2'].set_voltage_range(1.5)
dut['SMU2'].set_current_nlpc(10)
dut['SMU2'].set_voltage(1.0)
dut['SMU2'].set_current_limit(0.001)
dut['SMU2'].set_current_sense_range(0.00001)

#settings for sensor depletion source
#dut['SMU1'].off()
#dut['SMU1'].source_volt()
#dut['SMU1'].set_voltage_range(1.5)
#dut['SMU1'].set_current_nlpc(10)
#dut['SMU1'].set_voltage(-80.0)
#dut['SMU1'].set_current_limit(0.001)
#dut['SMU1'].set_current_sense_range(0.00001)

dut['SEQ'].reset()
dut['SEQ'].set_clk_divide(1)
dut['SEQ'].set_repeat_start(0) 
dut['SEQ'].set_repeat(0) 
dut['SEQ'].set_size(seq_size)

dut['SEQ']['CLK_0'][0:seq_size-1] =  bitarray('0100') 
dut['SEQ']['CLK_1'][0:seq_size-1] =  bitarray('0100') 
dut['SEQ']['CLK_2'][0:seq_size-1] =  bitarray('0001') 
dut['SEQ']['CLK_3'][0:seq_size-1] =  bitarray('0001')

dut['SEQ'].write()
dut['SEQ'].start()

dut['SMU3'].on()
dut['SMU2'].on()
#dut['SMU1'].on()

time.sleep(15)

#measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
for i in range(0, 20):
    c3 = dut['SMU3'].get_reading()
    c2 = dut['SMU2'].get_reading() 
    print('c3:', c3)
    print('c2:', c2)
    time.sleep(1)
    
#route through sensor matrix without edges
row_start =  2
row_stop  =  39
col_start =  1
col_stop  =  38

row_range = range(row_stop, row_start-1, -1)
col_range = range(col_start, col_stop+1)


freq_sweep_array = np.arange(1, 4.1, 1) #.astype(np.float) # [MHz]
table_first_row = ["row\col"]
table_first_row.extend(col_range)
table_row = []
table_storage = [] #some additional list to store results during measurement

data_file = open("./pixcap_full_data_image1.txt", "w")
#data_file.write(','.join(map(str,table_first_row))+'\n')

for i_row in row_range:
    table_row = []
    for i_col in col_range:
        current_array1 = []
        current_array2 = []
        fit_params = [] 
        table_storage = []
        dut.disable_all_pixels()
        dut.disable_all_columns()
        
        #enable columns of pixel under test and surrounding pixels
        dut.enable_column(i_col, c.EN_EOC_2 | c.EN_EOC_1 | c.EN_EOC_3) 
        dut.enable_column(i_col + 1 | i_col - 1, c.EN_EOC_1 | c.EN_EOC_3) 
        #dut.enable_column(i_col + 1, c.EN_EOC_1 | c.EN_EOC_3)
        #dut.enable_column(i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)

        time.sleep(1) 

        #enable pixel under test
        dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_2 | c.EN_CLK_0) 

        #enable pixels surrounding pixel under test
        dut.enable_pixel_clk(i_col, i_row+1, c.EN_CLK_1 | c.EN_CLK_3)
        dut.enable_pixel_clk(i_col+1, i_row+1, c.EN_CLK_1 | c.EN_CLK_3)
        dut.enable_pixel_clk(i_col+1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
        dut.enable_pixel_clk(i_col+1, i_row-1, c.EN_CLK_1 | c.EN_CLK_3)
        dut.enable_pixel_clk(i_col, i_row-1, c.EN_CLK_1 | c.EN_CLK_3)
        dut.enable_pixel_clk(i_col-1, i_row-1, c.EN_CLK_1 | c.EN_CLK_3)
        dut.enable_pixel_clk(i_col-1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
        dut.enable_pixel_clk(i_col-1, i_row+1, c.EN_CLK_1 | c.EN_CLK_3)
       
        
        for freq in freq_sweep_array: 
            temp = freq * seq_size
            dut['MIO_PLL'].setFrequency(temp)

            result1 = dut['SMU3'].get_reading()
            current_array1.append(float(result1.split(',')[1]))

            result2 = dut['SMU2'].get_reading()
            current_array2.append(float(result2.split(',')[1]))

        #apply linear fit to measured current values; also returns covariance matrix
        matrix1 = np.polyfit(freq_sweep_array, current_array1, 1, cov=True) 
        matrix2 = np.polyfit(freq_sweep_array, current_array2, 1, cov=True) 
        
        a, b = matrix1[0][0], matrix1[0][1] #fit parameters of reference pixel
        fit_params.append(a)
        fit_params.append(b)

        e, d = matrix2[0][0], matrix2[0][1] #fit parameters of neighbouring pixels
        fit_params.append(e)
        fit_params.append(d)

        print(i_col, i_row, a, e) 

        #data structure in txt file: "slope, offset (y-intercept), slope, offset (y-intercept)"
        table_storage.append(fit_params)
        table_row.append(table_storage)


        freq_sweep_array_plot = np.arange(0, 5.1, 1)

        fit_fn = a*freq_sweep_array_plot + b
        pl.plot(freq_sweep_array, current_array1, 'o', label = 'COL({i_col})PIX({i_row}), I3'.format(i_col=i_col, i_row=i_row))
        pl.plot(freq_sweep_array_plot, fit_fn, label = 'a={a:.3E}, b={b:.3E}'.format(a=a, b=b))

        fit_fn = e*freq_sweep_array_plot + d
        pl.plot(freq_sweep_array, current_array2, 'o', label = 'COL({i_col})PIX({i_row}), I2'.format(i_col=i_col, i_row=i_row))
        pl.plot(freq_sweep_array_plot, fit_fn, label = 'c={a:.3E}, d={b:.3E}'.format(a=e, b=d))

        
    #print(','.join(map(str, table_row)))
    #data_file.write(','.join(map(str, table_row)) + '\n')
    data_file.write('\n'.join(map(str, table_row)) + '\n') #write data in new lines 
      
data_file.close()

#time.sleep(300)

#dut['SMU1'].off()
dut['SMU2'].off()
dut['SMU3'].off()


pl.legend(loc='best')
pl.xlabel('Freq [MHz]')
pl.ylabel('I [A]')
pl.show()
dut.close()


