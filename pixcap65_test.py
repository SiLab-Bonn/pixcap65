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
from basil.dut import Dut

import matplotlib.pyplot as plt
import numpy as np

#from Keithley import KeithleySMU2400Series

import SiLibUSB    
import time
from bitarray import bitarray
import logging
logging.getLogger().setLevel(logging.DEBUG)


dut = pixcap65("pixcap65.yaml")
dut.init()


#tti = Dut("ttiQl355tp_local.yaml")
#tti.init()

#print tti['power'].get_info()

#tti['power'].set_enable(True, 1)
#time.sleep(1)
#print format( tti['power'].get_current(1), '.3f') + ' A'
#print format( tti['power'].get_voltage(1), '.3f') + ' V'


doExit = 0
power_on = True
power_off = False
COL_NMAX = 39 
PIX_NMAX = 40
meas_freq = 0.5  # [MHz]
seq_size  = 20 # granularity of the clock sequencer

dut.init_config()

dut['GPIO']['LED1'] = 1
dut['GPIO'].write()
time.sleep(0.05)
dut['GPIO']['LED2'] = 1
dut['GPIO'].write()
time.sleep(0.05)
dut['GPIO']['LED3'] = 1
dut['GPIO'].write()
time.sleep(0.05)
dut['GPIO']['LED4'] = 1
dut['GPIO'].write()
time.sleep(0.05)
dut['GPIO']['LED5'] = 1
dut['GPIO'].write()
   
dut['MIO_PLL'].setFrequency(meas_freq * seq_size)
dut.switch_on_power_supply_voltages(power_on)

dut['GPIO']['RST_B'] = 0
dut['GPIO'].write()
time.sleep(0.1)
dut['GPIO']['RST_B'] = 1
dut['GPIO'].write()


COL_DUT = 18
PIX_DUT = 0

dut['SPI'].set_size(9960)
dut['SPI']['COL'][COL_NMAX - COL_DUT]['EOC'] = '100'
dut['SPI']['COL'][COL_NMAX - COL_DUT]['PIX'][PIX_NMAX - PIX_DUT]['SEL'] = '00'
dut['SPI']['COL'][COL_NMAX - COL_DUT]['PIX'][PIX_NMAX - PIX_DUT]['CLK_EN'] = '1001'
# for i_pix in range(0, 1):
#     dut['SPI']['COL'][COL_NMAX - 0]['PIX'][PIX_NMAX - i_pix]['SEL'] = '00'
#     dut['SPI']['COL'][COL_NMAX - 0]['PIX'][PIX_NMAX - i_pix]['CLK_EN'] = '1001'
#dut['SPI'][9959:0] =  9960 * '1'
dut['SPI'].write()
dut['SPI'].start()
time.sleep(0.5)
dut['SPI'].start()

dut['SEQ'].reset()
dut['SEQ'].set_clk_divide(1)
dut['SEQ'].set_repeat_start(0) 
dut['SEQ'].set_repeat(0) 
dut['SEQ'].set_size(seq_size)
dut['SEQ']['CLK_0'][0:seq_size-1] =  bitarray('01111111100000000000') 
#dut['SEQ']['CLK_1'][0:seq_size-1] =  bitarray('00000000000111111110') 
#dut['SEQ']['CLK_2'][0:seq_size-1] =  bitarray('01111111110000000000') 
dut['SEQ']['CLK_3'][0:seq_size-1] =  bitarray('00000000000111111110') 
dut['SEQ'].write()
dut['SEQ'].start()


#def restart_seq():
#    while (not doExit):
#        if(dut['SEQ'].is_done()):
#            dut['SEQ'].start() # restart sequence

#thread.start_new_thread(restart_seq, ())

print("Press 'q' to exit.")
while 1:

  key = raw_input(": ")
  if key == 'q':
    break

doExit = 1
dut['SEQ'].clear()
dut.switch_on_power_supply_voltages(power_off)

dut['GPIO']['LED1'] = 0
dut['GPIO'].write()
time.sleep(0.05)
dut['GPIO']['LED2'] = 0
dut['GPIO'].write()
time.sleep(0.05)
dut['GPIO']['LED3'] = 0
dut['GPIO'].write()
time.sleep(0.05)
dut['GPIO']['LED4'] = 0
dut['GPIO'].write()
time.sleep(0.05)
dut['GPIO']['LED5'] = 0
dut['GPIO'].write()

dut.close()


