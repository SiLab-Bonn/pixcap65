#
# ------------------------------------------------------------
# Copyright (c) SILAB , Physics Institute of Bonn University
# ------------------------------------------------------------
#
# SVN revision information:
#  $Rev:: 418                   $:
#  $Author:: HK    $:
#  $Date:: 2015-01-04 10:56:36 #$:
#

import time

from basil.dut import Dut


from pylab import *

import matplotlib.pyplot as plt
import numpy as np



class pixcap65(Dut):


	def switch_on_power_supply_voltages(self,pwr_en):
		"""
		Switches on default supply voltages
		"""
		if(pwr_en):
			self['VDD'].set_current_limit(100, unit='mA')
		#Power
			self['VDD'].set_voltage(1.2, unit='V')
			self['VDD'].set_enable(pwr_en)

			time.sleep(0.1)
			print ''
			print 'VDD:\t', format(self['VDD'].get_voltage(unit='V'), '.3f'), 'V\t', format(self['VDD'].get_current(), '.3f'), 'mA'
			print ''
		else:
			self['VDD'].set_enable(pwr_en)
		return
	
	def get_status(self):
		status = {}
		status['Time'] = time.strftime("%d %M %Y %H:%M:%S")
		status['VDD'] = {'voltage(V)':format(self['VDD'].get_voltage(unit='V'), '.3f'), 'current(mA)':  format(self['VDD'].get_current(), '.3f' )}
		return status
	
	def init_config(self):
		BIAS_SEL = list([0,0])
		CLK_EN   = list([0,0,0,0])
		PIX      = array([BIAS_SEL, CLK_EN])
		EOC      = list([0,0,0])
		COL      = array([EOC, PIX * 41])
		CHIP     = array([COL * 40])
		return 
		
	



	

