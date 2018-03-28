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

import pixcap65_constants as c

class pixcap65(Dut):
	def init(self, init_conf=None, **kwargs):
		Dut.init(self, init_conf=init_conf, **kwargs)
		self.switch_on_power_supply_voltages(1)
		self['SPI'].set_size(9960)
		self.reset_chip()
		return
		
	def close(self):
		Dut.close(self)		
		self.switch_on_power_supply_voltages(0)
		self['SEQ'].clear()
		return


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
	
	def enable_pixel_clk(self, i_col, i_pix, clk_mask):
		self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['CLK_EN'] = clk_mask
		self.update_spi()
		return

	def enable_pixel_bias(self, i_col, i_pix, bias_mask):
		self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['SEL'] = bias_mask
		self.update_spi()
		return
	
	def enable_column(self, i_col, EOC_MASK):
		self['SPI']['COL'][c.COL_NMAX - i_col]['EOC'] = EOC_MASK
		self.update_spi()
		return
	
	def disbale_all_pixels(self):
		for i_col in range(0, c.COL_NMAX):
			for i_pix in range(0, c.PIX_NMAX):
				self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['CLK_EN'] = 0
				self['SPI']['COL'][c.COL_NMAX - i_col]['PIX'][c.PIX_NMAX - i_pix]['SEL'] = 0
		self.update_spi()		
		return		
	
	def update_spi(self):
		self['SPI'].write()
		self['SPI'].start()
		return
	
	def disable_all_columns(self):
		for i_col in range(0, c.COL_NMAX):
			self['SPI']['COL'][c.COL_NMAX - i_col]['EOC'] = 0
		self.update_spi()		
		return
	
	def reset_chip(self):
		# reset shift register
		self['GPIO']['RST_B'] = 0
		self['GPIO'].write()
		time.sleep(0.1)
		self['GPIO']['RST_B'] = 1
		self['GPIO'].write()
		return
	
	def init_config(self):
		self['SPI'].set_size(9960)
		self.reset_chip()
		return 
		
	



	

