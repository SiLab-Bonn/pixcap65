"""
The latest version of the Pixcap65 test script for measuring the total pixel capacitance.

Changes compared to original script:
- Remote control of depletion voltage source
- Reading some current values before actual measurement to avoid incorrect currents due to initial oscillation effects of SMU
- Vary the order of column/row routing and switching frequency using the reversed arrays (uncomment corresponding lines in code)
- Fit also returns covariance matrix in order to extract the errors of the fit parameters if needed
- Output in txt file also includes offset (y-intercept) next to the slope
"""

import tables as tb
from collections import OrderedDict

from pixcap65 import pixcap65
import pixcap65_constants as c
from analysis import analyze_data
from plotting import plot_data
from basil.dut import Dut

import numpy as np

import time
from bitarray import bitarray
import logging
logging.getLogger().setLevel(logging.INFO)


def store_scan_par_values(scan_parameters, scan_param_id, **kwargs):
    '''
        Manually store the scan parameter values for the scan parameter id
        This allows to reconstruct the scan parameter values for a given parameter state vector
    '''
    if scan_parameters.get(scan_param_id) and scan_parameters.get(scan_param_id) != kwargs:
        raise ValueError('You cannot change the scan parameter value of a scan parameter id')
    scan_parameters[scan_param_id] = kwargs


def _store_scan_par_values(h5_file, scan_parameters):
    '''
        Create scan_params table after a scan
    '''
    # Create parameter description
    keys = set()  # find all keys to make the table column names
    for par_values in scan_parameters.values():
        keys.update(par_values.keys())
    fields = [('scan_param_id', np.uint32)]
    # FIXME only float32 supported so far
    fields.extend([(name, np.float32) for name in keys])

    scan_par_table = h5_file.create_table(h5_file.root, name='scan_params', title='Scan parameter values per scan parameter id', description=np.dtype(fields))
    for par_id, par_values in scan_parameters.items():
        a = np.full(shape=(1,), fill_value=np.NaN).astype(np.dtype(fields))
        for key, val in par_values.items():
            a['scan_param_id'] = par_id
            a[key] = np.float32(val)
        scan_par_table.append(a)


scan_configuration = {
    'start_column': 10,
    'stop_column': 30,
    'start_row': 10,
    'stop_row': 30,

    'Vin': 1.0,  # input voltage in V
    'frequency_range': np.arange(1, 4.1, 1)  # frequency sweep in MHz
}


class PixCap65TotalCap(object):
    def __init__(self, scan_config, output_file):
        self.dut = pixcap65("pixcap65.yaml")
        self.dut.init()

        self.scan_config = scan_config

        self.output_file = output_file
        self.out_file_h5 = tb.open_file(self.output_file, mode='w')

        self.scan_parameters = OrderedDict()

        self.hist_current = np.full(shape=(40, 40, len(scan_config['frequency_range'])), fill_value=np.nan)  # current value for each measured frequency per pixel

    def configure(self):
        self.seq_size = 4  # granularity of the clock sequencer

        # settings for sensor depletion source
        # self.dut['SMU1'].off()
        # self.dut['SMU1'].source_volt()
        # self.dut['SMU1'].set_voltage_range(1.5)
        # self.dut['SMU1'].set_current_nlpc(10)
        # self.dut['SMU1'].set_voltage(-80.0)
        # self.dut['SMU1'].set_current_limit(0.001)
        # self.dut['SMU1'].set_current_sense_range(0.00001)

        self.dut['SMU'].off()
        self.dut['SMU'].source_volt()
        self.dut['SMU'].set_voltage_range(1.5)
        self.dut['SMU'].set_current_nlpc(10)
        self.dut['SMU'].set_voltage(self.scan_config['Vin'])
        self.dut['SMU'].set_current_limit(0.001)
        self.dut['SMU'].set_current_sense_range(0.00001)

        self.dut['SEQ'].reset()
        self.dut['SEQ'].set_clk_divide(1)
        self.dut['SEQ'].set_repeat_start(0)
        self.dut['SEQ'].set_repeat(0)
        self.dut['SEQ'].set_size(self.seq_size)
        self.dut['SEQ']['CLK_0'][0:self.seq_size - 1] = bitarray('1000')
        self.dut['SEQ']['CLK_3'][0:self.seq_size - 1] = bitarray('0010')
        # self.dut['SEQ']['CLK_1'][0:self.seq_size - 1] =  bitarray('00000000000111111110')
        # self.dut['SEQ']['CLK_2'][0:self.seq_size - 1] =  bitarray('01111111110000000000')
        self.dut['SEQ'].write()
        self.dut['SEQ'].start()

        self.dut['SMU'].on()

        # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
        logging.debug('Waiting for settling of SMU...')
        for i in range(0, 30):
            c3 = self.dut['SMU'].get_current()
            current = float(c3.split(',')[1])
            logging.debug('Current: %.3e' % current)
            time.sleep(1)

        self.dut['SMU'].get_current()

    def scan(self):
        row_range = range(self.scan_config['start_row'], self.scan_config['stop_row'])
        col_range = range(self.scan_config['start_column'], self.scan_config['stop_column'])
        frequency_range = self.scan_config['frequency_range']

        for i_row in row_range:
            for i_col in col_range:
                logging.info('Measuring pixel (%i, %i)...' % (i_col, i_row))
                self.dut.disable_all_pixels()
                self.dut.disable_all_columns()

                self.dut.enable_column(i_col, c.EN_EOC_3)
                self.dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)

                for k, freq in enumerate(frequency_range):
                    freq_conv = freq * self.seq_size
                    self.dut['MIO_PLL'].setFrequency(freq_conv)
                    time.sleep(1)
                    result = self.dut['SMU'].get_current()
                    current = float(result.split(',')[1])
                    self.hist_current[i_col, i_row, k] = current
                    store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq)

        # Save raw data
        _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.scan_parameters)
        self.out_file_h5.create_carray(self.out_file_h5.root,
                                       name='HistCurr',
                                       title='Current Histogram',
                                       obj=self.hist_current,
                                       filters=tb.Filters(complib='blosc',
                                                          complevel=5,
                                                          fletcher32=False))

        logging.info('Done')

    def close(self):
        self.out_file_h5.close()
        self.dut['SMU'].off()
        self.dut.close()


if __name__ == '__main__':
    output_file = "./TEST.h5"
    pix = PixCap65TotalCap(scan_configuration, output_file)
    pix.configure()
    pix.scan()
    pix.close()

    # Analyse and plot data
    analyze_data(output_file)
    plot_data(output_file)
