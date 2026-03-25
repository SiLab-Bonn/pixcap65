"""
The latest version of the Pixcap65 test script for measuring the total pixel capacitance.

Changes compared to original script:
- Remote control of depletion voltage source
- Reading some current values before actual measurement to avoid incorrect currents due to initial oscillation effects of SMU
- Vary the order of column/row routing and switching frequency using the reversed arrays (uncomment corresponding lines in code)
- Fit also returns covariance matrix in order to extract the errors of the fit parameters if needed
- Output in txt file also includes offset (y-intercept) next to the slope
"""

import logging
import time
from collections import OrderedDict

import numpy as np
import tables as tb
from bitarray import bitarray

import pixcap65_constants as c
from analysis import analyze_data
from configs.config_handler import extract_smu_current_error
from pixcap65 import pixcap65
from plotting import plot_data

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_handler = logging.FileHandler('pixcap_65_test.log')
log_formater = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
log_handler.setFormatter(log_formater)
logger.addHandler(log_handler)


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
    # float64 should be available now
    # fields.extend([(name, np.float64) for name in keys])
    fields.extend([(name, np.float32) for name in keys])

    scan_par_table = h5_file.create_table(h5_file.root, name='scan_params',
                                          title='Scan parameter values per scan parameter id',
                                          description=np.dtype(fields))
    for par_id, par_values in scan_parameters.items():
        a = np.full(shape=(1,), fill_value=np.nan).astype(np.dtype(fields))
        for key, val in par_values.items():
            a['scan_param_id'] = par_id
            a[key] = np.float32(val)
        scan_par_table.append(a)


scan_configuration = {
    'start_column': 0,
    'stop_column': 40,
    'start_row': 0,
    'stop_row': 40,

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

        self.n_frequencies = len(scan_config['frequency_range'])
        if "double_sweep" in scan_config and scan_config["double_sweep"]:
            self.n_frequencies *= 2
        self.hist_current = np.full(shape=(40, 40, self.n_frequencies),
                                    fill_value=np.nan)  # current value for each measured frequency per pixel
        self.n_measurements = scan_config.get("average_measurements", 8)
        self.hist_individual_currents = np.full(shape=(40, 40, self.n_frequencies, self.n_measurements), fill_value=np.nan)
        if "average_measurements" not in scan_config or scan_config["average_measurements"] < 1:
            self.n_measurements = -1

        self.current_sense_range = 0.00001
        with open("configs/2410_Range.yaml", "r") as f:
            import yaml
            self.smu_range_config = yaml.safe_load(f)
        self.hist_current_errors = np.full(shape=(40, 40, self.n_frequencies), fill_value=np.nan)
        self.scan_indices = np.ndindex((40, 40))
        # depending on the scan configuration it should be possible to swap the indices.
        # TODO: implement this

    def configure(self):
        self.seq_size = 4  # granularity of the clock sequencer

        # settings for sensor depletion source
        # self.init_bias_voltage(voltage=-80.0)

        self.init_smu()

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

        self.smu_on()

        # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
        logging.debug('Waiting for settling of SMU...')
        for i in range(0, 30):
            current = self.get_source_current()
            logging.debug('Current: {}'.format(current))
            time.sleep(1)

        # changed to simplify changes in the used SMU
        self.get_source_current()

    def scan(self):
        # select the group to write the analysis results to
        data_group = self.out_file_h5.root
        data_group._f_setattr('frequencies', self.n_frequencies)

        row_range = range(self.scan_config['start_row'], self.scan_config['stop_row'])
        col_range = range(self.scan_config['start_column'], self.scan_config['stop_column'])
        frequency_range = self.scan_config['frequency_range']

        # Addition by Dominik to perform also a down sweep in frequency
        if "double_sweep" in self.scan_config and self.scan_config["double_sweep"]:
            frequency_range = np.concatenate((frequency_range, np.flip(frequency_range)))
        if "average_measurements" in self.scan_config and self.scan_config["average_measurements"] > 1:
            n_measurements = self.scan_config["average_measurements"]
            # Added for convenience of the averaged measurements.
            # self.pixcap['SMU'].set_number_measurements(n_measurements, **self.smu_kwargs)
            individual_currents_shape = (40, 40, self.n_frequencies, n_measurements)
            if self.hist_individual_currents.shape != individual_currents_shape:
                self.hist_individual_currents = np.full(shape=(40, 40, self.n_frequencies, n_measurements), fill_value=np.nan)

            logging.info("Average over multiple measurements!")
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
                        self.hist_individual_currents[i_col, i_row, k, :] = self.get_source_current_multiple(
                            n_measurements)[:]
                        store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq)

            average_currents = np.nanmean(self.hist_individual_currents, axis=3, keepdims=True)
            self.hist_current = average_currents[:, :, :, 0]
            self.hist_current_errors = np.nanstd(self.hist_individual_currents, axis=3, mean=average_currents)
        else:
            logging.info('Scan pixel by single measurements.')
            logger.info('Scan pixel by single measurements.')
            for i_row in row_range:
                for i_col in col_range:
                    logging.info('Measuring pixel (%i, %i)...' % (i_col, i_row))
                    logger.info('Measuring pixel (%i, %i)...' % (i_col, i_row))
                    self.dut.disable_all_pixels()
                    self.dut.disable_all_columns()

                    self.dut.enable_column(i_col, c.EN_EOC_3)
                    self.dut.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)

                    for k, freq in enumerate(frequency_range):
                        freq_conv = freq * self.seq_size
                        self.dut['MIO_PLL'].setFrequency(freq_conv)
                        time.sleep(1)
                        current = self.get_source_current()
                        self.hist_current[i_col, i_row, k] = current
                        if np.isnan(current):
                            logging.warning(f'nan result for {i_col}, {i_row}, {k}')
                            logger.warning(f'nan result for {i_col}, {i_row}, {k}')
                        if (not np.isnan(current) and np.isnan(self.hist_current[i_col, i_row, k])):
                            logging.warning('There was a difference after saving the data.')
                            logger.warning('There was a difference after saving the data.')
                        if (k == 0):
                            logging.debug('%f' % (current))
                            logger.debug('%f' % (current))
                        store_scan_par_values(scan_parameters=self.scan_parameters, scan_param_id=k, frequency=freq)

        # Save raw data
        _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.scan_parameters)
        if np.all(np.isnan(self.hist_current)):
            raise Exception("UNEXPECTED: All measurement entries are still NaN.")
        self.out_file_h5.create_carray(data_group,
                                       name='HistCurr',
                                       title='Current Histogram',
                                       obj=self.hist_current,
                                       filters=tb.Filters(complib='blosc',
                                                          complevel=5,
                                                          fletcher32=False))

        # need the additional entries for the advanced averaging implementation
        if "average_measurements" in self.scan_config and self.scan_config["average_measurements"] > 1:
            self.out_file_h5.create_carray(data_group,
                                           name='HistCurrValues',
                                           title='Multiple Current Histogram',
                                           obj=self.hist_individual_currents,
                                           )
        else:
            try:
                self.hist_current_errors = extract_smu_current_error(self.smu_range_config, self.hist_current, self.current_sense_range)
            except Exception as e:
                logging.error(e.args)

        if hasattr(self, "hist_current_errors") and not np.all(np.isnan(self.hist_current_errors)):
            self.out_file_h5.create_carray(data_group,
                                           name='HistCurrErr',
                                           title='Current Error Histogram',
                                           obj=self.hist_current_errors,
                                           )


        # TODO: make it possible to directly export it also in a root tree.

        logging.info('Done')

    def close(self):
        self.out_file_h5.close()
        self.dut['SMU'].off(**self.smu_kwargs)
        self.dut.close()

    def __enter__(self):
        self.configure()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @property
    def pixcap(self):
        return self.dut

    # Handle the SMU!
    # these will now just forward the commands to the pixcap object
    def init_smu(self, voltage_range=1.5, current_limit=0.001, plc=10):
        self.pixcap.init_smu(self.scan_config['Vin'], self.current_sense_range, voltage_range, current_limit, plc)

    def smu_on(self):
        self.pixcap.smu_on()

    def get_source_current(self) -> float:
        return self.pixcap.get_source_current

    def get_source_current_multiple(self, n: int):
        return self.pixcap.get_source_current_multiple(n)

    def smu_off(self):
        self.pixcap.smu_off()

    # Handle the biasing supply
    def init_bias_voltage(self, voltage: float = -80.0):
        self.pixcap.init_bias_voltage(voltage)

    def set_bias_on(self):
        self.pixcap.set_bias_on()

    def set_bias_off(self):
       self.pixcap.set_bias_off()

    def set_bias_voltage(self, voltage: float):
        self.pixcap.bias_voltage = voltage

    @property
    def smu_kwargs(self):
        return self.pixcap.smu_kwargs


if __name__ == '__main__':
    output_file = "./TEST.h5"
    # try:
    #     pix = PixCap65TotalCap(scan_configuration, output_file)
    #     pix.configure()
    #     pix.scan()
    # finally:
    #     pix.close()

    output_file_2 = "./TEST_2.h5"
    with PixCap65TotalCap(scan_configuration, output_file_2) as pix:
        pix.scan()


    # Analyse and plot data
    # analyze_data(output_file)
    # plot_data(output_file)

    analyze_data(output_file_2)
    plot_data(output_file_2)
