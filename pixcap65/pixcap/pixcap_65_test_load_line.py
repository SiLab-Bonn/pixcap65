"""
Script for measuring the load line of a test capacitor
Outputs in txt file the number of bits m that were set to 1 and the corresponding measured current 
Calculate t_charge with t_charge = m/seq_size * 1/f_rep
"""

# TODO: Analysis and plotting of this kind of measurement is still missing.
# we need to plot the behaviour of the current for different frequencies.

import logging
import numpy as np
import tables as tb
import time
from bitarray import bitarray

from pixcap65.analysis_util.utility import HIST_CURRENT_MEAS_UNIT
from pixcap65.pixcap.pixcap65_measurement import ScanConfigurationKeys, CapType, MeasurementAttributes
from pixcap65.pixcap_65_total_cap import PixCap65Measurement, _store_scan_par_values
from pixcap65.utility import pixcap65_constants as c
from pixcap65.utility.tables_util import set_group_attribute

LOAD_LINE_SEQ_SIZE = 128

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# this will only investigate a particular pixel!
# noinspection SpellCheckingInspection
scan_configuration = {
    'start_column': 12,
    'stop_column': 12,
    'start_row': 0,
    'stop_row': 0,

    'Vin': 1.0,  # input voltage in V
    # 'frequency_range': np.arange(1, 4.1, 1)  # .astype(np.float) # [MHz]
    'frequency_range': [1.0]
}


class Pixcap65LoadLine(PixCap65Measurement):
    """
    Measurement class for line-loading tests with the PixCap65 chip.
    Could measure the average current in dependence on the charging time and switching-frequency.
    """
    def handle_measurement_errors(self, unit):
        if self.averaging:
            average_currents = np.nanmean(self.hist_current_values, axis=4, keepdims=True)
            self.hist_current = average_currents[:, :, :, :, 0]
            self.hist_current_errors = np.nanstd(self.hist_current_values, axis=4, mean=average_currents)
        else:
            self.hist_current_errors = self.determine_measurement_uncertainty(self.pixcap.primary_smu_key,
                                                                              self.hist_current)

    def store_measurement_data(self, data_group, sequence_call, unit=None):
        try:
            if "HistCurr" in data_group:
                data_group.HistCurr[:] = self.hist_current[:]
                if np.any(np.isfinite(self.hist_current_errors)):
                    data_group.HistCurrErr[:] = self.hist_current_errors[:]
            else:
                self.create_carray(where=data_group, name="HistCurr", title="Current Histogram",
                                   obj=self.hist_current, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT)
                if np.any(np.isfinite(self.hist_current_errors)):
                    self.create_carray(data_group, name='HistCurrErr', title='Current Error Histogram',
                                       obj=self.hist_current_errors, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT)

            if self.averaging:
                if "HistCurrValues" in data_group:
                    data_group.HistCurrValues[:] = self.hist_current_values[:]
                else:
                    self.create_carray(data_group, name='HistCurrValues', title='Multiple Current Histogram',
                                       obj=self.hist_current_values, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT)
        finally:
            assert isinstance(data_group, tb.Group)
            _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.scan_parameters, group=data_group)
            self.out_file_h5.flush()

    def __init__(self, scan_config, out_file):
        # granularity of the clock sequencer must be set upfront.
        self.seq_size = LOAD_LINE_SEQ_SIZE
        super(Pixcap65LoadLine).__init__(scan_config, out_file)

        # prepare the measurement fields
        self.hist_current = np.full(shape=(40, 41, self.n_frequencies + 1, int(self.seq_size / 2 - 1)),
                                    fill_value=np.nan)
        self.hist_current_errors = np.full(shape=(40, 41, self.n_frequencies + 1, int(self.seq_size / 2 - 1)),
                                           fill_value=np.nan)
        self.hist_current_values = np.full(
            shape=(40, 41, self.n_frequencies + 1, int(self.seq_size / 2 - 1), self.n_measurements),
            fill_value=np.nan)

        self.handle_measurement = self._handle_single_measurement
        self.cnt = 0

    def update_config(self, new_config=None):
        super(Pixcap65LoadLine).update_config(new_config)
        self.hist_current = np.full(shape=(40, 41, self.n_frequencies + 1, int(self.seq_size / 2 - 1)),
                                    fill_value=np.nan)
        self.hist_current_errors = np.full(shape=(40, 41, self.n_frequencies + 1, int(self.seq_size / 2 - 1)),
                                           fill_value=np.nan)
        self.hist_current_values = np.full(
            shape=(40, 41, self.n_frequencies + 1, int(self.seq_size / 2 - 1), self.n_measurements),
            fill_value=np.nan)

    def scan(self, data_group_spec=None, sequence_call=False):
        data_group = self.get_data_group(data_group_spec, CapType.TOTAL_PIXEL)
        set_group_attribute(data_group, MeasurementAttributes.N_FREQUENCIES, self.n_frequencies)

        # not handled by `configure` as these are adapted by the measurement.
        m = self.seq_size / 2 - 1  # define index of the last 1 in order to create a non-overlapping clock sequence
        assert isinstance(m, int), "m has to be an integer!"
        self.cnt = self.seq_size / 2 - 1  # counter for numbering in output file

        # create initial bit arrays for a given sequencer size
        bit_array_clk_3 = bitarray(self.seq_size)
        bit_array_clk_3.setall(0)
        bit_array_clk_3[1:m] = 1

        bit_array_clk_0 = bitarray(self.seq_size)
        bit_array_clk_0.setall(0)
        bit_array_clk_0[m + 1:-1] = 1

        # vary charging time by looping over the number of bits in bit_array_clk_3 that are set to 1;
        # number is reduced by one in every step
        for i in range(m, 0, -1):
            self.pixcap.seq_init(clk_0=bit_array_clk_0, clk_3=bit_array_clk_3)

            # maybe this step does not have to be within the loop over m
            # did not know if SMU has to be set on after changing the clock sequencer
            self.smu_on()
            self.get_source_current()

            i_row = 0
            for i_col, i_row in self.measurement_procedure(data_group, sequence_call,
                                                           reversed_order=self.scan_config(ScanConfigurationKeys.INVERT,
                                                                                           False)):
                # handle enabled pixels
                self.pixcap.disable_all_pixels()
                self.pixcap.disable_all_columns()
                self.pixcap.enable_column(i_col, c.EN_EOC_3)
                time.sleep(1)
                self.pixcap.enable_pixel_clk(i_col, i_row, c.EN_CLK_0 | c.EN_CLK_3)

                # perform the measurement.
                for k, freq in enumerate(self.frequency_range):
                    self.pixcap.cvm_frequency = freq

                    # no verification of averaged current stability!
                    self.handle_measurement(i_col, i_row, k)
                    self.store_iteration_parameters(freq, k)

            logger.info(bit_array_clk_3)
            logger.info(self.hist_current[self.col_range[-1], i_row, :, self.cnt - 1])

            # reduce charging time with every iteration by setting last bit 1 -> 0 and decrement counter
            bit_array_clk_3[i] = 0
            self.cnt = self.cnt - 1

    def _handle_single_measurement(self, col, row, k):
        result = self.get_source_current()
        # Why this special format?
        self.hist_current[col, row, 0, self.cnt - 1] = self.cnt

        # not necessary anymore, as these is handled by the measurement function
        self.hist_current[col, row, k + 1, self.cnt - 1] = result

    def _handle_averaged_measurement(self, col, row, k):
        result = self.pixcap.get_advanced_current_multiple(self.n_measurements)[:]
        self.hist_current_values[col, row, 0, self.cnt - 1, :] = self.cnt
        self.hist_current_values[col, row, k + 1, self.cnt - 1, :] = result

    def close(self):
        self.smu_off()
        super(Pixcap65LoadLine, self).close()
        logger.debug("Done and closed the pixcap system.")

    def analyze(self):
        logger.info("There is nothing to analyze for the load line test.")
        # how to analyse all of this?
        # is there any sense in investigating the capacitance in this case?
        # of course we could use the standard procedure for that, but would it help at all?

    def plot(self):
        logger.info("There is nothing to plot for the load line test.")
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        # we will need one plot per frequency handled and then current in dependence an that also for each pixel measured;
        # perhaps print all the different frequencies into just a single plot?
        with PdfPages(self.output_file[:-3] + '_load_line.pdf') as pdf_file:
            # need to iterate over-all the pixels
            for row, col in np.ndindex((40,41)):
                fig, ax = plt.subplots()

                if not np.isfinite(self.hist_current[col, row, 1, 0]):
                    continue

                # need to iterate over all the frequencies to create line plots from them.
                for k, frequency in enumerate(self.frequency_range):
                    ax.errorbar(self.hist_current[col, row, 0, :], self.hist_current[col, row, k + 1, :],label="f={}".format(frequency))

                pdf_file.savefig(fig, bbox_inches='tight')



    def storage_exception_handler(self, temp_id):
        with open("error_storage_configuration_{}_line_test.yaml".format(temp_id), 'w') as f:
            import yaml
            yaml.safe_dump(self.scan_config, f)
        with open("error_storage_scan_parameters_{}_line_test.yaml".format(temp_id), 'w') as f:
            import yaml
            yaml.safe_dump(self.scan_parameters, f)

        np.save("error_storage_currents_{}_line_test".format(temp_id), self.hist_current)
        np.save("error_storage_current_errors_{}_line_test".format(temp_id), self.hist_current_errors)
        np.save("error_storage_individual_currents_{}_line_test".format(temp_id), self.hist_individual_currents)

    # properties of the measurement class
    @property
    def row_range(self):
        return range(self.row_stop, self.row_start - 1, -1)

    @property
    def col_range(self):
        return range(self.col_start, self.col_stop + 1)


if __name__ == "__main__":
    output_file = "./pixcap_full_data_image1.h5"
    with Pixcap65LoadLine(scan_configuration, output_file) as pix:
        pix.scan()

        # but the analysis and plotting procedures are not included within the legacy script.
        pix.analyze()
        pix.plot()
