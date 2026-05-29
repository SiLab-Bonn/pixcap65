"""
Script for measuring Inter Pixel Capacitance 
"""

import logging
import numpy as np
import tables as tb
import time

from pixcap65.analysis_util import GENERAL_PIXCAP_SHAPE
from pixcap65.analysis_util.utility import HIST_CURRENT_MEAS_UNIT
from pixcap65.pixcap_65_test_total_cap import PixCap65Measurement, MEASURING_PIXEL_TEXT, \
    _store_scan_par_values, ScanConfigurationKeys
from pixcap65.utility import pixcap65_constants as c
from pixcap65.utility.tables_util import set_group_attribute
from pixcap65.utility.tqdm_logging_utils import advanced_tqdm_iterator

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_handler = logging.FileHandler('pixcap_65_test.log')
log_formater = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s')
log_handler.setFormatter(log_formater)
logger.addHandler(log_handler)
logger.propagate = True

scan_configuration = {
    'start_column': 1,
    'stop_column': 38,
    'start_row': 2,
    'stop_row': 38,

    'Vin': 1.0,  # input voltage in V
    'frequency_range': np.arange(1.0, 6.1, 0.75),  # .astype(np.float) # [MHz]
    'bias': -80,

    'data_path': "Reference/R1",
    "out_file_mode": "append",
}


class InterCap(tb.IsDescription):
    col = tb.Int32Col(pos=0)
    row = tb.Int32Col(pos=1)
    inter_cap_a = tb.Float64Col(pos=2)
    inter_cap_b = tb.Float64Col(pos=3)
    total_cap_b = tb.Float64Col(pos=4)
    leakage_a = tb.Float64Col(pos=5)
    leakage_b = tb.Float64Col(pos=6)
    leakage_total = tb.Float64Col(pos=7)


class Pixcap65InterCap(PixCap65Measurement):
    def pre_scan_handler(self, unit):
        if self.averaging:
            self.n_measurements = self.scan_config[ScanConfigurationKeys.AVERAGE_MEASUREMENTS]
            individual_currents_shape = (*GENERAL_PIXCAP_SHAPE, self.n_frequencies, self.n_measurements)
            if self.total_hist_individual_currents.shape != individual_currents_shape:
                self.total_hist_individual_currents = np.full(shape=individual_currents_shape, fill_value=np.nan)
                self.inter_hist_individual_currents_2 = np.full(shape=individual_currents_shape, fill_value=np.nan)
                self.inter_hist_individual_currents_1 = np.full(shape=individual_currents_shape, fill_value=np.nan)

            self.mode_logging_text = "Average over multiple measurements!"
            self.handle_measurement = self._handle_averaged_measurement
        else:
            self.mode_logging_text = 'Scan pixel by single measurements.'
            self.handle_measurement = self._handle_single_measurement

    def __init__(self, scan_config, out_file, **kwargs):
        # FIXME: Issue with the naming convention for the output file!
        if "output_file" in kwargs:
            out_file = kwargs.pop("output_file")
        super(Pixcap65InterCap, self).__init__(scan_config, out_file, **kwargs)

        # prepare the data fields for the measurement
        self.inter_hist_current_1 = np.full(shape=(40, 40, self.n_frequencies),
                                            fill_value=np.nan)  # current value for each measured frequency per pixel
        self.inter_hist_current_2 = np.full(shape=(40, 40, self.n_frequencies),
                                            fill_value=np.nan)  # current value for each measured frequency per pixel
        self.total_hist_current = np.full(shape=(40, 40, self.n_frequencies),
                                          fill_value=np.nan)  # current value for each measured frequency per pixel

        # current value for each measured frequency per pixel
        self.inter_hist_current_1_error = np.full(shape=(40, 40, self.n_frequencies),
                                                  fill_value=np.nan)
        # current value for each measured frequency per pixel
        self.inter_hist_current_2_error = np.full(shape=(40, 40, self.n_frequencies),
                                                  fill_value=np.nan)
        # current value for each measured frequency per pixel
        self.total_hist_current_error = np.full(shape=(40, 40, self.n_frequencies),
                                                fill_value=np.nan)

        self.inter_hist_individual_currents_1 = np.full(shape=(40,40,self.n_frequencies, 1), fill_value=np.nan)
        self.inter_hist_individual_currents_2 = np.full(shape=(40, 40, self.n_frequencies, 1), fill_value=np.nan)
        self.total_hist_individual_currents = np.full(shape=(40, 40, self.n_frequencies, 1), fill_value=np.nan)


        # this kind of setup is somewhat misplaced.
        self.pixcap.seq_size = 4
        self.handle_measurement = self._handle_single_measurement

    def configure(self):
        # already done by super-class
        super(Pixcap65InterCap, self).configure()
        self.init_smu(smu=self.pixcap.vm2_smu_key, current_range=0.000001)
        self.init_smu(smu=self.pixcap.vm1_smu_key)

        # self.pixcap.seq_init(clk_0='0100', clk_1='0100', clk_2='0001', clk_3='0001')
        # simplify switch the order of the clocks for once.
        self.pixcap.seq_init(clk_0='0100', clk_1='0001', clk_2='0001', clk_3='0100')

        self.inter_hist_individual_currents_1 = np.full(shape=(40, 40, self.n_frequencies, self.n_measurements), fill_value=np.nan)
        self.inter_hist_individual_currents_2 = np.full(shape=(40, 40, self.n_frequencies, self.n_measurements), fill_value=np.nan)
        self.total_hist_individual_currents = np.full(shape=(40, 40, self.n_frequencies, self.n_measurements), fill_value=np.nan)

        self.pixcap.vm3_on()
        self.pixcap.vm2_on()
        self.pixcap.vm1_on()

        # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
        logging.debug('Waiting for settling of SMU...')
        for _ in range(0, 20):
            c3 = self.pixcap.vm3_measure_current()
            c2 = self.pixcap.vm2_measure_current()
            c1 = self.pixcap.vm1_measure_current()
            logging.debug('C3: {}'.format(c3))
            logging.debug('C2: {}'.format(c2))
            logging.debug('C1: {}'.format(c1))
            time.sleep(1)

    def scan(self, data_group_spec=None, sequence_call: bool = False):
        """
        scan
        Performs the scan over the pixels on the sensor and measures the requested quantities in dependence on some
        other quantities.
        Will scan the specified frequency range for each pixel specified by the scan configuration and measure the
        current at all three SMU channels to obtain information about the pixel capacitance and the inter-pixel
        capacitance.

        :param data_group_spec: specifier of the data group in hdf file where the measurements are stored.
        :param sequence_call: boolean to indicate whether this function is called in a sequence of scan calls from another scan procedure.
        """
        # some further setup to be done right before the measurement
        data_group = self.get_data_group(data_group_spec, "inter_cap")
        set_group_attribute(data_group, "frequencies", self.n_frequencies)

        for i_col, i_row in self.measurement_procedure(data_group, False):
            logging.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
            logger.info(MEASURING_PIXEL_TEXT % (i_col, i_row))

            self.pixcap.disable_all_pixels()
            self.pixcap.disable_all_columns()

            # enable columns of pixel under test and surrounding pixels
            self.pixcap.enable_column(i_col, c.EN_EOC_1 | c.EN_EOC_2 | c.EN_EOC_3)
            self.pixcap.enable_column(i_col + 1, c.EN_EOC_1 | c.EN_EOC_3)
            self.pixcap.enable_column(i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)

            # enable pixel under test
            self.pixcap.enable_pixel_clk(i_col, i_row, c.EN_CLK_2 | c.EN_CLK_0)

            # enable pixels surrounding pixel under test
            self.pixcap.enable_pixel_clk(i_col, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col + 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col + 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col + 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col - 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col - 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col - 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)

            # Which SMU takes which role here?
            for k, freq in enumerate(self.frequency_range):
                self.pixcap.cvm_frequency = freq
                self.verify_stable_current(self.pixcap.vm2_smu_key)

                # TODO: Absorb this into a measurement handler
                self.inter_hist_current_1[i_col, i_row, k] = self.pixcap.vm3_measure_current()

                # perhaps the wrong capacitance!
                self.total_hist_current[i_col, i_row, k] = self.pixcap.vm2_measure_current()

                # extract the inter pix current
                self.inter_hist_current_2[i_col, i_row, k] = self.pixcap.vm1_measure_current()

                self.store_iteration_parameters(freq, k)

    def store_measurement_data(self, data_group: tb.Group, sequence_call: bool, unit=None):
        # TODO 2026-05-14 dominikfischer: this make multiple measurements for the inter-pix capacitance measurement impossible
        if unit == "regular":
            _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.scan_parameters, group=data_group)
            self.create_carray(data_group, name='TotalHistCurr',
                               title='Current Histogram for the total capacitance measurement', obj=self.total_hist_current,
                               filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT, input="VM2")
            self.create_carray(data_group, name='TotalHistCurrErr',
                               title='Error Histogram of the current for the total capacitance measurement',
                               obj=self.total_hist_current_error, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT,
                               input="VM2")
            self.create_carray(data_group, name='InterHistCurrA',
                               title='Current Histogram for the inter capacitance measurement',
                               obj=self.inter_hist_current_1, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT,
                               input="VM3")
            self.create_carray(data_group, name='InterHistCurrErrA',
                               title='Error Histogram of inter current A for the inter capacitance measurement',
                               obj=self.inter_hist_current_1_error, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT,
                               input="VM3")
            self.create_carray(data_group, name='InterHistCurrB',
                               title='Current Histogram for the inter capacitance measurement',
                               obj=self.inter_hist_current_2, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT,
                               input="VM1")
            self.create_carray(data_group, name='InterHistCurrErrB',
                               title='Error Histogram of inter current B for the inter capacitance measurement',
                               obj=self.inter_hist_current_2_error, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT,
                               input="VM1")

    def handle_measurement_errors(self, unit=None):
        # make sure the measurement points will have uncertainties.
        self.inter_hist_current_1_error = self.determine_measurement_uncertainty(self.pixcap.vm3_smu_key,
                                                                                 self.inter_hist_current_1)
        self.inter_hist_current_2_error = self.determine_measurement_uncertainty(self.pixcap.vm1_smu_key,
                                                                                 self.inter_hist_current_2)
        self.total_hist_current_error = self.determine_measurement_uncertainty(self.pixcap.vm2_smu_key,
                                                                               self.total_hist_current)

    def _handle_single_measurement(self, col, row, k):
        self.inter_hist_current_1[col, row, k] = self.pixcap.vm3_measure_current()
        self.total_hist_current[col, row, k] = self.pixcap.vm2_measure_current()
        self.inter_hist_current_2[col, row, k] = self.pixcap.vm1_measure_current()

    def _handle_averaged_measurement(self, col, row, k):
        self.pixcap.vm3_initiate_multiple_current(self.n_measurements)
        self.pixcap.vm2_initiate_multiple_current(self.n_measurements)
        self.pixcap.vm1_initiate_multiple_current(self.n_measurements)

        self.inter_hist_individual_currents_1[col, row, k, :] = self.pixcap.vm3_read_multiple_current(self.n_measurements)
        self.inter_hist_individual_currents_2[col, row, k, :] = self.pixcap.vm1_read_multiple_current(self.n_measurements)
        self.total_hist_individual_currents[col, row, k, :] = self.pixcap.vm2_read_multiple_current(self.n_measurements)


    def analyze(self):
        pass

    def plot(self):
        pass

    def close(self):
        self.pixcap.vm1_off()
        self.pixcap.vm2_off()
        self.pixcap.vm3_off()
        super(Pixcap65InterCap, self).close()

    # region Pixcap Properties
    # specialized for the inter capacitance measurement.
    @property
    def col_range(self):
        """Get the range of columns to scan the pixels for. (specialized for inter-pix)"""
        return range(self.col_start, self.col_stop + 1)

    @property
    def row_range(self):
        """Get the range of rows to scan the pixels for. (specialized for inter-pix)"""
        return range(self.row_stop, self.row_start - 1, -1)

    # endregion

    @property
    def current_sense_range(self):
        return 0.000010
    # endregion


if __name__ == "__main__":
    output_file = "../RX-Interpixel_Scan.h5"
    # with Pixcap65InterCap(scan_configuration, output_file) as pix:
    # pix.scan(data_group_spec="demo_measurement_1_80_V")
    from pixcap65.utils import PixCapSetup

    del scan_configuration['bias']
    with PixCapSetup(scan_configuration, output_file, measurement=Pixcap65InterCap) as pix:
        pix.pixcap.frequency_settling = 0.4
        pix.scan(data_group_spec="demo_measurement_1_unbiased_1_discharge")

    scan_configuration['bias'] = -80
    with PixCapSetup(scan_configuration, output_file, measurement=Pixcap65InterCap) as pix:
        pix.pixcap.frequency_settling = 0.6
        pix.scan(data_group_spec="demo_measurement_2_biased_80_V_1_discharge")

    scan_configuration['bias'] = -40
    with PixCapSetup(scan_configuration, output_file, measurement=Pixcap65InterCap) as pix:
        pix.pixcap.frequency_settling = 0.4
        pix.scan(data_group_spec="demo_measurement_3_biased_40_V_1_discharge")

    # scan_configuration['bias'] = -5
    # with PixCapSetup(scan_configuration, output_file, measurement=Pixcap65InterCap) as pix:
    #     pix.pixcap.frequency_settling = 0.4
    #     pix.scan(data_group_spec="demo_measurement_4_biased_05_V_1_discharge")
