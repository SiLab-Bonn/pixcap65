"""
Script for measuring Inter Pixel Capacitance

Defines a data structures and measurement classes specifically designed for inter-pixel capacitance measurements
and if necessary their analysis.

The form of the measurement class was derived from the original inter-pixel measurement script
:py:mod:`legacy.pixcap_65_test_inter_cap`.
"""

import logging
import numpy as np
import tables as tb
import time
import warnings
from contextlib import contextmanager

from pixcap65.analysis import analysis_data_handle
from pixcap65.analysis_util import GENERAL_PIXCAP_SHAPE
from pixcap65.analysis_util.utility import HIST_CURRENT_MEAS_UNIT, handle_analysis_mix_up
from pixcap65.pixcap.pixcap65_measurement import ScanConfigurationKeys, declare_logger, NUMBER_INITIAL_MEASUREMENTS, \
    MeasurementAttributes, CapType, REDUCED_NPLC, STANDARD_NLPC
from pixcap65.pixcap_65_total_cap import PixCap65Measurement, MEASURING_PIXEL_TEXT, \
    _store_scan_par_values
from pixcap65.utility import pixcap65_constants as c
from pixcap65.utility.tables_util import set_group_attribute
from pixcap65.utility.utils_2 import walk_to_node

logger = declare_logger()

scan_configuration = {
    'start_column': 1,
    'stop_column': 38,
    'start_row': 2,
    'stop_row': 38,

    'Vin': 1.0,  # input voltage in V
    'frequency_range': np.arange(1.0, 6.1, 0.75),  # .astype(np.float) # [MHz]
    'bias': -80,

    'data_path': "Reference/Demo",
    "out_file_mode": "append",
}


class InterCap(tb.IsDescription):
    """
    :py:class:`pytables.IsDescription` data structure to store the results of a inter-pixel capacitance measurement and
    analysis in a :py:class:`pytables.Table` object with one row for each investigated pixel on the active sensor.

    Class variables are to be understand as fields/entries of the table.

    :cvar col: column coordinate of the investigated pixel.
    :vartype col: int
    :cvar row: row coordinate of the investigated pixel.
    :vartype row: int
    :cvar inter_cap_a: inter-pixel capacitance measurement on channel A (the capacitance)
    :vartype inter_cap_a: float
    :cvar inter_cap_b: inter-pixel capacitance measurement on channel B (the capacitance)
    :vartype inter_cap_b: float
    :cvar total_cap_b: total capacitance measurement (the capacitance)
    :vartype total_cap_b: float
    :cvar leakage_a: determined leakage current of channel A for this pixel
    :vartype leakage_a: float
    :cvar leakage_b: determined leakage current of channel B for this pixel
    :vartype leakage_b: float
    :cvar leakage_total: determined leakage current of the channels used for total cap estimation!
    :vartype leakage_total: float
    """
    col = tb.Int32Col(pos=0)
    row = tb.Int32Col(pos=1)
    inter_cap_a = tb.Float64Col(pos=2)
    inter_cap_b = tb.Float64Col(pos=3)
    total_cap_b = tb.Float64Col(pos=4)
    leakage_a = tb.Float64Col(pos=5)
    leakage_b = tb.Float64Col(pos=6)
    leakage_total = tb.Float64Col(pos=7)


class Pixcap65InterCap(PixCap65Measurement):
    """
    Measurement class for the inter-pixel capacitance measurement.
    The implemented procedure will only actively measure the in-pix or backplane capacitance (to be more precise:
    the currents for different frequencies in order to determine these capacitances).

    :ivar inter_hist_current_1: matrix to store the current measurements for each pixel and frequency for channel A to
        determine the inter-pixel capacitance.
    :ivar inter_hist_current_2: matrix to store the current measurements for each pixel and frequency for channel B to
        determine the inter-pixel capacitance.
    :ivar total_hist_current: matrix to store the current measurements for each pixel and frequency to
        determine the total capacitance.
    :ivar inter_hist_current_1_error: matrix to store the statistical uncertainty of the  current measurements for
        each pixel and frequency for channel A to determine the inter-pixel capacitance.
    :ivar inter_hist_current_2_error: matrix to store the statistical uncertainty of the the current measurements
        for each pixel and frequency for channel B to determine the inter-pixel capacitance.
    :ivar total_hist_current_error: matrix to store the statistical uncertainty of the current measurements
        for each pixel and frequency to determine the total capacitance.
    :ivar inter_hist_individual_currents_1: matrix to store the current measurements for each pixel and frequency
        for channel A to determine the inter-pixel capacitance. Only used when for each frequency and pixel multiple
        current measurements are performed to store all the measurement results.
    :ivar inter_hist_individual_currents_2: matrix to store the current measurements for each pixel and frequency
        for channel B to determine the inter-pixel capacitance.  Only used when for each frequency and pixel multiple
        current measurements are performed to store all the measurement results.
    :ivar total_hist_individual_currents: matrix to store the current measurements for each pixel and frequency to
        determine the total capacitance.  Only used when for each frequency and pixel multiple
        current measurements are performed to store all the measurement results.
    """
    def pre_scan_handler(self, unit=None):
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

    def __init__(self, scan_config, output_file, **kwargs):
        super(Pixcap65InterCap, self).__init__(scan_config, output_file, **kwargs)

        # prepare the data fields for the measurement
        measurements_shape = tuple([*GENERAL_PIXCAP_SHAPE, self.n_frequencies])
        self.inter_hist_current_1 = np.full(shape=measurements_shape,
                                            fill_value=np.nan)  # current value for each measured frequency per pixel
        self.inter_hist_current_2 = np.full(shape=measurements_shape,
                                            fill_value=np.nan)  # current value for each measured frequency per pixel
        self.total_hist_current = np.full(shape=measurements_shape,
                                          fill_value=np.nan)  # current value for each measured frequency per pixel

        # current value for each measured frequency per pixel
        self.inter_hist_current_1_error = np.full(shape=measurements_shape,
                                                  fill_value=np.nan)
        # current value for each measured frequency per pixel
        self.inter_hist_current_2_error = np.full(shape=measurements_shape,
                                                  fill_value=np.nan)
        # current value for each measured frequency per pixel
        self.total_hist_current_error = np.full(shape=measurements_shape,
                                                fill_value=np.nan)

        self.n_measurements = scan_config.get(ScanConfigurationKeys.AVERAGE_MEASUREMENTS, 1)
        measurement_values_shape = (*measurements_shape, 1)
        self.inter_hist_individual_currents_1 = np.full(shape=measurement_values_shape, fill_value=np.nan)
        self.inter_hist_individual_currents_2 = np.full(shape=measurement_values_shape, fill_value=np.nan)
        self.total_hist_individual_currents = np.full(shape=measurement_values_shape, fill_value=np.nan)

        # this kind of setup is somewhat misplaced.
        self.pixcap.seq_size = 4
        self.handle_measurement = self._handle_single_measurement

    def configure(self):
        """
        configure

        @author: Dominik Fischer
        last update: 2026-08-27

        Handling the configuration of the pixcap measurement object and the physical setup
        All information additionally required will be fetched from the scan configuration.

        In Addition to the basic configuration of the primary smu also the two additional SMUs are configured for the
        measurement of inter-pixel capacitances.
        For one of these SMUs a slightly higher current measurement range will be used.

        As a last step some measurements are performed with the connected SMUs to make sure that we get correct current
        values and prevent effects by initial oscillation effects of the SMU.
        """
        # already done by super-class
        super(Pixcap65InterCap, self).configure()
        self.init_smu(smu=self.pixcap.vm2_smu_key, current_range=self.total_current_sense_range)
        self.init_smu(smu=self.pixcap.vm1_smu_key)

        # self.pixcap.seq_init(clk_0='0100', clk_1='0100', clk_2='0001', clk_3='0001')
        # simplify switch the order of the clocks for once.
        self.pixcap.seq_init(clk_0='0100', clk_1='0001', clk_2='0001', clk_3='0100')

        measurement_values_shape = tuple([*GENERAL_PIXCAP_SHAPE, self.n_frequencies, self.n_measurements])
        self.inter_hist_individual_currents_1 = np.full(shape=measurement_values_shape, fill_value=np.nan)
        self.inter_hist_individual_currents_2 = np.full(shape=measurement_values_shape, fill_value=np.nan)
        self.total_hist_individual_currents = np.full(shape=measurement_values_shape, fill_value=np.nan)

        self.pixcap.vm3_on()
        self.pixcap.vm2_on()
        self.pixcap.vm1_on()

        # measure some current values; avoid measuring incorrect currents due to initial oscillation effects of SMU
        logging.debug('Waiting for settling of SMU...')
        for _ in range(0, NUMBER_INITIAL_MEASUREMENTS):
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
        global logger
        # some further setup to be done right before the measurement
        data_group = self.get_data_group(data_group_spec, CapType.INTER_PIXEL)
        set_group_attribute(data_group, MeasurementAttributes.N_FREQUENCIES, self.n_frequencies)

        for i_col, i_row in self.measurement_procedure(data_group, False):
            logging.info(MEASURING_PIXEL_TEXT % (i_col, i_row))
            logger.info(MEASURING_PIXEL_TEXT % (i_col, i_row))

            self.pixcap.disable_all_pixels()
            self.pixcap.disable_all_columns()

            self.enable_measurement_pixels(i_col, i_row)

            # Which SMU takes which role here?
            for k, freq in enumerate(self.frequency_range):
                self.pixcap.cvm_frequency = freq
                self.verify_stable_current(self.pixcap.vm2_smu_key)

                self.handle_measurement(i_col, i_row, k)

                self.store_iteration_parameters(freq, k)
                logger.info("Get the nlpc values")
                logger.info(self.pixcap[self.pixcap.vm2_smu_key].get_current_nlpc())
                logger.info(self.pixcap[self.pixcap.vm3_smu_key].get_current_nlpc())

    def enable_measurement_pixels(self, i_col: int, i_row: int):
        """
        enable_measurement_pixels

        @author: Dominik Fischer
        last update: 2026-08-27

        Enable the pixels which should be measured.
        These are in general the pixel to be measured for the in-pix measurement SMU (or its corresponding clock signal
        CLK 2), and a selection of directly neighbouring pixels.
        The selection of neighbouring pixels activated for CLK1 and CLK3 might depend on the actual implementation by
        further subclasses.

        :param i_col: column coordinate of the pixel we are interested in.
        :param i_row: row coordinate of the pixel we are interested in.
        """
        # enable columns of pixel under test and surrounding pixels
        self.pixcap.enable_column(i_col, c.EN_EOC_1 | c.EN_EOC_2 | c.EN_EOC_3)
        self.pixcap.enable_column(i_col + 1, c.EN_EOC_1 | c.EN_EOC_3)
        self.pixcap.enable_column(i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)

        # enable pixel under test
        self.pixcap.enable_pixel_clk(i_col, i_row, c.EN_CLK_2 | c.EN_CLK_0)

        # enable pixels surrounding pixel under test
        if self.scan_config.get("Inter_Pix_diagonals", True):
            self.pixcap.enable_pixel_clk(i_col + 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col + 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col - 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col - 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
        if self.scan_config.get("Inter_Pixel_Sides", True):
            self.pixcap.enable_pixel_clk(i_col, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
        if self.scan_config.get("Inter_Pixel_Tops", True):
            self.pixcap.enable_pixel_clk(i_col + 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
            self.pixcap.enable_pixel_clk(i_col - 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)

    def store_measurement_data(self, data_group: tb.Group, sequence_call: bool, unit=None):
        """
        store_measurement_data

        :author: Dominik Fischer
        :date: 2026-05-14

        last update: 2026-08-27

        Stores the measured data into a hdf file.
        There will be individual arrays/matrices for each channel and separated for currents and their uncertainties.
        In `regular` case (must be provided by `unit`) these are written directly from the measurement if no multiple
        current measurements are available.
        If multiple measurements are available the uncertainty is calculated from the standard deviation of the
        measurements.

        There are no further implementations besides the regular one.

        :param data_group: hdf file's group where the data should be stored.
        :param sequence_call: boolean, indicating this is called from another scan.
        :param unit: measurement mode used.
        """
        try:
            if unit == "regular":
                self.create_carray(data_group, name='TotalHistCurr',
                                   title='Current Histogram for the total capacitance measurement',
                                   obj=self.total_hist_current,
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
                                   obj=self.inter_hist_current_1_error, filters=self.filters,
                                   unit=HIST_CURRENT_MEAS_UNIT,
                                   input="VM3")
                self.create_carray(data_group, name='InterHistCurrB',
                                   title='Current Histogram for the inter capacitance measurement',
                                   obj=self.inter_hist_current_2, filters=self.filters, unit=HIST_CURRENT_MEAS_UNIT,
                                   input="VM1")
                self.create_carray(data_group, name='InterHistCurrErrB',
                                   title='Error Histogram of inter current B for the inter capacitance measurement',
                                   obj=self.inter_hist_current_2_error, filters=self.filters,
                                   unit=HIST_CURRENT_MEAS_UNIT,
                                   input="VM1")
                if self.is_unit_averaging(unit):
                    self.create_carray(data_group, name="TotalHistCurrValues",
                                       title='Multiple current histogram for the total capacitance measurement',
                                       obj=self.total_hist_individual_currents, filters=self.filters,
                                       unit=HIST_CURRENT_MEAS_UNIT, input="VM2")
                    self.create_carray(data_group, name="InterHistCurrValuesA",
                                       title='Multiple current histogram for the inter capacitance measurement A',
                                       obj=self.inter_hist_individual_currents_1, filters=self.filters,
                                       unit=HIST_CURRENT_MEAS_UNIT, input="VM3")
                    self.create_carray(data_group, name="InterHistCurrValuesA",
                                       title='Multiple current histogram for the inter capacitance measurement B',
                                       obj=self.inter_hist_individual_currents_2, filters=self.filters,
                                       unit=HIST_CURRENT_MEAS_UNIT, input="VM1")
        finally:
            assert isinstance(data_group, tb.Group)
            if unit == "regular":
                _store_scan_par_values(h5_file=self.out_file_h5, scan_parameters=self.scan_parameters, group=data_group)

    def handle_measurement_errors(self, unit=None):
        if self.averaging:
            inter_average_currents_1 = np.nanmean(self.inter_hist_individual_currents_1, axis=3, keepdims=True)
            self.inter_hist_current_1 = inter_average_currents_1[:, :, :, 0]
            self.inter_hist_current_1_error = np.nanstd(self.inter_hist_individual_currents_1, axis=3,
                                                        mean=inter_average_currents_1)
            inter_average_currents_2 = np.nanmean(self.inter_hist_individual_currents_2, axis=3, keepdims=True)
            self.inter_hist_current_2 = inter_average_currents_2[:, :, :, 0]
            self.inter_hist_current_2_error = np.nanstd(self.inter_hist_individual_currents_2, axis=3,
                                                        mean=inter_average_currents_2)
            total_average_currents = np.nanmean(self.total_hist_individual_currents, axis=3, keepdims=True)
            self.total_hist_current = total_average_currents[:, :, :, 0]
            self.total_hist_current_error = np.nanstd(self.total_hist_individual_currents, axis=3,
                                                        mean=total_average_currents)

        else:
            # make sure the measurement points will have uncertainties.
            self.inter_hist_current_1_error = self.determine_measurement_uncertainty(self.pixcap.vm3_smu_key,
                                                                                     self.inter_hist_current_1,
                                                                                     sense_range=self.current_sense_range)
            self.inter_hist_current_2_error = self.determine_measurement_uncertainty(self.pixcap.vm1_smu_key,
                                                                                     self.inter_hist_current_2,
                                                                                     sense_range=self.total_current_sense_range)
            self.total_hist_current_error = self.determine_measurement_uncertainty(self.pixcap.vm2_smu_key,
                                                                                   self.total_hist_current,
                                                                                   sense_range=self.total_current_sense_range)

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

    def analyze(self, data_group_spec=None, **kwargs):
        # handle deprecated keyword arguments.
        correction_key_value = kwargs.pop("use_corrected", None)
        if correction_key_value is not None:
            msg = "keyword argument `use_corrected` is deprecated, use `apply_correction` instead. If `apply_correction` is also present this value will take precedence, otherwise the provided value will be used. This keyword argument will be removed in the future."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            kwargs.setdefault("apply_correction", correction_key_value)

        # handle the additional PDF file in case of plotting enabled
        fit_plot_pdf_name = kwargs.pop('fit_plot_pdf_name', None)
        if "plot" in kwargs and kwargs["plot"] and fit_plot_pdf_name is not None:
            from matplotlib.backends.backend_pdf import PdfPages
            assert "fit_plot_pdf_name" not in kwargs
            with PdfPages(fit_plot_pdf_name) as pdf:
                kwargs['fit_plot_pdf'] = pdf
                self.analyze(data_group_spec=data_group_spec, **kwargs)
                return

        kargs = kwargs.copy()
        kargs.pop("plot", None)
        kargs.pop("fit_plot_pdf", None)
        kargs.pop("use_kafe2", None)
        kargs.pop("output_pdf", None)
        kargs['no_plot'] = True

        # extract the hdf file groups to perform the analysis on.
        base_group, _ = walk_to_node(self.base_group, data_group_spec, create=False, verify_create=True)

        # special handling for C-V characterization.
        reference_group = base_group.inter_cap

        handle_analysis_mix_up(reference_group)
        ana_group, _ = walk_to_node(reference_group, "analysis", create=True, verify_create=True)
        assert isinstance(ana_group, tb.Group)
        analysis_data_handle(self.out_file_h5, reference_group.measurements, ana_group, is_inter_pixel=True, **kwargs)

    def plot(self, data_group_spec=None, **kwargs):
        from matplotlib.backends.backend_pdf import PdfPages
        from pixcap65.plotting import get_analysis_group
        from pixcap65.plotting_util.inter_pixel import plot_inter_pix_data_delegate
        from pixcap65.plotting_util.utility import get_pdf_name

        suffix = kwargs.pop("suffix", "general_data_intern")
        if kwargs.get("use_corrected", False):
            suffix = "{}_corrected".format(suffix)
        use_group = kwargs.pop("use_group", False)

        # determine the pdf file
        pdf_name = get_pdf_name(data_group_spec, self.output_file, suffix, use_group)
        with PdfPages(pdf_name) as output_pdf:
            plot_group, _ = walk_to_node(self.base_group, data_group_spec, create=False, verify_create=True)
            plot_inter_pix_data_delegate(plot_group.inter_cap.measurements,
                               get_analysis_group(plot_group.inter_cap, **kwargs),
                               output_pdf, **kwargs)

    def close(self):
        self.pixcap.vm1_off()
        self.pixcap.vm2_off()
        self.pixcap.vm3_off()
        super(Pixcap65InterCap, self).close()

    def storage_exception_handler(self, temp_id):
        np.save("error_storage_bias_currents_{}".format(temp_id), self.hist_bias_current)
        np.save("error_storage_bias_current_errors_{}".format(temp_id), self.hist_bias_current_errors)
        np.save("error_storage_bias_currents_individual_{}".format(temp_id), self.hist_bias_individual_currents)
        with open("error_storage_configuration_{].yaml".format(temp_id), 'w') as f:
            import yaml
            yaml.safe_dump(self.scan_config, f)
        with open("error_storage_scan_parameters_{}.yaml".format(temp_id), 'w') as f:
            import yaml
            yaml.safe_dump(self.scan_parameters, f)
        if hasattr(self, 'bias_scan_parameters'):
            with open("error_storage_bias_parameters_{}.yaml".format(temp_id), 'w') as f:
                import yaml
                yaml.safe_dump(self.bias_scan_parameters, f)

        np.save("error_storage_currents_total_{}".format(temp_id), self.total_hist_current)
        np.save("error_storage_current_errors_total_{}".format(temp_id), self.total_hist_current_error)
        np.save("error_storage_individual_currents_total_{}".format(temp_id), self.total_hist_individual_currents)

        np.save("error_storage_currents_inter_a_{}".format(temp_id), self.inter_hist_current_1)
        np.save("error_storage_current_errors_inter_a_{}".format(temp_id), self.inter_hist_current_1_error)
        np.save("error_storage_individual_currents_inter_a_{}".format(temp_id), self.inter_hist_individual_currents_1)

        np.save("error_storage_currents_inter_b_{}".format(temp_id), self.inter_hist_current_2)
        np.save("error_storage_current_errors_inter_b_{}".format(temp_id), self.inter_hist_current_2_error)
        np.save("error_storage_individual_currents_inter_b_{}".format(temp_id), self.inter_hist_individual_currents_2)

    @contextmanager
    def enhanced_readout_mode(self):
        try:
            if self.n_measurements > 5:
                self.pixcap[self.pixcap.vm2_smu_key].set_current_nlpc(REDUCED_NPLC)
                self.pixcap[self.pixcap.vm3_smu_key].set_current_nlpc(REDUCED_NPLC)
            yield self
        finally:
            self.pixcap[self.pixcap.vm2_smu_key].set_current_nlpc(STANDARD_NLPC)
            self.pixcap[self.pixcap.vm3_smu_key].set_current_nlpc(STANDARD_NLPC)

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

    @property
    def current_sense_range(self):
        """Current sense range used for the inter-pixel capacitance measurement channels."""
        return 0.000010
    # endregion

    @property
    def total_current_sense_range(self):
        """Current sense range used for the total-pixel capacitance measurement channels."""
        return 0.000001


class Pixcap65InterCapSides(Pixcap65InterCap):
    """
    Implementation to measure the inter-pixel capacitance to the neighbour pixels to the sides.
    This means along the `x-axis`.
    Still only direct neighbours are considered.
    """
    def enable_measurement_pixels(self, i_col: int, i_row: int):
        # enable columns of pixel under test and surrounding pixels
        self.pixcap.enable_column(i_col, c.EN_EOC_1 | c.EN_EOC_2 | c.EN_EOC_3)

        # enable pixel under test
        self.pixcap.enable_pixel_clk(i_col, i_row, c.EN_CLK_2 | c.EN_CLK_0)

        # enable pixels surrounding pixel under test
        self.pixcap.enable_pixel_clk(i_col, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
        self.pixcap.enable_pixel_clk(i_col, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)


class Pixcap65InterCapTops(Pixcap65InterCap):
    """
    Implementation to measure the inter-pixel capacitance to the neighbour pixels to the tops.
    This means along the `y-axis`.
    Still only direct neighbours are considered.
    """
    def enable_measurement_pixels(self, i_col: int, i_row: int):
        # enable columns of pixel under test and surrounding pixels
        self.pixcap.enable_column(i_col, c.EN_EOC_1 | c.EN_EOC_2 | c.EN_EOC_3)
        self.pixcap.enable_column(i_col + 1, c.EN_EOC_1 | c.EN_EOC_3)
        self.pixcap.enable_column(i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)

        # enable pixel under test
        self.pixcap.enable_pixel_clk(i_col, i_row, c.EN_CLK_2 | c.EN_CLK_0)

        # enable pixels surrounding pixel under test
        self.pixcap.enable_pixel_clk(i_col + 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)
        self.pixcap.enable_pixel_clk(i_col - 1, i_row, c.EN_CLK_1 | c.EN_CLK_3)


class Pixcap65InterCapDiagonals(Pixcap65InterCap):
    """
    Implementation to measure the inter-pixel capacitance to the neighbouring pixels along the diagonals of the grid.
    Still only direct neighbours are considered.
    """
    def enable_measurement_pixels(self, i_col: int, i_row: int):
        # enable columns of pixel under test and surrounding pixels
        self.pixcap.enable_column(i_col, c.EN_EOC_1 | c.EN_EOC_2 | c.EN_EOC_3)
        self.pixcap.enable_column(i_col + 1, c.EN_EOC_1 | c.EN_EOC_3)
        self.pixcap.enable_column(i_col - 1, c.EN_EOC_1 | c.EN_EOC_3)

        # enable pixel under test
        self.pixcap.enable_pixel_clk(i_col, i_row, c.EN_CLK_2 | c.EN_CLK_0)

        # enable pixels surrounding pixel under test
        self.pixcap.enable_pixel_clk(i_col + 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)
        self.pixcap.enable_pixel_clk(i_col + 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
        self.pixcap.enable_pixel_clk(i_col - 1, i_row - 1, c.EN_CLK_1 | c.EN_CLK_3)
        self.pixcap.enable_pixel_clk(i_col - 1, i_row + 1, c.EN_CLK_1 | c.EN_CLK_3)


if __name__ == "__main__":
    output_file = "test.h5"
    with Pixcap65InterCap(scan_configuration, output_file) as pix:
        pix.scan(data_group_spec="demo_measurement")
