from typing import Iterable

import numpy as np
import tables as tb
import time
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analysis import analyze_data
from analysis_util.utility import get_base_group
from pixcap65.pixcap_65_test_total_cap import scan_configuration, NUMBER_AVERAGE_MEASUREMENTS_KEY, PixCap65TotalCap
from utility.utils_2 import create_carray
from utils import PixCapSetup


def test_frequency_settling_2(settling_range: Iterable, file_name: str):
    configuration = {
        'start_column': 20,
        'stop_column': 24,
        'start_row': 20,
        'stop_row': 24,
        "average_measurements": 5,

        'Vin': 1.0,  # input voltage in V
        'frequency_range': np.arange(1, 4.1, 0.5),  # frequency sweep in MHz

        'data_path': "Equipment/Test/frequency",
        "out_file_mode": "append",
    }
    settling_range = np.asarray(settling_range)
    settling_path = {}
    with PixCapSetup.PixCapSetup(configuration) as pix:
        for settling_time in settling_range:
            pix.pixcap.frequency_settling = settling_time
            settling_path[settling_time] = "settling-time-{}".format(settling_time).replace('.', '_')
            try:
                pix.pixcap.binary_active = True
                pix._smu_setup(pix.pixcap.primary_smu_key).binary_format()
                pix.pixcap[pix.pixcap.primary_smu_key].set_current_nlpc(10)
                pix.pixcap.scan(data_group_spec=settling_path[settling_time])
            finally:
                pix.pixcap[pix.pixcap.primary_smu_key].set_current_nlpc(10)
                pix._smu_setup(pix.pixcap.primary_smu_key).text_format()
                pix.pixcap.binary_active = False

    # perform the analysis of these different paths.
    with PdfPages(file_name[:-3] + '.pdf') as output_pdf:
        for settling_time in settling_range:
            analyze_data(raw_data=file_name, base_path=configuration['data_path'] + settling_path[settling_time],
                         is_advanced=True, full_model=True, plot=True, apply_contour=True, fit_plot_pdf=output_pdf)

    # extract the dependencies for the chip from the measured error
    error_data = np.full((40, 40, settling_range.shape[0]), fill_value=np.nan)
    with tb.open_file(file_name, 'a') as h5_file, PdfPages(file_name[:-3] + "comparison" + '.pdf') as output_pdf:
        create_carray(h5_file, h5_file.root, "HistSettlingTime", obj=settling_range)
        for k, settling_time in enumerate(settling_range):
            group = get_base_group(configuration['data_path'] + settling_path[settling_time], h5_file)
            sensor_error_data = group.total_cap.analysis.HistCapErr[:]
            error_data[:, :, k] = sensor_error_data

        create_carray(h5_file, h5_file.root, "HistErrSettling", obj=error_data)

        # last plot the dependency for every pixel
        for ii, jj in np.ndindex((40, 40)):
            fig, ax = plt.subplots()
            ax.plot(settling_range, error_data[ii, jj, :])
            ax.set_title(f"Frequency settling results for pixel ({ii}, {jj}).")
            ax.set_xlabel("Settling time in s")
            ax.set_ylabel("Error of C")
            output_pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def test_frequency_settling(settling_range: Iterable, file_name: str):
    special_config = scan_configuration.copy()
    special_config["double_sweep"] = True
    special_config[NUMBER_AVERAGE_MEASUREMENTS_KEY] = 10
    with PixCap65TotalCap(special_config, file_name) as pix:
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib import cm
        cmap = cm.get_cmap('viridis')
        with PdfPages("Settling_Measurement.pdf") as output_pdf:
            for t in settling_range:
                settling_name = f"{t}_frequency_settling".replace(" ", "__").replace("-", "_")
                pix.pixcap.frequency_settling = t
                pix.scan(data_group_spec=settling_name)
            for t in settling_range:
                print(f"Testing the settling time for {t}")
                settling_name = f"{t}_frequency_settling".replace(" ", "__").replace("-", "_")
                # prepare the plotting here
                # Read pixel map
                current_hist = pix.out_file_h5.root[settling_name].total_cap.measurements.HistCurr[:]
                # Read scan parameters
                scan_parameters = pix.out_file_h5.root[settling_name].total_cap.measurements.scan_params[:]
                frequencies = np.asarray(scan_parameters['frequency'])
                n_full_freq = frequencies.shape[0]
                n_freq = n_full_freq // 2
                first_frequencies = frequencies[:n_freq]
                second_frequencies = frequencies[n_freq:]
                for col in range(0, current_hist.shape[0]):
                    for row in range(0, current_hist.shape[1]):
                        if np.isfinite(current_hist[col, row, 0]):
                            fig, ax = plt.subplots()
                            ax.plot(first_frequencies, current_hist[col, row, :n_freq] * 1e9, marker='o', ls='',
                                    label='Pixel({i_col},{i_row}) 1'.format(i_col=col, i_row=row), color=cmap(0.2))
                            ax.plot(second_frequencies, current_hist[col, row, n_freq:] * 1e9, marker='o', ls='',
                                    label='Pixel({i_col},{i_row}) 2'.format(i_col=col, i_row=row), color=cmap(0.6))
                            print("operating for pixel {i_col},{i_row}".format(i_col=col, i_row=row))
                            print(current_hist[col, row, :n_freq] - np.flip(current_hist[col, row, n_freq:]))
                            print(frequencies[:n_freq] - np.flip(frequencies[n_freq:]))
                            print(pix.out_file_h5.root[settling_name].total_cap.measurements.HistCurrValues[
                                      col, row, 1, :])
                            print(
                                pix.out_file_h5.root[settling_name].total_cap.measurements.HistCurrValues[
                                    col, row, -2, :])
                            ax.set_ylabel('Current / nA')
                            ax.set_xlabel('Frequency / MHz')
                            ax.set_title(f"Frequency settling test for currents and settling time {t}")
                            ax.legend()
                            ax.grid()
                            output_pdf.savefig(fig, bbox_inches='tight')


def test_source_settling(settling_range: Iterable, file_name: str):
    settling_times = np.asarray(settling_range)
    with PixCap65TotalCap(scan_configuration, file_name) as pix:
        pix.pixcap.bias_voltage = -0.1
        pix.pixcap.bias_on()
        for _ in range(30):
            pix.pixcap.bias_measure_current()
            time.sleep(1)
        pix.pixcap[pix.pixcap.bias_smu_key].set_number_triggers(1)
        pix.pixcap[pix.pixcap.bias_smu_key].set_number_measurements(10)
        pix.pixcap[pix.pixcap.bias_smu_key].set_number_triggers(10)
        current_data = np.full_like(settling_times, fill_value=np.nan)
        for k, t in enumerate(settling_times):
            pix.pixcap.source_settling_time = t
            pix.pixcap[pix.pixcap.bias_smu_key].set_number_triggers(1)
            pix.pixcap[pix.pixcap.bias_smu_key].set_number_measurements(10)
            pix.pixcap[pix.pixcap.bias_smu_key].set_number_triggers(10)
            pix.pixcap.bias_voltage = -20
            start = time.time()
            result = pix.pixcap.bias_advanced_current_multiple(10)[1::2]
            print(f"It takes {time.time() - start} seconds.")
            print(result)
            pix.pixcap.bias_voltage = -0.1
            time.sleep(1)
            pix.pixcap[pix.pixcap.bias_smu_key].set_number_triggers(1)
            pix.pixcap[pix.pixcap.bias_smu_key].set_number_measurements(1)
            pix.pixcap[pix.pixcap.bias_smu_key].set_number_triggers(1)
            pix.pixcap.bias_voltage = -20
            pix.pixcap.bias_measure_current()
            current_data[k] = pix.pixcap.bias_measure_current()
            pix.pixcap.bias_voltage = -0.1

        pix.pixcap.bias_off()
        fig, ax = plt.subplots()
        ax.plot(settling_times, current_data)
        fig.savefig("source_settling_test.png")


def test_reading_speed(smu, file_name: str, config: dict, reading_range: Iterable, n_tests=10):
    reading_range = np.asarray(reading_range)
    with PixCap65TotalCap(config, file_name) as pix:
        try:
            pix.pixcap[pix.pixcap.smu_setup_devices[smu]].text_format()
        except ValueError:
            pass
        pix.pixcap.bias_voltage = -0.1
        pix.pixcap.bias_on()
        pix.pixcap.set_smu_measurements(smu, 1)
        start = time.time()
        pix.pixcap[smu].get_current()
        diff = time.time() - start
        print(f"Single measurement takes {diff} seconds.")
        for n_readings in reading_range:
            print(f"Test for {n_readings} readings.")
            start = time.time()
            for _ in range(n_tests):
                pix.pixcap.smu_advanced_current_multiple(n_readings, smu)
            diff = time.time() - start
            print(f"{n_readings} take {diff / n_tests} seconds each.")

        pix.pixcap[pix.pixcap.smu_setup_devices[smu]].drain_error_queue()
        pix.pixcap.bias_off()


def test_reading_speed_adv(smu, file_name: str, config: dict, reading_range: Iterable, n_tests=10):
    reading_range = np.asarray(reading_range)
    from basil.dut import Base
    basis = Base("pixcap65.yaml")
    dut_config = basis._conf.copy()
    for hw in dut_config["hw_drivers"]:
        if hw["name"] != "SMU":
            continue
        hw["init"]["enable_binary_commands"] = True
    with PixCap65TotalCap(config, file_name, pix_config=dut_config) as pix:
        internal_smu = pix.pixcap[smu]
        try:
            internal_smu.disable_filter()
        except ValueError:
            pass
        try:
            pix.pixcap[pix.pixcap.smu_setup_devices[smu]].binary_format()
            internal_smu.set_current_nlpc(1)
            for n_readings in reading_range:
                print(f"Test for {n_readings} readings.")
                internal_smu.set_number_measurements(n_readings)
                try:
                    internal_smu.set_number_triggers(n_readings)
                except ValueError:
                    pass
                start = time.time()
                for _ in range(n_tests):
                    internal_smu.get_advanced_current(binary_enabled=True, data_points=n_readings)
                diff = time.time() - start
                print(f"{n_readings} take {diff / n_tests} seconds each.")

        finally:
            pix.pixcap[pix.pixcap.smu_setup_devices[smu]].text_format()
            pix.pixcap[pix.pixcap.smu_setup_devices[smu]].drain_error_queue()
            internal_smu.set_current_nlpc(10)


if __name__ == "__main__":
    test_frequency_settling(np.linspace(0, 1, 20), "freq_settling_test.h5")
    print(np.linspace(0, 2, 30))
    test_source_settling(np.linspace(0, 2, 30), "source_settling_test.h5")

    print("start reading test.")
    test_reading_speed("VM3", "reading_speed_test.h5", scan_configuration, np.arange(3, 20.1, 1), n_tests=20)
    print("start advanced test")
    test_reading_speed_adv("VM3", "reading_speed_test_advanced.h5", scan_configuration,
                           np.arange(3, 20.1, 1), n_tests=20)
