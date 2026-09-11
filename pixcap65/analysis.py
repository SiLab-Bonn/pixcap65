"""
Analysis of Pixcap65 data. Fits freq vs current to extract the capacitance. A 2D histogram containing the capacitance
for each pixel is stored.
"""
import locale
import logging
import numpy as np
import tables as tb
from iminuit.warnings import IMinuitWarning
from tables.exceptions import NaturalNameWarning

from pixcap65.analysis_util import analyze_data
from pixcap65.analysis_util.delegation.distribution import analyze_capacitance_distribution

try:
    # noinspection PyCompatibility
    from collections.abc import Callable, Sized, Iterable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Callable, Sized, Iterable
from warnings import filterwarnings

from pixcap65.utility import synchronized_process_open_file

# TODO: make sure to not import in the implementation files anything contained within module __init__.py

logger = logging.getLogger(__name__)

cap_counter = 0

filterwarnings("ignore", category=NaturalNameWarning)
filterwarnings("ignore", category=IMinuitWarning)
filterwarnings("ignore", category=np.exceptions.RankWarning, module="jacobi")


def get_test_capacitance_data(group: tb.Group, **kwargs):
    """
    get_test_capacitance_data

    @author: Dominik Fischer
    @date: 2026-08-12

    Utility function to print out the determined test capacitance values and their corresponding (statistical) uncertainties for the provided sensor.

    :param group: analysis group of the sensor and measurement series from which to take the test capacitances.
    :keyword print_result: whether to print out the test capacitance values on the standard output (default: True)
    :type print_result: bool
    :return: tuple of array of the test capacitances and their uncertainties.
    """
    locale.setlocale(locale.LC_NUMERIC, "de_DE")
    test_cap = group.HistCap[:][:, 0]
    test_cap_error = group.HistCapErr[:][:, 0]
    if kwargs.get("print_result", True):
        for k, (cap, err) in enumerate(zip(test_cap, test_cap_error)):
            eff_cap = cap * 1e15
            eff_err = err * 1e15
            print(k, f"{eff_cap:.3n}+-{eff_err:.3n}")

    test_cap[16] = np.nan
    test_cap_error[16] = np.nan
    return test_cap[np.isfinite(test_cap)], test_cap_error[np.isfinite(test_cap_error)]


def bare_analysis_handler(tb_lock):
    with synchronized_process_open_file("packaged/Reference_Bare_renewed.h5", 'a', lock=tb_lock) as h5_file:
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_31_renew", newname="unbiased_31_renew_full_model",
                          overwrite=True, recursive=True)

    with synchronized_process_open_file("packaged/data/Bare_Sample_05_Extended_Scan.h5", 'a', lock=tb_lock) as h5_file:
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_model",
                          overwrite=True, recursive=True)
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_kafe2",
                          overwrite=True, recursive=True)
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_extended",
                          overwrite=True, recursive=True)
        h5_file.copy_node(where="/Reference/Bare", name="unbiased_full", newname="unbiased_full_quad",
                          overwrite=True, recursive=True)

    analyze_data(raw_data="packaged/Reference_Bare_renewed.h5", base_path="Reference/Bare/unbiased_31_renew",
                 is_advanced=True, full_model=False, lock=tb_lock)
    analyze_data(raw_data="packaged/Reference_Bare_renewed.h5",
                 base_path="Reference/Bare/unbiased_31_renew_full_model", is_advanced=True,
                 full_model=True, lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5", base_path="Reference/Bare/unbiased_full",
                 is_advanced=True, full_model=False, lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_model", is_advanced=True,
                 full_model=True, lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_kafe2", is_advanced=True,
                 full_model=True, lock=tb_lock, use_kafe2=True)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_extended", is_advanced=True,
                 full_model='extended', lock=tb_lock)
    analyze_data(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                 base_path="Reference/Bare/unbiased_full_quad", is_advanced=True,
                 full_model='quad', lock=tb_lock)
    analyze_capacitance_distribution(raw_data="packaged/Reference_Bare_renewed.h5",
                                     base_path="Reference/Bare/unbiased_31_renew",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_renew_parasitic.pdf", lock=tb_lock)
    analyze_capacitance_distribution(raw_data="packaged/Reference_Bare_renewed.h5",
                                     base_path="Reference/Bare/unbiased_31_renew_full_model",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_renew_parasitic_full_model.pdf", lock=tb_lock)
    analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5',
                                     base_path="Reference/bare/unbiased_8_full_model",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_parasitic_full_model.pdf", lock=tb_lock)
    analyze_capacitance_distribution(raw_data="packaged/data/Bare_Sample_05_Extended_Scan.h5",
                                     base_path="Reference/Bare/unbiased_full",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_parasitic_extended_renew_full_model.pdf",
                                     lock=tb_lock)


if __name__ == '__main__':
    # analyse_data(raw_data='/home/silab/git/pixcap65/pixcap_full_data_image1.h5')
    # some usage examples
    from pixcap65.utility.homogenize_plots import set_params
    # hold this for now as it simplifies synchronization between the two devices!
    # analyze_data(raw_data="data/3D_Sensor_221_W13_X_Scan.h5", base_path="Thesis/ATLAS_ITk/X3/C_V_Characteristic",
    #              is_cv=True)
    # analyze_data(raw_data="data/3D_Sensor_221_W6_j_Scan.h5", base_path="Thesis/ATLAS_ITk/X5/C_V_Characteristic",
    #              is_cv=True)
    # analyze_data(raw_data="data/3D_Sensor_I14_S24_Scan.h5", base_path="Thesis/ATLAS_ITk/X6/C_V_Characteristic",
    #              is_cv=True)
    # analyze_data(raw_data="data/3D_Sensor_H23_S24_Scan.h5", base_path="Thesis/ATLAS_ITk/X7/C_V_Characteristic",
    #              is_cv=True)
    # analyze_data(raw_data="data/argparser.h5", base_path="Reference/R11/C_V_Characteristic", is_cv=True)
    # analyze_data(raw_data="data/3D_Sensor_221_W5_S_Scan.h5", base_path="Thesis/ATLAS_ITk/X4/C_V_Characteristic",
    #              is_cv=True)
    try:
        from subprocess import run

        run_result = run(['pdflatex', '--version'], check=True, capture_output=True)
        has_latex = True
        logger.info("The latex compiler to use is: %s", run_result.stdout.decode("utf-8"))
    except (FileNotFoundError, ImportError):
        # proceed as if no latex exists
        logger.exception("Could not verify whether latex exists.")
        has_latex = False
    set_params(latex=has_latex,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors="
                           r"{stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}"
                           r"\sisetup{retain-zero-uncertainty}")
