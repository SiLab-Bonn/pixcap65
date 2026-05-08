"""
Analysis and plotting script evaluate all the data taking from the beginning!
"""
import numpy as np
import tables as tb
from matplotlib.backends.backend_pdf import PdfPages

from analysis_util.utility import get_base_group
from pixcap65.analysis import analyze_data, analyze_capacitance_distribution
from pixcap65.utility.homogenize_plots import set_params
from plotting import plot_data, plot_combined_data, plot_bias_data
from utility.tables_util import get_group_attribute

if __name__ == '__main__':
    # additional setup
    # noinspection GrazieInspectionRunner
    set_params(latex=True,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={"
                           r"stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{"
                           r"retain-zero-uncertainty}", fig_width=8.26772, fig_height=11.69291, )
    bare_correction_args = {
        "apply_correction": True,
        "bare_file": "Bare_Repeat_2_Scan.h5",
        "bare_hdf_path": "Reference/bare/unbiased_8/total_cap",
    }

    doping_investigation_args = {

    }

    # investigate the bare pixcap ship and it's distribution!
    print("Analyze the Bare samples for calibration of the pixcap chips")
    analyze_data(raw_data='pixcap65/Data/bare-measurement/TEST.h5', is_advanced=False)
    from matplotlib import pyplot as plt
    print(plt.get_fignums())
    analyze_data(raw_data='pixcap65/Data/advanced-bare-measurement/TEST.h5', is_advanced=True, is_cv=False)
    print(plt.get_fignums())
    analyze_data(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8", is_advanced=True, is_cv=False,
                 full_model=False)
    print(plt.get_fignums())
    analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=True,
                                     fit_plot_pdf_name="Bare_analysis_parasitic_kafe2.pdf")
    print(plt.get_fignums())
    analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_parasitic.pdf")
    # investigate the bump capacitance
    from matplotlib import pyplot as plt

    print(plt.get_fignums())
    with tb.open_file("Bare_Repeat_2_Scan.h5", 'r') as f:
        base_group = get_base_group("Reference/bare/unbiased_8", f)
        try:
            bump_caps = np.concat(
                (base_group.total_cap.analysis.HistCap[:5, 0], base_group.total_cap.analysis.HistCap[35:, 0],))
            bump_errors = np.concat(
                (base_group.total_cap.analysis.HistCapErr[:5, 0], base_group.total_cap.analysis.HistCapErr[35:, 0],))
            weights = np.reciprocal(bump_errors ** 2)
            average_bump_cap = np.average(bump_caps, weights=weights)
            statistical_bump_error = np.shape(bump_errors)[0] / np.sum(weights)
            parasitic = get_group_attribute(base_group.total_cap.analysis, "parasitic")
            parasitic_error = get_group_attribute(base_group.total_cap.analysis, "parasitic_error")
            print(f"bump capacitance: ({parasitic - average_bump_cap}+-{statistical_bump_error}+-{parasitic_error})")
            print(np.std(bump_caps))
        except:
            print(f)
            raise

    # analysis section/calibration
    print("Analyze the R13 reference sample.")
    from matplotlib import pyplot as plt
    print(plt.get_fignums())
    analyze_data(raw_data='pixcap65/Data/TEST_2.h5', is_advanced=False)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/data.h5', is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/TEST.h5', is_advanced=False, **bare_correction_args)
    # r13-measurements/R13_BIAS_CV_2.h5 could not be directly investigated as it is incomplete.
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_2.h5', is_advanced=False, is_cv=True,
                 use_corrected=True, apply_doping=True, chip_group_name="sensor",
                 **bare_correction_args)
    # Data/r13-measurement/R13_BIAS_CV_COMBI_3.h5 no further investigation possible as data set is incomplete!
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_5.h5', is_advanced=False, is_cv=True,
                 first_boundaries=(-100, -40),
                 second_boundaries=(-8, -0), use_corrected=True, apply_doping=True, chip_group_name="sensor",
                 **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', is_advanced=False, is_cv=True,
                 first_boundaries=(-100, -40),
                 second_boundaries=(-10, 0), apply_doping=True, chip_group_name="sensor", use_corrected=True,
                 **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', is_advanced=True,
                 **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1",
                 is_advanced=True, **bare_correction_args)
    print("Analyze Evelyn reference sample.")
    from matplotlib import pyplot as plt

    print(plt.get_fignums())
    analyze_data(raw_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/unbiased_4_full", is_advanced=True,
                 **bare_correction_args)

    print("Analyze the ATLAS ITk samples.")
    from matplotlib import pyplot as plt

    print(plt.get_fignums())
    analyze_data(raw_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_1",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_2",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_1",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_2",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/ATLAS ITk/New_1_Initial_Scan.h5', is_advanced=True, **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_1",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_2", full_model=False,
                 is_advanced=True, **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3", is_advanced=True,
                 **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_4",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic",
                 is_advanced=False, is_cv=True, use_corrected=True, apply_doping=True,
                 chip_group_name="ATLAS ITk/sensor",
                 first_boundaries=[(-60, -40), (-80, -75)], second_boundaries=[(-5, 0), (-70, -65)],
                 **bare_correction_args)
    analyze_data(raw_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V",
                 is_advanced=True,
                 **bare_correction_args)

    print("Analyze the ATLAS sample X2")
    print("unbiased 1")
    from matplotlib import pyplot as plt
    print(plt.get_fignums())
    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1", is_advanced=True,
                 **bare_correction_args)
    print("C-V-Characteristic")
    print(plt.get_fignums())
    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic", is_advanced=True, is_cv=True,
                 first_boundaries=[(-60, -40), (-80, -75)], second_boundaries=[(-5, 0), (-70, -65)], use_corrected=True,
                 **bare_correction_args)
    print("biased 80")
    print(plt.get_fignums())
    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/biased_80_V", is_advanced=True,
                 **bare_correction_args)
    print("C-V-Characteristic refined")
    print(plt.get_fignums())
    with PdfPages("New_2_Scan_CV_refined_distribution.pdf") as pdf:
        analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic_refined", is_advanced=True,
                     is_cv=True,
                     first_boundaries=(-60, -20), second_boundaries=(-5, 0), use_corrected=True, apply_doping=True,
                     chip_group_name="ATLAS_Itk/X2/sensor", distribution=True, set_parasitic=False,
                     distribution_output_pdf=pdf, **bare_correction_args)

    # plotting section
    print("Plots for the reference sample BARE 5 BUMPS")
    plot_data(interpreted_data='pixcap65/Data/bare-measurement/TEST.h5', suffix="general_bare_data_1-1",
              use_group=False)
    plot_data(interpreted_data='pixcap65/Data/advanced-bare-measurement/TEST.h5', suffix="general_bare_data_2-1",
              use_group=False)
    plot_data(interpreted_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
              suffix="general_bare_data_3", use_group=True,
              exclude_test_cap=True)
    plot_data(interpreted_data='pixcap65/Data/TEST_2.h5', suffix="test_run", use_group=False)

    print("Plots for the reference sample R13")
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/data.h5', suffix="test_run", use_group=False)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/data.h5', suffix="test_run", use_group=False,
              use_corrected=True)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/Test.h5', suffix="test-general_run", use_group=False)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/Test.h5', suffix="test-general_run", use_group=False,
              use_corrected=True)
    plot_bias_data(interpreted_data='pixcap65/Data/r13-measurement/R13_BIAS_2.h5')
    # Data/r13-measurement/R13_BIAS_CV_COMBI_2.h5 no further investigation possible as data set is incomplete!
    # Data/r13-measurement/R13_BIAS_CV_COMBI_3.h5 no further investigation possible as data set is incomplete!
    plot_combined_data(interpreted_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_5.h5', first_lower=-100,
                       first_upper=-40, second_lower=-8, second_upper=0)
    plot_combined_data(interpreted_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_5.h5', first_lower=-100,
                       first_upper=-40, second_lower=-8, second_upper=0, use_corrected=True, apply_doping=True)
    plot_combined_data(interpreted_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', first_lower=-100,
                       first_upper=-40, second_lower=-10, second_upper=0)
    plot_combined_data(interpreted_data='pixcap65/Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', first_lower=-100,
                       first_upper=-40, second_lower=-10, second_upper=0, use_corrected=True, apply_doping=True)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data",
              use_group=False)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data",
              use_group=False,
              use_corrected=True, exclude_test_cap=True, distribution=True)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1",
              suffix="unbiased_full_measurement", use_group=True)
    plot_data(interpreted_data='pixcap65/Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1",
              suffix="unbiased_full_measurement", use_group=True, use_corrected=True)

    print("Plots for the reference sample E1")
    # plot_data(interpreted_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/unbiased_1_test", use_group=True)
    # plot_data(interpreted_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/unbiased_2_test", use_group=True)
    # plot_data(interpreted_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/unbiased_3_test", use_group=True)
    plot_data(interpreted_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/unbiased_4_full", use_group=True)
    plot_data(interpreted_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/unbiased_4_full", use_group=True,
              use_corrected=True, distribution=True, exclude_test_cap=True)
    plot_bias_data(interpreted_data="Reference_Evelyn_Scan.h5", base_path="Reference/E1/I_V_Characteristic",
                   use_group=True)

    print("Plots for ATLAS sample X1")
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_1", suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_1", suffix="test_run",
              use_group=True, use_corrected=True)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_2", suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_2", suffix="test_run",
              use_group=True, use_corrected=True)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_1",
              suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_1",
              suffix="test_run",
              use_group=True, use_corrected=True)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_2",
              suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_2",
              suffix="test_run",
              use_group=True, use_corrected=True)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_Scan.h5', suffix="test_run", use_group=True)
    plot_data(interpreted_data='pixcap65/Data/ATLAS ITk/New_1_Initial_Scan.h5', suffix="test_run", use_group=True,
              use_corrected=True)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_1",
              suffix="general_data_test_test", use_group=True)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_1",
              suffix="general_data_test_test", use_group=True, use_corrected=True)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_2",
              suffix="general_data", use_group=True)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_2",
              suffix="general_data", use_group=True, use_corrected=True)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
              suffix="general_data", use_group=True, exclude_test_cap=True, mask_pixel=[[39, 39], [38, 39]],
              distribution=True)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
              suffix="general_data", use_group=True, use_corrected=True, mask_pixel=[[39, 39], [38, 39]],
              exclude_test_cap=True, distribution=True)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_4",
              suffix="general_data", use_group=True)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_4",
              suffix="general_data", use_group=True, use_corrected=True)
    plot_bias_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/I_V_Characteristic",
                   use_group=True)
    plot_combined_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                       base_path="ATLAS ITk/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0)
    plot_combined_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5',
                       base_path="ATLAS ITk/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-60, second_upper=0,
                       use_corrected=True,
                       apply_doping=True)
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V",
              suffix="general_data", use_group=True, exclude_test_cap=True, mask_pixel=[[39, 39], [38, 39]], )
    plot_data(interpreted_data='pixcap65/Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V",
              suffix="general_data", use_group=True, use_corrected=True, apply_doping=True, exclude_test_cap=True,
              mask_pixel=[[39, 39], [38, 39]], distribution=True)

    print("Plots for ATLAS sample X2")
    plot_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1", suffix="general_data_80V",
              use_group=True, exclude_test_cap=True)
    plot_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1", suffix="general_data_80V",
              use_group=True, use_corrected=True, exclude_test_cap=True)
    plot_bias_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/I_V_Characteristic", use_group=True)
    plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0)
    plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-60, second_upper=0,
                       use_corrected=True)
    plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic_refined",
                       use_group=True, first_lower=-60, first_upper=-20, second_lower=-5, second_upper=0)
    plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic_refined",
                       use_group=True, first_lower=-60, first_upper=-20, second_lower=-5, second_upper=0,
                       use_corrected=True,
                       apply_doping=True, distribution=True)
    plot_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/biased_80_V", suffix="general_data_80V",
              use_group=True, exclude_test_cap=True)
    plot_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/biased_80_V", suffix="general_data_80V",
              use_group=True, use_corrected=True, apply_doping=True, exclude_test_cap=True)
