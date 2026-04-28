"""
Analysis and plotting script evaluate all of the data taking from the beginning!
"""
from analysis import analyze_data, analyze_capacitance_distribution
from plotting import plot_data, plot_combined_data, plot_bias_data
from utility.homogenize_plots import set_params

if __name__ == '__main__':
    # additional setup
    set_params(latex=True,
               latex_extra=r"\sisetup{separate-uncertainty}\sisetup{locale = DE}\sisetup{uncertainty-descriptors={stat,sys}}\sisetup{uncertainty-descriptor-mode=subscript}\sisetup{retain-zero-uncertainty}")
    bare_correction_args = {
        "apply_correction": True,
        "bare_file": "Bare_Repeat_2_Scan.h5",
        "bare_hdf_path": "Reference/bare/unbiased_8/total_cap",
    }

    # investigate the bare pixcap ship and it's distribution!
    analyze_data(raw_data='Data/bare-measurement/TEST.h5', is_advanced=False)
    analyze_data(raw_data='Data/advanced-bare-measurement/TEST.h5', is_advanced=True, is_cv=False)
    analyze_data(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8", is_advanced=True, is_cv=False)
    analyze_capacitance_distribution(raw_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8",
                                     corrected_distribution=False,
                                     exclude_test_cap=True, use_kafe2=False,
                                     fit_plot_pdf_name="Bare_analysis_parasitic.pdf")

    # analysis section/calibration
    analyze_data(raw_data='Data/TEST_2.h5', is_advanced=False)


    analyze_data(raw_data='Data/r13-measurement/data.h5', is_advanced=False, **bare_correction_args)
    # r13-measurements/R13_BIAS_CV_2.h5 could not be directly investigated as it is incomplete.
    analyze_data(raw_data='Data/r13-measurement/R13_BIAS_CV_COMBI_2.h5', is_advanced=False, is_cv=True,
                 **bare_correction_args)
    # Data/r13-measurement/R13_BIAS_CV_COMBI_3.h5 no further investigation possible as data set is incomplete!
    analyze_data(raw_data='Data/r13-measurement/R13_BIAS_CV_COMBI_5.h5', is_advanced=False, is_cv=True, first_boundaries=(-100, -40),
                 second_boundaries=(-8, -0), **bare_correction_args)
    analyze_data(raw_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', is_advanced=False, is_cv=True, first_boundaries=(-100, -40),
                 second_boundaries=(-10, 0), apply_doping=True, chip_group_name="sensor", use_corrected=True, **bare_correction_args)
    analyze_data(raw_data='Data/r13-measurement/R13_Full_Scan_80V.h5', is_advanced=True, **bare_correction_args)
    analyze_data(raw_data='Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1",
                 is_advanced=True, **bare_correction_args)

    analyze_data(raw_data='Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_1",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_2",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_1",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_2",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='Data/ATLAS ITk/New_1_Initial_Scan.h5', is_advanced=True, **bare_correction_args)
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_1",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_2", full_model=False,
                 is_advanced=True, **bare_correction_args)
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3", is_advanced=True,
                 **bare_correction_args)
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_4",
                 is_advanced=False, **bare_correction_args)
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic",
                 is_advanced=False, is_cv=True,
                 first_boundaries=(-60, -40), second_boundaries=(-5, 0), **bare_correction_args)
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V", is_advanced=True,
                 **bare_correction_args)

    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1", is_advanced=True, **bare_correction_args)
    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic", is_advanced=True, is_cv=True,
                 first_boundaries=(-60, -40), second_boundaries=(-5, 0), **bare_correction_args)
    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/biased_80_V", is_advanced=True,
                 **bare_correction_args)
    analyze_data(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic_refined", is_advanced=True, is_cv=True,
                 first_boundaries=(-60, -20), second_boundaries=(-5, 0), **bare_correction_args)


    # correct all results for the parasitic and intrinsic capacitances

    # plotting section
    plot_data(interpreted_data='Data/bare-measurement/TEST.h5', suffix="general_bare_data_1-1", use_group=False)
    plot_data(interpreted_data='Data/advanced-bare-measurement/TEST.h5', suffix="general_bare_data_2-1", use_group=False)
    plot_data(interpreted_data='Bare_Repeat_2_Scan.h5', base_path="Reference/bare/unbiased_8", suffix="general_bare_data_3", use_group=True,
              exclude_test_cap=True)
    plot_data(interpreted_data='Data/TEST_2.h5', suffix="test_run", use_group=False)


    plot_data(interpreted_data='Data/r13-measurement/data.h5', suffix="test_run", use_group=False)
    plot_data(interpreted_data='Data/r13-measurement/data.h5', suffix="test_run", use_group=False, use_corrected=True)
    plot_bias_data(interpreted_data='Data/r13-measurement/R13_BIAS_2.h5')
    # Data/r13-measurement/R13_BIAS_CV_COMBI_2.h5 no further investigation possible as data set is incomplete!
    # Data/r13-measurement/R13_BIAS_CV_COMBI_3.h5 no further investigation possible as data set is incomplete!
    plot_combined_data(interpreted_data='Data/r13-measurement/R13_BIAS_CV_COMBI_5.h5', first_lower=-100,
                       first_upper=-40, second_lower=-8, second_upper=0)
    plot_combined_data(interpreted_data='Data/r13-measurement/R13_BIAS_CV_COMBI_5.h5', first_lower=-100,
                       first_upper=-40, second_lower=-8, second_upper=0, use_corrected=True)
    plot_combined_data(interpreted_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', first_lower=-100,
                       first_upper=-40, second_lower=-10, second_upper=0)
    plot_combined_data(interpreted_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', first_lower=-100,
                       first_upper=-40, second_lower=-10, second_upper=0, use_corrected=True, apply_doping=True)
    plot_data(interpreted_data='Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data", use_group=False)
    plot_data(interpreted_data='Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data", use_group=False, use_corrected=True)
    plot_data(interpreted_data='Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1",
              suffix="unbiased_full_measurement", use_group=True)
    plot_data(interpreted_data='Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1",
              suffix="unbiased_full_measurement", use_group=True, use_corrected=True)


    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_1", suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_1", suffix="test_run",
              use_group=True, use_corrected=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_2", suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_2", suffix="test_run",
              use_group=True, use_corrected=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_1", suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_1", suffix="test_run",
              use_group=True, use_corrected=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_2", suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_2", suffix="test_run",
              use_group=True, use_corrected=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_Scan.h5', suffix="test_run", use_group=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_Scan.h5', suffix="test_run", use_group=True, use_corrected=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_1",
              suffix="general_data_test_test", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_1",
              suffix="general_data_test_test", use_group=True, use_corrected=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_2",
              suffix="general_data", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_2",
              suffix="general_data", use_group=True, use_corrected=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
              suffix="general_data", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
              suffix="general_data", use_group=True, use_corrected=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_4",
              suffix="general_data", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_4",
              suffix="general_data", use_group=True, use_corrected=True)
    plot_bias_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/I_V_Characteristic",
                   use_group=True)
    plot_combined_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0)
    plot_combined_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0, use_corrected=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V",
              suffix="general_data", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V",
              suffix="general_data", use_group=True, use_corrected=True)


    plot_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1", suffix="general_data_80V",
              use_group=True)
    plot_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1", suffix="general_data_80V",
              use_group=True, use_corrected=True)
    plot_bias_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/I_V_Characteristic", use_group=True)
    plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0)
    plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0, use_corrected=True)
    plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic_refined",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0)
    # plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic_refined",
    #                    use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0, use_corrected=True)
    plot_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/biased_80_V", suffix="general_data_80V",
              use_group=True)
    plot_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/biased_80_V", suffix="general_data_80V",
              use_group=True, use_corrected=True)

