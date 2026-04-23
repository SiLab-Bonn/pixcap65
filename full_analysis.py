"""
Analysis and plotting script evaluate all of the data taking from the beginning!
"""

from analysis import analyze_data, advanced_analysis
from plotting import plot_data, plot_combined_data, plot_bias_data

if __name__ == '__main__':
    # analysis section/calibration
    analyze_data(raw_data='Data/TEST_2.h5')

    analyze_data(raw_data='Data/r13-measurement/data.h5')
    # r13-measurements/R13_BIAS_CV_2.h5 could not be directly investigated as it is incomplete.
    analyze_data(raw_data='Data/r13-measurement/R13_BIAS_CV_COMBI_2.h5', is_cv=True)
    # Data/r13-measurement/R13_BIAS_CV_COMBI_3.h5 no further investigation possible as data set is incomplete!
    analyze_data(raw_data='Data/r13-measurement/R13_BIAS_CV_COMBI_5.h5', is_cv=True, first_boundaries=(-100, -40),
                 second_boundaries=(-8, -0))
    analyze_data(raw_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', is_cv=True, first_boundaries=(-100, -40),
                 second_boundaries=(-10, 0))
    advanced_analysis(raw_data='Data/r13-measurement/R13_Full_Scan_80V.h5', )
    advanced_analysis(raw_data='Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1")

    analyze_data(raw_data='Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_1")
    analyze_data(raw_data='Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_2")
    analyze_data(raw_data='Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_1")
    analyze_data(raw_data='Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_2")
    advanced_analysis(raw_data='Data/ATLAS ITk/New_1_Initial_Scan.h5')
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_1")
    advanced_analysis(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_2", full_model=False)
    advanced_analysis(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3")
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_4")
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic", is_cv=True,
                 first_boundaries=(-60, -40), second_boundaries=(-5, 0), )
    advanced_analysis(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V")

    advanced_analysis(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1")
    advanced_analysis(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic", is_cv=True,
                      first_boundaries=(-60, -40), second_boundaries=(-5, 0), )
    advanced_analysis(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/biased_80_V")
    advanced_analysis(raw_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic_refined", is_cv=True,
                      first_boundaries=(-60, -20), second_boundaries=(-5, 0), )

    # correct all results for the parasitic and intrinsic capacitances

    # plotting section
    plot_data(interpreted_data='Data/TEST_2.h5', suffix="test_run", use_group=False)

    plot_data(interpreted_data='Data/r13-measurement/data.h5', suffix="test_run", use_group=False)
    plot_bias_data(interpreted_data='Data/r13-measurement/R13_BIAS_2.h5')
    # Data/r13-measurement/R13_BIAS_CV_COMBI_2.h5 no further investigation possible as data set is incomplete!
    # Data/r13-measurement/R13_BIAS_CV_COMBI_3.h5 no further investigation possible as data set is incomplete!
    plot_combined_data(interpreted_data='Data/r13-measurement/R13_BIAS_CV_COMBI_5.h5', first_lower=-100,
                       first_upper=-40, second_lower=-8, second_upper=0)
    plot_combined_data(interpreted_data='Data/r13-measurement/R13_BIAS_CV_COMBI_6.h5', first_lower=-100,
                       first_upper=-40, second_lower=-10, second_upper=0)
    plot_data(interpreted_data='Data/r13-measurement/R13_Full_Scan_80V.h5', suffix="general_data", use_group=False)
    plot_data(interpreted_data='Data/r13-measurement/R13_Initial_3_Scan.h5', base_path="ATLAS ITk/unbiased_1",
              suffix="unbiased_full_measurement", use_group=True)

    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_1", suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_2_Scan.h5', base_path="run_2", suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_1", suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_3_Scan.h5', base_path="ATLAS ITk/run_2", suffix="test_run",
              use_group=True)
    plot_data(interpreted_data='Data/ATLAS ITk/New_1_Initial_Scan.h5', suffix="test_run", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_1",
              suffix="general_data_test_test", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_2",
              suffix="general_data", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
              suffix="general_data", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_4",
              suffix="general_data", use_group=True)
    plot_bias_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/I_V_Characteristic",
                   use_group=True)
    plot_combined_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/full_biased_80_V",
              suffix="general_data", use_group=True)

    plot_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/unbiased_1", suffix="general_data_80V",
              use_group=True)
    plot_bias_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/I_V_Characteristic", use_group=True)
    plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0)
    plot_combined_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/C_V_Characteristic_refined",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0)
    plot_data(interpreted_data='New_2_Scan.h5', base_path="ATLAS_Itk/X2/biased_80_V", suffix="general_data_80V",
              use_group=True)
