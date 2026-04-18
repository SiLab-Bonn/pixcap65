"""
Analysis and plotting script evaluate all of the data taking from the beginning!
"""

from analysis import analyze_data, advanced_analysis
from plotting import plot_data, plot_combined_data, plot_bias_data

if __name__ == '__main__':
    # analysis section/calibration
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_1")
    advanced_analysis(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_2", full_model=False)
    advanced_analysis(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3")
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_4")
    analyze_data(raw_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic", is_cv=True,
                 first_boundaries=(-60, -40), second_boundaries=(-5, 0), )

    # correct all results for the parasitic and intrinsic capacitances

    # plotting section
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_1",
              suffix="general_data_test_test", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/run_2",
              suffix="general_data_test_test", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_3",
              suffix="general_data_test_test", use_group=True)
    plot_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/unbiased_4",
              suffix="general_data_test_test", use_group=True)
    plot_bias_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/I_V_Characteristic",
                   use_group=True)
    plot_bias_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/I_V_Characteristic",
                   use_group=True)
    plot_combined_data(interpreted_data='Data/New_1_Initial_6_Scan.h5', base_path="ATLAS ITk/C_V_Characteristic",
                       use_group=True, first_lower=-60, first_upper=-40, second_lower=-5, second_upper=0)
