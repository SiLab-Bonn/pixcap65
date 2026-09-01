# ----------------------------------------------------------
#  Copyright (c) 2018-2026. All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------
"""
Plotting of Pixcap65 data.
"""
import logging

try:
    # noinspection PyCompatibility
    from collections.abc import Iterable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Iterable

# seemingly unused
# before deleting them, just comment them in case they are still required for one of the plotting handlers.
# HISTOGRAM_SHAPE_FORMAT = "The histograms shape is {}"
# E1_SCAN_FILE = "Reference_Evelyn_Scan.h5"
# X2_SCAN_FILE = 'New_2_Scan.h5'
# X1_SCAN_2_FILE = "packaged/data/X1_4_Renew_Scan.h5"
# REFERENCE_TEST_FILE = "packaged/Reference_Demo.h5"

if __name__ == '__main__':
    # plot_data(interpreted_data=os.path.expanduser('~/git/pixcap65/pixcap_LF_50x50_DC_R3_80V_HV.h5'))

    # some usage examples
    import matplotlib
    matplotlib.use('PDF')

    logging.basicConfig(level=logging.INFO)
    # plot_bias_data(interpreted_data="data/3D_Sensor_221_Scan.h5", base_path="Thesis/ATLAS_ITk/X3/I_V_Characteristic",
    #                use_group=True)
    # plot_bias_data(interpreted_data="data/3D_Sensor_221_W13_X_Scan.h5", base_path="Thesis/ATLAS_ITk/X3/I_V_Characteristic",
    #                use_group=True)
    # plot_combined_data(interpreted_data="data/3D_Sensor_221_W13_X_Scan.h5",
    #                    base_path="Thesis/ATLAS_ITk/X3/C_V_Characteristic", use_group=True)
    # plot_bias_data(interpreted_data="data/3D_Sensor_221_W6_j_Scan.h5", base_path="Thesis/ATLAS_ITk/X5/I_V_Characteristic",
    #                use_group=True)
    # plot_combined_data(interpreted_data="data/3D_Sensor_221_W6_j_Scan.h5",
    #                    base_path="Thesis/ATLAS_ITk/X5/C_V_Characteristic", use_group=True)
    # plot_bias_data(interpreted_data="data/3D_Sensor_I14_S24_Scan.h5", base_path="Thesis/ATLAS_ITk/X6/I_V_Characteristic",
    #                use_group=True)
    # plot_combined_data(interpreted_data="data/3D_Sensor_I14_S24_Scan.h5",
    #                    base_path="Thesis/ATLAS_ITk/X6/C_V_Characteristic", use_group=True)
    # plot_bias_data(interpreted_data="data/3D_Sensor_H23_S24_Scan.h5", base_path="Thesis/ATLAS_ITk/X7/I_V_Characteristic",
    #                use_group=True)
    # plot_combined_data(interpreted_data="data/3D_Sensor_H23_S24_Scan.h5",
    #                    base_path="Thesis/ATLAS_ITk/X7/C_V_Characteristic", use_group=True)
    # plot_bias_data(interpreted_data="data/argparser.h5", base_path="Reference/R11/I_V_Characteristic", use_group=True)
    # plot_combined_data(interpreted_data="data/argparser.h5", base_path="Reference/R11/C_V_Characteristic", use_group=True)
    # plot_combined_data(interpreted_data="data/3D_Sensor_221_W5_S_Scan.h5", base_path="Thesis/ATLAS_ITk/X4/C_V_Characteristic",
    #                    use_group=True)
