# ----------------------------------------------------------
#  Copyright (c) 2026. SiLab, Institute of Physics, University of Bonn.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ----------------------------------------------------------
from os.path import join as hdf

from examples import data_constants
from examples.data_constants import E1_SCAN_FILE
from pixcap65.plotting_util import threaded_plotting, plot_data


def e1_plotter_first(tb_lock):
    name = "E1"
    display_name = name
    top_ref = "Reference"
    print("Plotting", display_name)
    plot_data(interpreted_data=E1_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_4_full'),
              use_group=True, test_cap_exclusion=True, mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock, )
    plot_data(interpreted_data=E1_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_4_full'),
              use_group=True, use_corrected=True, distribution=True, test_cap_exclusion=True,
              mask_pixel=data_constants.e1_pixel_mask, lock=tb_lock, )
    plot_data(interpreted_data=E1_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_1_test'),
                                use_group=True, lock=tb_lock)
    plot_data(interpreted_data=E1_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_2_test'),
                                use_group=True, lock=tb_lock)
    plot_data(interpreted_data=E1_SCAN_FILE, base_path=hdf(top_ref, name, 'unbiased_3_test'),
                                use_group=True, lock=tb_lock)
    threaded_plotting.plot_bias_data(interpreted_data=E1_SCAN_FILE,
                                     base_path=hdf(top_ref, name, 'I_V_Characteristic'),
                                     use_group=True, lock=tb_lock)
    print(display_name, "- CV")
    print(display_name, "- CV")
    threaded_plotting.plot_combined_data(interpreted_data=E1_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, distribution=True, lock=tb_lock, )
    threaded_plotting.plot_combined_data(interpreted_data=E1_SCAN_FILE,
                                         base_path=hdf(top_ref, name, 'C_V_Characteristic'),
                                         use_group=True, use_corrected=True, apply_doping=False, distribution=False,
                                         lock=tb_lock, )
    threaded_plotting.joint_plotting()
    print("Finished -", display_name)
