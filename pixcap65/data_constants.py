# ----------------------------------------------------------
#  Copyright (c) 2026.
#   All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------
import numpy as np

X1_SCAN_2_FILE = "packaged/data/X1_4_Renew_Scan.h5"
X2_SCAN_FILE = 'New_2_Scan.h5'
X2_SCAN_2_FILE = "packaged/data/X2_2_Scan.h5"
X4_SCAN_FILE = "packaged/data/3D_Sensor_221_W5_S_Scan.h5"
X5_SCAN_FILE = "packaged/data/3D_Sensor_221_W6_J_Scan.h5"
X6_SCAN_FILE = "packaged/data/3D_Sensor_I14_S24_Full_Scan.h5"
X7_SCAN_FILE = "packaged/data/3D_Sensor_H23_S24_Full_Scan.h5"

E1_SCAN_FILE = "Reference_Evelyn_Scan.h5"
E1_2_SCAN_FILE = "packaged/E1_Renew_Scan.h5"
R13_2_SCAN_FILE = "packaged/R13_Renew_Scan.h5"
R11_SCAN_FILE = "packaged/data/R11_R1_RX_Scan.h5"

x1_second_pixel_mask = [[39, 39], [38, 39]]
x4_pixel_mask_temp = np.load("x4_mask.npy")
x4_pixel_mask = []
for entry in x4_pixel_mask_temp:
    x4_pixel_mask.append([*entry])

x4_pixel_mask.append([1, 3])
x4_pixel_mask.append([1, 2])
x4_pixel_mask.append([2, 3])
x4_pixel_mask.append([2, 4])
x5_second_pixel_mask = [
    [0, 39],
    [0, 38],
    [0, 37],
    [0, 36],
    [0, 35],
    [0, 34],
    [0, 33],
    [0, 32],
    [0, 31],
    [0, 29],
    [1, 39],
    [1, 38],
    [1, 37],
    [1, 36],
    [1, 35],
    [1, 34],
    [1, 33],
    [2, 39],
    [2, 38],
    [2, 37],
    [2, 36],
    [2, 35],
    [2, 33],
    [3, 39],
    [3, 37],
    [3, 36],
    [3, 34],
    [4, 39],
    [4, 37],
    [4, 36],
    [5, 39],
    [5, 38],
    [5, 37],
    [6, 39],
    [10, 39],
    [37, 39],
]
x5_pixel_mask_temp = np.load("x5_mask.npy")
x5_pixel_mask = x5_second_pixel_mask.copy()
for entry in x4_pixel_mask_temp:
    x5_pixel_mask.append([*entry])
x6_second_pixel_mask = [
    [0, 38],
]

r1_pixel_mask = [
    [35, 2],
    [35, 3],
]
e1_pixel_mask = [[39, 1]]

e1_pixel_groups = {
    "dnw30_50": {
        "columns": [(0, 15), ],
        "rows": [(2, 33), ]
    },
    "nw15_50": {
        "columns": [(31, 39), ],
        "rows": [(33, 40), ]
    },
    "nw20_50": {
        "columns": [(23, 31), ],
        "rows": [(33, 40), ]
    },
    "nw25_50": {
        "columns": [(15, 23), ],
        "rows": [(33, 40), ]
    },
    "nw30_50": {
        "columns": [(0, 15), (0, 40), (39, 40),],
        "rows": [(33, 40), (1, 2), (1, 40), ]
    },
    "dnw15_50": {
        "columns": [(31, 39),],
        "rows": [(2, 33), ]
    },
    "dnw20_50": {
        "columns": [(23, 31), ],
        "rows": [(2, 33), ]
    },
    "dnw25_50": {
        "columns": [(15, 23), ],
        "rows": [(2, 33), ]
    },
}


# this here are implantation sizes but also the pitch might matter for the capacitance modelling!
# in particular the difference between these two should impact the inter-pixel capacitance!
e1_pixel_dimensions = {
    "dnw30_50": 30,
    "nw15_50": 15,
    "nw20_50": 20,
    "nw25_50": 25,
    "nw30_50": 30,
    "dnw15_50": 15,
    "dnw20_50": 20,
    "dnw25_50": 25,
}

e1_pixel_depletion_args = {
    "dnw30_50": {
        "first_boundaries": (-100, -25),
        "second_boundaries": (-7, -2),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "pixel_mask": e1_pixel_mask,
        "apply_doping": True,
        "chip_group_name": "Reference/E1/sensor",
    },
    "nw15_50": {
        "first_boundaries": (-100, -30),
        "second_boundaries": (-15, -2.5),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "pixel_mask": e1_pixel_mask,
        "apply_doping": True,
        "chip_group_name": "Reference/E1/sensor",
    },
    "nw20_50": {
        "first_boundaries": (-100, -30),
        "second_boundaries": (-17.5, -2.5),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "pixel_mask": e1_pixel_mask,
        "apply_doping": True,
        "chip_group_name": "Reference/E1/sensor",
    },
    "nw25_50": {
        "first_boundaries": (-100, -30),
        "second_boundaries": (-10, -2),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "pixel_mask": e1_pixel_mask,
        "apply_doping": True,
        "chip_group_name": "Reference/E1/sensor",
    },
    "nw30_50": {
        "first_boundaries": (-100, -25),
        "second_boundaries": (-10, -2),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "pixel_mask": e1_pixel_mask,
        "apply_doping": True,
        "chip_group_name": "Reference/E1/sensor",
    },
    "dnw15_50": {
        "first_boundaries": (-100, -40),
        "second_boundaries": (-12, -2),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "pixel_mask": e1_pixel_mask,
        "apply_doping": True,
        "chip_group_name": "Reference/E1/sensor",
    },
    "dnw20_50": {
        "first_boundaries": (-100, -40),
        "second_boundaries": (-12, -3),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "pixel_mask": e1_pixel_mask,
        "apply_doping": True,
        "chip_group_name": "Reference/E1/sensor",
    },
    "dnw25_50": {
        "first_boundaries": (-100, -25),
        "second_boundaries": (-11, -2),
        "distribution": True,
        "apply_contour": False,
        "apply_contours": False,
        "pixel_mask": e1_pixel_mask,
        "apply_doping": True,
        "chip_group_name": "Reference/E1/sensor",
    },
}

# multiprocessing_key = b"\x1f\xb8MC3\xf8@7\x11\x9fh7,\x11\xb5\xd4JI\x06e\xdc<b\x0f\x92\x04\xb9C\xfb\xb0\xc8\xa2"
