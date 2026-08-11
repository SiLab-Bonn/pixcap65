
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

import numpy as np

from compare_plots import linear_model


def inter_cap_model(xy, a0, a1, a2, a3, a4):
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(- a4 * separation_x) + a1 * A + a2 * W + a3 * p

def inter_cap_model_2(xy, a0, a3):
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(a3 * p)

def inter_cap_model_3(xy, a0, a1, a2, a3):
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(a3 * p / separation_x) + a1 * A + a2 * W

def inter_cap_model_4(xy, a0, a1, a2, a3, a4, a5):
    # exclude this model as it leads to small cost functions but with parameter uncertainties which are in general
    # larger than the parameters itself.
    from detailed_fits import quadratic_model
    A, d, p, separation_x, separation_y = xy
    return quadratic_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5))

def inter_cap_model_5(xy, a0, a1, a2, a3, a4, a5):
    from detailed_fits import quadratic_model
    A, d, p, separation_x, separation_y = xy
    return quadratic_model(p / separation_x, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5))

def inter_cap_model_6(xy, a0, a1, a2, a3):
    A, d, p, separation_x, separation_y = xy
    return linear_model(p / separation_x, linear_model(d, a0, a1), linear_model(d, a2, a3))

def inter_cap_model_7(xy, a0, a1, a2, a3, a4, a5, a6, a7):
    # exclude this model as it leads to small cost functions but with parameter uncertainties which are in general
    # larger than the parameters itself.
    from detailed_fits import quadratic_model
    A, d, p, separation_x, separation_y = xy
    return quadratic_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5)) + linear_model(d, a6, a7) / separation_x
