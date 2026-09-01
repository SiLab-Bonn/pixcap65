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
"""
Utility module collecting the different attempts to sufficiently model the total pixel capacitance for the different
pixel geometries and implantation sizes during my bachelor's thesis. Provided just as a example on how the results
of the PixCap65 capacitance measurements could be used further.
"""

from examples.general_model import exponential_model, quadratic_model, linear_model


def simplified_cap_model(xy, a0, a1, a2):
    A, W = xy
    return a0 + a1 * A + a2 * W


def extended_cap_model(xy, a0, a1, a2, a3, a4, a5):
    A, d, p, separation_x, separation_y = xy
    return quadratic_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5))


def extended_cap_model_2(xy, a0, a1, a2, a3):
    A, d, p, separation_x, separation_y = xy
    return exponential_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3))


def extended_cap_model_3(xy, a0, a1, a2, a3, a4, a5):
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return exponential_model(p, a0 + a1 * A + a2 * d + a3 * A * d, linear_model(d, a4, a5))


def extended_cap_model_4(xy, a0, a1, a2, a3, a4, a5):
    from examples.detailed_fits import quadratic_model
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return quadratic_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5)) / w_x


def extended_cap_model_5(xy, a0, a1, a2, a3, a4, a5, a6, a7):
    from examples.detailed_fits import quadratic_model
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return quadratic_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5)) + linear_model(d, a6, a7) / w_x


def extended_cap_model_6(xy, a0, a1, a2, a3, a4, a5):
    from examples.detailed_fits import quadratic_model
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return quadratic_model(p / w_x, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5))


def extended_cap_model_7(xy, a0, a1, a2, a3):
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return linear_model(p / w_x, linear_model(d, a0, a1), linear_model(d, a2, a3))


def extended_cap_model_8(xy, a0, a1, a2, a3, a4, a5, a6, a7):
    from examples.detailed_fits import quadratic_model
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return quadratic_model(p, linear_model(d, a0, a1) / w_x, linear_model(d, a2, a3), linear_model(d, a4, a5)) + linear_model(
        d, a6, a7) / w_x
