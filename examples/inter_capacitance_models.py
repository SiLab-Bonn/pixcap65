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
Collection of inter-pixel capacitance models attempted in my bachelor thesis.
"""

import numpy as np

from examples.general_model import quadratic_model, linear_model


def inter_cap_model(xy, a0, a1, a2, a3, a4):
    """
    exponential inter-pixel capacitance model 01.

    :author: Dominik FIscher
    :date: 2026-07-04

    last update: 2026-09-17

    Simple model for inter-pixel capacitances' using a exponential decay of the capacitance
    for increasing distance between/separation of neighboured pixels in combination with a linear
    increase by the implantations area, depth and perimeter.

    :param xy: tuple of the dependent data (area, depth, perimeter, pixel separation in both dimensions).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :param a4: fifth phenomenological model parameter.
    :return: model prediction.
    """
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(- a4 * separation_x) + a1 * A + a2 * W + a3 * p

def inter_cap_model_2(xy, a0, a3):
    """
    exponential inter-pixel capacitance model 02.

    :author: Dominik FIscher
    :date: 2026-07-04

    last update: 2026-09-17

    Model for inter-pixel capacitances' using a exponential decay of the capacitance
    for decreasing pixel implanation perimeter (description explicitly inverts the signs
    of the parameters).

    :param xy: tuple of the dependent data (area, depth, perimeter, pixel separation in both dimensions).
    :param a0: first phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :return: model prediction.
    """
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(a3 * p)

def inter_cap_model_3(xy, a0, a1, a2, a3):
    """
    exponential inter-pixel capacitance model.

    :author: Dominik FIscher
    :date: 2026-07-04

    last update: 2026-09-17

    Simple model for inter-pixel capacitances' using a exponential decay of the capacitance
    for decreasing the quantity perimeter/separation in combination with a linear
    increase by the implantations area, depth.
    The pixel separation could be defined here as the distance between neighbouring pixels.
    The model accounts only for the pixel separation in one of the two dimensions.

    :param xy: tuple of the dependent data (area, depth, perimeter, pixel separation in both dimensions).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :return: model prediction.
    """
    A, W, p, separation_x, separation_y = xy
    return a0 * np.exp(a3 * p / separation_x) + a1 * A + a2 * W

def inter_cap_model_4(xy, a0, a1, a2, a3, a4, a5):
    """
    quadratic inter-pixel capacitance model 01.

    :author: Dominik FIscher
    :date: 2026-07-04

    last update: 2026-09-17

    Simple model for inter-pixel capacitances' using a quadratic dependency of the capacitance
    on the distance between/separation of neighboured pixels.
    The model parameters feature a (weak) dependency on the depth of the
    pixel implantation.

    :param xy: tuple of the dependent data (area, depth, perimeter, pixel separation in both dimensions).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :param a4: fifth phenomenological model parameter.
    :param a5: sixth phenomenological model parameter.
    :return: model prediction.
    """
    # exclude this model as it leads to small cost functions but with parameter uncertainties which are in general
    # larger than the parameters itself.
    A, d, p, separation_x, separation_y = xy
    return quadratic_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5))

def inter_cap_model_5(xy, a0, a1, a2, a3, a4, a5):
    """
    quadratic inter-pixel capacitance model 03.

    :author: Dominik Fischer
    :date: 2026-07-04

    last update: 2026-09-17

    Extended capacitance model which considers besides the implanation area and the depth also the perimeter of the
    pixel implanation and distance between/separation of neighboured pixels is considered.
    The area dependence is ignored mostly and the depth only considered as a weak linear dependence on other model
    parameters.
    The perimeter dependence is modelled as a quadratic function.
    This model only accounts for the combined quantity perimeter/separation.

    :param xy: tuple of the dependent data (area, depth, perimeter, x-separation, y-separation).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :param a4: fifth phenomenological model parameter.
    :param a5: sixth phenomenological model parameter.
    :return: model prediction.
    """
    from examples.capacitance_models import extended_cap_model_6
    return extended_cap_model_6(xy, a0, a1, a2, a3, a4, a5)

def inter_cap_model_6(xy, a0, a1, a2, a3):
    """
    linear inter-pixel capacitance model 01.

    :author: Dominik FIscher
    :date: 2026-07-04

    last update: 2026-09-17

    Simple model for inter-pixel capacitances' using a linear dependency of the capacitance
    on the quantity (perimeter/distance between/separation of neighboured pixels).
    The model parameters feature a (weak) dependency on the depth of the
    pixel implantation.

    :param xy: tuple of the dependent data (area, depth, perimeter, pixel separation in both dimensions).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :return: model prediction.
    """
    A, d, p, separation_x, separation_y = xy
    return linear_model(p / separation_x, linear_model(d, a0, a1), linear_model(d, a2, a3))

def inter_cap_model_7(xy, a0, a1, a2, a3, a4, a5, a6, a7):
    """
    quadratic inter-pixel capacitance model 02.

    :author: Dominik FIscher
    :date: 2026-07-04

    last update: 2026-09-17

    Simple model for inter-pixel capacitances' using a quadratic dependency of the capacitance
    on the distance between/separation of neighboured pixels in combination with an additive reciprocal
    dependency on the distance between/separation of neighboured pixels.
    The model parameters feature a (weak) dependency on the depth of the
    pixel implantation.

    :param xy: tuple of the dependent data (area, depth, perimeter, pixel separation in both dimensions).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :param a4: fifth phenomenological model parameter.
    :param a5: sixth phenomenological model parameter.
    :return: model prediction.
    """
    # exclude this model as it leads to small cost functions but with parameter uncertainties which are in general
    # larger than the parameters itself.
    A, d, p, separation_x, separation_y = xy
    return quadratic_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5))\
        + linear_model(d, a6, a7) / separation_x
