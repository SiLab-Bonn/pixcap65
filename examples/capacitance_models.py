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
    """
    simplified capacitance model

    :author: Dominik Fischer
    :date: 2026-07-04

    last update: 2026-09-17

    Simplified model of the total pixel capacitance assuming a linear increate with depth and pixel/implantation area.

    :param xy: tuple of the dependent data (area, depth) (is it really depth?).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :return: model prediction.
    """
    A, W = xy
    return a0 + a1 * A + a2 * W


def extended_cap_model(xy, a0, a1, a2, a3, a4, a5):
    """
    extended capacitance model 01

    :author: Dominik Fischer
    :date: 2026-07-04

    last update: 2026-09-17

    Extended capacitance model which considers besides the implanation area and the depth also the perimeter of the
    pixel implanation is considered.
    The area dependence is ignored mostly and the depth only considered as a weak linear dependence on other model
    parameters.
    The perimeter dependence is modelled as a quadratic function.

    :param xy: tuple of the dependent data (area, depth, perimeter, x-separation, y-separation).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :param a4: fifth phenomenological model parameter.
    :param a5: sixth phenomenological model parameter.
    :return: model prediction.
    """
    A, d, p, separation_x, separation_y = xy
    return quadratic_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5))


def extended_cap_model_2(xy, a0, a1, a2, a3):
    """
    extended capacitance model 02

    :author: Dominik Fischer
    :date: 2026-07-04

    last update: 2026-09-17

    Extended capacitance model which considers besides the implanation area and the depth also the perimeter of the
    pixel implanation is considered.
    The area dependence is ignored mostly and the depth only considered as a weak linear dependence on other model
    parameters.
    The perimeter dependence is modelled as a exponential decay (negative parameter).

    :param xy: tuple of the dependent data (area, depth, perimeter, x-separation, y-separation).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :return: model prediction.
    """
    A, d, p, separation_x, separation_y = xy
    return exponential_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3))


def extended_cap_model_3(xy, a0, a1, a2, a3, a4, a5):
    """
    extended capacitance model 03

    :author: Dominik Fischer
    :date: 2026-07-04

    last update: 2026-09-17

    Extended capacitance model which considers besides the implanation area and the depth also the perimeter of the
    pixel implanation is considered.
    The area dependence and the depth only considered as a weak linear dependence on other model
    parameters.
    The perimeter dependence is modelled as a exponential decay (negative parameter).

    :param xy: tuple of the dependent data (area, depth, perimeter, x-separation, y-separation).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :param a4: fifth phenomenological model parameter.
    :param a5: sixth phenomenological model parameter.
    :return: model prediction.
    """
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return exponential_model(p, a0 + a1 * A + a2 * d + a3 * A * d, linear_model(d, a4, a5))


def extended_cap_model_4(xy, a0, a1, a2, a3, a4, a5):
    """
    extended capacitance model 04

    :author: Dominik Fischer
    :date: 2026-08-11

    last update: 2026-09-17

    Extended capacitance model which considers besides the implanation area and the depth also the perimeter of the
    pixel implanation and the distance between/separtion of neighboured pixels is considered.
    The area dependence is ignored mostly and the depth only considered as a weak linear dependence on other model
    parameters.
    The perimeter dependence is modelled as a quadratic function relative to the pixel separation.
    (quadratic form) / (pixel separation)

    :param xy: tuple of the dependent data (area, depth, perimeter, x-separation, y-separation).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :param a4: fifth phenomenological model parameter.
    :param a5: sixth phenomenological model parameter.
    :return: model prediction.
    """
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return quadratic_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5)) / w_x


def extended_cap_model_5(xy, a0, a1, a2, a3, a4, a5, a6, a7):
    """
    extended capacitance model 05

    :author: Dominik Fischer
    :date: 2026-07-04

    last update: 2026-09-17

    Extended capacitance model which considers besides the implanation area and the depth also the perimeter of the
    pixel implanation and the difference between/separation of neighboured pixels is considered.
    The area dependence is ignored mostly and the depth only considered as a weak linear dependence on other model
    parameters.
    The perimeter dependence is modelled as a quadratic function, while the dependence on the pixel separation is
    modelled by an additive reciprocal term.

    :param xy: tuple of the dependent data (area, depth, perimeter, x-separation, y-separation).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :param a4: fifth phenomenological model parameter.
    :param a5: sixth phenomenological model parameter.
    :param a6: seventh phenomenological model parameter.
    :param a7: eighth phenomenological model parameter.
    :return: model prediction.
    """
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return quadratic_model(p, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5))\
        + linear_model(d, a6, a7) / w_x


def extended_cap_model_6(xy, a0, a1, a2, a3, a4, a5):
    """
    extended capacitance model 06

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
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return quadratic_model(p / w_x, linear_model(d, a0, a1), linear_model(d, a2, a3), linear_model(d, a4, a5))


def extended_cap_model_7(xy, a0, a1, a2, a3):
    """
    extended capacitance model 07

    :author: Dominik Fischer
    :date: 2026-07-04

    last update: 2026-09-17

    Extended capacitance model which considers besides the implanation area and the depth also the perimeter of the
    pixel implanation and the distance between/separation of neighboured pixels is considered.
    The area dependence is ignored mostly and the depth only considered as a weak linear dependence on other model
    parameters.
    The perimeter dependence is modelled as a linear function.
    This model only accounts for the combined quantity perimeter/separation.

    :param xy: tuple of the dependent data (area, depth, perimeter, x-separation, y-separation).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :return: model prediction.
    """
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return linear_model(p / w_x, linear_model(d, a0, a1), linear_model(d, a2, a3))


def extended_cap_model_8(xy, a0, a1, a2, a3, a4, a5, a6, a7):
    """
    extended capacitance model 08

    :author: Dominik Fischer
    :date: 2026-07-04

    last update: 2026-09-17

    Extended capacitance model which considers besides the implanation area and the depth also the perimeter of the
    pixel implanation and the distance between/separation of neighboured pixels is considered.
    The area dependence is ignored mostly and the depth only considered as a weak linear dependence on other model
    parameters.
    The perimeter dependence is modelled as a quadratic function.
    The separation is modelled by an additive reciprocal term, which parameter depends linearly on the depth.
    Where is the difference to :py:func:`examples.capacitance_models.extended_cap_model_5`.

    :param xy: tuple of the dependent data (area, depth, perimeter, x-separation, y-separation).
    :param a0: first phenomenological model parameter.
    :param a1: second phenomenological model parameter.
    :param a2: third phenomenological model parameter.
    :param a3: fourth phenomenological model parameter.
    :param a4: fifth phenomenological model parameter.
    :param a5: sixth phenomenological model parameter.
    :return: model prediction.
    """
    # make a try in not using depletion width at all, but in using the pixel separation here!
    A, d, p, w_x, w_y = xy
    return quadratic_model(p, linear_model(d, a0, a1) / w_x, linear_model(d, a2, a3), linear_model(d, a4, a5)) + linear_model(
        d, a6, a7) / w_x
