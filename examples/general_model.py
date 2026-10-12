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
Collection of simple models used together for the capacitance modelling in my bachelor thesis.
"""

import numpy as np
from typing import Iterable


def exponential_model(x, a, b):
    """
    Simple exponential model.

    :author: Dominik Fischer
    :date: 2026-07-04

    last update: 2026-09-17

    :param x: independent variable.
    :param a: first phenomenological parameter.
    :param b: second phenomenological parameter.
    :return: model prediction.
    """
    return a * np.exp(b * x)


def quadratic_model(x, a, b, c):
    """
    Quadratic model.

    :author: Dominik Fischer
    :date: 2026-09-01

    last update: 2026-09-17

    :param x: independent variable.
    :param a: first phenomenological parameter.
    :param b: second phenomenological parameter.
    :param c: third phenomenological parameter.
    :return: model prediction.
    """
    return a + b * x + c * x ** 2


def quadratic_model_grad(x, a, b, c):
    """
    Parameter space gradient of the (simple) quadratic model
    :py:func:`quadratic_model`.

    :author: Dominik Fischer
    :date: 2026-09-01

    last update: 2026-09-17

    :param x: independent variable.
    :param a: first phenomenological parameter.
    :param b: second phenomenological parameter.
    :param c: third phenomenological parameter.
    :return: gradient vector corresponding to model prediction.
    """
    if isinstance(x, Iterable):
        return np.vstack([np.array([1, val, val ** 2]) for val in x]).T
    else:
        return np.array([1, x, x ** 2])


def linear_model_grad(x, a, b):
    """
    Parameter space gradient of the (simple) linear model
    :py:func:`linear_model`.

    :author: Dominik Fischer
    :date: 2026-09-01

    last update: 2026-09-17

    :param x: independent variable.
    :param a: first phenomenological parameter.
    :param b: second phenomenological parameter.
    :return: gradient vector corresponding to model prediction.
    """
    if isinstance(x, Iterable):
        return np.vstack([np.array([1, val]) for val in x]).T
    else:
        return np.array([1, x])


def exponential_model_grad(x, a, b):
    """
    Parameter space gradient of the exponential model
    :py:func:`exponential_model`.

    :author: Dominik Fischer
    :date: 2026-09-01

    last update: 2026-09-17

    :param x: independent variable.
    :param a: first phenomenological parameter.
    :param b: second phenomenological parameter.
    :param c: third phenomenological parameter.
    :return: gradient vector corresponding to model prediction.
    """
    if isinstance(x, Iterable):
        return np.vstack([np.array([np.exp(b * val), a * b * np.exp(b * val)]) for val in x]).T
    else:
        return np.array([np.exp(b * x), a * b * np.exp(b * x)])


def get_polynomial_model(dof):
    """
    Polynomial model of arbitrary degree.

    :author: Dominik Fischer
    :date: 2026-09-01

    last update: 2026-09-17

    :param dof: degree of the polynomial. (highest power used for the independent variable)
    :return: callable to evaluate the model (its predictions)
    """
    def _method(x, *args):
        return np.polyval(args, x)

    return _method


def linear_model(x, a, b):
    """
    linear model (straight line modelling).

    :author: Dominik Fischer
    :date: 2026-09-01

    last update: 2026-09-17

    :param x: independent variable.
    :param a: first phenomenological parameter.
    :param b: second phenomenological parameter.
    :return: model prediction.
    """
    return a + x * b


def reciprocal_model(x, a, b):
    """
    reciprocal model (1/x).

    :author: Dominik Fischer
    :date: 2026-09-01

    last update: 2026-09-17

    :param x: independent variable.
    :param a: first phenomenological parameter.
    :param b: second phenomenological parameter.
    :return: model prediction.
    """
    return a + b / x


def reciprocal_deriv(x, a, b):
    """
    Derivative in x of a reciprocal model.

    :author: Dominik Fischer
    :date: 2026-09-01

    last update: 2026-09-17

    :param x: independent variable.
    :param a: first phenomenological parameter.
    :param b: second phenomenological parameter.
    :return: model prediction.
    """
    return - b / x ** 2


def inverted_reciprocal_model(y, a, b):
    """
    Helper function to invert a reciprocal model.

    :author: Dominik Fischer
    :date: 2026-09-01

    last update: 2026-09-17

    :param y: independent variable.
    :param a: first phenomenological parameter.
    :param b: second phenomenological parameter.
    :return: model prediction. (Inversion)
    """
    return b / (y - a)


def combined_model(x, a, b, c):
    """
    Combination of linear model and a reciprocal model (additive).

    :author: Dominik Fischer
    :date: 2026-09-01

    last update: 2026-09-17

    :param x: independent variable.
    :param a: first phenomenological parameter.
    :param b: second phenomenological parameter.
    :param c: third phenomenological parameter.
    :return: model prediction.
    """
    return a + b * x + c / x
