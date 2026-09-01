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
    return a * np.exp(b * x)


def quadratic_model(x, a, b, c):
    return a + b * x + c * x ** 2


def quadratic_model_grad(x, a, b, c):
    if isinstance(x, Iterable):
        return np.vstack([np.array([1, val, val ** 2]) for val in x]).T
    else:
        return np.array([1, x, x ** 2])


def linear_model_grad(x, a, b):
    if isinstance(x, Iterable):
        return np.vstack([np.array([1, val]) for val in x]).T
    else:
        return np.array([1, x])


def exponential_model_grad(x, a, b):
    if isinstance(x, Iterable):
        return np.vstack([np.array([np.exp(b * val), a * b * np.exp(b * val)]) for val in x]).T
    else:
        return np.array([np.exp(b * x), a * b * np.exp(b * x)])


def get_polynomial_model(dof):
    def method(x, *args):
        return np.polyval(args, x)

    return method


def linear_model(x, a, b):
    return a + x * b


def reciprocal_model(x, a, b):
    return a + b / x


def reciprocal_deriv(x, a, b):
    return - b / x ** 2


def inverted_reciprocal_model(y, a, b):
    return b / (y - a)


def combined_model(x, a, b, c):
    return a + b * x + c / x
