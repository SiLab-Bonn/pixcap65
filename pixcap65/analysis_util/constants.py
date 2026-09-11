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

try:
    # noinspection PyCompatibility
    from collections.abc import Iterable
except ImportError:
    # python 2.7
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from collections import Iterable
finally:
    from typing import Union, Tuple



SI_MOBILITY = 1450
UNITS_ATTRIBUTE_KEY = "Units"
BOUNDARY_TYPE = Union[Tuple, Iterable[Tuple]]
ADVANCED_PARAMETER_TYPE = Union[bool, Iterable[bool]]
SYSTEMATICS_SAMPLE_SIZE = 500  # perhaps this should better be an keyword argument?
REDUCED_SYSTEMATICS_SAMPLE_SIZE = 20
RANDOM_SEED = 42
DISPERSION_PARASITIC_DEVIATION = 3.e-16
BIAS_VOLTAGE_ACCESS_IDX = 0
SLOPE_RESISTIVITY_CONVERSION = 1e12
DOPING_RESULT_TYPE = Tuple[np.ndarray, np.ndarray, int]
