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

from pixcap65.analysis_util import RANDOM_SEED


def __generate_gaussian_samples(loc: np.ndarray, scale: float, size: int, rng: np.random.Generator) -> np.ndarray:
    result_shape = tuple((*loc.shape, size))
    result_data = np.zeros(result_shape, dtype=np.float64)
    for indices in np.ndindex(*loc.shape):
        result_data[indices] = rng.normal(loc[indices], scale, size)

    return result_data


def __second_generate_gaussian_samples(loc: np.ndarray, scale: float, size: int,
                                       rng: np.random.Generator) -> np.ndarray:
    result_shape = tuple((*loc.shape, size))
    result_data = np.zeros(result_shape, dtype=np.float64)
    for indices in np.ndindex(*loc.shape):
        result_data[indices] = rng.normal(loc[indices], scale, size)

    return np.moveaxis(result_data, -1, 0)


global_rng = np.random.default_rng(RANDOM_SEED)


def get_rng():
    """
    get_rng

    @author: Dominik Fischer
    @date: 2026-08-12

    Helper function to spawn a new random number generator for each boostrapping step within the analysis.

    :return: requested numpy-based random number generator.
    """
    return global_rng.spawn(1)[0]
