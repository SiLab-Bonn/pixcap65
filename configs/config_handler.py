import enum
import sys
import os
from collections.abc import Iterable
from typing import Callable

import yaml
import numpy as np

# TODO: add the functions doc strings

CONFIG_KIND_CURRENT = 'current'
CONFIG_KIND_VOLTAGE = 'voltage'

class CONFIG_ELEMENTS(enum.Enum):
    NORMALISED_RANGE = 'normalised'
    RANGE_UNIT = 'unit'
    RESOLUTION = 'resolution'
    UNIT_PREFIX_VALUE = 'prefix'
    RANGE = 'range'

class RESOLUTION_ELEMENTS(enum.Enum):
    ACCURACY = 'accuracy'
    AMPS = 'amps'

# we could use the range configurations to estimate the measurement error of the SMU in use.
def update_smu_range_configuration(config_file):
    range_config = {}
    with open(config_file, 'r') as f:
        range_config = yaml.safe_load(f)
        print(range_config)
        key_spec = CONFIG_KIND_VOLTAGE
        if key_spec in range_config:
            for idx in range(len(range_config[key_spec])):
                if CONFIG_ELEMENTS.NORMALISED_RANGE in range_config[key_spec][idx]:
                    print(f"The current normalisation type is {type(range_config[key_spec][idx][CONFIG_ELEMENTS.NORMALISED_RANGE])}")
                if isinstance(range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE], float):
                    unit_prefix = range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE]
                else:
                    unit_prefix = float(range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE])
                    range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE] = unit_prefix
                    assert isinstance(range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE], float)

                if isinstance(range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE], float):
                    range_spec = range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE]
                else:
                    range_spec = float(range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE])
                    range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE] = range_spec
                    assert isinstance(range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE], float)

                normalised_range = range_spec * unit_prefix
                assert isinstance(normalised_range, float)
                range_config[key_spec][idx][CONFIG_ELEMENTS.NORMALISED_RANGE] = normalised_range
                print(type(range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE]))
                print(type(range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE]))
                print(range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE])

        key_spec = CONFIG_KIND_CURRENT
        if key_spec in range_config:
            for idx in range(len(range_config[key_spec])):
                if CONFIG_ELEMENTS.NORMALISED_RANGE in range_config[key_spec][idx]:
                    print(
                        f"The current normalisation type is {type(range_config[key_spec][idx][CONFIG_ELEMENTS.NORMALISED_RANGE])}")
                if isinstance(range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE], float):
                    unit_prefix = range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE]
                else:
                    unit_prefix = float(range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE])
                    range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE] = unit_prefix
                    assert isinstance(range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE], float)

                if isinstance(range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE], float):
                    range_spec = range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE]
                else:
                    range_spec = float(range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE])
                    range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE] = range_spec
                    assert isinstance(range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE], float)

                normalised_range = range_spec * unit_prefix
                assert isinstance(normalised_range, float)
                range_config[key_spec][idx][CONFIG_ELEMENTS.NORMALISED_RANGE] = normalised_range
                print(type(range_config[key_spec][idx][CONFIG_ELEMENTS.RANGE]))
                print(type(range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE]))
                print(range_config[key_spec][idx][CONFIG_ELEMENTS.UNIT_PREFIX_VALUE])

    with open(config_file, 'w') as f:
        yaml.safe_dump(range_config, f, encoding='utf-8', allow_unicode=True, sort_keys=False)

def smu_handler(smu_file: str, handler: Callable, **kwargs):
    if not os.path.exists(smu_file):
        raise FileNotFoundError(f"The SMU configuration file {smu_file} does not exist")
    with open(smu_file, 'r') as f:
        smu_config = yaml.safe_load(f)
        return handler(smu_config, **kwargs)


def extract_smu_range_error(smu_config: dict, data, range: float, kind: str):
    assert kind in smu_config
    for entry in smu_config[kind]:
        if entry[CONFIG_ELEMENTS.NORMALISED_RANGE] == range:
            reading_error = float(entry[CONFIG_ELEMENTS.RESOLUTION][RESOLUTION_ELEMENTS.ACCURACY])
            absolute_error = float(entry[CONFIG_ELEMENTS.RESOLUTION][RESOLUTION_ELEMENTS.ACCURACY.AMPS])
            break
    else:
        raise ValueError(f"Could not find range {range} in SMU configuration")

    if isinstance(data, np.ndarray):
        return reading_error * data + absolute_error
    elif isinstance(data, Iterable):
        return np.array([reading_error * value + absolute_error for value in data])
    elif isinstance(data, float) or isinstance(data, int) or isinstance(data, np.float64):
        return reading_error * data + absolute_error
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def extract_smu_current_error(smu_config: dict, current_data, range:float):
    return extract_smu_range_error(smu_config, current_data, range, CONFIG_KIND_CURRENT)


def extract_smu_voltage_error(smu_config: dict, voltage_data, range:float):
    return extract_smu_range_error(smu_config, voltage_data, range, CONFIG_KIND_VOLTAGE)


if __name__ == '__main__':
    script_file = sys.argv[0]
    print(os.path.splitext(script_file))
    print(os.path.dirname(script_file))
    config_file = os.listdir(os.path.dirname(script_file))
    for file in config_file:
        print(file)
        if file.endswith('_Range.yaml'):
            update_smu_range_configuration(file)



