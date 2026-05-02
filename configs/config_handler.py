import enum
import logging.config
import logging.handlers
import os

import sys

try:
    from collections.abc import Iterable
except ImportError:
    # python 2.7 and < python 3.3
    # noinspection PyProtectedMember
    from collections import Iterable

from typing import Callable

import numpy as np
import yaml

CONFIG_KIND_CURRENT = 'current'
CONFIG_KIND_VOLTAGE = 'voltage'
effective_floating_type = np.longdouble
result_floating_type = float


class ConfigElements(enum.StrEnum):
    """
    Keys for the range configuration of the smu channels.
    """
    NORMALISED_RANGE = 'normalised'
    RANGE_UNIT = 'unit'
    RESOLUTION = 'resolution'
    UNIT_PREFIX_VALUE = 'prefix'
    RANGE = 'range'


class ResolutionElements(enum.StrEnum):
    """
    Keys for the resolution configuration of the smu channels.
    """
    ACCURACY = 'accuracy'
    AMPS = 'amps'


# we could use the range configurations to estimate the measurement error of the SMU in use.
def update_smu_range_configuration(config_file):
    """
    update_smu_range_configuration

    This will take a prepared configuration file and add or update the values for the normalised range values,
    which are the ones used for the communication with the Lab device. The changed config will replace the old one.
    The configuration file will be overwritten in the end.
    :param config_file: path of the file containing the range
        and resolution configuration of a Lab Device.
    """
    range_config = {}
    with open(config_file, 'r') as f:
        range_config = yaml.safe_load(f)
        print(range_config)
        handle_range_configuration(CONFIG_KIND_VOLTAGE, range_config)

        handle_range_configuration(CONFIG_KIND_CURRENT, range_config)

    with open(config_file, 'w') as f:
        yaml.safe_dump(range_config, f, encoding='utf-8', allow_unicode=True, sort_keys=False)


def handle_range_configuration(key_spec: str, range_config, config_precision=2):
    """
    handle_range_configuration

    Helper function to handle the range configuration for a particular kind of measurement. It will verify the
    existence of the range prefixes and range quantifiers convert all string representations to float values.
    Afterwards the normalised range value will be calculated and stored in the configuration (as a scientific
    notation string).


    :param key_spec: kind of measurement (current or voltage)
    :param range_config: range configuration mapping
    :param config_precision: precision of the normalised range value in scientific notation.
    """
    if key_spec in range_config:
        for idx in range(len(range_config[key_spec])):
            if ConfigElements.NORMALISED_RANGE in range_config[key_spec][idx]:
                print(
                    "The current normalisation type is "
                    f"{type(range_config[key_spec][idx][ConfigElements.NORMALISED_RANGE])}")
            unit_prefix = verify_range_prefix(idx, key_spec, range_config)

            if isinstance(range_config[key_spec][idx][ConfigElements.RANGE], float):
                range_spec = range_config[key_spec][idx][ConfigElements.RANGE]
            else:
                range_spec = float(range_config[key_spec][idx][ConfigElements.RANGE])
                range_config[key_spec][idx][ConfigElements.RANGE] = range_spec
                assert isinstance(range_config[key_spec][idx][ConfigElements.RANGE], float)

            normalised_range = range_spec * unit_prefix
            assert isinstance(normalised_range, float)
            range_config[key_spec][idx][ConfigElements.NORMALISED_RANGE] = np.format_float_scientific(normalised_range,
                                                                                                      config_precision)
            print(type(range_config[key_spec][idx][ConfigElements.RANGE]))
            print(type(range_config[key_spec][idx][ConfigElements.UNIT_PREFIX_VALUE]))
            print(range_config[key_spec][idx][ConfigElements.UNIT_PREFIX_VALUE])


def verify_range_prefix(idx: int, key_spec: str, range_config) -> float:
    """
    verify_range_prefix

    Helper function to verify the existence of a range prefix and convert it to a float value.
    Will correct the types in the range mapping if necessary.
    :param idx: identifier of the range in the range configuration
    :param key_spec: kind of measurement (current or voltage)
    :param range_config: range configuration mapping
    :return: floating point value of the range prefix
    """
    if isinstance(range_config[key_spec][idx][ConfigElements.UNIT_PREFIX_VALUE], float):
        unit_prefix = range_config[key_spec][idx][ConfigElements.UNIT_PREFIX_VALUE]
    else:
        unit_prefix = float(range_config[key_spec][idx][ConfigElements.UNIT_PREFIX_VALUE])
        range_config[key_spec][idx][ConfigElements.UNIT_PREFIX_VALUE] = unit_prefix
        assert isinstance(range_config[key_spec][idx][ConfigElements.UNIT_PREFIX_VALUE], float)
    return unit_prefix


def smu_handler(smu_file: str, handler: Callable, **kwargs):
    """
    smu_handler

    Wrapper to read the SMU configuration file containing information about the available measurement ranges and
    their resolution and applies the provided delegation handler onto it, to retrieve the requested transformation of
    the supplied data.

    :param smu_file: path to the smu range configuration file
    :param handler: Callable, handler function to apply onto the data while respecting the configuration
    :param kwargs: keyword arguments to be forwarded to the handler callable.
    :return: result of handler application.
    """
    if not os.path.exists(smu_file):
        raise FileNotFoundError(f"The SMU configuration file {smu_file} does not exist")
    with open(smu_file, 'r') as f:
        smu_config = yaml.safe_load(f)
        return handler(smu_config, **kwargs)


def extract_smu_range_error(smu_config: dict, data, range_spec: float, kind: str):
    """
    extract_smu_range_error

    Extract the measurement error of the supplied SMU measurement (current or voltage) for the given range.
    To get the correct error for the given range a comparison to values in the SMU configuration is performed.
    This comparison may suffer from (numerical) rounding errors in the process.

    The error is assumed to composed from a relative part of the measurement (reading error) and an absolute part
    defined by the selected measurement range. It is also possible to perform the calculation directly for an
    Iterable of measurements. In this case the errors are returned as a numpy array.


    :param smu_config: configuration mapping to retrieve the measurement error parameters from.
    :param data: measurement data to calculate the error for. (maybe Iterable)
    :param range_spec: float, range for which the error should be calculated.
    :param kind: kind of the measurement (current or voltage)
    :return: depending on the input data type it's a scalar containing the error or a numpy array containing the errors
        for each measurement.
    """
    assert kind in smu_config
    for entry in smu_config[kind]:
        if np.isclose(effective_floating_type(entry[ConfigElements.NORMALISED_RANGE]), range_spec):
            reading_error = result_floating_type(entry[ConfigElements.RESOLUTION][ResolutionElements.ACCURACY])
            absolute_error = result_floating_type(
                entry[ConfigElements.RESOLUTION][ResolutionElements.ACCURACY.AMPS]
            ) * range_spec
            break
    else:
        raise ValueError(f"Could not find range {range_spec} in SMU configuration")

    if isinstance(data, np.ndarray):
        return reading_error * data + absolute_error
    elif isinstance(data, Iterable):
        return np.array([reading_error * value + absolute_error for value in data])
    elif isinstance(data, float) or isinstance(data, int) or isinstance(data, np.float64):
        return reading_error * data + absolute_error
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def extract_smu_current_error(smu_config: dict, current_data, range_spec: float):
    """
    extract_smu_current_error

    Convenience function to extract the current measurement error for the given range.
    For further information see the documentation of extract_smu_range_error.
    :see: extract_smu_range_error
    :param smu_config: configuration mapping to retrieve the measurement error parameters from.
    :param current_data: measurement data to calculate the error for. (maybe Iterable)
    :param range_spec: float, range for which the error should be calculated.
    :return: depending on the input data type it's a scalar containing the error or a numpy array containing the errors
        for each measurement.
    """
    return extract_smu_range_error(smu_config, current_data, range_spec, CONFIG_KIND_CURRENT)


def extract_smu_voltage_error(smu_config: dict, voltage_data, range_spec: float):
    """
    extract_smu_voltage_error

    Convenience function to extract the voltage measurement error for the given range.
    For further information see the documentation of extract_smu_range_error.
    :param smu_config: configuration mapping to retrieve the measurement error parameters from.
    :param voltage_data: measurement data to calculate the error for. (maybe Iterable)
    :param range_spec: float, range for which the error should be calculated.
    :return: depending on the input data type it's a scalar containing the error or a numpy array containing the errors 
        for each measurement.
    """
    return extract_smu_range_error(smu_config, voltage_data, range_spec, CONFIG_KIND_VOLTAGE)


if __name__ == '__main__':
    script_file = sys.argv[0]
    print(os.path.splitext(script_file))
    print(os.path.dirname(script_file))
    config_file = os.listdir(os.path.dirname(script_file))
    for file in config_file:
        print(file)
        if file.endswith('_Range.yaml'):
            update_smu_range_configuration(file)

    with open("../pixcap_logging.yml", 'r') as f:
        logging.config.dictConfig(yaml.safe_load(f))

    for logger in logging.getLogger().getChildren():
        print(logger.name)
