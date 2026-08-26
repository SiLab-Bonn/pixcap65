"""
Definitions for the structure of a Pixcap65 measurement setup.
Providing in particular the allowed configuration keys as enumerations.
"""
from enum import StrEnum


class BasilConfigKeys(StrEnum):
    """
    Enumeration of configuration keys to simplify the access to the `basil` configuration file for a lab setup.
    """
    TRANSFER_LAYER = 'transfer_layer'
    HARDWARE_LAYER = 'hw_drivers'
    REGISTER_LAYER = 'registers'
