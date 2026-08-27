"""
Collection of utility function to help with the interaction with the `basil` framework.
"""
# ----------------------------------------------------------
#  Copyright (c) .
#   All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------

from __future__ import annotations

import logging

from pixcap65.pixcap.pixcap_structure import BasilConfigKeys


def extract_basil_layers(adjusted_config) -> tuple[dict, dict, dict]:
    """
    Extract the three layers of a basil configuration (Transfer layer, hardware driver, register) into three different
    mappings.
    These mapping map the name of such a layer element against its specific configuration.

    @author: Dominik Fischer
    last updated: 2026-08-26

    :param adjusted_config: configuration from which to extract the layers.
    :return: tuple of the three layers.
    """
    rl_mapping = {}
    tl_mapping = {}
    hl_mapping = {}

    def _handle_basil_component(configuration, component, result, name):
        if component in configuration:
            for idx, layer in enumerate(configuration[component]):
                if "name" in layer:
                    result[layer["name"]] = idx
                else:
                    logging.info("%s at %i has no name. Will skip it.", name, idx)

    _handle_basil_component(adjusted_config, BasilConfigKeys.TRANSFER_LAYER, tl_mapping, "Transfer layer")
    _handle_basil_component(adjusted_config, BasilConfigKeys.HARDWARE_LAYER, hl_mapping, "Hardware driver")
    _handle_basil_component(adjusted_config, BasilConfigKeys.REGISTER_LAYER, rl_mapping, "Register")
    return hl_mapping, tl_mapping, rl_mapping
