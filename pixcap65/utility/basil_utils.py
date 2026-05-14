
# ----------------------------------------------------------
#  Copyright (c) .
#   All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------

from __future__ import annotations

from pixcap.pixcap_structure import BasilConfigKeys
from pixcap_65_test_total_cap import logger


def extract_basil_layers(adjusted_config) -> tuple[dict, dict, dict]:
    rl_mapping = {}
    tl_mapping = {}
    hl_mapping = {}

    def handle_basil_component(configuration, component, result, name):
        if component in configuration:
            for idx, layer in enumerate(configuration[component]):
                if "name" in layer:
                    result[layer["name"]] = idx
                else:
                    logger.info("%s at %i has no name. Will skip it.", name, idx)

    handle_basil_component(adjusted_config, BasilConfigKeys.TRANSFER_LAYER, tl_mapping, "Transfer layer")
    handle_basil_component(adjusted_config, BasilConfigKeys.HARDWARE_LAYER, hl_mapping, "Hardware driver")
    handle_basil_component(adjusted_config, BasilConfigKeys.REGISTER_LAYER, rl_mapping, "Register")
    return hl_mapping, tl_mapping, rl_mapping
