# ----------------------------------------------------------
#  Copyright (c) .
#   All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------
"""
Analysis module to implement the correction of measured capacitances for the parasitic capacitances' of the measurement
circuit.
"""
import logging

import numpy as np
import tables as tb
import threading
from typing import Optional, Any

logger = logging.getLogger(__name__)
from pixcap65.analysis_util.utility import get_base_group, PARASITIC_SUBTRACTION
from pixcap65.utility import synchronized_process_open_file
from pixcap65.utility.tables_util import get_parent_group, group_get_file, get_groups, get_leaves, copy_node, \
    list_attributes, get_group_attributes, get_group_attribute
from pixcap65.utility.utils_2 import GroupType, prevent_group_mix_up, walk_to_node


def apply_correction(raw_data, base_path=None, bare_data_path=None, bare_group=None, **tb_kwargs):
    """
    apply_correction

    @author: Dominik Fischer
    last update: 2026-08-12

    Wrapper function to handle the file access, when correcting the capacitance data tables.

    :param raw_data: hdf file containing the measurements and investigation of a pix cap sample which needs correction
    :param base_path: hdf files group with the analysis data/capacitance data to be corrected.
    :param bare_data_path: file containing the analysis of the bare pix cap sample to be used for
        capacitance correction.
    :param bare_group: hdf files group for the bare analysis (holding the parasitic capacitance information)
    """
    # first extract the parasitic capacitance
    assert bare_data_path is not None
    with synchronized_process_open_file(raw_data, mode='a', **tb_kwargs) as in_file_h5_inner:
        base_group = get_base_group(base_path, in_file_h5_inner)
        apply_correction_simple(bare_data_path, bare_group, base_group.analysis)


def apply_correction_simple(bare_data_path: str, bare_path: str, analysis_group: tb.Group, **kwargs):
    """
    apply_correction_simple

    @author: Dominik Fischer
    last update: 2026-08-12

    Minimal wrapper for the correction of the measured capacitance for the capacitance of the bump-bond and the
    switching circuit itself.
    It will open the data file and analysis group of the bare pix cap measurement, extract the parasitic capacitance
    (including their uncertainties which will be applied as systematic uncertainties to the measurement), correct for
    these effect and write all the results of the current analysis with the correction applied back.
    As the bump-bond capacitance and the intrinsic capacitance of the switching circuit can be considered as being
    parallel no further effects needs to be accounted for. Also, these capacitance could be considered being parallel
    to the pixel sensors capacitance. So the correction could be done by a simple subtraction.


    :param bare_data_path: path to the hdf file of the bare pix cap measurement to obtain information about parasitic
        capacitance.
    :param bare_path: hdf files hierarchy path to the group containing the bare pix cap analysis with the information
        about the parasitic after investigating the capacitance distribution.
    :param analysis_group: hdf files hierarchy group with the analysis results which needs to be corrected for the
        intrinsic and parasitic effects.
    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    """
    with synchronized_process_open_file(bare_data_path, mode='r', **kwargs) as in_file_h5:
        parasitic, parasitic_error = _handle_parasitic_cap(bare_path, in_file_h5)
        # second perform correction of the capacitance data
        apply_correction_delegate(parasitic, parasitic_error, analysis_group)


def apply_correction_delegate(parasitic, parasitic_error, group: GroupType):
    """
    apply_correction_delegate

    @author: Dominik Fischer
    last update: 2026-08-12

    Implementation of the capacitance correction for the parasitic capacitance of the bump-bond and the intrinsic
    capacitance of the switching circuit.
    As the bump-bond capacitance and the intrinsic capacitance of the switching circuit can be considered as being
    parallel no further effects needs to be accounted for. Also, these capacitance could be considered being parallel
    to the pixel sensors capacitance. So the correction could be done by a simple subtraction.

    To make it possible to still plot all the data later on, the new tables get also the parasitic capacitance
    subtracted as an attribute.

    :param parasitic: parasitic capacitance to correct for.
    :param parasitic_error: (systematic) error of the parasitic capacitance to correct for.
    :param group: hdf files group containing the capacitance data to be corrected.
    """
    assert isinstance(group, tb.Group)
    # define the corrected analysis group
    prevent_group_mix_up(get_parent_group(group), "analysis_correction")

    file_h5 = group_get_file(group)
    correction_group = file_h5.create_group(where=get_parent_group(group), name="analysis_correction")

    # first extract the capacitance histograms (errors shall be copied to the new group
    _correct_data(correction_group, file_h5, group, parasitic, parasitic_error)

    # will need to handle also the subgroups (only one level down)!
    for group_key, next_group in get_groups(group):
        next_correction_group = file_h5.create_group(where=correction_group, name=group_key)
        _correct_data(next_correction_group, file_h5, next_group, parasitic, parasitic_error)


def _correct_data(correction_group: tb.Group, file_h5: tb.File, group: tb.Group, parasitic, parasitic_error):
    for key, value in get_leaves(group):
        assert isinstance(value, tb.Leaf)
        if "Cap" in key:
            logger.debug("Found the key %s for transferring capacitance data.", key)
            if "Err" in key:
                copy_node(value, newparent=correction_group)
            else:
                assert isinstance(value, tb.Array) or isinstance(value, tb.Table)
                cap_data_hist = value[:]
                assert isinstance(cap_data_hist, np.ndarray)
                corrected_cap_data = np.where(np.isfinite(cap_data_hist), cap_data_hist - parasitic * 1.e-15, np.nan)
                corrected_cap_systematics = np.full_like(corrected_cap_data, fill_value=parasitic_error)
                temp_data_array = file_h5.create_carray(where=correction_group, name=key, title=value.title,
                                                        filters=value.filters, obj=corrected_cap_data)
                # careful the parasitic capacitance are provided in fF were all other capacitance are saved in F
                temp_data_array.attrs[PARASITIC_SUBTRACTION] = parasitic
                temp_error_array = file_h5.create_carray(where=correction_group, name="{}Systematics".format(key),
                                                         title=value.title,
                                                         filters=value.filters, obj=corrected_cap_systematics)
                assert isinstance(value, tb.Leaf)
                for name in list_attributes(value):
                    temp_data_array.attrs[name] = value.attrs[name]
                    temp_error_array.attrs[name] = value.attrs[name]
        else:
            copy_node(value, newparent=correction_group)


def _handle_cap_correction(result_group: tb.Group, **kwargs):
    """
    _handle_cap_correction

    @author: Dominik Fischer
    last update: 2026-08-12

    (Internal) Utility function to correct the measured capacitance for the parasitic ones by the measurement circuit.

    :param result_group: hdf files' group where the analysis results were written to.
    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :keyword apply_correction: boolean, apply_correction: boolean, indicates whether the measured capacitance should be corrected
        immediately; Will require the presence of further arguments as information about the parasitic capacitance needs to be
        submitted. (data corrected for parasitic capacitances of PixCap65, default: False)
    :keyword bare_file: hdf file containing the measurements and investigation of a bare pix cap sample to obtain
        information about intrinsic and parasitic capacitance. (Only required for the correction procedure, but in
        this case it must be present)
    :keyword bare_path: hdf files hierarchy path to the group containing the bare pix cap analysis with the information
        about the parasitic after investigating the capacitance distribution. (Only required for
        the correction procedure, but in this case it must be present)
    """
    global cap_counter
    cap_counter += 1
    if "lock" not in kwargs:
        try:
            lock = threading.RLock()
            kwargs["lock"] = lock
            _handle_cap_correction(result_group, **kwargs)
            return
        finally:
            del lock

    if "apply_correction" in kwargs and kwargs["apply_correction"]:
        assert 'bare_file' in kwargs
        assert 'bare_hdf_path' in kwargs
        apply_correction_simple(kwargs['bare_file'], kwargs['bare_hdf_path'], result_group, lock=kwargs['lock'])


def _extract_table_data(key: str, is_corrected: bool, table: np.ndarray, ):
    """
    extract_table_data

    @author: Dominik Fischer
    last update: 2026-08-13

    Wrapper for the extraction of table column data. Only relevant to simplify the implementation for the usage of
    corrected capacitance data. It replaces some of the keys to extract capacitance data from the table by the corrected
    table columns.

    :type key: str
    :param key: name of the original column to extract data from
    :type is_corrected: bool
    :param is_corrected: boolean, whether to use the corrected capacitance data.
    :param table: table (as a structured numpy array) to extract data from
    :return: extracted data
    """
    if is_corrected:
        match (key):
            case 'capacitance':
                return table['cap_corrected']
            case 'cap_std':
                return table['cap_corrected_err']
            case _:
                return table[key]
    else:
        return table[key]


def _handle_parasitic_cap(bare_path: Optional[str], in_file_h5: tb.File) -> tuple[Any, Any]:
    """
    _handle_parasitic_cap

    @author: Dominik Fischer
    last update: 2026-08-13

    (Internal) helper function to fetch the parasitic capacitance and it's spread over a full sensor/chip from the
    dedicated analysis file.

    :param bare_path: path of the analysis group within the provided file.
    :param in_file_h5: path/filename of the .h5-file storing the analysis data of a full measurement of a bare PixCap65 chip.
    :return: tuple of parasitic capacitance and it's spread over a single sensor.
    :rtype: tuple
    """
    if bare_path is None:
        bare_group = in_file_h5.root
    else:
        bare_group, _ = walk_to_node(in_file_h5.root, bare_path, verify_create=True)

    assert "parasitic" in get_group_attributes(bare_group.analysis)
    assert "parasitic_error" in get_group_attributes(bare_group.analysis)
    parasitic = get_group_attribute(bare_group.analysis, "parasitic")
    parasitic_error = get_group_attribute(bare_group.analysis, "parasitic_error")
    return parasitic, parasitic_error
