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
from typing import Optional, Any

from pixcap65.analysis_util.correction import _handle_parasitic_cap
from pixcap65.analysis_util.delegation.depletion import __depletion_iterator_implementation
from pixcap65.pixcap.pixcap_structure import CAPACITANCE_CONVERSION_FACTOR
from pixcap65.utility import synchronized_process_open_file


def __mp_init_distribution_delegate(cap_data, kargs):
    global mp_shared_cap_ref, mp_shared_keyword_args
    mp_shared_cap_ref = cap_data
    mp_shared_keyword_args = kargs
    if "lock" in kargs:
        lock = kargs["lock"]
        print("found a lock:", type(lock))
        if hasattr(lock, "_token"):
            print(lock._token)
        if hasattr(lock, "_serial"):
            print(lock._serial)


def __mp_handle_distribution_delegate(offset):
    from pixcap65.analysis_util.delegation.distribution import analyze_capacitance_distribution_delegate

    global mp_shared_cap_ref, mp_shared_keyword_args
    return analyze_capacitance_distribution_delegate(None, None, capacitance=mp_shared_cap_ref + offset, convert=False,
                                                     **mp_shared_keyword_args)


def _handle_mp_parasitic_cap(bare_path: Optional[str], bare_file: Optional[str], **kwargs) -> tuple[Any, Any]:
    """
    _handle_mp_parasitic_cap

    @author: Dominik Fischer
    last update: 2026-08-13

    (Internal) helper function to fetch the parasitic capacitance and it's spread over a full sensor/chip from the
    dedicated analysis file in case of using for multiprocessing for speeding up analysis.
    As in this case we must prevent any simultaneous access to the bare data file from multiple processes.


    :param bare_path: path of the analysis group within the provided file.
    :param bare_file: path/filename of the .h5-file storing the analysis data of a full measurement of a bare PixCap65 chip.
    :keyword lock: synchronization object to prevent multiple overlapping accesses to the pytables api and simultaneously
        write/read operations on the same file.
    :param kwargs: provide further keywords for opening a pytables.File object.
    :return: tuple of parasitic capacitance and it's spread over a single sensor.
    :rtype: tuple
    """
    if bare_file is None:
        return 0, 0
    lock = kwargs.get('lock', None)
    keywords = {}
    if lock is not None:
        keywords["lock"] = lock
    with synchronized_process_open_file(bare_file, mode='r', **keywords) as correction_h5:
        assert isinstance(bare_path, (str, None))
        parasitic, parasitic_error = _handle_parasitic_cap(bare_path, correction_h5)
        if parasitic < 1.0:
            print("Unfortunatley the parasitic capacitance vanishs.")
        parasitic /= CAPACITANCE_CONVERSION_FACTOR
        parasitic_error /= CAPACITANCE_CONVERSION_FACTOR

        return parasitic, parasitic_error


def __depletion_iterator_mp_implementation(storage, index, **kwargs):
    global mp_shared_cap_data, mp_shared_cap_error_data, mp_shared_voltage_data, mp_shared_keywords
    global mp_shared_lower_limit, mp_shared_upper_limit
    key_args = mp_shared_keywords.copy()
    key_args.update(kwargs)
    __depletion_iterator_implementation(index, mp_shared_cap_data, mp_shared_cap_error_data, mp_shared_lower_limit,
                                        mp_shared_upper_limit, mp_shared_voltage_data, storage, **key_args)


def __init_mo_depletion_iterator_implementation(cap_data, cap_error_data, voltage_data, keywords, lower, upper):
    global mp_shared_cap_data, mp_shared_cap_error_data, mp_shared_voltage_data, mp_shared_keywords
    mp_shared_cap_data = cap_data
    mp_shared_cap_error_data = cap_error_data
    mp_shared_voltage_data = voltage_data
    mp_shared_keywords = keywords.copy()
    global mp_shared_lower_limit, mp_shared_upper_limit
    mp_shared_lower_limit = lower
    mp_shared_upper_limit = upper
    if "lock" in mp_shared_keywords:
        print("Found a locking/semaphore object within the keywords; this might be part of the leakage issue!")


def init_pool(storage):
    """
    init_pool

    @author Dominik Fischer
    last update: 2026-08-13

    Utility function to initialize the multiprocessing worker pool with the required storage object, when trying to
    speed things up a bit by using multiprocessing.

    CAUTION: the multiprocessing implementation in the analysis does not work properly.

    :param storage: iterable with an data_store.py storage object at index 0.
    """
    global fit_result_mp_storage
    fit_result_mp_storage = storage[0]


def get_manager_keywords(**kwargs):
    """
    get_manager_keywords

    @author: Dominik Fischer
    @date: 2026-08-12

    extract the multiprocessing.Manager keyword arguments from the provided keyword arguments.

    :keyword address: address of the socket of the multiprocessing.Manager object we want to connect to.
    :keyword authkey: authentication key necessary to connect to the socket. (It is recommended not to use this parameter as
        it is not pickable)
    :return: dict-like mapping of keys suitable to instantiate a multiprocessing.Manager object.
    """
    return {key: value for key, value in kwargs.items() if key in ("address", "authkey")}
