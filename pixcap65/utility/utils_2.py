import contextlib
import logging
import numpy as np
import sys
import tables as tb
import time
from time import sleep
# noinspection PyProtectedMember
from tqdm.contrib import DummyTqdmFile
from typing import Union, Tuple
from warnings import warn

from pixcap65.utility.tables_util import group_get_file, set_group_attribute, get_children_bare, hdf_get_or_create_path

GroupType = Union[tb.Group, tb.Node, tb.Leaf]


def walk_to_node(parent: GroupType, path: str, create=False, verify_create=False) -> tb.Group | Tuple[tb.Group, bool]:
    assert verify_create
    result = parent
    assert isinstance(parent, tb.Group)
    file_h5 = group_get_file(parent)
    already_exits = True
    for element in path.split('/'):
        assert isinstance(result, tb.Group)
        if element in result:
            result = result[element]
        elif create:
            already_exits = False
            assert isinstance(result, tb.Group)
            result = file_h5.create_group(where=result, name=element)
        else:
            print(file_h5)
            raise AssertionError(f"The requested path '{path}' does not exist and creating the path is disabled.")
    assert isinstance(result, tb.Group)
    if verify_create:
        return result, not already_exits,
    warn_msg = ("The usage of the parameter `verify_create` is deprecated. The parameter must be specified to be true."
                "The old behaviour with implicit False will be removed and this warning is mean to identify code where"
                "still the old behaviour is expected.")
    warn(warn_msg, category=DeprecationWarning)
    return result


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


logger = logging.getLogger(__name__)


def create_carray(h5: tb.File, where: Union[tb.Group, str], name: str, **kwargs):
    input_dut = kwargs.pop('input', None)
    unit = kwargs.pop('unit', None)
    if isinstance(where, str):
        where = hdf_get_or_create_path(h5, where)
        assert isinstance(where, tb.Group)
    result = create_update_array(h5, where, name, **kwargs)
    if result is None:
        logger.error("Failed to create or update the array. Could not adjust the attributes.")
    elif isinstance(result, tb.Leaf):
        if input_dut is not None:
            result.attrs["Input"] = input_dut
        if unit is not None:
            result.attrs[UNITS_ATTRIBUTE_KEY] = unit
    elif isinstance(result, tb.Group):
        if input_dut is not None:
            set_group_attribute(result, "Input", input_dut)
        if unit is not None:
            set_group_attribute(result, UNITS_ATTRIBUTE_KEY, unit)
    return result


def create_update_array(h5, where: tb.Group, name: str, **kwargs):
    max_iter = kwargs.get("max_iter", 10)
    if name in get_children_bare(where):
        current_array = get_children_bare(where)[name]
        assert isinstance(current_array, tb.CArray)
        current_shape = current_array.shape
        assert current_shape is not None
        if current_shape == kwargs['obj'].shape:
            current_array[:] = kwargs['obj']
            current_array.flush()
            return current_array
        elif np.all(current_shape >= kwargs['obj'].shape):
            logger.warning(
                "The shape of the arrays to be updated are not the same. Only a partial update is performed.")
            for indices in np.ndindex(kwargs['obj'].shape):
                current_array[indices] = kwargs['obj'][indices]

            current_array.flush()
            return current_array

    create_iteration = 0
    while create_iteration < max_iter:
        create_iteration += 1
        try:
            new_array = h5.create_carray(where, name=name, **kwargs)
            new_array.flush()
            return new_array
        except tb.exceptions.NodeError as e:
            if "already has a child node named" in str(e):
                logger.warning("Unexpectedly found that the child node already exists.")
                assert hasattr(where[name], "rename")
                # noinspection PyUnresolvedReferences
                where[name].rename("{old}_backing".format(old=name))
                sleep(1)
            else:
                print("Array creation failed.")
                logger.error(e)
                logger.exception(e.args, e.__traceback__)
                sleep(create_iteration // 2)
        except Exception as e:
            print("Array creation failed.")
            logger.error(e)
            logger.exception(e.args, e.__traceback__)
            sleep(create_iteration // 2)

    logger.warning(f"Failed to create or update the array {name}")
    return None


def prevent_group_mix_up(parent: tb.Group, node: str):
    if node in parent:
        # noinspection PyProtectedMember
        parent[node]._f_remove(recursive=True)
        time.sleep(1)


UNITS_ATTRIBUTE_KEY = "Units"
