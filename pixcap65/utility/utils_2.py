"""
Utilities to create tables and arrays using `pytables`.
"""
import contextlib
import logging
import numpy as np
import sys
import tables as tb
import time
# noinspection PyProtectedMember
from tqdm.contrib import DummyTqdmFile
from typing import Union, Tuple
from warnings import warn

from pixcap65.utility.tables_util import group_get_file, set_group_attribute, get_children_bare, hdf_get_or_create_path

GroupType = Union[tb.Group, tb.Node, tb.Leaf]
UNITS_ATTRIBUTE_KEY = "Units"


def walk_to_node(parent: GroupType, path: str, create=False, verify_create=False) -> Union[tb.Group, Tuple[tb.Group, bool]]:
    """
    walk_to_node

    @author: Dominik Fischer
    last update: 2026-08-27

    Utility function to navigate from group `parent` along `path` to the desired node, or creating a new group no node
    exists.

    :param parent: group from which to start to reach the specified path.
    :type parent: GroupType
    :param path: path to the desired group starting from `parent`.
    :type path: str
    :param create: whether to create the group if no node exists at the specified location.
    :type create: bool
    :param verify_create: MUST BE TRUE; Verify that a group was actually created.
    :type verify_create: bool
    :return: fetched node or tuple of fetched node and a boolean whether the group was createde.
        Actually only the last one is allowed by now.
    """
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
            raise AssertionError("The requested path '{path}' does not exist and creating the path is disabled.".format(path=path))
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
    """
    std_out_err_redirect_tqdm

    @author: Dominik Fischer
    last update: 2026-08-27

    Redirects the standard output and standard error streams to a `tqdm.tqdm` object.
    This might be necessary to not break progress bars by `print`.

    :return: yields the original error stream (defined as standard)
    """
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


logger = logging.getLogger(__name__)


def create_carray(h5: tb.File, where: Union[tb.Group, str], name: str, **kwargs):
    """
    create_carray

    @author: Dominik Fischer+
    last update: 2026-08-27

    Utility function to create a :py:class:`tables.carray` objects with specific enhanced properties.
    The newly created array is also flushed immediately.
    After creating it is possible to set some specific attributes to the newly created array before flushing.
    If an array of suitable dimension already exists its values are updated.
    For further information on that see :py:func:`pixcap65.utility.utils_2.create_update_array`.


    :param h5: pytables hdf file object in which to create the new array.
    :type h5: tb.File
    :param where: parent group object of the newly created array or path to this group.
    :type where: tb.Group | str
    :param name: name of the new array.
    :type name: str
    :param kwargs: additional keyword arguments mostly propagated to :py:class:`tables.carray`.
    :keyword input: specifies the `Input` attribute which will only be set for leafs.
    :keyword unit: unit of the values residing inside the array. (attribute name will be `Units`)
    :return: created array object.
    """
    # fetch the specific attributes.
    input_dut = kwargs.pop('input', None)
    unit = kwargs.pop('unit', None)
    if isinstance(where, str):
        where = hdf_get_or_create_path(h5, where)
        assert isinstance(where, tb.Group)

    # create the array.
    result = create_update_array(h5, where, name, **kwargs)

    # post process the created array.
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
    """
    create_update_array

    @author: Dominik Fischer
    last update: 2026-08-27

    Creates a new :py:class:`tables.carray` object inside `h5`.
    If an array with the specified `name` already exists in the specified parent group `where`m it will be tried to
    update this array with the new values given by the keyword argument `obj`.
    But this requires matching shapes of the arrays.
    If shape of the already existing array is larger the values are update anyway, otherwise the creation of the
    array will be attempted until it was created successfully, or the maximum number of attempts allowed is exceeded.
    If an attempt fails due to an already existing node with this name, that node will be backed by appending a suitable
    number of `_backing` to its name.
    Any other failure will only to a waiting period before starting a new attempt.

    For further information on the available keyword arguments take a look at
    :py:meth:`tables.File.create_carray`.


    :param h5: pytables hdf file object in which to create the new array.
    :type h5: tb.File
    :param where: parent group object of the newly created array or path to this group.
    :type where: tb.Group | str
    :param name: name of the new array.
    :type name: str
    :param kwargs: further keyword arguments mostly propagated to :py:meth:`tables.File.create_carray`.
    :keyword max_iter: maximum number of attempts to create the array. (default: 10)

    :return: created or update array object.
    :rtype: :py:class:`tables.CArray`, optional
    """
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
                time.sleep(1)
            else:
                print("Array creation failed.")
                logger.error(e)
                logger.exception(e.args, e.__traceback__)
                time.sleep(create_iteration // 2)
        except Exception as e:
            print("Array creation failed.")
            logger.error(e)
            logger.exception(e.args, e.__traceback__)
            time.sleep(create_iteration // 2)

    logger.warning("Failed to create or update the array {name}".format(name=name))
    return None


def prevent_group_mix_up(parent: tb.Group, node: str):
    """
    prevent_group_mix_up

    @author: Dominik Fischer
    last update: 2026-08-27

    In particular when performing analysis steps it is necessary to prevent existing nodes data mixing with newly
    generated results (arrays or tables).
    Therefore, it is necessary to remove/delete already existing nodes from the group if we intend to write new data to
    this name.

    :param parent: hdf files group for which to prevent the mix-up.
    :param node: name of the node, we wish to create later-on.
    """
    if node in parent:
        # noinspection PyProtectedMember
        parent[node]._f_remove(recursive=True)
        time.sleep(1)
