import contextlib
import logging
import sys
from time import sleep
from typing import Union

import numpy as np
import tables as tb
from tqdm.contrib import DummyTqdmFile

GroupType = Union[tb.Group, tb.Node, tb.Leaf, tb.RootGroup]


def walk_to_node(parent: tb.Node, path: str, create=False, verify_create=False):
    result = parent
    already_exits = True
    for element in path.split('/'):
        if element in result:
            result = result[element]
        elif create:
            already_exits = False
            result = parent._v_file.create_group(where=result, name=element)
        else:
            raise AssertionError("The requested path does not exist and creating the path is disabled.")
    if verify_create:
        return result, not already_exits
    return result


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err

logger = logging.getLogger(__name__)

def create_update_array(h5, where: tb.Group, name: str, **kwargs):
    max_iter = kwargs.get("max_iter", 10)
    if name in where._v_children:
        current_array = where._v_children[name]
        assert isinstance(current_array, tb.CArray)
        if current_array.shape == kwargs['obj'].shape:
            current_array[:] = kwargs['obj']
            current_array.flush()
            return current_array
        elif np.all(current_array.shape >= kwargs['obj'].shape):
            logger.warning("The shape of the arrays to be updated are not the same. Only a partial update is performed.")
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
