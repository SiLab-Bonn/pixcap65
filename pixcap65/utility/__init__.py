from multiprocessing import RLock

import sys
from contextlib import contextmanager
from tables import open_file

tables_lock = RLock()

tables_process_lock = RLock()


@contextmanager
def synchronized_open_file(*args, **kwargs):
    with tables_lock:
        file_handle = open_file(*args, **kwargs)
    try:
        yield file_handle.__enter__()
    finally:
        with tables_lock:
            file_handle.__exit__(*sys.exc_info())


@contextmanager
def synchronized_process_open_file(*args, **kwargs):
    with tables_process_lock:
        file_handle = open_file(*args, **kwargs)
    try:
        yield file_handle.__enter__()
    finally:
        with tables_process_lock:
            file_handle.__exit__(*sys.exc_info())

