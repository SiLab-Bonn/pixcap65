import concurrent.futures as concurrency
import sys
import time
from contextlib import contextmanager
from tables import open_file, File

tables_lock = None

import atexit
def release_tables_lock():
    global tables_lock
    tables_lock.acquire()
    tables_lock.release()
    del tables_lock
    tables_lock = None
    import gc
    atexit.unregister(release_tables_lock)
    gc.collect()


def get_tables_lock():
    from multiprocessing import RLock
    global tables_lock
    if tables_lock is None:
        tables_lock = RLock()
        # atexit.register(release_tables_lock)
    return tables_lock


@contextmanager
def synchronized_open_file(*args, **kwargs):
    file_handle = __synchronized_tables_open_file(*args, **kwargs)
    try:
        yield file_handle.__enter__()
    finally:
        with kwargs.get("lock", get_tables_lock()):
            file_handle.__exit__(*sys.exc_info())

def __synchronized_tables_open_file(*args, **kwargs) -> File:
    lock = kwargs.pop("lock", get_tables_lock())
    start_time = time.time()
    use_timeout = kwargs.get('max_time', None) is not None
    max_time = kwargs.pop('max_time', 0)
    while not use_timeout or max_time > time.time() - start_time:
        try:
            with lock:
                file_handle = open_file(*args, **kwargs)
            break
        except ValueError as e:
            if "already opened" in str(e):
                print("Opening failed. Will try again later.")
                time.sleep(30)
            else:
                raise
    else:
        raise TimeoutError("Opening the file handle timed out.")
    return file_handle


@contextmanager
def synchronized_process_open_file(*args, **kwargs):
    lock = kwargs.pop("lock", get_tables_lock())
    file_handle = __synchronized_tables_open_file(*args, lock=lock, **kwargs)
    try:
        yield file_handle.__enter__()
    finally:
        with lock:
            file_handle.__exit__(*sys.exc_info())

def synchronized_process_open_file(*args, **kwargs) -> File:
    kwargs.setdefault("lock", get_tables_lock())
    if kwargs.get("lock", None) is None:
        kwargs["lock"] = get_tables_lock()
    file_handle = __synchronized_tables_open_file(*args, **kwargs)
    return file_handle

def create_concurrent_wrapper(*args, iterable_position=0, iterable_name=None, **kwargs):
    if iterable_name is None:
        concurrent_iterator = args[iterable_position]
        new_arguments = list(args)
        del new_arguments[iterable_position]
        spec = iterable_position
    else:
        concurrent_iterator = kwargs.pop(iterable_name)
        new_arguments = list(args)
        spec = iterable_name
    return [tuple((current, *new_arguments, kwargs, spec)) for current in concurrent_iterator]

def generate_concurrent_function(function):
    def method(arguments):
        key_args = arguments[-2]
        positional_args = list(arguments[1:-2])
        spec = arguments[-1]
        if isinstance(spec, str):
            key_args[spec] = arguments[0]
        else:
            positional_args.insert(spec, arguments[0])

        return function(*positional_args, **key_args)
    return method

def concurrent_map(fn, executor: concurrency.Executor, *args, iterable_position=0, iterable_name=None, **kwargs):
    eff_iterable = create_concurrent_wrapper(*args, iterable_position=iterable_position, iterable_name=iterable_name, **kwargs)
    chunksize = kwargs.pop('chunksize', 1)
    return executor.map(generate_concurrent_function(fn), eff_iterable, chunksize=chunksize)

