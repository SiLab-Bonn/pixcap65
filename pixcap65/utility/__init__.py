"""
Implementations to enhance concurrency features.
Implements a at least particularly thread-safe version for the usage of pytables.

Enhances some of the functionality of :py:mod:`concurrent.futures`.
"""
import atexit
import concurrent.futures as concurrency
import sys
import time
from contextlib import contextmanager
from tables import open_file, File

tables_lock = None

def release_tables_lock():
    """
    Make sure that all semaphore objects which are globally initialized and acquired on module level are released
    accordingly such that no semaphore objects can leak outside the process when the current process exits.

    @author: Dominik Fischer
    last update: 2026-08-26
    """
    global tables_lock
    tables_lock.acquire()
    tables_lock.release()
    del tables_lock
    tables_lock = None
    import gc
    atexit.unregister(release_tables_lock)
    gc.collect()


def get_tables_lock():
    """
    Retrieve a synchronization primitve to make sure that two processes or thread are not able to simultaneously
    access the pytables api or even write to the same file which would lead to a data race.

    @author: Dominik Fischer
    last update: 2026-08-26
    :return: locking object suitable for multiple processes or threads
    """
    from multiprocessing import RLock
    global tables_lock
    if tables_lock is None:
        tables_lock = RLock()
        # atexit.register(release_tables_lock)
    return tables_lock


@contextmanager
def synchronized_open_file(*args, **kwargs):
    """
    Attempt of a thread-safe and multiprocessing-safe implementation to open a hdf file by using pytables.
    For this reason the access to file opening api needs to be protected by locks.
    If no lock is specified explicitly as a keyword argument, the module level lock will be used.
    This might help against issues with multiple threads but not with multiple processes.
    Implementation is done here by yielding a context manager to close the file handle after usage.
    The synchronization does not put the pytables guards back in when running in different processes.

    @author: Dominik Fischer
    last update: 2026-08-26

    :param args: positional arguments to be used for opening a hdf file
    :param kwargs: keyword arguments for some further controlls and to be propagated to open the file.
    :keyword lock: synchronization primitve object (or a Proxy to it) (also called a lock) to make sure that there are no data races when accessing pytables.
    :keyword max_time: maximum time to wait for acquiring the lock or getting access to the file handle.
    :return: yields a handle of hdf file object.
    """
    file_handle = __synchronized_tables_open_file(*args, **kwargs)
    try:
        yield file_handle.__enter__()
    finally:
        with kwargs.get("lock", get_tables_lock()):
            file_handle.__exit__(*sys.exc_info())

def __synchronized_tables_open_file(*args, **kwargs) -> File:
    # CHECK: perhaps it would be a better implementation when using a multiprocessing manager?
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
    """
    Attempt of a thread-safe and multiprocessing-safe implementation to open a hdf file by using pytables.
    For this reason the access to file opening api needs to be protected by locks.
    If no lock is specified explicitly as a keyword argument, the module level lock will be used.
    This might help against issues with multiple threads but not with multiple processes.
    Implementation is done here by yielding a context manager to close the file handle after usage.
    The synchronization does not put the pytables guards back in when running in different processes.

    @author: Dominik Fischer
    last update: 2026-08-26

    :param args: positional arguments to be used for opening a hdf file
    :param kwargs: keyword arguments for some further controlls and to be propagated to open the file.
    :keyword lock: synchronization primitve object (or a Proxy to it) (also called a lock) to make sure that there are no data races when accessing pytables.
    :keyword max_time: maximum time to wait for acquiring the lock or getting access to the file handle.
    :return: yields a handle of hdf file object.
    """
    lock = kwargs.pop("lock", get_tables_lock())
    file_handle = __synchronized_tables_open_file(*args, lock=lock, **kwargs)
    try:
        yield file_handle.__enter__()
    finally:
        with lock:
            file_handle.__exit__(*sys.exc_info())

def synchronized_process_open_file(*args, **kwargs) -> File:
    """
    Attempt of a thread-safe and multiprocessing-safe implementation to open a hdf file by using pytables.
    For this reason the access to file opening api needs to be protected by locks.
    If no lock is specified explicitly as a keyword argument, the module level lock will be used.
    This might help against issues with multiple threads but not with multiple processes.
    The synchronization does not put the pytables guards back in when running in different processes.

    @author: Dominik Fischer
    last update: 2026-08-26
    :param args: positional arguments to be used for opening a hdf file
    :param kwargs: keyword arguments for some further controlls and to be propagated to open the file.
    :keyword lock: synchronization primitve object (or a Proxy to it) (also called a lock) to make sure that there are no data races when accessing pytables.
    :keyword max_time: maximum time to wait for acquiring the lock or getting access to the file handle.
    :return: handle of hdf file object.
    """
    kwargs.setdefault("lock", get_tables_lock())
    if kwargs.get("lock", None) is None:
        kwargs["lock"] = get_tables_lock()
    file_handle = __synchronized_tables_open_file(*args, **kwargs)
    return file_handle

def create_concurrent_wrapper(*args, iterable_position=0, iterable_name=None, **kwargs):
    """
    Generates a list of parameters to be iterated over to make it possible to use arbitrary function signatures with the
    module :py:mod:`concurrent.futures`.

    @author: Dominik Fischer
    last update: 2026-08-26

    :param args: positional arguments to be passed to the function over the iterable except for the iterable parameter itself.
        The iterable argument must be given as a position argument, too.
    :param iterable_position: index at which the current element needs to be inserted in the positional arguments when iterating over it.
    :param iterable_name: name of the keyword argument which is the current element of the iterable.
    :param kwargs: keyword arguments to be propagated down to the wrapped function.
    :return: generates argument tuples (current iterable element, upcoming positional arguments, keywords to be passed down to the wrapped function and the name of the keyword) for each element of the iterable.
    """
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
    """
    Generate a function compatible with the api of the module :py:mod:`concurrent.futures`, while allowing for an
    arbitrary function signature.
    :param function: callable to be wrapped
    :return: wrapper to the callable
    """
    def _method(arguments):
        key_args = arguments[-2]
        positional_args = list(arguments[1:-2])
        spec = arguments[-1]
        if isinstance(spec, str):
            key_args[spec] = arguments[0]
        else:
            positional_args.insert(spec, arguments[0])

        return function(*positional_args, **key_args)
    return _method

def concurrent_map(fn, executor: concurrency.Executor, *args, iterable_position=0, iterable_name=None, **kwargs):
    """
    Enhanced version of :py:func:`concurrent.futures.map` that allows for arbitrary function signatures.
    The arguments as well as the callable/function itself will be wrapped appropriately before passing it to the
    implementation by the module :py:mod:`concurrent.futures`.

    @author: Dominik Fischer
    last update: 2026-08-26

    :param fn: callable for which concurrent execution of iterable arguments is desired.
    :param executor: executor pool handled by  :py:mod:`concurrent.futures` which should be used for evaluating the
        function.
    :param args: positional arguments to be passed to the function. If one of these arguments is concurrent iterable,
        the iterable must be included here and the index must be given.
    :param iterable_position: index of the iterated positional argument.
    :param iterable_name: if the iterated argument is a keyword argument, we need to specify the name here.
    :param kwargs: keyword arguments to be passed down to the wrapped function.
    :return: result of the concurrent execution of iterable arguments
    :rtype: concurrent.futures.Future
    """

    eff_iterable = create_concurrent_wrapper(*args, iterable_position=iterable_position, iterable_name=iterable_name, **kwargs)
    chunksize = kwargs.pop('chunksize', 1)
    return executor.map(generate_concurrent_function(fn), eff_iterable, chunksize=chunksize)

