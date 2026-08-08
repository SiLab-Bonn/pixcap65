# ----------------------------------------------------------
#  Copyright (c) .
#   All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------
import atexit
import inspect
import multiprocessing as mp
import numpy as np
import sys
from contextlib import contextmanager
from multiprocessing.managers import SyncManager, BaseManager

from pixcap65.analysis_util.data_store import DepletionArrayStoreMP
from pixcap65.concurrency import manager
from pixcap65.concurrency import proxy
from pixcap65.concurrency.manager import ExtendedSyncManager, DepletionMPManager

__manager = None
# What about about doing such things here directly within the Extended manager in the concurrency module?
__depletion_manager = None

# we need to make sure that this here defered correctly!
__manager_handling_lock = mp.Lock()

def defer_module():
    global __manager_handling_lock
    del __manager_handling_lock


atexit.register(defer_module)

def exit_manager():
    global __manager
    with __manager_handling_lock:
        if __manager is not None:
            __manager.__exit__(*sys.exc_info())
            __manager = None

# make sure the manager object will be closed ordinarily on exit.
__started_manager = True

@contextmanager
def get_context_manager(**kwargs):
    try:
        yield get_manager(**kwargs)
    finally:
        close_manager()

def get_manager(**kwargs):
    with __manager_handling_lock:
        global __manager, __started_manager
        if __manager is None:
            __manager = ExtendedSyncManager()
            if "address" in kwargs:
                if __manager.address is None:
                    __manager._address = kwargs["address"]
                __manager.connect()
                __started_manager = False
            else:
                __manager.__enter__()
                atexit.register(close_manager)
                __started_manager = True

    return __manager

def close_manager():
    global __manager, __started_manager
    with __manager_handling_lock:
        if __manager is not None and isinstance(__manager, BaseManager) and __started_manager:
            if hasattr(__manager, "shutdown"):
                __manager.shutdown()
            else:
                from warnings import warn
                warn("Unfortunately the manager could not be shutted down.", stacklevel=3)
            __manager = None
            atexit.unregister(close_manager)


@contextmanager
def get_mp_context_manager(**kwargs):
    try:
        yield get_mp_manager(**kwargs)
    finally:
        close_mp_manager()


def get_mp_manager(**kwargs):
    with __manager_handling_lock:
        global __depletion_manager
        if __depletion_manager is None:
            __depletion_manager = proxy.DepletionMPManager(**kwargs)
            if "address" in kwargs:
                __depletion_manager.connect()
            else:
                __depletion_manager.__enter__()
                atexit.register(close_mp_manager)

        return __depletion_manager


def close_mp_manager():
    global __depletion_manager
    with __manager_handling_lock:
        if __depletion_manager is not None:
            __depletion_manager.__exit__(None, None, None)
            __depletion_manager = None
            atexit.unregister(close_mp_manager)

try:
    from matplotlib.backends.backend_pdf import PdfPages

    class PdfPagesProxy(mp.managers.BaseProxy):
        _exposed_ = ('__enter__', '__exit__')

        def __enter__(self):
            return self._callmethod("__enter__")

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._callmethod("__exit__", (None, None, None,))

    ExtendedSyncManager.register("PdfPages", PdfPages, PdfPagesProxy)

    class ThreadedPdfPages:
        def __init__(self, *args, **kwargs):
            self.lock = get_manager().RLock()
            self.pdf = get_manager().PdfPages(*args, **kwargs)

        def __enter__(self):
            self.pdf.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            with self.lock:
                return self.pdf.__exit__(exc_type, exc_val, exc_tb)

        def __getattr__(self, name):
            pdf = object.__getattribute__(self, 'pdf')
            lock = object.__getattribute__(self, 'lock')
            pdf_attr = object.__getattribute__(pdf, name)
            if callable(pdf_attr):
                def method(*args, **kwargs):
                    with lock:
                        return pdf_attr(*args, **kwargs)
                return method
            with lock:
                return pdf_attr

    class ThreadedPdfPagesProxy(mp.managers.BaseProxy):
        _exposed_ = ('__enter__', '__exit__', '__getattr__')

        def __exit__(self, *args):
            return self._callmethod("__exit__", (None, None, None,))

        def __enter__(self):
            return self._callmethod("__enter__")

    ExtendedSyncManager.register("ThreadedPdfPages", ThreadedPdfPages, ThreadedPdfPagesProxy)
except ImportError:
    pass

def initialize_worker_manager(**kwargs):
    _ = get_manager(**kwargs)

# we need to make sure that also our primary manager could be equipped with the relevant data store objects to only
# start a single manager when to perform all our tasks
# from pixcap65.analysis_util.data_store import NumpyProxy, DepletionArrayStore, DepletionArrayStoreProxy, register_proxy
# import numpy as np
# ExtendedSyncManager.register('full', np.full, NumpyProxy)
# register_proxy("DepletionArrayStorage", DepletionArrayStore, DepletionArrayStoreProxy,
#                ExtendedSyncManager)

def register_proxy(name, cls, proxy, manager_cls=manager.DepletionMPManager):
    setattr(proxy, name, proxy)
    for attr in dir(cls):
        if "lock" in attr.lower():
            continue
        if inspect.ismethod(getattr(cls, attr)) and not attr.startswith("__"):
            proxy._exposed_ += (attr,)
            setattr(proxy, attr, lambda s: object.__getattribute__(s, '_callmethod')(attr))
    manager_cls.register(name, cls, proxy)


# we should close these here aways such that they are not imported multiple times?
manager.DepletionMPManager.register("full", np.full, proxy.NumpyProxy)
# the question now is whether a already started manager will be affected by a change of registered methods?
# registration must be completed before the manager is started at all.
manager.ExtendedSyncManager.register('full', np.full, proxy.NumpyProxy)

register_proxy("DepletionArrayStorage", DepletionArrayStoreMP, proxy.DepletionArrayStoreProxy)
register_proxy("DepletionArrayStorage", DepletionArrayStoreMP, proxy.DepletionArrayStoreProxy,
               manager.ExtendedSyncManager)


@contextmanager
def get_mp_context_manager(**kwargs):
    try:
        yield get_mp_manager(**kwargs)
    finally:
        close_mp_manager()


def get_mp_manager(**kwargs):
    with __manager_handling_lock:
        global __depletion_manager
        if __depletion_manager is None:
            __depletion_manager = manager.DepletionMPManager(**kwargs)
            if "address" in kwargs:
                __depletion_manager.connect()
            else:
                __depletion_manager.__enter__()
                atexit.register(close_mp_manager)

        return __depletion_manager


def close_mp_manager():
    global __depletion_manager
    with __manager_handling_lock:
        if __depletion_manager is not None:
            __depletion_manager.__exit__(None, None, None)
            __depletion_manager = None
            atexit.unregister(close_mp_manager)
