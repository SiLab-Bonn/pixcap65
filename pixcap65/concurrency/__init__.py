"""
Implementations to simplify the usage of pythons concurrency features in particular concering multiple processes,
shared resources and synchronization between different processes.
"""
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
from typing import Optional

from pixcap65.analysis_util.modelling.data_store import DepletionArrayStoreMP
from pixcap65.concurrency import manager
from pixcap65.concurrency import proxy
from pixcap65.concurrency.manager import ExtendedSyncManager, DepletionMPManager

__manager: Optional[BaseManager] = None
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
                    # noinspection unresolved-references
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
        """
        Multi-Threading capable version of :py:class:`~matplotlib.backends.backend_pdf.PdfPages`.
        (I'm not 100% sure whether this implementation is actually/really thread-safe.
        Nevertheless even if it is completely thread-safe there could be made no promise w.r.t. the ordering of the
        individual figures/pages of the output pdf document.
        The implementation is directly based upon the matplotlib backends implementation, except for the affect that
        `transactions` are protected by synchronization primitives.

        For information on the arguments and keyword arguments/attributes of this class see
        :py:class:`matplotlib.backends.backend_pdf.PdfPages` as any positional and keyword argument is propagate to the
        backend instance.
        """
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
        """
        Proxy class to enable threaded PdfPages objects handled by a multiprocessing.Manager object.
        Here only a minimal set of functionality will be exposed to the calling to enable saving for figures in a
        multiprocessing context.

        Effectively only the context manager implementation is exposed at all.
        """
        _exposed_ = ('__enter__', '__exit__', '__getattr__')

        def __exit__(self, *args):
            return self._callmethod("__exit__", (None, None, None,))

        def __enter__(self):
            return self._callmethod("__enter__")

    ExtendedSyncManager.register("ThreadedPdfPages", ThreadedPdfPages, ThreadedPdfPagesProxy)
except ImportError:
    pass

def initialize_worker_manager(**kwargs):
    """
    Alternative to instantiate a manager object within this module than
    :py:func:`pixcap65.concurrency.get_manager`.
    In opposite to that implementation this here is not intended to actually fetch a manager object but to create own at
    process initialization and remember it at the module level as a global constant such that certain manager arguments
    do not need to be propagated through all the following code.

    @author: Dominik Fischer
    last update: 2026-08-25

    :param kwargs: further arguments to connect to a already running manager or create a new one with well-defined parameters. For this keywords you may also take a look at :py:func:`pixcap65.analysis.get_manager_keywords`.
    :key address: address of the socket of the multiprocessing.Manager object we want to connect to.
    :key authkey: authentication key necessary to connect to the socket. (It is recommended not to use this parameter as
        it is not pickable)
    """
    _ = get_manager(**kwargs)

# we need to make sure that also our primary manager could be equipped with the relevant data store objects to only
# start a single manager when to perform all our tasks.

def register_proxy(name, cls, proxy, manager_cls=manager.DepletionMPManager):
    """
    More specialized function to register a new data type and corresponding proxy into a multiprocessing.Manager object
    like by the classmethod :py:meth:`multiprocessing.manager.BaseManager.register`.

    @author: Dominik Fischer
    last update: 2026-08-25

    :param name: typeid, name under which to register the proxy and which could be used to init a new object from the
        manager object
    :param cls: class of the object which should be registered to be managed as a shared resource. This could also
        a callable which implements initialization of such an object.
    :param proxy: proxy class to use for this python type (after registration). The proxy will be exposed.
    :param manager_cls: class of the manager to which it should be registered.
    """
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
