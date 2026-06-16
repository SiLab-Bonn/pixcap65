# ----------------------------------------------------------
#  Copyright (c) .
#   All rights reserved
#  SiLab, Institute of Physics, University of Bonn
# ----------------------------------------------------------

import atexit
import multiprocessing as mp
import sys
from multiprocessing.managers import SyncManager

__manager = None

__manager_handling_lock = mp.Lock()

def exit_manager():
    global __manager
    with __manager_handling_lock:
        if __manager is not None:
            __manager.__exit__(*sys.exc_info())
            __manager = None

# make sure the manager object will be closed ordinarily on exit.

class ExtendedSyncManager(SyncManager):
    pass

def get_manager(**kwargs):
    with __manager_handling_lock:
        global __manager
        if __manager is None:
            __manager = ExtendedSyncManager(**kwargs)
            if "address" in kwargs:
                __manager.connect()
            else:
                __manager.__enter__()
                atexit.register(__manager.shutdown)

    return __manager

def close_manager():
    global __manager
    with __manager_handling_lock:
        if __manager is not None:
            __manager.shutdown()
            __manager = None

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


