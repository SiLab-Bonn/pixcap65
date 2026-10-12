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
"""
Additional implementations of :py:class:`multiprocessing.managers.BaseManager` designed for the purpose of enabling
multiprocessing in the depletion analysis of the :py:mod:`pixcap6` framework.
"""


from multiprocessing.managers import SyncManager, BaseManager

class ExtendedSyncManager(SyncManager):
    """
    Implementation of an :py:class:`multiprocessing.managers.BaseManager` designed to provide the same proxies like
    :py:class:`multiprocessing.managers.SyncManager` and in Addition proxies for some numpy classes/types like
    :py:class:`numpy.full`.
    So effectively it is the synthesis of :py:class:`multiprocessing.managers.SyncManager` and
    :py:class:`pixcap65.concurrency.manager.DepletionMPManager`.
    """
    pass


class DepletionMPManager(BaseManager):
    """
    Implementation of an :py:class:`multiprocessing.managers.BaseManager` particularily designed for the purpose of
    speeding some parts of the depletion analysis of the :py:mod:`pixcap6` framework.

    The following objects could also be proxied and used as shared resources if :py:mod:`pixcap65.concurrency` is loaded.

    * :py:class:`numpy.full`: provides access to :py:class:`numpy.ndarray` objects
    * :py:class:`pixcap65.analysis_util.modelling.data_store.DepletionArrayStoreMP`: provides access and shares objects
        of type :py:class:`pixcap65.analysis_util.modelling.data_store.DepletionArrayStore` or it's multiprocessing
        optimised counter-part between multiple processes.
    """
    pass


class ManagerDummy:
    """
    Pseudo-Implementation of an :py:class:`~multiprocessing.managers.SyncManager` designed to act as a dummy which
    provides access to the corresponding standard-python objects instead of creating these in a separate manager process
    and give only proxy-objects of these to the caller.

    Only the types needed by the depletion analysis data structure and data store implementations are actually
    implemented by now.
    """
    def dict(self):
        """
        Provides an empty standard-python dictionary.
        :return: empty dictionary
        :rtype: dict
        """
        return {}

    def full(self, *args, **kwargs):
        """
        Provides an (empty) numpy array filled with a specific entry.
        :param args: positional arguments to be propagated to the numpy initializer.
        :param kwargs: keyword arguments to be propagated to the numpy initializer.
        :return: empty numpy array
        """
        import numpy as np
        return np.full(*args, **kwargs)

    def RLock(self):
        """
        Provides an re-entrant safe lock to synchronize multiple threads.
        :return: RLock
        """
        import threading
        return threading.RLock()

    def list(self):
        """
        Provides an empty standard-python list.
        :return: empty list
        :rtype: list
        """
        return []
