import multiprocessing as mp

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

from multiprocessing.managers import SyncManager

class ExtendedSyncManager(SyncManager):
    pass


class DepletionMPManager(mp.managers.BaseManager):
    pass

class ManagerDummy:
    def dict(self):
        return {}

    def full(self, *args, **kwargs):
        import numpy as np
        return np.full(*args, **kwargs)

    def RLock(self):
        import threading
        return threading.RLock()

    def list(self):
        return []
