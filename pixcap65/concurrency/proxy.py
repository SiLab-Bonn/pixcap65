
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

import multiprocessing as mp
from multiprocessing.managers import BaseProxy


class ProxyBase(mp.managers.NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__')


class NumpyProxy(BaseProxy):
    _exposed_ = ('__getattr__', '__setattr__', '__delattr__', '__getitem__', '__setitem__', 'shape', )

    def __getitem__(self, *args):
        return self._callmethod('__getitem__', args)

    def __setitem__(self, *args):
        self._callmethod('__setitem__', args)

    def shape(self):
        self._callmethod('shape')

    def __len__(self):
        return self._callmethod('__len__')

    def __contains__(self, *args):
        return self._callmethod('__contains__', args)

    def __iter__(self):
        return self._callmethod('__iter__')

    def __index__(self):
        return self._callmethod('__index__')

    def __lt__(self, other):
        return self._callmethod('__lt__', other)

    def __le__(self, other):
        return self._callmethod('__le__', other)

    def __gt__(self, other):
        return self._callmethod('__gt__', other)

    def __ge__(self, other):
        return self._callmethod('__ge__', other)
    def __eq__(self, other):
        return self._callmethod('__eq__', other)
    def __ne__(self, other):
        return self._callmethod('__ne__', other)


class DepletionArrayStoreProxy(ProxyBase) : pass
