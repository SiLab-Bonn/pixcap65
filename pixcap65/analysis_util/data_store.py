import atexit
import inspect
import multiprocessing as mp
import numpy as np
import tables as tb
import threading
from multiprocessing.managers import BaseProxy

import pixcap65.concurrency
from pixcap65.analysis_util import GENERAL_PIXCAP_SHAPE


# we should close these here aways such that they are not imported multiple times?
class DepletionMPManager(mp.managers.BaseManager):
    pass


class ProxyBase(mp.managers.NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__')


class DepletionArrayStoreProxy(ProxyBase): pass


def register_proxy(name, cls, proxy, manager_cls=DepletionMPManager):
    setattr(proxy, name, proxy)
    for attr in dir(cls):
        if "lock" in attr.lower():
            continue
        if inspect.ismethod(getattr(cls, attr)) and not attr.startswith("__"):
            proxy._exposed_ += (attr,)
            setattr(proxy, attr, lambda s: object.__getattribute__(s, '_callmethod')(attr))
    manager_cls.register(name, cls, proxy)

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


DepletionMPManager.register("full", np.full, NumpyProxy)
pixcap65.concurrency.ExtendedSyncManager.register('full', np.full, NumpyProxy)


class DepletionDataStore:
    def store_data(self, key, data):
        """
        temporarily store the data for the given key into the internal data structures.
        :param key: kind of data to store
        :param data: data set to store
        """
        # Stub function for further implementation
        pass

    def flush_data(self):
        """
        Write the data stored in the internal data structures to a file/disk what ever.
        This will depend on the actual subclass used.
        """
        # stub function for further implemenation by more specialized subclasses.
        pass


class DepletionTableStore(DepletionDataStore):
    def __init__(self, table: tb.Table, n_depletions=None):
        self.table = table
        self.entry = self.table.row
        self.depletion_reg = None

    remap = [0, 1, 3.0, 2.0, 3.1, 2.1, 5.0, 4.0, 5.1, 4.1]
    match_fields = ("depletion",
                    "depletion_error",
                    "fit_result_first:0",
                    "fit_error_first:0",
                    "fit_result_first:1",
                    "fit_error_first:1",
                    "fit_result_second:0",
                    "fit_error_second:0",
                    "fit_result_second:1",
                    "fit_error_second:1",)

    def store_data(self, key, data):
        match key:
            case "depletion":
                self.entry["Ubi"] = data
            case "depletion_error":
                self.entry["Ubi_error"] = data
            case "fit_result_first":
                self.entry["a"] = data[0]
                self.entry["b"] = data[1]
            case "fit_error_first":
                self.entry["a_error"] = data[0]
                self.entry["b_error"] = data[1]
            case "fit_result_second":
                self.entry["c"] = data[0]
                self.entry["d"] = data[1]
            case "fit_error_second":
                self.entry["c_error"] = data[0]
                self.entry["d_error"] = data[1]
            case _:
                raise ValueError(f"The provided storage key is unknown: {key}")

    def flush_data(self):
        self.entry.append()

class DepletionNumpyStore(DepletionDataStore):
    def __init__(self, dtp: np.dtype):
        self.entry = {}
        self.depletion_reg = None
        self.data_temp = []
        self.dtype = dtp
        self.fields = dtp.names


    def store_data(self, key, data):
        match key:
            case "depletion":
                self.entry["Ubi"] = data
            case "depletion_error":
                self.entry["Ubi_error"] = data
            case "fit_result_first":
                self.entry["a"] = data[0]
                self.entry["b"] = data[1]
            case "fit_error_first":
                self.entry["a_error"] = data[0]
                self.entry["b_error"] = data[1]
            case "fit_result_second":
                self.entry["c"] = data[0]
                self.entry["d"] = data[1]
            case "fit_error_second":
                self.entry["c_error"] = data[0]
                self.entry["d_error"] = data[1]
            case _:
                raise ValueError(f"The provided storage key is unknown: {key}")

    def set_depletion_region(self, i):
        assert i < self.n_values
        self.depletion_reg = i

    def flush_data(self):
        entries = tuple(self.entry.pop(name, np.nan) for name in self.fields)
        self.data_temp.extend([entries, ])
        self.entry.clear()

    @property
    def table(self):
        return np.rec.array(self.data_temp, dtype=self.dtype)


# What about about doing such things here directly within the Extended manager in the concurrency module?

__manager = None


def get_mp_manager():
    global __manager
    if __manager is None:
        __manager = DepletionMPManager()
        __manager.__enter__()
        atexit.register(__manager.__exit__, None, None, None)

    return __manager

class DepletionArrayStore(DepletionDataStore):
    def __init__(self, n_depletions=None):
        general_shape = GENERAL_PIXCAP_SHAPE if n_depletions is None else tuple([*GENERAL_PIXCAP_SHAPE, n_depletions])
        parameter_shape = tuple([*GENERAL_PIXCAP_SHAPE, 4]) if n_depletions is None else tuple(
            [*GENERAL_PIXCAP_SHAPE, n_depletions, 4])
        def create_full(*args, **kwargs):
            return get_mp_manager().full(*args, **kwargs)
        self.depletion_voltage = create_full(shape=general_shape, fill_value=np.nan)
        self.depletion_error = create_full(shape=general_shape, fill_value=np.nan)
        self.fit_parameter_estimators = create_full(shape=parameter_shape, fill_value=np.nan)
        self.fit_parameter_errors = create_full(shape=parameter_shape, fill_value=np.nan)
        self.lock = pixcap65.concurrency.get_manager().RLock()
        self._pixel_row = pixcap65.concurrency.get_manager().dict()
        self._pixel_col = pixcap65.concurrency.get_manager().dict()
        # could use the thread id to identify
        self._depletion_reg = pixcap65.concurrency.get_manager().dict()
        self.systematic_errors = create_full(shape=general_shape, fill_value=np.nan)
        self.systematic_dispersion = create_full(shape=general_shape, fill_value=np.nan)

    @property
    def pixel_row(self):
        with self.lock:
            return self._pixel_row.get(threading.get_native_id(), 1)

    @property
    def pixel_col(self):
        with self.lock:
            return self._pixel_col.get(threading.get_native_id(), 1)

    @property
    def depletion_reg(self):
        with self.lock:
            return self._depletion_reg.get(-1, None)

    @pixel_row.setter
    def pixel_row(self, i):
        with self.lock:
            self._pixel_row[threading.get_native_id()] = i

    @pixel_col.setter
    def pixel_col(self, j):
        with self.lock:
            self._pixel_col[threading.get_native_id()] = j

    @depletion_reg.setter
    def depletion_reg(self, i):
        with self.lock:
            self._depletion_reg[-1] = i

    @pixel_row.deleter
    def pixel_row(self):
        with self.lock:
            if threading.get_native_id() in self._pixel_row:
                del self._pixel_row[threading.get_native_id()]

    @pixel_col.deleter
    def pixel_col(self):
        with self.lock:
            if threading.get_native_id() in self._pixel_col:
                del self._pixel_col[threading.get_native_id()]

    @depletion_reg.deleter
    def depletion_reg(self):
        with self.lock:
            if threading.get_native_id() in self._depletion_reg:
                del self._depletion_reg[-1]

    def set_pixel(self, i_row, i_col):
        with self.lock:
            self.pixel_row = i_row
            self.pixel_col = i_col

    def set_depletion_region(self, i):
        with self.lock:
            self.depletion_reg = i

    def store_data(self, key, data):
        # this part is not protected anyway from any access.
        with self.lock:
            if self.depletion_reg is None:
                match key:
                    case "depletion":
                        self.depletion_voltage[self.pixel_col, self.pixel_row] = data
                    case "depletion_error":
                        self.depletion_error[self.pixel_col, self.pixel_row] = data
                    case "fit_result_first":
                        try:
                            self.fit_parameter_estimators[self.pixel_col, self.pixel_row, :2] = data
                        except:
                            print(data)
                            print(self.fit_parameter_errors.shape)
                            print(self.fit_parameter_errors[self.pixel_col, self.pixel_row, :2].shape)
                            print(data.shape)
                            print(threading.get_native_id())
                            print(self._depletion_reg)
                            print(self.pixel_col)
                            print(self.pixel_row)
                            raise
                    case "fit_error_first":
                        self.fit_parameter_errors[self.pixel_col, self.pixel_row, :2] = data
                    case "fit_result_second":
                        self.fit_parameter_estimators[self.pixel_col, self.pixel_row, 2:] = data
                    case "fit_error_second":
                        self.fit_parameter_errors[self.pixel_col, self.pixel_row, 2:] = data
                    case "systematic":
                        self.systematic_errors[self.pixel_col, self.pixel_row] = data
                    case "dispersion":
                        self.systematic_dispersion[self.pixel_col, self.pixel_row] = data
                    case _:
                        raise ValueError(f"The provided storage key is unknown: {key}")
            else:
                match key:
                    case "depletion":
                        self.depletion_voltage[self.pixel_col, self.pixel_row, self.depletion_reg] = data
                    case "depletion_error":
                        self.depletion_error[self.pixel_col, self.pixel_row, self.depletion_reg] = data
                    case "fit_result_first":
                        self.fit_parameter_estimators[self.pixel_col, self.pixel_row, self.depletion_reg, :2] = data
                    case "fit_error_first":
                        self.fit_parameter_errors[self.pixel_col, self.pixel_row, self.depletion_reg, :2] = data
                    case "fit_result_second":
                        self.fit_parameter_estimators[self.pixel_col, self.pixel_row, self.depletion_reg, 2:] = data
                    case "fit_error_second":
                        self.fit_parameter_errors[self.pixel_col, self.pixel_row, self.depletion_reg, 2:] = data
                    case "systematic":
                        self.systematic_errors[self.pixel_col, self.pixel_row, self.depletion_reg] = data
                    case "dispersion":
                        self.systematic_dispersion[self.pixel_col, self.pixel_row, self.depletion_reg] = data
                    case _:
                        raise ValueError(f"The provided storage key is unknown: {key}")


class DopingArrayStore(DepletionDataStore):
    def __init__(self, doping_shape):
        self.pixel_row = 1
        self.pixel_col = 1
        self.depletion_width_plate = np.full(shape=doping_shape, fill_value=np.nan)
        self.depletion_width_plate_error = np.full(shape=doping_shape, fill_value=np.nan)
        self.depletion_fit_parameter_table = np.full((40, 40, 4), fill_value=np.nan)
        self.depletion_fit_parameter_error_table = np.full((40, 40, 4), fill_value=np.nan)
        self.depletion_fit_covariance_table = np.full((40, 40, 4, 4), fill_value=np.nan)
        self.effective_doping_table = np.full(shape=doping_shape, fill_value=np.nan)

    def set_pixel(self, i_row, i_col):
        self.pixel_row = i_row
        self.pixel_col = i_col

    def store_data(self, key, data):
        match key:
            case "width":
                self.depletion_width_plate[self.pixel_col, self.pixel_row] = data
            case "width_error":
                self.depletion_width_plate_error[self.pixel_col, self.pixel_row] = data
            case "fit_parameters":
                self.depletion_fit_parameter_table[self.pixel_col, self.pixel_row] = data
            case "fit_parameters_error":
                self.depletion_fit_parameter_error_table[self.pixel_col, self.pixel_row] = data
            case "fit_covariance":
                self.depletion_fit_covariance_table[self.pixel_col, self.pixel_row] = data
            case "doping":
                self.effective_doping_table[self.pixel_col, self.pixel_row] = data
            case _:
                raise ValueError(f"Unknown data storage key {key}")

class SummaryTable(tb.IsDescription):
    sensor = tb.StringCol(10, pos=0)
    inj_cap = tb.Float64Col(pos=1)
    inj_cap_2 = tb.Float64Col(pos=2)


register_proxy("DepletionArrayStorage", DepletionArrayStore, DepletionArrayStoreProxy)
register_proxy("DepletionArrayStorage", DepletionArrayStore, DepletionArrayStoreProxy,
               pixcap65.concurrency.ExtendedSyncManager)

