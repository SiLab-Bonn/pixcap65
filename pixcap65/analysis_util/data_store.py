import datetime
import multiprocessing as mp
import numpy as np
import tables as tb
import threading

from pixcap65.analysis_util import GENERAL_PIXCAP_SHAPE


class DepletionDataStore:
    def store_data(self, key, data):
        """
        temporarily store the data for the given key into the internal data structures.
        For the depletion storage units

        * depletion: stores the estimation of the (full or surface) depletion voltage
        * depletion_error: estimation of the statistical uncertainty of the depletion voltage
        * fit_result_first: parameter vector for fit of the first linear region (high voltage limit)
        * fit_error_first: vector of parameter errors for fit of the first linear region (high voltage limit)
        * fit_result_second: parameter vector for fit of the second linear region (low voltage limit)
        * fit_error_second: vector of parameter errors for fit of the second linear region (high voltage limit)
        * fit_result_cov_first: covariance matrix of the first linear region (high voltage limit)
        * fit_result_cov_second: covariance matrix of the second linear region (low voltage limit)
        * systematic: estimation of the (general) systematic uncertainty of the depletion voltage
        * dispersion: estimation of the systematic uncertainty of the depletion voltage by dispersion of parasitic capacitances between differen pixcap chips.

        For doping storage units:

        * width: estimation of the depletion width for the applied bias voltages
        * width_error: (statistical) uncertainties for the depletion width for the applied bias voltages
        * fit_parameters: estimation of the fit parameters to model the depletion width for the applied bias voltages
        * fit_parameters_error: estimation of the fit parameter errors to model the depletion width for the applied bias voltages
        * fit_covariance: covariance matrix of the fit parameters to model the depletion width for the applied bias voltages
        * doping: effective doping concentration for the applied bias voltages estimated from the depletion width and capacitance behaviour
        * resistivity: resistivity of the substrate estimated from effective doping concentration
        * res_mod: resistivity of the substrate estimated from fit to low voltage limit region of the C-V-curve
        * res_mod_error: (statistical) uncertainty of the substrate's resistivity estimated from fit to low voltage limit region of the C-V-curve.


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
            case "fit_result_cov_first":
                self.entry["first_covariance"] = data
            case "fit_result_cov_second":
                self.entry["second_covariance"] = data
            case "systematic":
                self.entry["cap_systematic_error"] = data
            case "dispersion":
                self.entry["cap_systematic_dispersion"] = data
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
            case "fit_result_cov_first":
                self.entry["first_covariance"] = data
            case "fit_result_cov_second":
                self.entry["second_covariance"] = data
            case "systematic":
                self.entry["cap_systematic_error"] = data
            case "dispersion":
                self.entry["cap_systematic_dispersion"] = data
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

class DepletionArrayStore(DepletionDataStore):
    @property
    def mp_manager(self):
        from pixcap65.concurrency.manager import ManagerDummy
        return ManagerDummy()

    def __init__(self, n_depletions=None, **manager_kwargs):
        general_shape = GENERAL_PIXCAP_SHAPE if n_depletions is None else tuple([*GENERAL_PIXCAP_SHAPE, n_depletions])
        parameter_shape = tuple([*GENERAL_PIXCAP_SHAPE, 4]) if n_depletions is None else tuple(
            [*GENERAL_PIXCAP_SHAPE, n_depletions, 4])
        cov_parameter_shape = tuple([*GENERAL_PIXCAP_SHAPE, 4, 4]) if n_depletions is None else tuple(
            [*GENERAL_PIXCAP_SHAPE, n_depletions, 4, 4])

        self.depletion_voltage = self.mp_manager.full(shape=general_shape, fill_value=np.nan)
        self.depletion_error = self.mp_manager.full(shape=general_shape, fill_value=np.nan)
        self.fit_parameter_estimators = self.mp_manager.full(shape=parameter_shape, fill_value=np.nan)
        self.fit_parameter_errors = self.mp_manager.full(shape=parameter_shape, fill_value=np.nan)
        self.fit_parameter_covariances = self.mp_manager.full(shape=cov_parameter_shape, fill_value=np.nan)
        self.lock = self.mp_manager.RLock()
        self._pixel_row = self.mp_manager.dict()
        self._pixel_col = self.mp_manager.dict()
        # could use the thread id to identify
        self._depletion_reg = self.mp_manager.dict()
        self.systematic_errors = self.mp_manager.full(shape=general_shape, fill_value=np.nan)
        self.systematic_dispersion = self.mp_manager.full(shape=general_shape, fill_value=np.nan)

    def __del__(self):
        # need to cleanup all the manager objects
        del self.depletion_voltage
        del self.depletion_error
        del self.fit_parameter_estimators
        del self.fit_parameter_errors
        del self.fit_parameter_covariances
        del self._pixel_col
        del self._pixel_row
        del self._depletion_reg
        del self.systematic_errors
        del self.systematic_dispersion
        del self.lock

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
                    case "fit_result_cov_first":
                        self.fit_parameter_covariances[self.pixel_col, self.pixel_row, :2, :2] = data
                    case "fit_result_cov_second":
                        self.fit_parameter_covariances[self.pixel_col, self.pixel_row, 2:, 2:] = data
                    case "systematic":
                        self.systematic_errors[self.pixel_col, self.pixel_row] = data
                    case "dispersion":
                        self.systematic_dispersion[self.pixel_col, self.pixel_row] = data
                    case _:
                        raise ValueError(f"The provided storage key is unknown: {key}")
            else:
                match key:
                    case "depletion":
                        self.depletion_voltage[
                            self.pixel_col, self.pixel_row, self.depletion_reg
                        ] = data
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
                    case "fit_result_cov_first":
                        self.fit_parameter_covariances[self.pixel_col, self.pixel_row, self.depletion_reg, :2, :2] = data
                    case "fit_result_cov_second":
                        self.fit_parameter_covariances[self.pixel_col, self.pixel_row, self.depletion_reg, 2:, 2:] = data
                    case "systematic":
                        self.systematic_errors[self.pixel_col, self.pixel_row, self.depletion_reg] = data
                    case "dispersion":
                        self.systematic_dispersion[self.pixel_col, self.pixel_row, self.depletion_reg] = data
                    case _:
                        raise ValueError(f"The provided storage key is unknown: {key}")


class DepletionArrayStoreMP(DepletionArrayStore):
    @property
    def mp_manager(self):
        from pixcap65.concurrency import get_manager
        manager = get_manager(**self.manager_args)
        if "address" not in self.manager_args:
            self.manager_args["address"] = manager.address

        if int(self.manager_debug_information) > 2:
            with open("depletion_manager_instance_{}.txt".format(mp.current_process().pid), 'a') as f:
                date_obj = datetime.datetime.now()
                full_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                date = date_obj.strftime("%Y-%m-%d")
                time_str = date_obj.strftime("%H:%M:%S")
                print(date, time_str, manager.address, manager._process.pid,  file=f)
        return manager

    def __init__(self, n_depletions=None, **manager_kwargs):
        self.manager_args = manager_kwargs.copy()
        self.manager_debug_information = self.manager_args.pop("debug_information", False)
        # we need to make sure that NO authkeys are transmitted by pickling between processes
        if "authkey" in self.manager_args:
            del self.manager_args["authkey"]

        super(DepletionArrayStoreMP, self).__init__(n_depletions, **manager_kwargs)
        if self.manager_debug_information:
            with open("depletion_manager_information_{}.txt".format(mp.current_process().pid), 'a') as f:
                date_obj = datetime.datetime.now()
                full_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                date = date_obj.strftime("%Y-%m-%d")
                time_str = date_obj.strftime("%H:%M:%S")
                print(date, mp.current_process().pid, time_str,
                      mp.current_process().name, mp.current_process().authkey, file=f)

class DopingArrayStore(DepletionDataStore):
    def __init__(self, doping_shape, n_depletions=1):
        shape_list = [*GENERAL_PIXCAP_SHAPE, 4]
        general_shape = tuple(shape_list)
        matrix_shape = tuple(shape_list + [4])
        depletion_shape = tuple([*GENERAL_PIXCAP_SHAPE, n_depletions])
        self.pixel_row = 1
        self.pixel_col = 1
        self.depletion_width_plate = np.full(shape=doping_shape, fill_value=np.nan)
        self.depletion_width_plate_error = np.full(shape=doping_shape, fill_value=np.nan)
        self.depletion_fit_parameter_table = np.full(general_shape, fill_value=np.nan)
        self.depletion_fit_parameter_error_table = np.full(general_shape, fill_value=np.nan)
        self.depletion_fit_covariance_table = np.full(matrix_shape, fill_value=np.nan)
        self.effective_doping_table = np.full(shape=doping_shape, fill_value=np.nan)
        self.resistivity_table = np.full(shape=doping_shape, fill_value=np.nan)
        self.second_resistivities = np.full(shape=depletion_shape, fill_value=np.nan)
        self.second_resistivities_err = np.full(shape=depletion_shape, fill_value=np.nan)

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
            case "resistivity":
                self.resistivity_table[self.pixel_col, self.pixel_row] = data
            case "res_mod":
                self.second_resistivities[self.pixel_col, self.pixel_row] = data
            case "res_mod_err":
                self.second_resistivities_err[self.pixel_col, self.pixel_row] = data
            case _:
                raise ValueError(f"Unknown data storage key {key}")

class SummaryTable(tb.IsDescription):
    sensor = tb.StringCol(10, pos=0)
    inj_cap = tb.Float64Col(pos=1)
    inj_cap_2 = tb.Float64Col(pos=2)
