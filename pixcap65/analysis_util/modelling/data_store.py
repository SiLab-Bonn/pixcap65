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
Defines and implements some data structures used for more complicated analysis like the cv characterization
of a (full) sensor.
Also data structures used for summarising results are defined here.
"""

import datetime
import multiprocessing as mp
import threading

import numpy as np
import tables as tb

from pixcap65.analysis_util.utility import GENERAL_PIXCAP_SHAPE


class DepletionDataStore:
    """
    Stub/abstract storage/structure class to save/store results of the depletion analysis for each pixel.
    This includes data about the depletion width and resistivity and others in dependence on the applied bias voltages.
    The methods of this class have no 'real' implementation.
    """
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
        * dispersion: estimation of the systematic uncertainty of the depletion voltage by dispersion of
            parasitic capacitances between differen pixcap chips.

        For doping storage units:

        * width: estimation of the depletion width for the applied bias voltages
        * width_error: (statistical) uncertainties for the depletion width for the applied bias voltages
        * fit_parameters: estimation of the fit parameters to model the depletion width for the applied bias voltages
        * fit_parameters_error: estimation of the fit parameter errors to model the depletion width for the applied
            bias voltages
        * fit_covariance: covariance matrix of the fit parameters to model the depletion width for the applied
            bias voltages
        * doping: effective doping concentration for the applied bias voltages estimated from the depletion width and
            capacitance behaviour
        * resistivity: resistivity of the substrate estimated from effective doping concentration
        * res_mod: resistivity of the substrate estimated from fit to low voltage limit region of the C-V-curve
        * res_mod_error: (statistical) uncertainty of the substrate's resistivity estimated from fit to
            low voltage limit region of the C-V-curve.


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
    """
    Data structure to store the results of the depletion analysis for each pixel.
    This includes data about the depletion width and resistivity and others in dependence on the applied bias voltages.

    The data is stored in a table like form using a :py:class:`pytables.Table` to implement
    :py:class:`pixcap65.analysis_util.modelling.DepletionDataStore`.

    The data will not be constructed by this class but must be provided.
    The table description must be :py:class:`pixcap65.analysis_util.utility.DepletionData`.
    """
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
        # noinspection compatibility
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
                raise ValueError("The provided storage key is unknown: {key}".format(key=key))

    def flush_data(self):
        self.entry.append()


class DepletionNumpyStore(DepletionDataStore):
    """
    Data structure to store the results of the depletion analysis for each pixel.
    This includes data about the depletion width and resistivity and others in dependence on the applied bias voltages.

    The data is stored in a table like form using a :py:class:`numpy.rec.array`
    The dtype must be explicitly provided but can be derived from the corresponding
    :py:class:`pytables.Description` implementation.
    The table description must be :py:class:`pixcap65.analysis_util.utility.DepletionData`.
    """
    def __init__(self, dtp: np.dtype):
        self.entry = {}
        self.depletion_reg = None
        self.data_temp = []
        self.dtype = dtp
        self.fields = dtp.names

    def store_data(self, key, data):
        # perhaps use this here as the super-implementation and map the table implementation downwards!
        # noinspection compatibility
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
                raise ValueError("The provided storage key is unknown: {key}".format(key=key))

    def set_depletion_region(self, i):
        """
        set_depletion_region(i)

        @author Dominik Fischer
        last update: 2026-08-24

        For some sensors there is not a single region which reasons a depletion fit in order to obtain a/the
        depletion voltage but multiple.
        The results from fits to all these regions must be stored.
        Therefore it is necessary to set the internal property telling the data structure which depletion region is
        processed currently.

        :param i: index of the depletion region to be processed.
        """
        assert i < self.n_values
        self.depletion_reg = i

    def flush_data(self):
        assert self.fields is not None
        entries = tuple(self.entry.pop(name, np.nan) for name in self.fields)
        self.data_temp.extend([entries, ])
        self.entry.clear()

    @property
    def table(self):
        """
        table

        @author Dominik Fischer
        last update: 2026-08-24

        retrieve the tabular representation of all the processed depletion regions.
        :return: tabular representation of the data obtained.
        :rtype: numpy.rec.array
        """
        return np.rec.array(self.data_temp, dtype=self.dtype)


class DepletionArrayStore(DepletionDataStore):
    """
    Data structure to store the results of the depletion analysis for each pixel.
    This includes data about the depletion width and resistivity and others in dependence on the applied bias voltages.

    The data is stored in :py:class:`numpy.ndarray` object and could be retrieved by dedicated names.
    The names of the fields are as follows:

    * `DepletionArrayStore.depletion_voltage`: stores the estimation of the (full or surface) depletion voltage.
    * `DepletionArrayStore.depletion_error`: estimation of the statistical uncertainty of the depletion voltage.
    * `DepletionArrayStore.fit_parameter_estimators`: parameter vector for fit of the both linear regions.
    * `DepletionArrayStore.fit_parameter_errors`: vector of parameter errors for fit of both linear region.
    * `DepletionArrayStore.fit_parameter_covariances`: covariance matrix of both linear regions.
    * `DepletionArrayStore.systematic_errors`: estimation of the (general) systematic uncertainty of
        the depletion voltage.
    * `DepletionArrayStore.systematic_dispersion`: estimation of the systematic uncertainty of the depletion voltage
    by dispersion of parasitic capacitances between differen pixcap chips.

    """
    @property
    def mp_manager(self):
        """
        Fetch a multiprocessing manager of type :py:class:`multiprocessing.Manager` or some subclass of it.
        The actual class of the multiprocessing manager will depend on the implementation.
        It must provide (utility) methods to create dicts (mapping), lists, numpy.ndarray objects and locks.

        :return: multiprocessing.manager object to be used to organize the fields of this data structure.
        """
        # In this case it is just a dummy implementation as this implementation of DepletionDataStore is build to
        # handle multiprocessing.
        from pixcap65.concurrency.manager import ManagerDummy
        return ManagerDummy()

    def __init__(self, n_depletions=None, **manager_kwargs):
        self.no_additional_depletion = n_depletions is None or n_depletions == 1
        if n_depletions is not None and n_depletions <= 1:
            n_depletions = None
        general_shape = GENERAL_PIXCAP_SHAPE if self.no_additional_depletion else tuple([*GENERAL_PIXCAP_SHAPE, n_depletions])
        parameter_shape = tuple([*GENERAL_PIXCAP_SHAPE, 4]) if self.no_additional_depletion else tuple(
            [*GENERAL_PIXCAP_SHAPE, n_depletions, 4])
        cov_parameter_shape = tuple([*GENERAL_PIXCAP_SHAPE, 4, 4]) if self.no_additional_depletion else tuple(
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
        """
        Get the current pixel row for which the data should be stored. (or set it)
        :return: pixel row currently beeing processed.
        """
        with self.lock:
            return self._pixel_row.get(threading.get_native_id(), 1)

    @property
    def pixel_col(self):
        """
        Get the current pixel column for which the data should be stored. (or set it)
        :return: pixel column currently beeing processed.
        """
        with self.lock:
            return self._pixel_col.get(threading.get_native_id(), 1)

    @property
    def depletion_reg(self):
        """
        For some sensors there is not a single region which reasons a depletion fit in order to obtain a/the
        depletion voltage but multiple.
        The results from fits to all these regions must be stored.
        Therefore it is necessary to set the internal property telling the data structure which depletion region is
        processed currently.

        :return: index of the currently processed depletion region.
        """
        with self.lock:
            return self._depletion_reg.get(-1, None)

    @pixel_row.setter
    def pixel_row(self, i):
        """
        Set the current pixel row for which the data should be stored.

        :param i: pixel row to be processed next.
        """
        with self.lock:
            self._pixel_row[threading.get_native_id()] = i

    @pixel_col.setter
    def pixel_col(self, j):
        """
        Set the current pixel column for which the data should be stored.
        :param j: pixel column to be processed next.
        """
        with self.lock:
            self._pixel_col[threading.get_native_id()] = j

    @depletion_reg.setter
    def depletion_reg(self, i):
        """
        set_depletion_region(i)

        @author Dominik Fischer
        last update: 2026-08-24

        For some sensors there is not a single region which reasons a depletion fit in order to obtain a/the
        depletion voltage but multiple.
        The results from fits to all these regions must be stored.
        Therefore it is necessary to set the internal property telling the data structure which depletion region is
        processed currently.

        :param i: index of the depletion region to be processed.
        """
        with self.lock:
            if not self.no_additional_depletion:
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
        """
        set_pixel

        @author Dominik Fischer
        last update: 2026-08-24

        Set the pixel for which the analysis data is processed next.

        :param i_row: row index for the pixel
        :param i_col: column index for the pixel
        """
        with self.lock:
            self.pixel_row = i_row
            self.pixel_col = i_col

    def set_depletion_region(self, i):
        """
        set_depletion_region(i)

        @author Dominik Fischer
        last update: 2026-08-24

        For some sensors there is not a single region which reasons a depletion fit in order to obtain a/the
        depletion voltage but multiple.
        The results from fits to all these regions must be stored.
        Therefore it is necessary to set the internal property telling the data structure which depletion region is
        processed currently.

        :param i: index of the depletion region to be processed.
        """
        with self.lock:
            self.depletion_reg = i

    def store_data(self, key, data):
        # this part is not protected anyway from any access.
        with self.lock:
            if self.depletion_reg is None or self.no_additional_depletion:
                # noinspection compatibility
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
                        raise ValueError("The provided storage key is unknown: {key}".format(key=key))
            else:
                # noinspection compatibility
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
                        raise ValueError("The provided storage key is unknown: {key}".format(key=key))


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
                date = date_obj.strftime("%Y-%m-%d")
                time_str = date_obj.strftime("%H:%M:%S")
                print(date, mp.current_process().pid, time_str,
                      mp.current_process().name, mp.current_process().authkey, file=f)


class DopingArrayStore(DepletionDataStore):
    """
    Data structure to store the results of the doping analysis for each pixel.
    This includes data about the doping profile in dependence on the applied bias voltages.

    The data is stored in :class:`numpy.ndarray object and could be retrieved by dedicated names.
    The structure of the names is as follows:
`
    * `DopingArrayStore.depletion_width_plate`: estimation of the depletion width for the applied bias voltages.
    * `DopingArrayStore.depletion_width_plate_error`: (statistical) uncertainties for the depletion width for the
    applied bias voltages.
    * `DopingArrayStore.depletion_fit_parameter_table`: estimation of the fit parameters to model the depletion width
    for the applied bias voltages.
    * `DopingArrayStore.depletion_fit_parameter_error_table`: estimation of the fit parameter errors to model the
    depletion width for the applied bias voltages.
    * `DopingArrayStore.depletion_fit_covariance_table`: covariance matrix of the fit parameters to model
    the depletion width for the applied bias voltages.
    * `DopingArrayStore.effective_doping_table`: effective doping concentration for the applied bias voltages estimated
    from the depletion width and capacitance behaviour.
    * `DopingArrayStore.resistivity_table`: resistivity of the substrate estimated from effective doping concentration
    * `DopingArrayStore.second_resistivities`: resistivity of the substrate estimated from fit to low voltage limit
    region of the C-V-curve.
    * `DopingArrayStore.second_resistivities_err`: (statistical) uncertainty of the substrate's resistivity estimated
    from fit to low voltage limit region of the C-V-curve.
    """
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
        """
        set_pixel

        @author Dominik Fischer
        last update: 2026-08-24

        Set the pixel for which the analysis data is processed next.

        :param i_row: row index for the pixel
        :param i_col: column index for the pixel
        """
        self.pixel_row = i_row
        self.pixel_col = i_col

    def store_data(self, key, data):
        # noinspection compatibility
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
                raise ValueError("Unknown data storage key {key}".format(key=key))
