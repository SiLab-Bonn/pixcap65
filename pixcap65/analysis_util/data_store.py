import numpy as np
import tables as tb

from pixcap65.analysis_util import GENERAL_PIXCAP_SHAPE


class DepletionDataStore:
    def store_data(self, key, data):
        # TODO document why this method is empty
        pass

    def flush_data(self):
        # TODO document why this method is empty
        pass


class DepletionTableStore(DepletionDataStore):
    def __init__(self, table: tb.Table):
        self.table = table
        self.entry = self.table.row

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


class DepletionArrayStore(DepletionDataStore):
    def __init__(self, n_depletions=None):
        general_shape = GENERAL_PIXCAP_SHAPE if n_depletions is None else tuple([*GENERAL_PIXCAP_SHAPE, n_depletions])
        parameter_shape = tuple([*GENERAL_PIXCAP_SHAPE, 4]) if n_depletions is None else tuple(
            [*GENERAL_PIXCAP_SHAPE, n_depletions, 4])
        self.depletion_voltage = np.full(shape=general_shape, fill_value=np.nan)
        self.depletion_error = np.full(shape=general_shape, fill_value=np.nan)
        self.fit_parameter_estimators = np.full(shape=parameter_shape, fill_value=np.nan)
        self.fit_parameter_errors = np.full(shape=parameter_shape, fill_value=np.nan)
        self.pixel_row = 1
        self.pixel_col = 1
        self.depletion_reg = None

    def set_pixel(self, i_row, i_col):
        self.pixel_row = i_row
        self.pixel_col = i_col

    def set_depletion_region(self, i):
        self.depletion_reg = i

    def store_data(self, key, data):
        if self.depletion_reg is None:
            match key:
                case "depletion":
                    self.depletion_voltage[self.pixel_col, self.pixel_row] = data
                case "depletion_error":
                    self.depletion_error[self.pixel_col, self.pixel_row] = data
                case "fit_result_first":
                    self.fit_parameter_estimators[self.pixel_col, self.pixel_row, :2] = data
                case "fit_error_first":
                    self.fit_parameter_errors[self.pixel_col, self.pixel_row, :2] = data
                case "fit_result_second":
                    self.fit_parameter_estimators[self.pixel_col, self.pixel_row, 2:] = data
                case "fit_error_second":
                    self.fit_parameter_errors[self.pixel_col, self.pixel_row, 2:] = data
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
