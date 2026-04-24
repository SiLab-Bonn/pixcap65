import numpy as np

GENERAL_TRANSFORMATION_MATRIX = np.array(
    [[1.e-12, 1.e-6, 1.e3, 1.e-6], [1.e-6, 1, 1.e9, 1], [1.e3, 1.e9, 1.e18, 1.e9], [1.e-6, 1, 1.e9, 1]])
ANALYSIS_GROUP_NAME = "analysis"
ANALYSIS_CORRECTED_GROUP_NAME = "analysis_correction"


def transform_covariance(cov):
    """
    transform_covariance

    Transforms the returned covariance such that the units match the specified ones.
    For the more advanced fits also the contributions of the fixed reference voltage parameter are removed from the covariance matrix.

    :param cov: covariance matrix to be transformed.
    :return: transformed covariance matrix.
    """
    cov = np.asarray(cov)
    match (cov.shape):
        case (2, 2):
            assert cov.shape == (2, 2)
            mask = np.array([[True, False, True, False], [False, False, False, False], [True, False, True, False],
                             [False, False, False, False]])
            return cov * GENERAL_TRANSFORMATION_MATRIX[mask].reshape((2, 2))
        case (3, 3):
            assert cov.shape == (3, 3)
            # here it is necessary to reduce the parts from the covariance of u0 to
            mask = np.array([[True, False, True, False], [False, False, False, False], [True, False, True, False],
                             [False, False, False, False]])
            mask_reduction = np.array([[True, True, False], [True, True, False], [False, False, False]])
            return cov[mask_reduction].reshape((2, 2)) * GENERAL_TRANSFORMATION_MATRIX[mask].reshape((2, 2))
        case (4, 4):
            assert cov.shape == (4, 4)
            # here it is necessary to reduce the parts from the covariance of u0 to
            mask = np.array([[True, True, True, False], [True, True, True, False], [True, True, True, False],
                             [False, False, False, False]])
            return cov[mask].reshape((3, 3)) * GENERAL_TRANSFORMATION_MATRIX[mask].reshape((3, 3))
        case _:
            raise ValueError(
                "The dimension of the covariance matrix does not fit to any of the fitting functions and their parameters.")


def str_join(delimiter, *args):
    return delimiter.join(args)


def check_leaf_unit(leaf, unit: str):
    """
    check_leaf_unit(leaf, unit)

    Verify that the table or array has a units attribute and the unit is the one expected.
    As last step return the data structure requested.

    This may not work for tables with multiple units (one per column)
    :param leaf: Data structure to be verified and extracted.
    :param unit: Expected unit for the data structure.
    :return: array_like of the data structures contents.
    """
    if "Units" not in leaf.attrs or leaf.attrs["Units"] != unit:
        raise AssertionError
    return leaf[:]


default_analysis_keyword_arguments = {
    "cap_name": "HistCap",
    "cap_title": "Capacitance Histogram",
    "cap_err_name": "HistCapErr",
    "cap_err_title": "Capacitance Error Histogram",
    "leak_name": "HistLeak",
    "leak_title": "Leakage Current Histogram",
    "leak_error_name": "HistLeakErr",
    "leak_error_title": "Leakage Current Error Histogram",
    "resistor_name": "HistRes",
    "resistor_title": "On-Resistance Histogram",
    "resistor_error_name": "HistResErr",
    "resistor_error_title": "On-Resistance Error Histogram",
    "cov_name": "HistFitCov",
    "cov_title": 'Fit Covariance Matrix'
}
