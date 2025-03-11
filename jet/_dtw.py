import math
import numpy as np
from typing import Optional


def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
) -> float:
    if x.ndim == 1 and y.ndim == 1:
        bounding_matrix = create_bounding_matrix(
            x.shape[0], y.shape[0], window, itakura_max_slope
        )
        return _dtw_distance(x, y, bounding_matrix)
    raise ValueError("x and y must be 1D")


def _dtw_distance(x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray) -> float:
    return _dtw_cost_matrix(x, y, bounding_matrix)[x.shape[0] - 1, y.shape[0] - 1]


def _dtw_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray
) -> np.ndarray:
    x_size = x.shape[0]
    y_size = y.shape[0]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if bounding_matrix[i, j]:
                cost_matrix[i + 1, j + 1] = _univariate_squared_distance(
                    x[i], y[j]
                ) + min(
                    cost_matrix[i, j + 1],
                    cost_matrix[i + 1, j],
                    cost_matrix[i, j],
                )

    return cost_matrix[1:, 1:]


def _univariate_squared_distance(x: float, y: float) -> float:
    difference = x - y
    return difference * difference


def create_bounding_matrix(
    x_size: int,
    y_size: int,
    window: Optional[float] = None,
    itakura_max_slope: Optional[float] = None,
):
    if itakura_max_slope is not None:
        if itakura_max_slope < 0 or itakura_max_slope > 1:
            raise ValueError("itakura_max_slope must be between 0 and 1")
        return _itakura_parallelogram(x_size, y_size, itakura_max_slope)
    if window is not None and window != 1.0:
        if window < 0 or window > 1:
            raise ValueError("window must be between 0 and 1")
        return _sakoe_chiba_bounding(x_size, y_size, window)
    return np.full((x_size, y_size), True)


def _itakura_parallelogram(x_size: int, y_size: int, max_slope_percent: float):
    one_percent = min(x_size, y_size) / 100
    max_slope = math.floor((max_slope_percent * one_percent) * 100)
    min_slope = 1 / float(max_slope)
    max_slope *= float(x_size) / float(y_size)
    min_slope *= float(x_size) / float(y_size)

    lower_bound = np.empty((2, y_size))
    lower_bound[0] = min_slope * np.arange(y_size)
    lower_bound[1] = (
        (x_size - 1) - max_slope * (y_size - 1) + max_slope * np.arange(y_size)
    )
    lower_bound_ = np.empty(y_size)
    for i in range(y_size):
        lower_bound_[i] = max(round(lower_bound[0, i], 2), round(lower_bound[1, i], 2))
    lower_bound_ = np.ceil(lower_bound_)

    upper_bound = np.empty((2, y_size))
    upper_bound[0] = max_slope * np.arange(y_size)
    upper_bound[1] = (
        (x_size - 1) - min_slope * (y_size - 1) + min_slope * np.arange(y_size)
    )
    upper_bound_ = np.empty(y_size)
    for i in range(y_size):
        upper_bound_[i] = min(round(upper_bound[0, i], 2), round(upper_bound[1, i], 2))
    upper_bound_ = np.floor(upper_bound_ + 1)

    bounding_matrix = np.full((x_size, y_size), False)
    for i in range(y_size):
        bounding_matrix[int(lower_bound_[i]) : int(upper_bound_[i]), i] = True
    return bounding_matrix


def _sakoe_chiba_bounding(
    x_size: int, y_size: int, radius_percent: float
) -> np.ndarray:

    if x_size > y_size:
        return _sakoe_chiba_bounding(y_size, x_size, radius_percent).T

    matrix = np.full((x_size, y_size), False)  # Create a matrix filled with False

    max_size = max(x_size, y_size) + 1

    shortest_dimension = min(x_size, y_size)
    thickness = int(radius_percent * shortest_dimension)
    for step in range(max_size):
        x_index = math.floor((step / max_size) * x_size)
        y_index = math.floor((step / max_size) * y_size)

        upper = max(0, (x_index - thickness))
        lower = min(x_size, (x_index + thickness + 1))

        matrix[upper:lower, y_index] = True

    return matrix
