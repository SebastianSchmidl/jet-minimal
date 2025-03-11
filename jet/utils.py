from __future__ import annotations

import numpy as np
from typing import Callable, List, Type, Union, Optional, Any

from scipy.signal import correlate
from joblib import Parallel, delayed
from ._dtw import dtw_distance


DistanceMeasureFunctionType = Callable[[np.ndarray, np.ndarray, Optional[Any]], float]


class staticproperty(staticmethod):
    def __get__(self, *args):
        return self.__func__()


class JETMetric:
    def __init__(self, distance_measure: Type[DistanceMeasureFunctionType]) -> None:
        self.distance_measure = distance_measure

    def __call__(self, series_a: np.ndarray, series_b: np.ndarray, **kwargs) -> float:
        return self.distance_measure(series_a, series_b, **kwargs)

    @staticproperty
    def SHAPE_BASED_DISTANCE() -> JETMetric:
        return JETMetric(shape_based_distance)

    @staticproperty
    def DTW() -> JETMetric:
        return JETMetric(dtw_distance)

    @staticproperty
    def MSM() -> JETMetric:
        return JETMetric(move_split_merge)


def shape_based_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the shape-based distance between two time series."""
    return abs(
        float(
            1
            - np.max(
                correlate(x, y, method="fft") / np.sqrt(np.dot(x, x) * np.dot(y, y))
            )
        )
    )


def c(x_i: float, x_i_1: float, y_j: float, constant: float) -> float:
    if (x_i_1 <= x_i and x_i <= y_j) or (x_i_1 >= x_i and x_i >= y_j):
        return constant
    return float(constant + min(np.abs(x_i - x_i_1), np.abs(x_i - y_j)))


def move_split_merge(
    x: np.ndarray, y: np.ndarray, constant: Optional[float] = 0.5
) -> float:
    constant = constant or 0.5
    m = x.shape[0]
    n = y.shape[0]

    cost = np.zeros((m, n), dtype=np.float_)
    cost[0, 0] = np.abs(x[0] - y[0])
    for i in range(1, m):
        cost[i, 0] = cost[i - 1, 0] + c(x[i], x[i - 1], y[0], constant)

    for j in range(1, n):
        cost[0, j] = cost[0, j - 1] + c(y[j], x[0], y[j - 1], constant)

    for i in range(1, m):
        for j in range(1, n):
            cost[i, j] = min(
                cost[i - 1, j - 1] + np.abs(x[i] - y[j]),
                cost[i - 1, j] + c(x[i], x[i - 1], y[j], constant),
                cost[i, j - 1] + c(y[j], x[i], y[j - 1], constant),
            )

    return float(cost[m - 1, n - 1])


def z_matrix(
    series: Union[np.ndarray, List[np.ndarray]],
    n_jobs: int = 1,
    verbose: bool = False,
    metric: Union[str, JETMetric] = "shape_based_distance",
    c: float = 700,
) -> np.ndarray:
    if isinstance(metric, JETMetric):
        f = metric
    elif isinstance(metric, str):
        if metric == "msm":
            f = lambda x, y: JETMetric.MSM(x, y, constant=c)
        elif metric == "shape_based_distance":
            f = JETMetric.SHAPE_BASED_DISTANCE
        elif metric == "dtw":
            f = JETMetric.DTW
        else:
            raise ValueError(f"Unknown metric {metric}")
    else:
        raise ValueError(f"Unknown metric {metric}")

    z = Parallel(n_jobs=n_jobs)(
        delayed(f)(series[i], series[j])  # only 1d metrics so far
        for i in range(len(series) - 1)
        for j in range(i + 1, len(series))
    )
    return np.array(z).flatten()


def matrix(
    series: Union[np.ndarray, List[np.ndarray]],
    n_jobs: int = 1,
    verbose: bool = False,
    metric: Union[str, JETMetric] = "shape_based_distance",
    c: float = 700,
) -> np.ndarray:
    len_s = len(series)
    z = z_matrix(series, n_jobs=n_jobs, verbose=verbose, metric=metric, c=c)
    distance_matrix = np.zeros((len_s, len_s))
    last_end = 0
    for i in range(len_s):
        values = z[last_end : last_end + len_s - (i + 1)]
        distance_matrix[i, i + 1 :] = values
        distance_matrix[i + 1 :, i] = values
        last_end += len_s - (i + 1)
    return distance_matrix
