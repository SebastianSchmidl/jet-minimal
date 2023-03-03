from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from tsfresh.feature_extraction import extract_features


class FeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs: int = 1, verbose: bool = False):
        super().__init__()
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def fit(self, X: List[np.ndarray], y: Optional[np.ndarray] = None) -> FeatureEncoder:
        return self
    
    def transform(self, X: List[np.ndarray]) -> np.ndarray:
        features = self._extract_features(X)
        return features.values

    def fit_transform(self, X: List[np.ndarray], y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _extract_features(self, data: List[np.ndarray]) -> pd.DataFrame:
        trans = self._transform_format(data)
        feats = extract_features(
            trans,
            n_jobs=self.n_jobs,
            column_id="id",
            column_sort="time",
            column_kind=None,
            column_value="value",
            default_fc_parameters=FilteredFCParameters(data),
        )
        assert isinstance(feats, pd.DataFrame), "Features must be a DataFrame"
        feats = feats.dropna(axis=1)
        feats = StandardScaler().fit_transform(feats.values)
        feats = VarianceThreshold().fit_transform(feats)
        return pd.DataFrame(feats, columns=list(range(feats.shape[1])))

    def _transform_format(self, X: List[np.ndarray]) -> dict:
        return {
            "value-0": pd.DataFrame(
                [{"id": j, "time": i, "value": x_i} for j, x in enumerate(X) for i, x_i in enumerate(x)]
            )
        }
    

class FilteredFCParameters(dict):
    def __init__(self, data: List[np.ndarray]):
        name_to_param = {
            "agg_autocorrelation": [{"f_agg": "median", "maxlag": 40}],
            "agg_linear_trend": [{"attr": "rvalue", "chunk_len": 5, "f_agg": "max"}, {"attr": "rvalue", "chunk_len": 5, "f_agg": "min"}, {"attr": "rvalue", "chunk_len": 5, "f_agg": "mean"}, {"attr": "rvalue", "chunk_len": 5, "f_agg": "var"}, {"attr": "rvalue", "chunk_len": 10, "f_agg": "var"}, {"attr": "intercept", "chunk_len": 5, "f_agg": "min"}, {"attr": "intercept", "chunk_len": 5, "f_agg": "mean"}, {"attr": "intercept", "chunk_len": 10, "f_agg": "max"}, {"attr": "intercept", "chunk_len": 10, "f_agg": "min"}, {"attr": "intercept", "chunk_len": 10, "f_agg": "mean"}, {"attr": "intercept", "chunk_len": 10, "f_agg": "var"}, {"attr": "slope", "chunk_len": 5, "f_agg": "max"}, {"attr": "slope", "chunk_len": 5, "f_agg": "min"}, {"attr": "slope", "chunk_len": 5, "f_agg": "mean"}, {"attr": "slope", "chunk_len": 10, "f_agg": "min"}, {"attr": "slope", "chunk_len": 10, "f_agg": "var"}, {"attr": "stderr", "chunk_len": 5, "f_agg": "max"}, {"attr": "stderr", "chunk_len": 5, "f_agg": "min"}, {"attr": "stderr", "chunk_len": 5, "f_agg": "mean"}, {"attr": "stderr", "chunk_len": 5, "f_agg": "var"}, {"attr": "stderr", "chunk_len": 10, "f_agg": "max"}, {"attr": "stderr", "chunk_len": 10, "f_agg": "min"}, {"attr": "stderr", "chunk_len": 10, "f_agg": "mean"}],
            "approximate_entropy": [{"m": 2, "r": 0.1}, {"m": 2, "r": 0.5}, {"m": 2, "r": 0.7}, {"m": 2, "r": 0.9}],
            "ar_coefficient": [{"coeff": 2, "k": 10}, {"coeff": 4, "k": 10}, {"coeff": 5, "k": 10}, {"coeff": 10, "k": 10}],
            "augmented_dickey_fuller": [{"attr": "pvalue", "autolag": "AIC"}],
            "autocorrelation": [{"lag": 1}, {"lag": 3}, {"lag": 4}, {"lag": 5}, {"lag": 6}, {"lag": 7}, {"lag": 8}],
            "benford_correlation": None,
            "binned_entropy": [{"max_bins": 10}],
            "change_quantiles": [{"f_agg": "mean", "isabs": False, "qh": 0.2, "ql": 0.0}, {"f_agg": "mean", "isabs": True, "qh": 0.2, "ql": 0.0}, {"f_agg": "var", "isabs": False, "qh": 0.4, "ql": 0.0}, {"f_agg": "var", "isabs": True, "qh": 0.4, "ql": 0.0}, {"f_agg": "mean", "isabs": True, "qh": 0.6, "ql": 0.0}, {"f_agg": "var", "isabs": True, "qh": 0.6, "ql": 0.0}, {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.0}, {"f_agg": "mean", "isabs": False, "qh": 0.4, "ql": 0.2}, {"f_agg": "var", "isabs": False, "qh": 0.4, "ql": 0.2}, {"f_agg": "mean", "isabs": True, "qh": 0.4, "ql": 0.2}, {"f_agg": "mean", "isabs": True, "qh": 0.8, "ql": 0.2}, {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.2}, {"f_agg": "var", "isabs": True, "qh": 1.0, "ql": 0.2}, {"f_agg": "mean", "isabs": False, "qh": 0.8, "ql": 0.4}, {"f_agg": "var", "isabs": True, "qh": 0.8, "ql": 0.4}, {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.4}, {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.4}, {"f_agg": "var", "isabs": True, "qh": 1.0, "ql": 0.4}, {"f_agg": "mean", "isabs": False, "qh": 0.8, "ql": 0.6}, {"f_agg": "mean", "isabs": True, "qh": 0.8, "ql": 0.6}, {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.6}, {"f_agg": "var", "isabs": True, "qh": 1.0, "ql": 0.6}, {"f_agg": "mean", "isabs": False, "qh": 1.0, "ql": 0.8}, {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.8}, {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.8}],
            "count_below": [{"t": 0}],
            "cwt_coefficients": [{"coeff": 0, "w": 5, "widths": (2, 5, 10, 20)}, {"coeff": 0, "w": 10, "widths": (2, 5, 10, 20)}, {"coeff": 0, "w": 20, "widths": (2, 5, 10, 20)}, {"coeff": 1, "w": 2, "widths": (2, 5, 10, 20)}, {"coeff": 1, "w": 10, "widths": (2, 5, 10, 20)}, {"coeff": 1, "w": 20, "widths": (2, 5, 10, 20)}, {"coeff": 2, "w": 20, "widths": (2, 5, 10, 20)}, {"coeff": 3, "w": 2, "widths": (2, 5, 10, 20)}, {"coeff": 3, "w": 5, "widths": (2, 5, 10, 20)}, {"coeff": 4, "w": 2, "widths": (2, 5, 10, 20)}, {"coeff": 4, "w": 5, "widths": (2, 5, 10, 20)}, {"coeff": 4, "w": 10, "widths": (2, 5, 10, 20)}, {"coeff": 4, "w": 20, "widths": (2, 5, 10, 20)}, {"coeff": 5, "w": 2, "widths": (2, 5, 10, 20)}, {"coeff": 5, "w": 5, "widths": (2, 5, 10, 20)}, {"coeff": 5, "w": 10, "widths": (2, 5, 10, 20)}, {"coeff": 6, "w": 5, "widths": (2, 5, 10, 20)}, {"coeff": 6, "w": 10, "widths": (2, 5, 10, 20)}, {"coeff": 7, "w": 2, "widths": (2, 5, 10, 20)}, {"coeff": 7, "w": 10, "widths": (2, 5, 10, 20)}, {"coeff": 7, "w": 20, "widths": (2, 5, 10, 20)}, {"coeff": 8, "w": 2, "widths": (2, 5, 10, 20)}, {"coeff": 8, "w": 10, "widths": (2, 5, 10, 20)}, {"coeff": 9, "w": 5, "widths": (2, 5, 10, 20)}, {"coeff": 10, "w": 5, "widths": (2, 5, 10, 20)}, {"coeff": 10, "w": 20, "widths": (2, 5, 10, 20)}, {"coeff": 11, "w": 2, "widths": (2, 5, 10, 20)}, {"coeff": 11, "w": 10, "widths": (2, 5, 10, 20)}, {"coeff": 11, "w": 20, "widths": (2, 5, 10, 20)}, {"coeff": 12, "w": 5, "widths": (2, 5, 10, 20)}, {"coeff": 13, "w": 2, "widths": (2, 5, 10, 20)}, {"coeff": 13, "w": 5, "widths": (2, 5, 10, 20)}, {"coeff": 13, "w": 10, "widths": (2, 5, 10, 20)}, {"coeff": 14, "w": 2, "widths": (2, 5, 10, 20)}],
            "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 1}, {"num_segments": 10, "segment_focus": 2}, {"num_segments": 10, "segment_focus": 3}, {"num_segments": 10, "segment_focus": 5}, {"num_segments": 10, "segment_focus": 6}, {"num_segments": 10, "segment_focus": 7}, {"num_segments": 10, "segment_focus": 9}],
            "fft_aggregated": [{"aggtype": "kurtosis"}],
            "fft_coefficient": [{"attr": "real", "coeff": 2}, {"attr": "real", "coeff": 3}, {"attr": "real", "coeff": 4}, {"attr": "real", "coeff": 5}, {"attr": "imag", "coeff": 0}, {"attr": "imag", "coeff": 2}, {"attr": "imag", "coeff": 4}, {"attr": "imag", "coeff": 6}, {"attr": "abs", "coeff": 0}, {"attr": "abs", "coeff": 4}, {"attr": "abs", "coeff": 5}, {"attr": "angle", "coeff": 0}, {"attr": "angle", "coeff": 1}, {"attr": "angle", "coeff": 2}, {"attr": "angle", "coeff": 5}, {"attr": "angle", "coeff": 7}],
            "first_location_of_maximum": None,
            "first_location_of_minimum": None,
            "fourier_entropy": [{"bins": 2}, {"bins": 3}, {"bins": 5}, {"bins": 100}],
            "has_duplicate_max": None,
            "has_duplicate_min": None,
            "index_mass_quantile": [{"q": 0.1}, {"q": 0.2}, {"q": 0.3}, {"q": 0.4}, {"q": 0.7}],
            "kurtosis": None,
            "large_standard_deviation": [{"r": 0.05}, {"r": 0.15}, {"r": 0.3}, {"r": 0.35}, {"r": 0.45}, {"r": 0.55}, {"r": 0.7}, {"r": 0.8}, {"r": 0.85}, {"r": 0.9}],
            "last_location_of_maximum": None,
            "last_location_of_minimum": None,
            "linear_trend": [{"attr": "pvalue"}, {"attr": "rvalue"}, {"attr": "intercept"}],
            "mean": None,
            "mean_abs_change": None,
            "mean_second_derivative_central": None,
            "minimum": None,
            "number_crossing_m": [{"m": 0}, {"m": 1}] + [{"m": m} for m in self._generate_grid(data, n=20)],
            "number_cwt_peaks": [{"n": 1}, {"n": 5}],
            "number_peaks": [{"n": 3}, {"n": 5}, {"n": 50}],
            "partial_autocorrelation": [{"lag": 2}, {"lag": 4}],
            "percentage_of_reoccurring_values_to_all_values": None,
            "permutation_entropy": [{"dimension": 4, "tau": 1}, {"dimension": 5, "tau": 1}, {"dimension": 6, "tau": 1}],
            "quantile": [{"q": 0.4}, {"q": 0.7}, {"q": 0.8}, {"q": 0.9}],
            "range_count": [{"max": 1, "min": -1}, {"max": 1000000000000.0, "min": 0}],
            "ratio_beyond_r_sigma": [{"r": 0.5}, {"r": 1}, {"r": 1.5}, {"r": 2}, {"r": 10}],
            "ratio_value_number_to_time_series_length": None,
            "skewness": None,
            "standard_deviation": None,
            "sum_values": None,
            "symmetry_looking": [{"r": 0.05}, {"r": 0.3}, {"r": 0.45}, {"r": 0.5}, {"r": 0.6}, {"r": 0.7}, {"r": 0.8}, {"r": 0.9}, {"r": 0.95}],
            "time_reversal_asymmetry_statistic": [{"lag": 1}],
            "variance_larger_than_standard_deviation": None,
        }

        super().__init__(name_to_param)
    
    def _generate_grid(self, data: List[np.ndarray], n=10) -> List[float]:
        values = np.sort(np.concatenate(data))
        grid = np.linspace(values[0], values[-1], n)
        return grid.tolist()
        