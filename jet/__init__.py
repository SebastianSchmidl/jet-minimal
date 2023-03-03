from __future__ import annotations

from typing import Optional, List, Union
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from warnings import warn
from enum import Enum

from .feature_encoder import FeatureEncoder
from .pre_clustering import PreClustering
from .distance_matrix_estimator import DistanceMatrixEstimator
from .ward_clustering import WardClustering


class JETMetric(Enum):
    SHAPE_BASED_DISTANCE = "shape_based_distance"
    DTW = "dtw"
    MSM = "msm"


class JET(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters: int = 10, n_pre_clusters: Optional[int] = None, n_jobs: int = 1, verbose: bool = False, metric: Union[str, JETMetric] = "shape_based_distance", c: float = 700) -> None:
        super().__init__()

        self.n_clusters = n_clusters
        self.n_pre_clusters = n_pre_clusters
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.metric = metric if isinstance(metric, JETMetric) else JETMetric(metric)
        self.c = c

        self._feature_encoder = FeatureEncoder(n_jobs=self.n_jobs, verbose=self.verbose)
        self._pre_clustering = PreClustering(n_clusters=self.n_pre_clusters, n_jobs=self.n_jobs, verbose=self.verbose)
        self._distance_matrix_estimator = DistanceMatrixEstimator(n_jobs=self.n_jobs, verbose=self.verbose, metric=self.metric.value, c=self.c)
        self._ward_clustering = WardClustering(n_clusters=self.n_clusters, n_jobs=self.n_jobs, verbose=self.verbose)

    def fit(self, X: List[np.ndarray], y: Optional[np.ndarray] = None) -> JET:
        if len(X) <= 150:
            warn("The number of time series is less than 150. It's probably faster to use the exact algorithm.")
        
        encoded = self._feature_encoder.fit_transform(X)
        pre_labels = self._pre_clustering.fit_predict(encoded)
        estimated_distance_matrix = self._distance_matrix_estimator.fit_transform(X, pre_labels)
        self._ward_clustering.fit(estimated_distance_matrix)
        return self
    
    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        return self._ward_clustering.predict(X)
    
    def fit_predict(self, X: List[np.ndarray], y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).predict(X)
