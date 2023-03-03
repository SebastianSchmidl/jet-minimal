from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import Birch


class PreClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters: Optional[int], n_jobs: int = 1, verbose=False, birch_threshold: float = 0.1) -> None:
        super().__init__()

        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._clusterer = None
        self.n_clusters = n_clusters
        self.birch_threshold = birch_threshold
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> PreClustering:
        assert X.shape[0] > 9, "The number of time series is less than 10. It's probably faster to use the exact algorithm."
        n_clusters = self.n_clusters or int(np.sqrt(len(X)))*3
        self._clusterer = Birch(n_clusters=n_clusters, threshold=self.birch_threshold).fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clusterer.predict(X)
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X).predict(X)
