from __future__ import annotations

from typing import List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.cluster.hierarchy import cut_tree, ward


class WardClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters: int, n_jobs: int = 1, verbose: bool = False) -> None:
        super().__init__()

        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> WardClustering:
        self._linkage_matrix = ward(X)
        return self

    def predict(self, X: Optional[Any] = None) -> np.ndarray:
        return cut_tree(self._linkage_matrix, n_clusters=self.n_clusters).flatten()

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X).predict(X)
