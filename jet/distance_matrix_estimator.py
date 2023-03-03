from __future__ import annotations

from typing import List, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .utils import matrix


class DistanceMatrixEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs: int = 1, verbose: bool = False, metric: str = "shape_based_distance", c: float = 700) -> None:
        super().__init__()

        self.n_jobs = n_jobs
        self.verbose = verbose
        self.metric = metric
        self.c = c
        self._distance_matrix: Optional[np.ndarray] = None
        self._distance_matrices: List[np.ndarray] = []
        self._cluster_indices: List[np.ndarray] = []
        self._medoid_distances: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> DistanceMatrixEstimator:
        raise NotImplementedError
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def fit_transform(self, X: List[np.ndarray], y: np.ndarray) -> np.ndarray:
        self._build_medoid_distance_matrix(X, y)
        self._build_distance_matrix(X, y)
        assert self._distance_matrix is not None, 'Distance matrix is not built'
        return self._condense_distance_matrix()

    def _build_medoid_distance_matrix(self, X: List[np.ndarray], y: np.ndarray) -> None:
        medoids = []
        for c in np.unique(y):
            ids = np.where(y == c)[0]
            dm = matrix([X[i] for i in ids], n_jobs=self.n_jobs, verbose=self.verbose, metric=self.metric, c=self.c)
            self._distance_matrices.append(dm)
            self._cluster_indices.append(ids)
            medoid = np.argmin(dm.sum(axis=0))
            medoids.append(ids[medoid])
        self._medoid_distances = matrix([X[i] for i in medoids], n_jobs=self.n_jobs, verbose=self.verbose, metric=self.metric, c=self.c)
    
    def _build_distance_matrix(self, X: List[np.ndarray], y: np.ndarray) -> None:
        assert self._medoid_distances is not None, 'Medoid distance matrix is not built'
        self._distance_matrix = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            i_label = y[i]
            for j in range(i + 1, len(X)):
                j_label = y[j]
                if i_label == j_label:
                    pseudo_i, pseudo_j = (
                        np.where(self._cluster_indices[i_label] == i)[0][0],
                        np.where(self._cluster_indices[i_label] == j)[0][0],
                    )
                    distance = self._distance_matrices[i_label][pseudo_i, pseudo_j]
                else:
                    distance = self._medoid_distances[i_label, j_label]
                self._distance_matrix[i, j] = distance
                self._distance_matrix[j, i] = distance

    def _condense_distance_matrix(self) -> np.ndarray:
        assert self._distance_matrix is not None, 'Distance matrix is not built'
        condensed = self._distance_matrix[np.triu_indices(self._distance_matrix.shape[0], k=1)]
        return condensed