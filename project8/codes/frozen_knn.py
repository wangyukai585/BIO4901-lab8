"""Chunked exact cosine kNN, avoiding platform BLAS matmul warning artifacts."""

import numpy as np
from scipy.spatial.distance import cdist


class FrozenKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, x, y):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        if not np.isfinite(self.x).all():
            raise ValueError("Nonfinite embeddings")
        return self

    def predict_proba(self, x):
        result = []
        for start in range(0, len(x), 128):
            dist = cdist(
                np.asarray(x[start : start + 128], dtype=np.float64),
                self.x,
                metric="cosine",
            )
            if not np.isfinite(dist).all():
                raise ValueError("Nonfinite cosine distances")
            dist = np.maximum(dist, 0)
            k = min(self.k, len(self.x))
            idx = np.argsort(dist, axis=1, kind="stable")[:, :k]
            selected = np.take_along_axis(dist, idx, axis=1)
            weights = 1 / np.maximum(selected, 1e-15)
            zero = selected < 1e-15
            weights = np.where(zero.any(1, keepdims=True), zero, weights)
            probs = np.stack(
                [(weights * (self.y[idx] == c)).sum(1) for c in self.classes_], axis=1
            )
            result.append(probs / probs.sum(1, keepdims=True))
        return np.concatenate(result)
