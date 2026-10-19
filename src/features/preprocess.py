"""Fitted transforms that can only see training rows (CONSTRAINTS R2, R3).

`fit` takes the full array plus explicit training indices, so fitting on everything by accident
is a TypeError rather than a silent leak. Never normalise per day across all images.
"""

import numpy as np


class BandScaler:
    """Per-band min-max to [0, 1]; works on spectra (N, B) and cubes (N, H, W, B)."""

    def fit(self, X: np.ndarray, train_idx: np.ndarray) -> "BandScaler":
        bands = X[train_idx].reshape(-1, X.shape[-1])
        self.min_ = bands.min(axis=0)
        self.max_ = bands.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        span = np.where(self.max_ > self.min_, self.max_ - self.min_, 1.0)
        return ((X - self.min_) / span).astype(np.float32)


class OutlierFilter:
    """Fedotov's filter: drop training spectra whose correlation with the training mean is
    below `threshold`. Only training rows are ever examined or dropped."""

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def fit(self, spectra: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
        """Returns the kept training indices; `n_removed` and `removed_idx` are logged by
        callers into config.json."""
        train_idx = np.asarray(train_idx)
        X = spectra[train_idx]
        ref = X.mean(axis=0)
        Xc = X - X.mean(axis=1, keepdims=True)
        rc = ref - ref.mean()
        corr = (Xc @ rc) / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(rc))
        keep = corr >= self.threshold
        self.removed_idx = train_idx[~keep]
        self.n_removed = int((~keep).sum())
        return train_idx[keep]
