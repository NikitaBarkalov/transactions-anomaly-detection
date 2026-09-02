import numpy as np


class PerFeatureClipper:
    def __init__(self, percentile: float = 99.9) -> None:
        self.percentile = percentile
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PerFeatureClipper":
        lo_pct = 100.0 - self.percentile
        hi_pct = self.percentile
        self.lo_ = np.percentile(X, lo_pct, axis=0)
        self.hi_ = np.percentile(X, hi_pct, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lo_ is None or self.hi_ is None:
            raise RuntimeError("PerFeatureClipper must be fitted before transform")
        return np.clip(X, self.lo_, self.hi_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
