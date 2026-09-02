import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd


class BaseAnomalyDetector(ABC):
    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
        self.is_fitted = False

    @abstractmethod
    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None
    ) -> "BaseAnomalyDetector":
        pass

    @abstractmethod
    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        pass

    def predict_label(
        self, X: pd.DataFrame | np.ndarray, threshold: float | None = None
    ) -> np.ndarray:
        scores = self.predict_score(X)
        if threshold is None:
            threshold = self.config.get("anomaly_threshold", 0.5)
        return (scores >= threshold).astype(int)

    def fit_predict(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(X, y).predict_label(X)

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Model file not found at: {in_path.resolve()}")
        with open(in_path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded model is of type {type(model)}, expected {cls}")
        return model

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_fitted": self.is_fitted,
            "config": self.config,
        }
