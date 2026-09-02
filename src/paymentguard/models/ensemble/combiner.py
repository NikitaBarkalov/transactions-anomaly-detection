from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd

from paymentguard.core.base_detector import BaseAnomalyDetector


class MetaEnsembleDetector(BaseAnomalyDetector):
    def __init__(
        self,
        detectors: Sequence[BaseAnomalyDetector],
        mode: str = "soft_voting",
        weights: Optional[Sequence[float]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="MetaEnsembleDetector", config=config or {})
        self.detectors = list(detectors)
        self.mode = mode
        if weights is not None:
            self.weights = np.array(weights) / np.sum(weights)
        else:
            self.weights = np.ones(len(self.detectors)) / len(self.detectors)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None) -> "MetaEnsembleDetector":
        for det in self.detectors:
            if not det.is_fitted:
                det.fit(X, y)
        self.is_fitted = True
        return self

    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        all_scores = [det.predict_score(X) for det in self.detectors]
        fused_score = np.zeros(len(all_scores[0]))
        for score, w in zip(all_scores, self.weights):
            fused_score += score * w
        return fused_score

    def predict_label(self, X: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        if self.mode == "union":
            preds = [det.predict_label(X) for det in self.detectors]
            union_pred = np.zeros(len(preds[0]), dtype=bool)
            for p in preds:
                union_pred |= p.astype(bool)
            return union_pred.astype(int)

        elif self.mode == "hard_voting":
            preds = [det.predict_label(X) for det in self.detectors]
            pred_matrix = np.column_stack(preds)
            majority_threshold = len(self.detectors) / 2.0
            return (pred_matrix.sum(axis=1) >= majority_threshold).astype(int)

        else:
            scores = self.predict_score(X)
            thr = threshold if threshold is not None else self.config.get("anomaly_threshold", 0.50)
            return (scores >= thr).astype(int)
