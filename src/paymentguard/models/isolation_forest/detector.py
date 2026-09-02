from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from paymentguard.core.base_detector import BaseAnomalyDetector
from paymentguard.features.pipeline import FeaturePipeline

DEFAULT_N_ESTIMATORS = 300
DEFAULT_CONTAMINATION: float | str = "auto"
DEFAULT_MAX_SAMPLES = 50000
DEFAULT_MAX_FEATURES = 0.8
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_JOBS = -1


class IsolationForestDetector(BaseAnomalyDetector):
    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        contamination: float | str = DEFAULT_CONTAMINATION,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        max_features: float = DEFAULT_MAX_FEATURES,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_jobs: int = DEFAULT_N_JOBS,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="IsolationForestDetector", config=config or {})
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.feature_pipeline = FeaturePipeline(use_user_agg=True, use_sequential=True)
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None) -> "IsolationForestDetector":
        if isinstance(X, pd.DataFrame):
            _, X_matrix = self.feature_pipeline.fit_transform(X)
            self.feature_names_ = self.feature_pipeline.numeric_cols_
        else:
            X_matrix = X
            self.feature_names_ = [f"feat_{i}" for i in range(X_matrix.shape[1])]

        self.model.fit(X_matrix)
        self.is_fitted = True
        return self

    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict")

        if isinstance(X, pd.DataFrame):
            _, X_matrix = self.feature_pipeline.transform(X)
        else:
            X_matrix = X

        scores = -self.model.score_samples(X_matrix)
        return scores

    def predict_label(self, X: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict")

        if isinstance(X, pd.DataFrame):
            _, X_matrix = self.feature_pipeline.transform(X)
        else:
            X_matrix = X

        preds = (self.model.predict(X_matrix) == -1).astype(int)
        return preds

    def get_shap_explainer(self) -> Any:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        import shap
        return shap.TreeExplainer(self.model)
