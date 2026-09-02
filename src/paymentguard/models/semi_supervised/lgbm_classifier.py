from typing import Any, Optional
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from paymentguard.core.base_detector import BaseAnomalyDetector
from paymentguard.features.pipeline import FeaturePipeline
from paymentguard.utils.logger import get_logger

logger = get_logger("SemiSupervisedLGBM")

DEFAULT_N_SPLITS = 3
DEFAULT_N_ESTIMATORS = 5000
DEFAULT_EARLY_STOPPING_ROUNDS = 60
DEFAULT_LEARNING_RATE = 0.08
DEFAULT_NUM_LEAVES = 31
DEFAULT_MAX_DEPTH = 8
DEFAULT_SUBSAMPLE = 0.8
DEFAULT_COLSAMPLE_BYTREE = 0.8
DEFAULT_ANOMALY_THRESHOLD = 0.98
DEFAULT_RANDOM_STATE = 42
DEFAULT_LOG_PERIOD = 200


class SemiSupervisedLGBMDetector(BaseAnomalyDetector):
    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        num_leaves: int = DEFAULT_NUM_LEAVES,
        max_depth: int = DEFAULT_MAX_DEPTH,
        subsample: float = DEFAULT_SUBSAMPLE,
        colsample_bytree: float = DEFAULT_COLSAMPLE_BYTREE,
        anomaly_threshold: float = DEFAULT_ANOMALY_THRESHOLD,
        log_period: int = DEFAULT_LOG_PERIOD,
        random_state: int = DEFAULT_RANDOM_STATE,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="SemiSupervisedLGBMDetector", config=config or {})
        self.n_splits = n_splits
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.anomaly_threshold = anomaly_threshold
        self.log_period = log_period
        self.random_state = random_state

        self.feature_pipeline = FeaturePipeline(use_user_agg=True, use_sequential=True)
        self.models_: list[lgb.LGBMClassifier] = []
        self.best_iterations_: list[int] = []
        self.calibrated_threshold_: float = anomaly_threshold
        self._cached_scores: Optional[np.ndarray] = None

    def fit_pseudo(
        self,
        full_df: pd.DataFrame,
        pseudo_indices: np.ndarray,
        pseudo_y: np.ndarray,
    ) -> "SemiSupervisedLGBMDetector":
        _, full_X_matrix = self.feature_pipeline.fit_transform(full_df)

        X_train = full_X_matrix[pseudo_indices]
        y_train = np.asarray(pseudo_y, dtype=int)

        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos_weight = float(np.sqrt(n_neg / max(1, n_pos)))

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.models_ = []
        self.best_iterations_ = []
        logger.info(f"Training LightGBM ensemble on {len(X_train):,} samples across {self.n_splits} folds (LR={self.learning_rate}, scale_pos_weight={scale_pos_weight:.2f}, patience={self.early_stopping_rounds})...")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_va, y_va = X_train[val_idx], y_train[val_idx]

            clf = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                scale_pos_weight=scale_pos_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="binary_logloss",
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=True),
                        lgb.log_evaluation(period=self.log_period),
                    ],
                )
            self.models_.append(clf)
            self.best_iterations_.append(clf.best_iteration_)
            best_loss = clf.best_score_.get("valid_0", {}).get("binary_logloss", 0.0)
            logger.info(f"  Fold [{fold_idx}/{self.n_splits}] | Optimal tree {clf.best_iteration_:04d} | Val LogLoss: {best_loss:.5f}")

        self.is_fitted = True

        all_probs = np.mean([clf.predict_proba(full_X_matrix)[:, 1] for clf in self.models_], axis=0)
        self._cached_scores = all_probs
        self.calibrated_threshold_ = self.anomaly_threshold
        logger.info(f"LightGBM training complete. Best iterations: {self.best_iterations_}. Probability threshold: {self.calibrated_threshold_:.2f}")
        return self

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None) -> "SemiSupervisedLGBMDetector":
        if y is None:
            raise ValueError("SemiSupervisedLGBMDetector requires pseudo ground truth labels `y` for fitting")

        if isinstance(X, pd.DataFrame):
            _, X_matrix = self.feature_pipeline.fit_transform(X)
        else:
            X_matrix = X

        y_arr = np.asarray(y, dtype=int)
        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        scale_pos_weight = float(np.sqrt(n_neg / max(1, n_pos)))

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.models_ = []
        self.best_iterations_ = []

        for train_idx, val_idx in skf.split(X_matrix, y_arr):
            X_tr, y_tr = X_matrix[train_idx], y_arr[train_idx]
            X_va, y_va = X_matrix[val_idx], y_arr[val_idx]

            clf = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                scale_pos_weight=scale_pos_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="binary_logloss",
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=True),
                        lgb.log_evaluation(period=self.log_period),
                    ],
                )
            self.models_.append(clf)
            self.best_iterations_.append(clf.best_iteration_)

        self.is_fitted = True
        return self

    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted or not self.models_:
            raise RuntimeError("Model must be fitted before predict")

        if self._cached_scores is not None and isinstance(X, pd.DataFrame) and len(X) == len(self._cached_scores):
            scores = self._cached_scores
            self._cached_scores = None
            return scores

        if isinstance(X, pd.DataFrame):
            _, X_matrix = self.feature_pipeline.transform(X)
        else:
            X_matrix = X

        probs = np.mean([clf.predict_proba(X_matrix)[:, 1] for clf in self.models_], axis=0)
        return probs

    def predict_label(self, X: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        scores = self.predict_score(X)
        thr = threshold if threshold is not None else self.calibrated_threshold_
        return (scores >= thr).astype(int)
