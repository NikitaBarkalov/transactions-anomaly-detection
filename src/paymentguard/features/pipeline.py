from typing import Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np

from paymentguard.features.temporal import compute_temporal_features
from paymentguard.features.financial import compute_financial_features
from paymentguard.features.geo_security import compute_geo_security_features
from paymentguard.features.aggregations import UserAggregator, compute_sequential_user_features
from paymentguard.features.encoders import CategoricalEncoder


class FeaturePipeline:
    def __init__(self, use_user_agg: bool = True, use_sequential: bool = True) -> None:
        self.use_user_agg = use_user_agg
        self.use_sequential = use_sequential
        self.user_aggregator = UserAggregator() if use_user_agg else None
        self.cat_encoder = CategoricalEncoder()
        self.numeric_scaler = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
        self.numeric_cols_: list[str] = []
        self.is_fitted = False

    def transform_dataframes(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        df = compute_temporal_features(df)
        df = compute_financial_features(df)
        df = compute_geo_security_features(df)

        if self.use_sequential:
            df = compute_sequential_user_features(df)

        if self.use_user_agg and self.user_aggregator is not None:
            if is_train:
                df = self.user_aggregator.fit_transform(df)
            else:
                df = self.user_aggregator.transform(df)

        if is_train:
            df = self.cat_encoder.fit_transform(df)
        else:
            df = self.cat_encoder.transform(df)

        return df

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        enriched_df = self.transform_dataframes(df, is_train=True)

        ignore_cols = {
            "order_id", "user_id", "bank_id", "psp_id", "created_at", "processed_at",
            "currency", "payment_method", "order_type", "order_payment_type",
            "ip_country", "bin_country", "error_code", "status", "has_refund", "is_secured"
        }
        self.numeric_cols_ = [
            c for c in enriched_df.columns
            if c not in ignore_cols and np.issubdtype(enriched_df[c].dtype, np.number)
        ]

        scaled_matrix = self.numeric_scaler.fit_transform(enriched_df[self.numeric_cols_]).astype(np.float32)
        self.is_fitted = True
        return enriched_df, scaled_matrix

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        enriched_df = self.transform_dataframes(df, is_train=False)
        for col in self.numeric_cols_:
            if col not in enriched_df.columns:
                enriched_df[col] = 0.0

        scaled_matrix = self.numeric_scaler.transform(enriched_df[self.numeric_cols_]).astype(np.float32)
        return enriched_df, scaled_matrix
