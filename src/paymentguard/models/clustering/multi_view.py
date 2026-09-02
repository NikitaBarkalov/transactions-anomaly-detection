from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from paymentguard.core.base_detector import BaseAnomalyDetector
from paymentguard.core.constants import CONVERSION_COEFFS

DEFAULT_N_CLUSTERS_BEHAVIORAL = 25
DEFAULT_N_CLUSTERS_TECHNICAL = 100
DEFAULT_N_CLUSTERS_LATENCY = 15
DEFAULT_BEHAVIORAL_THRESHOLD = 0.40
DEFAULT_TECHNICAL_FAIL_THRESHOLD = 0.60
DEFAULT_LATENCY_THRESHOLD = 0.50
DEFAULT_BATCH_SIZE = 50000
DEFAULT_RANDOM_STATE = 42
DEFAULT_INCLUDE_RULE1 = True

BEHAVIORAL_FEATURES = [
    "log_amount", "amount_zscore", "amount_vs_user_avg",
    "log_latency", "is_latency_outlier",
    "is_geo_mismatch", "is_secured", "is_risky_combo",
    "is_night", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "refund_ratio", "is_full_refund", "has_error",
    "user_id_freq", "bank_id_freq", "psp_id_freq",
    "error_code_freq", "ip_country_freq", "bin_country_freq", "currency_freq",
    "user_tx_count", "user_fail_rate_10", "user_geo_rate_10",
    "log_seconds_since_last", "is_velocity_burst",
    "psp_id_fail_rate", "bank_id_fail_rate",
    "high_amt_geo_unsecured", "first_high_amount", "night_velocity",
]

TECHNICAL_FEATURES = [
    "log_amount", "amount_bucket", "is_amount_outlier",
    "is_secured",
    "is_applepay", "is_googlepay",
    "is_first_order",
    "is_rebill", "is_oneclick", "is_payment_missing",
    "psp_id_fail_rate", "bank_id_fail_rate",
    "first_high_amount",
]

LATENCY_FEATURES = [
    "latency",
    "log_latency",
    "is_latency_outlier",
    "log_seconds_since_last",
    "is_velocity_burst",
    "log_amount",
    "refund_ratio", "is_full_refund",
    "net_amount",
    "has_error", "error_code_freq",
    "is_geo_mismatch", "is_risky_combo",
]


class MultiViewClusteringDetector(BaseAnomalyDetector):
    def __init__(
        self,
        n_clusters_behavioral: int = DEFAULT_N_CLUSTERS_BEHAVIORAL,
        n_clusters_technical: int = DEFAULT_N_CLUSTERS_TECHNICAL,
        n_clusters_latency: int = DEFAULT_N_CLUSTERS_LATENCY,
        behavioral_threshold: float = DEFAULT_BEHAVIORAL_THRESHOLD,
        technical_fail_threshold: float = DEFAULT_TECHNICAL_FAIL_THRESHOLD,
        latency_threshold: float = DEFAULT_LATENCY_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
        include_rule1: bool = DEFAULT_INCLUDE_RULE1,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="MultiViewClusteringDetector", config=config or {})
        self.batch_size = batch_size
        self.random_state = random_state
        self.behavioral_threshold = behavioral_threshold
        self.technical_fail_threshold = technical_fail_threshold
        self.latency_threshold = latency_threshold
        self.include_rule1 = include_rule1

        self.km_behavioral = MiniBatchKMeans(n_clusters=n_clusters_behavioral, random_state=self.random_state, batch_size=self.batch_size)
        self.km_technical = MiniBatchKMeans(n_clusters=n_clusters_technical, random_state=self.random_state, batch_size=self.batch_size)
        self.km_latency = MiniBatchKMeans(n_clusters=n_clusters_latency, random_state=self.random_state, batch_size=self.batch_size)

        self.scaler_b = StandardScaler()
        self.scaler_t = StandardScaler()
        self.scaler_l = StandardScaler()

        self.anomaly_clusters_b_: set[int] = set()
        self.anomaly_clusters_t_: set[int] = set()
        self.anomaly_clusters_l_: set[int] = set()

        self.freq_maps_: dict[str, dict[Any, float]] = {}
        self.fail_rate_maps_: dict[str, dict[Any, float]] = {}
        self.lat_mean_: float = 0.0
        self.lat_std_: float = 0.0
        self.train_labels_: Optional[np.ndarray] = None

    def _engineer_features(self, df: pd.DataFrame, is_train: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
        df = df.copy()

        created_at = pd.to_datetime(df["created_at"], utc=True)
        processed_at = pd.to_datetime(df["processed_at"], utc=True)

        df["latency"] = (processed_at - created_at).dt.total_seconds()
        if is_train:
            self.lat_mean_ = float(df["latency"].mean())
            self.lat_std_ = float(df["latency"].std())

        df["is_latency_outlier"] = (df["latency"] > self.lat_mean_ + 3 * self.lat_std_).astype(int)
        df["log_latency"] = np.log1p(df["latency"].clip(lower=0))

        hours = created_at.dt.hour
        df["is_night"] = hours.between(0, 5).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        weekdays = created_at.dt.weekday
        df["dow_sin"] = np.sin(2 * np.pi * weekdays / 7)
        df["dow_cos"] = np.cos(2 * np.pi * weekdays / 7)

        df["is_geo_mismatch"] = (df["ip_country"] != df["bin_country"]).astype(int)
        df["is_secured"] = df["is_secured"].astype(int)
        df["is_risky_combo"] = ((df["is_geo_mismatch"] == 1) & (df["is_secured"] == 0)).astype(int)

        df["log_amount"] = np.log1p(df["amount"])
        df["net_amount"] = df["amount"] - df["refunded_amount"]
        df["refund_ratio"] = np.where(df["amount"] > 0, df["refunded_amount"] / df["amount"], 0.0)
        df["is_full_refund"] = (df["refund_ratio"] >= 1.0).astype(int)

        df["amount_usd"] = df["amount"] * df["currency"].map(CONVERSION_COEFFS).fillna(1.0)
        df["refunded_amount_usd"] = df["refunded_amount"] * df["currency"].map(CONVERSION_COEFFS).fillna(1.0)

        df["amount_bucket"] = pd.cut(
            df["amount_usd"],
            bins=[0, 5, 15, 50, 150, 500, np.inf],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(int)

        df["amount_zscore"] = df.groupby("currency")["amount_usd"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )
        df["is_amount_outlier"] = (df["amount_zscore"].abs() > 3).astype(int)

        df["is_fail"] = (df["status"] == "fail").astype(int)
        df["has_error"] = df["error_code"].notna().astype(int)

        df["is_applepay"] = (df["payment_method"] == "applepay").astype(int)
        df["is_googlepay"] = (df["payment_method"] == "googlepay").astype(int)
        df["is_first_order"] = (df["order_type"] == "first").astype(int)
        df["is_rebill"] = (df["order_payment_type"] == "rebill").astype(int)
        df["is_oneclick"] = (df["order_payment_type"] == "1-click").astype(int)
        df["is_payment_missing"] = df["order_payment_type"].isna().astype(int)

        freq_cols = ["user_id", "bank_id", "psp_id", "error_code", "ip_country", "bin_country", "currency"]
        for col in freq_cols:
            if is_train:
                freq = df[col].value_counts(normalize=True).to_dict()
                self.freq_maps_[col] = freq
            df[f"{col}_freq"] = df[col].map(self.freq_maps_[col]).fillna(0.0)

        orig_idx = np.arange(len(df))
        df["_orig_idx"] = orig_idx
        df["_created_at"] = created_at
        df = df.sort_values(["user_id", "_created_at"]).reset_index(drop=True)
        grp = df.groupby("user_id")

        df["user_tx_count"] = grp.cumcount()
        df["user_fail_rate_10"] = grp["is_fail"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0.0)
        df["user_geo_rate_10"] = grp["is_geo_mismatch"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0.0)

        time_diff = grp["_created_at"].diff().dt.total_seconds().fillna(0.0)
        df["log_seconds_since_last"] = np.log1p(time_diff.clip(lower=0.0))
        df["is_velocity_burst"] = (time_diff.between(0.1, 60.0)).astype(int)

        user_avg_10 = grp["amount_usd"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(df["amount_usd"].mean())
        df["amount_vs_user_avg"] = df["amount_usd"] / (user_avg_10 + 1e-9)

        for col in ["psp_id", "bank_id"]:
            if is_train:
                self.fail_rate_maps_[col] = df.groupby(col)["is_fail"].mean().to_dict()
            df[f"{col}_fail_rate"] = df[col].map(self.fail_rate_maps_[col]).fillna(0.0)

        amt_q90 = float(df["amount"].quantile(0.90))
        amt_q95 = float(df["amount"].quantile(0.95))

        df["high_amt_geo_unsecured"] = ((df["amount"] > amt_q90) & (df["is_geo_mismatch"] == 1) & (df["is_secured"] == 0)).astype(int)
        df["first_high_amount"] = ((df["is_first_order"] == 1) & (df["amount"] > amt_q95)).astype(int)
        df["night_velocity"] = ((df["is_night"] == 1) & (df["is_velocity_burst"] == 1)).astype(int)

        sorted_indices = df["_orig_idx"].values
        df = df.drop(columns=["_orig_idx", "_created_at"], errors="ignore")
        return df, sorted_indices

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None) -> "MultiViewClusteringDetector":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MultiViewClusteringDetector requires DataFrame input")

        enriched_df, sorted_indices = self._engineer_features(X, is_train=True)

        X1 = enriched_df[BEHAVIORAL_FEATURES].fillna(0.0)
        X1_scaled = self.scaler_b.fit_transform(X1)
        clusters_b = self.km_behavioral.fit_predict(X1_scaled)
        enriched_df["cluster_b"] = clusters_b

        profile_b = enriched_df.groupby("cluster_b").agg(
            fail_rate=("is_fail", "mean"),
            avg_amount=("amount", "mean"),
            avg_latency=("latency", "mean"),
            geo_mismatch=("is_geo_mismatch", "mean"),
            risky_combo=("is_risky_combo", "mean"),
            night_rate=("is_night", "mean"),
            refund_rate=("refund_ratio", "mean"),
            full_refund_rate=("is_full_refund", "mean"),
            secured_rate=("is_secured", "mean"),
            error_rate=("has_error", "mean"),
            velocity_rate=("is_velocity_burst", "mean"),
            amt_vs_avg=("amount_vs_user_avg", "mean"),
            psp_fail=("psp_id_fail_rate", "mean"),
        )

        def norm(s: pd.Series) -> pd.Series:
            return (s - s.min()) / (s.max() - s.min() + 1e-9)

        score_b = (
            norm(profile_b["refund_rate"]) * 0.20 +
            norm(profile_b["full_refund_rate"]) * 0.20 +
            norm(profile_b["risky_combo"]) * 0.15 +
            norm(profile_b["error_rate"]) * 0.10 +
            norm(profile_b["geo_mismatch"]) * 0.10 +
            norm(profile_b["velocity_rate"]) * 0.10 +
            norm(profile_b["avg_latency"]) * 0.05 +
            norm(profile_b["avg_amount"]) * 0.05 +
            norm(profile_b["night_rate"]) * 0.05
        )
        self.anomaly_clusters_b_ = set(score_b[score_b > self.behavioral_threshold].index)
        flag_b = enriched_df["cluster_b"].isin(self.anomaly_clusters_b_).values

        X2 = enriched_df[TECHNICAL_FEATURES].fillna(0.0)
        X2_scaled = self.scaler_t.fit_transform(X2)
        clusters_t = self.km_technical.fit_predict(X2_scaled)
        enriched_df["cluster_t"] = clusters_t
        profile_t = enriched_df.groupby("cluster_t")["is_fail"].mean()
        self.anomaly_clusters_t_ = set(profile_t[profile_t > self.technical_fail_threshold].index)
        flag_t = enriched_df["cluster_t"].isin(self.anomaly_clusters_t_).values

        X3 = enriched_df[LATENCY_FEATURES].fillna(0.0)
        X3_scaled = self.scaler_l.fit_transform(X3)
        clusters_l = self.km_latency.fit_predict(X3_scaled)
        enriched_df["cluster_l"] = clusters_l
        profile_l = enriched_df.groupby("cluster_l").agg(
            fail_rate=("is_fail", "mean"),
            avg_latency=("latency", "mean"),
            refund_rate=("refund_ratio", "mean"),
            full_refund_rate=("is_full_refund", "mean"),
            error_rate=("has_error", "mean"),
            geo_mismatch=("is_geo_mismatch", "mean"),
            velocity_rate=("is_velocity_burst", "mean"),
        )
        score_l = (
            norm(profile_l["avg_latency"]) * 0.40 +
            norm(profile_l["fail_rate"]) * 0.15 +
            norm(profile_l["velocity_rate"]) * 0.15 +
            norm(profile_l["full_refund_rate"]) * 0.15 +
            norm(profile_l["error_rate"]) * 0.10 +
            norm(profile_l["geo_mismatch"]) * 0.05
        )
        self.anomaly_clusters_l_ = set(score_l[score_l > self.latency_threshold].index)
        flag_l = enriched_df["cluster_l"].isin(self.anomaly_clusters_l_).values

        rule_1 = np.zeros(len(enriched_df), dtype=bool)
        if self.include_rule1:
            rule_1 = (
                (enriched_df["psp_id"] == "psp_alpha") &
                (enriched_df["order_type"] == "recurring") &
                (enriched_df["error_code"].astype(str).isin(["3.08", "3.8"])) &
                (enriched_df["amount_usd"] > 70.0)
            ).values

        sorted_is_anomaly = (flag_b | flag_t | flag_l | rule_1).astype(int)
        
        train_labels = np.zeros_like(sorted_is_anomaly)
        train_labels[sorted_indices] = sorted_is_anomaly
        self.train_labels_ = train_labels

        self.is_fitted = True
        return self

    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        labels = self.predict_label(X)
        return labels.astype(float)

    def fit_predict(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        if self.train_labels_ is not None:
            return self.train_labels_
        return self.predict_label(X)

    def predict_label(self, X: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MultiViewClusteringDetector requires DataFrame input")

        enriched_df, sorted_indices = self._engineer_features(X, is_train=False)

        c_b = self.km_behavioral.predict(self.scaler_b.transform(enriched_df[BEHAVIORAL_FEATURES].fillna(0.0)))
        c_t = self.km_technical.predict(self.scaler_t.transform(enriched_df[TECHNICAL_FEATURES].fillna(0.0)))
        c_l = self.km_latency.predict(self.scaler_l.transform(enriched_df[LATENCY_FEATURES].fillna(0.0)))

        flag_b = pd.Series(c_b).isin(self.anomaly_clusters_b_).values
        flag_t = pd.Series(c_t).isin(self.anomaly_clusters_t_).values
        flag_l = pd.Series(c_l).isin(self.anomaly_clusters_l_).values

        rule_1 = np.zeros(len(enriched_df), dtype=bool)
        if self.include_rule1:
            rule_1 = (
                (enriched_df["psp_id"] == "psp_alpha") &
                (enriched_df["order_type"] == "recurring") &
                (enriched_df["error_code"].astype(str).isin(["3.08", "3.8"])) &
                (enriched_df["amount_usd"] > 70.0)
            ).values

        sorted_is_anomaly = (flag_b | flag_t | flag_l | rule_1).astype(int)
        out_labels = np.zeros_like(sorted_is_anomaly)
        out_labels[sorted_indices] = sorted_is_anomaly
        return out_labels
