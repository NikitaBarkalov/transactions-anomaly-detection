from typing import Any

import numpy as np
import pandas as pd

from paymentguard.core.base_detector import BaseAnomalyDetector
from paymentguard.features.financial import compute_financial_features
from paymentguard.features.geo_security import compute_geo_security_features
from paymentguard.features.temporal import compute_temporal_features

DEFAULT_AMOUNT_IQR_MULTIPLIER = 3.0
DEFAULT_LATENCY_OUTLIER_THRESHOLD = 3000.0
DEFAULT_GAMMA_LATENCY_THRESHOLD = 1000.0
DEFAULT_COMPROMISED_BANK_ID = 777
DEFAULT_PSP_BETA_AUGUST_STORM = True
DEFAULT_PSP_ALPHA_RECURRING_70 = True


class RuleBasedAuditor(BaseAnomalyDetector):
    def __init__(
        self,
        amount_iqr_multiplier: float = DEFAULT_AMOUNT_IQR_MULTIPLIER,
        latency_outlier_threshold: float = DEFAULT_LATENCY_OUTLIER_THRESHOLD,
        gamma_latency_threshold: float = DEFAULT_GAMMA_LATENCY_THRESHOLD,
        compromised_bank_id: int = DEFAULT_COMPROMISED_BANK_ID,
        psp_beta_august_storm: bool = DEFAULT_PSP_BETA_AUGUST_STORM,
        psp_alpha_recurring_70: bool = DEFAULT_PSP_ALPHA_RECURRING_70,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name="RuleBasedAuditor", config=config or {})
        self.amount_iqr_multiplier = amount_iqr_multiplier
        self.latency_outlier_threshold = latency_outlier_threshold
        self.gamma_latency_threshold = gamma_latency_threshold
        self.compromised_bank_id = compromised_bank_id
        self.psp_beta_august_storm = psp_beta_august_storm
        self.psp_alpha_recurring_70 = psp_alpha_recurring_70
        self.layer_anomalies_: dict[str, set[Any]] = {}

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None
    ) -> "RuleBasedAuditor":
        self.is_fitted = True
        return self

    def evaluate_layers(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        df = df.copy()
        if "latency_sec" not in df.columns:
            df = compute_temporal_features(df)
        if "amount_usd" not in df.columns:
            df = compute_financial_features(df)
        if "is_geo_mismatch" not in df.columns:
            df = compute_geo_security_features(df)

        flags: dict[str, pd.Series] = {}

        flags["layer0_neg_latency"] = df["latency_sec"] < 0
        flags["layer1_zero_or_neg_amt"] = df["amount"] <= 0
        flags["layer1_amount_outlier"] = df["is_amount_outlier_3sigma"].astype(bool)
        flags["layer2_extreme_latency"] = df["latency_sec"] > self.latency_outlier_threshold
        flags["layer2_gamma_latency"] = (df["psp_id"] == "psp_gamma") & (
            df["latency_sec"] > self.gamma_latency_threshold
        )

        if "is_velocity_burst" in df.columns:
            flags["layer3_velocity_burst"] = df["is_velocity_burst"].astype(bool)

        flags["layer4_mismatch_first"] = df["mismatch_first_order"].astype(bool)
        flags["layer5_compromised_bank_777"] = df["bank_id"] == self.compromised_bank_id
        flags["layer6_success_with_error"] = (
            (df["status"] == "success")
            & df["error_code"].notna()
            & (df["error_code"] != "no_error")
        )
        flags["layer7_over_refund"] = df["refund_exceeds_amount"].astype(bool)
        flags["layer7_refund_on_fail"] = (df["has_refund"]) & (df["status"] == "fail")

        if self.psp_beta_august_storm:
            date_series = pd.to_datetime(df["created_at"]).dt.date
            aug_start = pd.to_datetime("2025-08-05").date()
            aug_end = pd.to_datetime("2025-08-09").date()
            flags["layer8_august_psp_beta_refund_storm"] = (
                (date_series >= aug_start)
                & (date_series <= aug_end)
                & (df["psp_id"] == "psp_beta")
                & (df["has_refund"])
            )

        if self.psp_alpha_recurring_70:
            flags["layer8_psp_alpha_recurring_glitch"] = (
                (df["psp_id"] == "psp_alpha")
                & (df["order_type"] == "recurring")
                & (df["error_code"].astype(str).isin(["3.08", "3.8"]))
                & (df["amount_usd"] > 70.0)
            )

        return flags

    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RuleBasedAuditor requires a pandas DataFrame input")

        layer_flags = self.evaluate_layers(X)
        flag_matrix = np.column_stack([flag.values.astype(float) for flag in layer_flags.values()])
        scores = np.clip(flag_matrix.sum(axis=1) / 3.0, 0.0, 1.0)
        return scores

    def predict_label(
        self, X: pd.DataFrame | np.ndarray, threshold: float | None = None
    ) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RuleBasedAuditor requires a pandas DataFrame input")

        layer_flags = self.evaluate_layers(X)
        combined = np.zeros(len(X), dtype=bool)
        for flag in layer_flags.values():
            combined |= flag.values
        return combined.astype(int)
