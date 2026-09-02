import numpy as np
import pandas as pd

from paymentguard.core.constants import CONVERSION_COEFFS


def compute_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_amount"] = np.log1p(df["amount"])

    df["amount_usd"] = df["amount"] * df["currency"].map(CONVERSION_COEFFS).fillna(1.0)
    df["refunded_amount_usd"] = df["refunded_amount"] * df["currency"].map(
        CONVERSION_COEFFS
    ).fillna(1.0)

    df["net_amount_usd"] = df["amount_usd"] - df["refunded_amount_usd"]

    df["refund_ratio"] = np.where(
        df["amount_usd"] > 0,
        df["refunded_amount_usd"] / (df["amount_usd"] + 1e-9),
        0.0,
    )
    df["is_full_refund"] = (df["refund_ratio"] >= 1.0).astype(np.int8)
    df["refund_exceeds_amount"] = (df["refunded_amount_usd"] > (df["amount_usd"] + 0.01)).astype(
        np.int8
    )

    df["is_success"] = (df["status"] == "success").astype(np.int8)
    df["is_fail"] = (df["status"] == "fail").astype(np.int8)

    df["amount_zscore_curr"] = (
        df.groupby("currency")["amount_usd"]
        .transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
        .fillna(0.0)
    )
    df["is_amount_outlier_3sigma"] = (df["amount_zscore_curr"].abs() > 3.0).astype(np.int8)

    return df
