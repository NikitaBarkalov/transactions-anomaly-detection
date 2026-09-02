import numpy as np
import pandas as pd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    latency_sec = (df["processed_at"] - df["created_at"]).dt.total_seconds()
    df["latency_sec"] = latency_sec
    df["log_latency"] = np.sign(latency_sec) * np.log1p(np.abs(latency_sec))

    hour = df["created_at"].dt.hour
    dow = df["created_at"].dt.dayofweek

    df["hour_of_day"] = hour.astype(np.int8)
    df["day_of_week"] = dow.astype(np.int8)
    df["is_weekend"] = (dow >= 5).astype(np.int8)
    df["month"] = df["created_at"].dt.month.astype(np.int8)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    df["is_night"] = hour.between(0, 5).astype(np.int8)

    return df
