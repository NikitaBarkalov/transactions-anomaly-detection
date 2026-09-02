import pandas as pd

REQUIRED_COLUMNS = [
    "order_id",
    "user_id",
    "bank_id",
    "psp_id",
    "created_at",
    "processed_at",
    "amount",
    "currency",
    "has_refund",
    "refunded_amount",
    "payment_method",
    "order_type",
    "ip_country",
    "bin_country",
    "is_secured",
    "status",
]


def validate_and_clean_data(df: pd.DataFrame, drop_invalid_dates: bool = True) -> pd.DataFrame:
    df = df.copy()

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["processed_at"] = pd.to_datetime(df["processed_at"], utc=True, errors="coerce")

    if drop_invalid_dates:
        df = df.dropna(subset=["created_at", "processed_at"]).reset_index(drop=True)

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df.loc[df["amount"] < 0, "amount"] = 0.0

    df["refunded_amount"] = pd.to_numeric(df["refunded_amount"], errors="coerce").fillna(0.0)
    df.loc[df["refunded_amount"] < 0, "refunded_amount"] = 0.0

    df["has_refund"] = df["has_refund"].astype(bool)
    df["is_secured"] = df["is_secured"].astype(bool)

    return df
