import numpy as np
import pandas as pd


class UserAggregator:
    def __init__(self) -> None:
        self.user_stats_: pd.DataFrame | None = None
        self.fallback_values_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "UserAggregator":
        df = df.copy()
        df["_log_amount"] = np.log1p(df["amount"].clip(lower=0.0))
        df["_is_fail"] = (df["status"] == "fail").astype(float)
        df["_is_refund"] = df["has_refund"].astype(float)
        df["_is_mismatch"] = (
            df["ip_country"].astype(str).str.upper() != df["bin_country"].astype(str).str.upper()
        ).astype(float)
        df["_is_secured"] = df["is_secured"].astype(float)

        g = df.groupby("user_id")

        aggs = g.agg(
            user_tx_count=("amount", "count"),
            user_amount_sum=("_log_amount", "sum"),
            user_amount_mean=("_log_amount", "mean"),
            user_amount_std=("_log_amount", "std"),
            user_fail_rate=("_is_fail", "mean"),
            user_refund_rate=("_is_refund", "mean"),
            user_mismatch_rate=("_is_mismatch", "mean"),
            user_secured_rate=("_is_secured", "mean"),
        )
        aggs["user_tx_count"] = np.log1p(aggs["user_tx_count"])
        aggs["user_amount_std"] = aggs["user_amount_std"].fillna(0.0)

        aggs["user_n_currencies"] = (
            df.drop_duplicates(subset=["user_id", "currency"]).groupby("user_id").size()
        )
        aggs["user_n_pay_methods"] = (
            df.drop_duplicates(subset=["user_id", "payment_method"]).groupby("user_id").size()
        )
        aggs["user_n_banks"] = (
            df.drop_duplicates(subset=["user_id", "bank_id"]).groupby("user_id").size()
        )
        aggs["user_n_psps"] = (
            df.drop_duplicates(subset=["user_id", "psp_id"]).groupby("user_id").size()
        )

        self.user_stats_ = aggs.reset_index()
        self.feature_cols = [c for c in self.user_stats_.columns if c != "user_id"]
        self.fallback_values_ = {
            col: float(self.user_stats_[col].median()) for col in self.feature_cols
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.user_stats_ is None:
            raise RuntimeError("UserAggregator must be fitted before calling transform()")

        merged = df.merge(self.user_stats_, on="user_id", how="left")
        for col in self.feature_cols:
            merged[col] = merged[col].fillna(self.fallback_values_[col])
        return merged

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def _fast_grouped_rolling_mean(
    user_ids: np.ndarray, values: np.ndarray, window: int = 10, default_fill: float = 0.0
) -> np.ndarray:
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    user_diff = np.empty(n, dtype=bool)
    user_diff[0] = True
    user_diff[1:] = user_ids[1:] != user_ids[:-1]
    group_starts = np.flatnonzero(user_diff)
    group_ends = np.append(group_starts[1:], n)

    for s, e in zip(group_starts, group_ends, strict=False):
        group_len = e - s
        if group_len == 1:
            out[s] = default_fill
        else:
            g_vals = values[s:e]
            out[s] = default_fill
            cumsum = np.cumsum(np.insert(g_vals[:-1], 0, 0.0))
            for i in range(1, group_len):
                w_start = max(0, i - window)
                out[s + i] = (cumsum[i] - cumsum[w_start]) / (i - w_start)
    return out


def compute_sequential_user_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user_id", "created_at"]).reset_index(drop=True).copy()

    user_ids = df["user_id"].values
    created_at = df["created_at"].values

    time_diff = np.zeros(len(df), dtype=np.float64)
    user_diff = np.empty(len(df), dtype=bool)
    user_diff[0] = True
    user_diff[1:] = user_ids[1:] != user_ids[:-1]

    dt_sec = (created_at[1:] - created_at[:-1]).astype("timedelta64[s]").astype(np.float64)
    time_diff[1:] = np.where(user_diff[1:], 0.0, np.maximum(0.0, dt_sec))

    df["seconds_since_prev_tx"] = time_diff
    df["log_seconds_since_prev_tx"] = np.log1p(time_diff)
    df["is_velocity_burst"] = ((time_diff >= 0.1) & (time_diff <= 60.0)).astype(np.int8)

    is_fail = (df["status"] == "fail").values.astype(np.float64)
    df["user_rolling_fail_rate_10"] = _fast_grouped_rolling_mean(
        user_ids, is_fail, window=10, default_fill=0.0
    )

    amount_usd = (
        df["amount_usd"].values
        if "amount_usd" in df.columns
        else df["amount"].values.astype(np.float64)
    )
    global_mean_amt = float(np.mean(amount_usd))
    user_hist_mean = _fast_grouped_rolling_mean(
        user_ids, amount_usd, window=10, default_fill=global_mean_amt
    )
    df["amount_vs_user_rolling_avg"] = amount_usd / (user_hist_mean + 1e-9)

    return df
