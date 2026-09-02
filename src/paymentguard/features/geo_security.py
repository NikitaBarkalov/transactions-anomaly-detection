import numpy as np
import pandas as pd

from paymentguard.core.constants import COUNTRY_CURRENCIES


def compute_geo_security_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ip_str = df["ip_country"].astype(str).str.upper()
    bin_str = df["bin_country"].astype(str).str.upper()
    df["is_geo_mismatch"] = (ip_str != bin_str).astype(np.int8)

    expected_curr = df["bin_country"].map(COUNTRY_CURRENCIES)
    df["is_currency_mismatch"] = (df["currency"] != expected_curr).astype(np.int8)

    df["is_secured_int"] = df["is_secured"].astype(np.int8)

    df["is_risky_combo"] = ((df["is_geo_mismatch"] == 1) & (df["is_secured_int"] == 0)).astype(
        np.int8
    )

    is_first = (df["order_type"] == "first").astype(np.int8)
    df["is_first_order"] = is_first
    df["mismatch_first_order"] = ((df["is_geo_mismatch"] == 1) & (is_first == 1)).astype(np.int8)

    if "is_night" in df.columns:
        df["night_unsecured_mismatch"] = (
            (df["is_night"] == 1) & (df["is_geo_mismatch"] == 1) & (df["is_secured_int"] == 0)
        ).astype(np.int8)

    return df
