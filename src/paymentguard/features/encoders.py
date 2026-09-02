from typing import Optional
import pandas as pd


class CategoricalEncoder:
    def __init__(self, max_ohe_cardinality: int = 15) -> None:
        self.max_ohe_cardinality = max_ohe_cardinality
        self.ohe_cols: list[str] = []
        self.freq_cols: list[str] = []
        self.freq_maps_: dict[str, dict[str, float]] = {}
        self.ohe_columns_: list[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, cat_cols: Optional[list[str]] = None) -> "CategoricalEncoder":
        if cat_cols is None:
            cat_cols = [
                "currency",
                "payment_method",
                "order_type",
                "order_payment_type",
                "ip_country",
                "bin_country",
                "psp_id",
                "error_code",
            ]

        self.ohe_cols = []
        self.freq_cols = []
        self.freq_maps_ = {}

        for col in cat_cols:
            if col not in df.columns:
                continue
            n_unique = df[col].nunique()
            if n_unique <= self.max_ohe_cardinality:
                self.ohe_cols.append(col)
            else:
                self.freq_cols.append(col)
                freq = df[col].value_counts(normalize=True).to_dict()
                self.freq_maps_[col] = freq

        if self.ohe_cols:
            dummies = pd.get_dummies(df[self.ohe_cols].astype(str), prefix=self.ohe_cols, drop_first=False)
            self.ohe_columns_ = list(dummies.columns)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before transform")

        df = df.copy()

        for col in self.freq_cols:
            if col in df.columns:
                df[f"{col}_freq"] = df[col].map(self.freq_maps_[col]).fillna(0.0).astype(float)

        if self.ohe_cols:
            available_ohe = [c for c in self.ohe_cols if c in df.columns]
            dummies = pd.get_dummies(df[available_ohe].astype(str), prefix=available_ohe, drop_first=False)
            dummies = dummies.reindex(columns=self.ohe_columns_, fill_value=0)
            df = pd.concat([df, dummies], axis=1)

        return df

    def fit_transform(self, df: pd.DataFrame, cat_cols: Optional[list[str]] = None) -> pd.DataFrame:
        return self.fit(df, cat_cols).transform(df)
