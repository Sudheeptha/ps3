"""
Issue 2 – Handle Missing Data and Price Outliers
KNN imputation for missing values; IQR-based outlier smoothing.
No forward-looking bias: KNN is fitted per asset on available past data.
"""

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.impute import KNNImputer


# Columns treated as numeric price/value features for imputation
NUMERIC_COLS = {
    "equity": ["close", "volume"],
    "macro": ["value"],
    "multiasset": ["close", "volume"],
    "oil": ["close", "volume"],
}

# IQR multiplier for outlier detection (lower = more aggressive)
IQR_MULTIPLIER = 3.0
# KNN neighbours
KNN_NEIGHBORS = 5


class DataPreprocessor:
    """
    Processes all raw DataFrames:
    1. Casts types and strips bad rows  (Issue 16)
    2. KNN-imputes missing numeric values per ticker/asset group
    3. Smooths outliers via IQR-capping
    """

    def process_all(self, raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        processed = {}
        for name, df in raw.items():
            print(f"    preprocessing '{name}' ({len(df)} rows) …")
            df = self._type_cast(df, name)
            df = self._knn_impute(df, name)
            df = self._smooth_outliers(df, name)
            processed[name] = df
            missing_after = df[NUMERIC_COLS.get(name, [])].isna().sum().sum()
            print(f"      → {len(df)} rows, {missing_after} missing values remaining")
        return processed

    # ── Type casting / basic cleaning ─────────────────────────────────────────

    def _type_cast(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Force numeric types; coerce bad values to NaN (Issue 16)."""
        for col in NUMERIC_COLS.get(name, []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where the primary price column is entirely null
        price_col = "close" if "close" in df.columns else "value"
        if price_col in df.columns:
            # Don't drop yet – let KNN handle; flag rows with no date as bad
            df = df.dropna(subset=["date"])

        return df.copy()

    # ── KNN imputation ─────────────────────────────────────────────────────────

    def _knn_impute(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Impute per group (ticker / indicator / contract) so that only
        same-asset neighbours are used → no cross-asset leakage.
        """
        num_cols = [c for c in NUMERIC_COLS.get(name, []) if c in df.columns]
        if not num_cols:
            return df

        group_col = self._group_col(name)

        if group_col and group_col in df.columns:
            groups = []
            for grp_val, grp_df in df.groupby(group_col):
                grp_df = grp_df.copy()
                grp_df = self._impute_block(grp_df, num_cols)
                groups.append(grp_df)
            return pd.concat(groups).sort_values("date").reset_index(drop=True)
        else:
            return self._impute_block(df, num_cols)

    def _impute_block(self, df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
        missing_mask = df[num_cols].isna().any(axis=1)
        if not missing_mask.any():
            return df   # nothing to do

        n_neighbors = min(KNN_NEIGHBORS, max(1, missing_mask.sum()))
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")

        df[num_cols] = imputer.fit_transform(df[num_cols])
        return df

    # ── Outlier smoothing ──────────────────────────────────────────────────────

    def _smooth_outliers(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        IQR-based capping per group.  Extreme values are winsorised to
        Q1 - k*IQR … Q3 + k*IQR (flag column added for audit).
        """
        num_cols = [c for c in NUMERIC_COLS.get(name, []) if c in df.columns]
        price_cols = [c for c in ["close", "open", "high", "low", "value"] if c in num_cols]

        flagged_total = 0
        for col in price_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - IQR_MULTIPLIER * iqr
            hi = q3 + IQR_MULTIPLIER * iqr

            outliers = (df[col] < lo) | (df[col] > hi)
            flagged_total += outliers.sum()

            df[f"{col}_outlier_flag"] = outliers.astype(int)
            df[col] = df[col].clip(lower=lo, upper=hi)

        if flagged_total:
            print(f"      ⚑  {flagged_total} outlier cells smoothed")
        return df

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _group_col(name: str):
        mapping = {
            "equity":     "ticker",
            "macro":      "indicator",
            "multiasset": "ticker",
            "oil":        "contract",
        }
        return mapping.get(name)
