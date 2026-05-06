"""
Issue 3 & 4 – Feature Engineering + Macro / Sentiment Integration
Produces per-asset rolling features used downstream by the signal engine.

Features calculated
─────────────────────────────────────────────────────────────────────
momentum_*      : price return over N days
volatility_*    : rolling std of log-returns over N days
trend_*         : EMA fast vs slow cross
rsi             : relative strength index
volume_zscore   : normalised volume
macro_*         : latest macro indicator values aligned to market dates
oil_z           : oil price z-score (cross-asset signal)
"""

import numpy as np
import pandas as pd
from typing import Dict


MOMENTUM_WINDOWS   = [5, 10, 21]    # trading days
VOLATILITY_WINDOWS = [10, 21]
EMA_SHORT = 12
EMA_LONG  = 26
RSI_PERIOD = 14


class FeatureEngineer:

    def build_features(self, clean: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Returns a dict where the primary key 'featured' holds a
        multi-ticker DataFrame with all derived features joined.
        Also returns the raw named dicts for reference.
        """
        featured = {}

        # Main price features from equity + multiasset
        for name in ("equity", "multiasset"):
            if name in clean:
                featured[name] = self._price_features(clean[name], name)

        # Oil z-score (cross-asset signal)
        if "oil" in clean:
            featured["oil"] = self._oil_features(clean["oil"])

        # Macro signal alignment
        if "macro" in clean:
            featured["macro"] = self._macro_pivot(clean["macro"])

        # Merge macro + oil signals onto equity features
        for name in ("equity", "multiasset"):
            if name in featured:
                featured[name] = self._merge_macro_oil(
                    featured[name],
                    featured.get("macro"),
                    featured.get("oil"),
                )

        return featured

    # ── Price features ─────────────────────────────────────────────────────────

    def _price_features(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        group_col = "ticker"
        if group_col not in df.columns:
            return df

        parts = []
        for ticker, gdf in df.groupby(group_col):
            gdf = gdf.sort_values("date").copy()
            gdf = self._add_returns(gdf)
            gdf = self._add_momentum(gdf)
            gdf = self._add_volatility(gdf)
            gdf = self._add_trend(gdf)
            gdf = self._add_rsi(gdf)
            gdf = self._add_volume_zscore(gdf)
            parts.append(gdf)

        return pd.concat(parts).reset_index(drop=True)

    # individual feature builders ──────────────────────────────────────────────

    @staticmethod
    def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
        safe_close = df["close"].clip(lower=1e-6)
        df["log_return"] = np.log(safe_close / safe_close.shift(1))
        df["daily_return"] = df["close"].pct_change()
        return df

    @staticmethod
    def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
        for w in MOMENTUM_WINDOWS:
            df[f"momentum_{w}d"] = df["close"].pct_change(w)
        return df

    @staticmethod
    def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
        safe_close = df["close"].clip(lower=1e-6)
        log_ret = np.log(safe_close / safe_close.shift(1))
        for w in VOLATILITY_WINDOWS:
            df[f"volatility_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)
        return df

    @staticmethod
    def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
        ema_s = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
        ema_l = df["close"].ewm(span=EMA_LONG,  adjust=False).mean()
        df["ema_cross"] = (ema_s - ema_l) / ema_l   # normalised crossover
        df["ema_short"] = ema_s
        df["ema_long"]  = ema_l
        return df

    @staticmethod
    def _add_rsi(df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def _add_volume_zscore(df: pd.DataFrame) -> pd.DataFrame:
        if "volume" in df.columns:
            mu  = df["volume"].rolling(21).mean()
            std = df["volume"].rolling(21).std()
            df["volume_zscore"] = (df["volume"] - mu) / std.replace(0, np.nan)
        return df

    # ── Oil features ───────────────────────────────────────────────────────────

    @staticmethod
    def _oil_features(df: pd.DataFrame) -> pd.DataFrame:
        oil = df[df["contract"] == "WTI"].copy() if "contract" in df.columns else df.copy()
        oil = oil.sort_values("date")
        mu  = oil["close"].rolling(21).mean()
        std = oil["close"].rolling(21).std()
        oil["oil_zscore"]  = (oil["close"] - mu) / std.replace(0, np.nan)
        oil["oil_return"]  = oil["close"].pct_change()
        return oil[["date", "oil_zscore", "oil_return"]].dropna()

    # ── Macro pivot (wide format, forward-filled to daily) ────────────────────

    @staticmethod
    def _macro_pivot(df: pd.DataFrame) -> pd.DataFrame:
        pivot = df.pivot_table(index="date", columns="indicator", values="value", aggfunc="last")
        pivot.columns = [f"macro_{c}" for c in pivot.columns]
        pivot = pivot.reset_index()
        # Forward-fill monthly data to daily frequency
        date_range = pd.date_range(pivot["date"].min(), pivot["date"].max(), freq="B")
        pivot = pivot.set_index("date").reindex(date_range).ffill().reset_index()
        pivot.rename(columns={"index": "date"}, inplace=True)
        return pivot

    # ── Merge external signals ─────────────────────────────────────────────────

    @staticmethod
    def _merge_macro_oil(
        df: pd.DataFrame,
        macro_df: pd.DataFrame | None,
        oil_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if macro_df is not None:
            df = pd.merge_asof(
                df.sort_values("date"),
                macro_df.sort_values("date"),
                on="date",
                direction="backward",
            )
        if oil_df is not None:
            df = pd.merge_asof(
                df.sort_values("date"),
                oil_df.sort_values("date").rename(
                    columns={"oil_zscore": "oil_zscore", "oil_return": "oil_return"}
                ),
                on="date",
                direction="backward",
            )
        return df
