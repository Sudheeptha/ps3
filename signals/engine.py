"""
Issue 8 – Trading Signal Generation Engine
Issue 14 – Explainable Strategy Logs

CORE IDEA
─────────────────────────────────────────────────────────────────────
Each asset gets a composite SIGNAL SCORE on [-1, +1]:

  score = Σ ( weight_i × normalised_metric_i )

The five sub-scores and their default weights:
  momentum_score   (0.30)  – price momentum across time windows
  volatility_score (0.20)  – inverse vol (low vol → bullish bias)
  trend_score      (0.25)  – EMA crossover + RSI regime
  macro_score      (0.15)  – macro indicator composite
  sentiment_score  (0.10)  – oil z-score as cross-asset proxy

Thresholds:
  score ≥  0.20  → BUY
  score ≤ -0.20  → SELL
  otherwise      → HOLD

All intermediate metric values are stored in the signal log for
full explainability (Issue 14).
"""

import numpy as np
import pandas as pd
from typing import Dict, List


BUY_THRESHOLD  =  0.15
SELL_THRESHOLD = -0.15


class SignalEngine:
    """
    Weighted score-based signal generator.
    weights: dict matching keys momentum_score, volatility_score,
             trend_score, macro_score, sentiment_score
    """

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self._validate_weights()

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(self, featured: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per (date, ticker) containing:
          - all sub-scores
          - composite signal_score
          - signal  {BUY, SELL, HOLD}
          - rationale  (human-readable explanation)
        """
        equity_df = featured.get("equity")
        if equity_df is None or equity_df.empty:
            equity_df = featured.get("multiasset")
        if equity_df is None or equity_df.empty:
            raise ValueError("No equity/multiasset data found for signal generation.")

        records: List[dict] = []

        for ticker, gdf in equity_df.groupby("ticker"):
            gdf = gdf.sort_values("date").copy()

            for _, row in gdf.iterrows():
                sub_scores = self._compute_sub_scores(row)
                composite  = self._weighted_sum(sub_scores)
                signal     = self._classify(composite)
                rationale  = self._explain(ticker, row, sub_scores, composite, signal)

                records.append({
                    "date":              row["date"],
                    "ticker":            ticker,
                    **sub_scores,
                    "signal_score":      round(composite, 4),
                    "signal":            signal,
                    "close":             row.get("close", np.nan),
                    "rationale":         rationale,
                })

        signals = pd.DataFrame(records)
        print(
            f"    → {len(signals)} signals generated | "
            f"BUY={( signals.signal=='BUY').sum()} | "
            f"SELL={(signals.signal=='SELL').sum()} | "
            f"HOLD={(signals.signal=='HOLD').sum()}"
        )
        return signals

    # ── Sub-score calculators ─────────────────────────────────────────────────

    def _compute_sub_scores(self, row: pd.Series) -> Dict[str, float]:
        return {
            "momentum_score":   self._momentum_score(row),
            "volatility_score": self._volatility_score(row),
            "trend_score":      self._trend_score(row),
            "macro_score":      self._macro_score(row),
            "sentiment_score":  self._sentiment_score(row),
        }

    @staticmethod
    def _momentum_score(row: pd.Series) -> float:
        """
        Average of normalised momentum across windows.
        Clipped to [-1, 1].  Positive → price rising.
        """
        vals = []
        for w in [5, 10, 21]:
            m = row.get(f"momentum_{w}d", np.nan)
            if pd.notna(m):
                # scale: ±10% move → ±1 score
                vals.append(np.clip(m / 0.10, -1, 1))
        return float(np.nanmean(vals)) if vals else 0.0

    @staticmethod
    def _volatility_score(row: pd.Series) -> float:
        """
        Inverse volatility: low volatility → positive score (prefer calm assets).
        Uses 21-day annualised vol.  Typical range: 0.10–0.60.
        """
        vol = row.get("volatility_21d", np.nan)
        if pd.isna(vol) or vol == 0:
            return 0.0
        # vol 0.10 → score +1;  vol 0.40 → score -1 (linear interpolation)
        score = 1.0 - (vol / 0.25)
        return float(np.clip(score, -1, 1))

    @staticmethod
    def _trend_score(row: pd.Series) -> float:
        """
        EMA crossover + RSI.
        ema_cross > 0  → uptrend; RSI < 30 → oversold (buy); RSI > 70 → overbought.
        """
        ema_cross = row.get("ema_cross", np.nan)
        rsi       = row.get("rsi", np.nan)

        trend = 0.0
        if pd.notna(ema_cross):
            trend += np.clip(ema_cross * 10, -0.5, 0.5)    # EMA component

        if pd.notna(rsi):
            # RSI: map [0,100] → [-1,+1], with oversold/overbought extremes
            rsi_score = (50 - rsi) / 50    # RSI 50 → 0; RSI 30 → +0.4; RSI 70 → -0.4
            trend += np.clip(rsi_score, -0.5, 0.5)

        return float(np.clip(trend, -1, 1))

    @staticmethod
    def _macro_score(row: pd.Series) -> float:
        """
        Composite of available macro indicator deviations.
        Positive GDP growth + low inflation + low unemployment → bullish.
        """
        components = []
        for col in row.index:
            if not col.startswith("macro_"):
                continue
            val = row[col]
            if pd.isna(val):
                continue
            indicator = col.replace("macro_", "")
            if "gdp" in indicator or "growth" in indicator:
                components.append(np.clip(val / 4.0, -1, 1))       # 4% GDP → +1
            elif "inflation" in indicator or "cpi" in indicator:
                components.append(np.clip(-val / 4.0, -1, 1))      # high inflation → negative
            elif "unemployment" in indicator:
                components.append(np.clip(-(val - 5.0) / 5.0, -1, 1))
            elif "rate" in indicator or "fed" in indicator:
                components.append(np.clip(-val / 5.0, -1, 1))      # high rates → bearish
            else:
                components.append(np.clip(val / 2.0, -1, 1))

        return float(np.nanmean(components)) if components else 0.0

    @staticmethod
    def _sentiment_score(row: pd.Series) -> float:
        """
        Uses oil z-score as a cross-asset sentiment proxy.
        High oil price → negative for most equities except energy.
        """
        oil_z = row.get("oil_zscore", np.nan)
        if pd.isna(oil_z):
            return 0.0
        # invert: rising oil z-score is modestly bearish for broad market
        return float(np.clip(-oil_z / 2.0, -1, 1))

    # ── Composite calculation ──────────────────────────────────────────────────

    def _weighted_sum(self, sub_scores: Dict[str, float]) -> float:
        total = 0.0
        for key, w in self.weights.items():
            total += w * sub_scores.get(key, 0.0)
        return float(np.clip(total, -1, 1))

    # ── Signal classification ──────────────────────────────────────────────────

    @staticmethod
    def _classify(score: float) -> str:
        if score >= BUY_THRESHOLD:
            return "BUY"
        elif score <= SELL_THRESHOLD:
            return "SELL"
        return "HOLD"

    # ── Explainability ─────────────────────────────────────────────────────────

    @staticmethod
    def _explain(
        ticker: str,
        row: pd.Series,
        sub_scores: Dict[str, float],
        composite: float,
        signal: str,
    ) -> str:
        """
        Human-readable rationale for the signal (Issue 14).
        Captures metric values and which sub-scores drove the decision.
        """
        dominant = max(sub_scores, key=lambda k: abs(sub_scores[k]))
        dominant_val = sub_scores[dominant]
        direction = "positive" if dominant_val > 0 else "negative"

        rsi  = row.get("rsi", "n/a")
        mom  = row.get("momentum_21d", "n/a")
        vol  = row.get("volatility_21d", "n/a")
        ema  = row.get("ema_cross", "n/a")

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        return (
            f"[{ticker}] Signal={signal} | Score={composite:.4f} | "
            f"Dominant driver: {dominant} ({direction}, {fmt(dominant_val)}) | "
            f"RSI={fmt(rsi)} | Momentum21d={fmt(mom)} | "
            f"Vol21d={fmt(vol)} | EMACross={fmt(ema)} | "
            f"Scores: " + " ".join(f"{k}={v:.3f}" for k, v in sub_scores.items())
        )

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate_weights(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Signal weights must sum to 1.0 (got {total:.4f}). "
                f"Weights: {self.weights}"
            )
