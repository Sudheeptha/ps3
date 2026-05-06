"""
Issue 18 – Dashboard & Metrics Visualization Data Builder
Aggregates NAV history, trade logs, and metrics into a JSON-ready report.
"""

import pandas as pd
from typing import Dict, Any


class DashboardReporter:

    def build(
        self,
        portfolio,
        metrics: dict,
        trade_log: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> Dict[str, Any]:

        nav_df = pd.DataFrame(portfolio.nav_history)

        return {
            "summary": {
                "initial_capital":    portfolio.initial_capital,
                "final_nav":          metrics.get("final_nav"),
                "total_return_pct":   metrics.get("total_return"),
                "annualised_return":  metrics.get("annualised_return"),
                "sharpe_ratio":       metrics.get("sharpe_ratio"),
                "sortino_ratio":      metrics.get("sortino_ratio"),
                "max_drawdown_pct":   metrics.get("max_drawdown_pct"),
                "var_1d_pct":         metrics.get("var_1d_pct"),
                "var_10d_pct":        metrics.get("var_10d_pct"),
                "cvar_1d_pct":        metrics.get("cvar_1d_pct"),
                "beta":               metrics.get("beta"),
                "alpha_annualised":   metrics.get("alpha_annualised"),
                "portfolio_volatility": metrics.get("portfolio_volatility"),
                "total_trades":       metrics.get("total_trades"),
            },
            "nav_timeseries": nav_df.to_dict(orient="records") if not nav_df.empty else [],
            "trade_log": trade_log.to_dict(orient="records") if not trade_log.empty else [],
            "signal_summary": self._signal_summary(signals),
            "top_signals": self._top_signals(signals),
        }

    @staticmethod
    def _signal_summary(signals: pd.DataFrame) -> dict:
        if signals.empty:
            return {}
        counts = signals["signal"].value_counts().to_dict()
        avg_scores = signals.groupby("signal")["signal_score"].mean().round(4).to_dict()
        return {"counts": counts, "avg_score_by_signal": avg_scores}

    @staticmethod
    def _top_signals(signals: pd.DataFrame, n: int = 20) -> list:
        if signals.empty:
            return []
        top = (
            signals[signals["signal"] != "HOLD"]
            .sort_values("signal_score", key=abs, ascending=False)
            .head(n)
        )
        return top[["date", "ticker", "signal", "signal_score", "rationale"]].to_dict(orient="records")
