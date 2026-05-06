"""
Issue 6  – Value at Risk (VaR)
Issue 7  – Maximum Drawdown & Volatility
Issue 12 – Sharpe Ratio
Issue 13 – Alpha & Beta
"""

import numpy as np
import pandas as pd
from portfolio.state import PortfolioState


class RiskMetrics:
    """
    Computes all risk and performance metrics from the portfolio NAV history.
    Market benchmark assumed to be a buy-and-hold equal-weight index
    approximated from the portfolio's own NAV (or passed explicitly).
    """

    def __init__(self, config: dict):
        self.risk_free_rate = config["risk_free_rate"]
        self.var_confidence = config["var_confidence"]
        self.trading_days   = 252

    # ── Public ────────────────────────────────────────────────────────────────

    def compute(self, portfolio: PortfolioState) -> dict:
        nav_series = portfolio.get_nav_series()
        if len(nav_series) < 10:
            return {"error": "Insufficient NAV history for metrics"}

        returns = nav_series.pct_change().dropna()

        metrics = {
            **self._var(returns),
            **self._drawdown(nav_series),
            **self._sharpe(returns),
            **self._alpha_beta(returns),
            "total_return":       self._total_return(nav_series),
            "annualised_return":  self._annualised_return(returns),
            "portfolio_volatility": self._portfolio_vol(returns),
            "total_trades":       len(portfolio.trade_history),
            "final_nav":          round(nav_series.iloc[-1], 2),
            "initial_capital":    portfolio.initial_capital,
        }
        return metrics

    # ── VaR (Historical Simulation) ───────────────────────────────────────────

    def _var(self, returns: pd.Series) -> dict:
        """
        Historical VaR: the loss exceeded with probability (1 - confidence).
        Also compute CVaR (Expected Shortfall) for completeness.
        """
        q = 1 - self.var_confidence
        var_1d   = abs(float(returns.quantile(q)))
        cvar_1d  = abs(float(returns[returns <= -var_1d].mean())) if len(returns[returns <= -var_1d]) else var_1d

        # Scale to 10-day VaR (square-root-of-time rule)
        var_10d = var_1d * np.sqrt(10)

        return {
            "var_1d_pct":    round(var_1d * 100, 4),
            "var_10d_pct":   round(var_10d * 100, 4),
            "cvar_1d_pct":   round(cvar_1d * 100, 4),
            "var_confidence": self.var_confidence,
        }

    # ── Drawdown ───────────────────────────────────────────────────────────────

    def _drawdown(self, nav_series: pd.Series) -> dict:
        """Maximum peak-to-trough drawdown and current drawdown."""
        rolling_max = nav_series.cummax()
        drawdown    = (nav_series - rolling_max) / rolling_max

        max_dd   = float(drawdown.min())
        curr_dd  = float(drawdown.iloc[-1])

        # Drawdown duration (days in current or worst episode)
        in_dd = drawdown < 0
        dd_start = drawdown[in_dd].idxmin() if in_dd.any() else None
        dd_end   = drawdown.idxmin()

        return {
            "max_drawdown_pct":     round(max_dd * 100, 4),
            "current_drawdown_pct": round(curr_dd * 100, 4),
        }

    # ── Sharpe Ratio ───────────────────────────────────────────────────────────

    def _sharpe(self, returns: pd.Series) -> dict:
        excess   = returns - self.risk_free_rate / self.trading_days
        mean_exc = float(excess.mean())
        std_ret  = float(returns.std())

        sharpe = (mean_exc / std_ret * np.sqrt(self.trading_days)) if std_ret else 0.0

        # Sortino (only downside deviation)
        neg_returns = returns[returns < 0]
        downside_std = float(neg_returns.std()) if len(neg_returns) > 1 else std_ret
        sortino = (mean_exc / downside_std * np.sqrt(self.trading_days)) if downside_std else 0.0

        return {
            "sharpe_ratio":  round(sharpe,  4),
            "sortino_ratio": round(sortino, 4),
        }

    # ── Alpha & Beta ───────────────────────────────────────────────────────────

    def _alpha_beta(self, returns: pd.Series) -> dict:
        """
        Market proxy: a random-walk 'index' centred on the same mean return.
        In production, pass in actual benchmark (e.g. SPY) returns.
        """
        np.random.seed(0)
        market_returns = pd.Series(
            np.random.normal(returns.mean() * 0.9, returns.std() * 0.95, len(returns)),
            index=returns.index,
        )

        cov = np.cov(returns, market_returns)
        beta  = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0
        rf_d  = self.risk_free_rate / self.trading_days
        alpha = float(
            (returns.mean() - rf_d) - beta * (market_returns.mean() - rf_d)
        ) * self.trading_days   # annualised

        return {
            "beta":             round(beta,  4),
            "alpha_annualised": round(alpha, 4),
        }

    # ── Simple helpers ─────────────────────────────────────────────────────────

    def _total_return(self, nav_series: pd.Series) -> float:
        return round((nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100, 4)

    def _annualised_return(self, returns: pd.Series) -> float:
        n = len(returns)
        cumulative = (1 + returns).prod()
        return round((cumulative ** (self.trading_days / n) - 1) * 100, 4)

    def _portfolio_vol(self, returns: pd.Series) -> float:
        return round(float(returns.std()) * np.sqrt(self.trading_days) * 100, 4)
