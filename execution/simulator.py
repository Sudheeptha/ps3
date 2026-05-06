"""
Issue 9  – Risk-Aware Position Sizing
Issue 10 – Transaction Costs & Slippage
Issue 11 – Periodic Portfolio Rebalancing
Issue 15 – Insufficient Capital Error Handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from portfolio.state import PortfolioState, InsufficientCapitalError


class TradeSimulator:
    """
    Iterates over the signal DataFrame day-by-day and executes trades.
    Position sizing is volatility-scaled and capped by max_position_pct.
    """

    def __init__(self, config: dict):
        self.tx_cost_bps  = config["transaction_cost_bps"]   # e.g. 10
        self.slippage_bps = config["slippage_bps"]           # e.g. 5
        self.max_pos_pct  = config["max_position_pct"]
        self.rebalance_freq = config.get("rebalance_frequency", "monthly")
        self._trade_log: List[dict] = []
        self._error_log: List[dict] = []

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        signals: pd.DataFrame,
        featured: Dict[str, pd.DataFrame],
        portfolio: PortfolioState,
    ) -> pd.DataFrame:

        price_pivot = self._build_price_pivot(featured)
        dates = sorted(signals["date"].unique())
        last_rebalance = None

        for date in dates:
            day_signals   = signals[signals["date"] == date]
            current_prices = self._prices_on(price_pivot, date)

            # Record NAV before any trades
            portfolio.record_nav(date, current_prices)

            # Rebalance check
            if self._should_rebalance(date, last_rebalance):
                self._rebalance(portfolio, current_prices, date)
                last_rebalance = date

            # Execute signals
            for _, sig in day_signals.iterrows():
                ticker = sig["ticker"]
                signal = sig["signal"]
                price  = current_prices.get(ticker, sig.get("close", np.nan))
                vol    = sig.get("volatility_21d", 0.20)

                if pd.isna(price):
                    continue

                if signal == "BUY":
                    self._execute_buy(portfolio, ticker, price, vol, date, sig["rationale"])
                elif signal == "SELL":
                    self._execute_sell(portfolio, ticker, price, date, sig["rationale"])

        return pd.DataFrame(self._trade_log)

    # ── Execution helpers ─────────────────────────────────────────────────────

    def _execute_buy(self, portfolio, ticker, price, vol, date, rationale):
        nav = portfolio.nav()
        if nav <= 0:
            return

        # Issue 9: volatility-scaled position sizing
        target_value = self._position_size(nav, vol)

        # Respect max position cap
        current_exposure = 0.0
        if ticker in portfolio.positions:
            current_exposure = portfolio.positions[ticker].shares * price
        additional = min(target_value - current_exposure, portfolio.cash * 0.95)

        if additional <= 0:
            return

        exec_price = self._apply_slippage(price, "BUY")
        commission = additional * (self.tx_cost_bps / 10_000)
        shares     = additional / exec_price

        try:
            portfolio.open_position(ticker, shares, exec_price, date, commission)
            self._trade_log.append({
                "date": date, "ticker": ticker, "action": "BUY",
                "shares": round(shares, 4), "exec_price": round(exec_price, 4),
                "value": round(additional, 2), "commission": round(commission, 2),
                "slippage_bps": self.slippage_bps,
                "rationale": rationale,
            })
        except InsufficientCapitalError as e:
            # Issue 15: log and continue
            self._error_log.append({"date": date, "ticker": ticker, "error": str(e)})

    def _execute_sell(self, portfolio, ticker, price, date, rationale):
        if ticker not in portfolio.positions:
            return

        pos = portfolio.positions[ticker]
        exec_price = self._apply_slippage(price, "SELL")
        proceeds   = pos.shares * exec_price
        commission = proceeds * (self.tx_cost_bps / 10_000)

        portfolio.close_position(ticker, pos.shares, exec_price, date, commission)
        self._trade_log.append({
            "date": date, "ticker": ticker, "action": "SELL",
            "shares": round(pos.shares, 4), "exec_price": round(exec_price, 4),
            "value": round(proceeds, 2), "commission": round(commission, 2),
            "slippage_bps": self.slippage_bps,
            "rationale": rationale,
        })

    # ── Rebalancing ───────────────────────────────────────────────────────────

    def _should_rebalance(self, date, last_rebalance) -> bool:
        if last_rebalance is None:
            return False
        if self.rebalance_freq == "monthly":
            return date.month != last_rebalance.month
        elif self.rebalance_freq == "quarterly":
            return (date.month - 1) // 3 != (last_rebalance.month - 1) // 3
        return False

    def _rebalance(self, portfolio, prices, date):
        """Trim any position exceeding max_position_pct by 5pp."""
        nav = portfolio.nav(prices)
        for ticker in list(portfolio.positions.keys()):
            price = prices.get(ticker, portfolio.positions[ticker].avg_cost)
            pct   = portfolio.position_pct(ticker, price, nav)
            if pct > self.max_pos_pct + 0.05:
                excess_value = (pct - self.max_pos_pct) * nav
                trim_shares  = excess_value / price
                exec_price   = self._apply_slippage(price, "SELL")
                commission   = excess_value * (self.tx_cost_bps / 10_000)
                portfolio.close_position(ticker, trim_shares, exec_price, date, commission)
                self._trade_log.append({
                    "date": date, "ticker": ticker, "action": "REBALANCE_TRIM",
                    "shares": round(trim_shares, 4), "exec_price": round(exec_price, 4),
                    "value": round(excess_value, 2), "commission": round(commission, 2),
                    "slippage_bps": self.slippage_bps,
                    "rationale": f"Rebalance: trimmed from {pct:.2%} to {self.max_pos_pct:.2%}",
                })

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _position_size(self, nav: float, vol: float) -> float:
        """
        Kelly-inspired inverse-vol sizing, capped at max_position_pct.
        Lower volatility → larger position.
        """
        base_pct = min(self.max_pos_pct, 0.02 / max(vol, 0.01))  # risk budget = 2% daily
        return nav * base_pct

    def _apply_slippage(self, price: float, direction: str) -> float:
        factor = self.slippage_bps / 10_000
        return price * (1 + factor) if direction == "BUY" else price * (1 - factor)

    @staticmethod
    def _build_price_pivot(featured: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for name in ("equity", "multiasset"):
            df = featured.get(name)
            if df is not None and "ticker" in df.columns and "close" in df.columns:
                frames.append(df[["date", "ticker", "close"]])
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames).drop_duplicates(["date", "ticker"])
        return combined.pivot(index="date", columns="ticker", values="close")

    @staticmethod
    def _prices_on(pivot: pd.DataFrame, date) -> Dict[str, float]:
        if pivot.empty or date not in pivot.index:
            return {}
        row = pivot.loc[date]
        return row.dropna().to_dict()
