from dataclasses import dataclass
import pandas as pd


class InsufficientCapitalError(Exception):
    pass


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float


class PortfolioState:

    def __init__(self, config):
        self.initial_capital = config["initial_capital"]
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.nav_history = []

    def open_position(self, ticker, shares, price, date, commission):
        cost = shares * price + commission

        if cost > self.cash:
            raise InsufficientCapitalError(
                f"Not enough cash to buy {ticker}"
            )

        self.cash -= cost

        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos.shares + shares
            avg_cost = (
                (pos.shares * pos.avg_cost)
                + (shares * price)
            ) / total_shares

            pos.shares = total_shares
            pos.avg_cost = avg_cost

        else:
            self.positions[ticker] = Position(
                ticker,
                shares,
                price
            )

        self.trade_history.append({
            "date": date,
            "ticker": ticker,
            "action": "BUY"
        })

    def close_position(self, ticker, shares, price, date, commission):
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]

        shares = min(shares, pos.shares)

        proceeds = shares * price - commission

        self.cash += proceeds

        pos.shares -= shares

        if pos.shares <= 0:
            del self.positions[ticker]

        self.trade_history.append({
            "date": date,
            "ticker": ticker,
            "action": "SELL"
        })

    def nav(self, prices=None):
        total = self.cash

        if prices:
            for ticker, pos in self.positions.items():
                total += (
                    pos.shares *
                    prices.get(ticker, pos.avg_cost)
                )

        return total

    def record_nav(self, date, prices):
        self.nav_history.append({
            "date": date,
            "nav": self.nav(prices),
            "cash": self.cash
        })

    def get_nav_series(self):
        df = pd.DataFrame(self.nav_history)

        return pd.Series(
            df["nav"].values,
            index=pd.to_datetime(df["date"])
        )

    def position_pct(self, ticker, price, nav):
        if ticker not in self.positions:
            return 0

        value = self.positions[ticker].shares * price

        return value / nav