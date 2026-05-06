# System Architecture – Hedge Fund Risk & Semi-Automated Trading System

## Overview

The system is a modular Python pipeline that ingests multi-asset market data,
preprocesses it, engineers features, generates weighted-score trading signals,
simulates execution with realistic market friction, and computes risk-adjusted
performance metrics.

```
Raw CSVs
  ↓
[1] ingestion/loader.py         – concurrent CSV loading, schema validation
  ↓
[2] preprocessing/cleaner.py    – KNN imputation, IQR outlier smoothing
  ↓
[3] features/engineer.py        – momentum, volatility, EMA, RSI, macro/oil merge
  ↓
[4] signals/engine.py           – WEIGHTED SCORE ENGINE  ← core logic
  ↓
[5] execution/simulator.py      – position sizing, tx costs, slippage, rebalancing
  ↓
[6] risk/metrics.py             – VaR, drawdown, Sharpe, Sortino, Alpha, Beta
  ↓
[7] reporting/dashboard.py      – JSON report for dashboard / API consumption
```

---

## Module Details

### [1] Data Ingestion  (`ingestion/loader.py`)
- Loads equity, macro, multiasset, oil CSVs concurrently (ThreadPoolExecutor)
- Column alias normalisation handles varied real-world column names
- Schema validation raises on missing required columns (Issue 16)
- Falls back to synthetic random-walk data when CSVs are absent (useful for dev/testing)
- Chunked reading (50 k rows/chunk) prevents memory bottlenecks on large files

### [2] Preprocessing  (`preprocessing/cleaner.py`)
- **KNN Imputation** (sklearn `KNNImputer`, k=5): imputes missing numeric values
  *per ticker/indicator group* so only same-asset neighbours are used — no cross-asset
  leakage or forward-looking bias
- **IQR Outlier Smoothing**: prices outside Q1 ± 3×IQR are winsorised; each smoothed
  cell gets an `*_outlier_flag` column for audit trail
- Type coercion with `pd.to_numeric(errors="coerce")` safely converts bad strings to NaN

### [3] Feature Engineering  (`features/engineer.py`)
Per-ticker rolling features:
| Feature              | Description                                     |
|----------------------|-------------------------------------------------|
| `momentum_{5,10,21}d`| Price return over N trading days                |
| `volatility_{10,21}d`| Rolling log-return std × √252 (annualised)      |
| `ema_cross`          | (EMA12 − EMA26) / EMA26 — normalised crossover  |
| `rsi`                | 14-day Relative Strength Index                  |
| `volume_zscore`      | Volume deviation from 21-day mean               |
| `macro_*`            | Macro indicators forward-filled to daily freq   |
| `oil_zscore`         | WTI oil price z-score (cross-asset signal)      |

### [4] Signal Engine  (`signals/engine.py`)  ← CORE
The engine computes a **composite score ∈ [−1, +1]** per asset per day:

```
signal_score = Σ ( weight_i × sub_score_i )

Sub-scores (all normalised to [−1, +1]):
  momentum_score   (default 30%)  — average of momentum_{5,10,21}d
  volatility_score (default 20%)  — inverse vol; low vol → bullish
  trend_score      (default 25%)  — EMA crossover + RSI regime
  macro_score      (default 15%)  — GDP, inflation, unemployment, rate composite
  sentiment_score  (default 10%)  — oil z-score proxy (inverted)

Thresholds:
  score ≥ +0.20 → BUY
  score ≤ −0.20 → SELL
  otherwise     → HOLD
```

Every signal record stores **all sub-scores + raw metric values** for full
explainability (Issue 14). Weights are configurable in `main.py`.

### [5] Execution Simulator  (`execution/simulator.py`)
- **Position sizing**: inverse-volatility scaled, capped at `max_position_pct` (default 20 %)
- **Transaction costs**: `tx_cost_bps` (default 10 bps) deducted per trade
- **Slippage**: `slippage_bps` (default 5 bps) applied to execution price
- **Rebalancing**: monthly trim of positions exceeding cap by > 5 pp
- **Insufficient capital** (Issue 15): caught, logged, simulation continues

### [6] Risk Metrics  (`risk/metrics.py`)
| Metric               | Method                                          |
|----------------------|-------------------------------------------------|
| VaR 1-day / 10-day   | Historical simulation, configurable confidence  |
| CVaR (Expected Shortfall) | Mean of losses beyond VaR threshold        |
| Max Drawdown         | Peak-to-trough from NAV series                  |
| Portfolio Volatility | Annualised std of daily returns                 |
| Sharpe Ratio         | Annualised excess return / vol                  |
| Sortino Ratio        | Annualised excess return / downside std         |
| Beta                 | Cov(portfolio, market) / Var(market)            |
| Alpha                | Annualised Jensen's alpha                       |

### [7] Dashboard Reporter  (`reporting/dashboard.py`)
Outputs `output/report.json` containing:
- `summary`: all risk/return metrics
- `nav_timeseries`: daily NAV, cash, position value
- `trade_log`: every executed trade with rationale
- `signal_summary`: BUY/SELL/HOLD counts and average scores
- `top_signals`: highest-conviction signals (by |score|)

---

## Configuration  (`main.py → CONFIG`)

```python
CONFIG = {
    "initial_capital":     10_000_000,
    "risk_free_rate":      0.05,
    "max_position_pct":    0.20,
    "var_confidence":      0.95,
    "rebalance_frequency": "monthly",
    "transaction_cost_bps": 10,
    "slippage_bps":         5,
    "signal_weights": {
        "momentum_score":   0.30,
        "volatility_score": 0.20,
        "trend_score":      0.25,
        "macro_score":      0.15,
        "sentiment_score":  0.10,
    },
    "data_paths": {
        "equity":     "data/equity.csv",
        "macro":      "data/macro.csv",
        "multiasset": "data/multiasset.csv",
        "oil":        "data/oil.csv",
    },
}
```

## Plugging In Real Data

Drop your CSVs into the `data/` folder. Column names are auto-normalised
via `COLUMN_ALIASES` in `ingestion/loader.py`. If your dataset uses different
column names, add them to that dict. The expected schemas are:

| Dataset    | Required columns                                            |
|------------|-------------------------------------------------------------|
| equity     | date, ticker, open, high, low, close, volume                |
| macro      | date, indicator, value                                      |
| multiasset | date, asset_class, ticker, close, volume                    |
| oil        | date, contract, open, high, low, close, volume              |

## Running

```bash
pip install -r requirements.txt
python main.py
```
