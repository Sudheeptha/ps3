import streamlit as st
import pandas as pd
import json
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hedge Fund Dashboard",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# LIGHT MODE STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #111111;
    }

    .metric-card {
        background-color: #f7f9fc;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e6eaf1;
        margin-bottom: 15px;
    }

    .section-header {
        font-size: 28px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #111111;
    }

    .sub-header {
        font-size: 20px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 10px;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD REPORT
# ============================================================

report_path = Path("output/report.json")
ai_path = Path("output/ai_analysis.txt")

if not report_path.exists():
    st.error("output/report.json not found. Run main.py first.")
    st.stop()

with open(report_path, "r") as f:
    report = json.load(f)

summary = report.get("summary", {})
trade_log = report.get("trade_log", [])
nav_series = report.get("nav_timeseries", [])
signals = report.get("top_signals", [])


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="section-header">📈 Hedge Fund Risk Modeling Dashboard</div>',
    unsafe_allow_html=True
)

st.write(
    "Quantitative trading, portfolio simulation, risk analytics, "
    "and AI-powered financial interpretation dashboard."
)


# ============================================================
# METRICS
# ============================================================

st.markdown(
    '<div class="section-header">📊 Portfolio Metrics</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Return (%)",
        round(summary.get("total_return", 0), 2)
    )

    st.metric(
        "Sharpe Ratio",
        round(summary.get("sharpe_ratio", 0), 3)
    )

with col2:
    st.metric(
        "Max Drawdown (%)",
        round(summary.get("max_drawdown_pct", 0), 2)
    )

    st.metric(
        "Portfolio Volatility",
        round(summary.get("portfolio_volatility", 0), 3)
    )

with col3:
    st.metric(
        "VaR 1D (%)",
        round(summary.get("var_1d_pct", 0), 3)
    )

    st.metric(
        "CVaR 1D (%)",
        round(summary.get("cvar_1d_pct", 0), 3)
    )

with col4:
    st.metric(
        "Final NAV",
        f"{summary.get('final_nav', 0):,.0f}"
    )

    st.metric(
        "Total Trades",
        int(summary.get("total_trades", 0))
    )


# ============================================================
# NAV CHART
# ============================================================

st.markdown(
    '<div class="section-header">📈 NAV Curve</div>',
    unsafe_allow_html=True
)

if nav_series:

    nav_df = pd.DataFrame(nav_series)

    if "date" in nav_df.columns:
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_df = nav_df.set_index("date")

    if "nav" in nav_df.columns:
        st.line_chart(nav_df["nav"])

else:
    st.info("No NAV history available.")


# ============================================================
# SIGNALS
# ============================================================

st.markdown(
    '<div class="section-header">🚦 Trading Signals</div>',
    unsafe_allow_html=True
)

if signals:

    signals_df = pd.DataFrame(signals)

    st.dataframe(
        signals_df,
        use_container_width=True
    )

else:
    st.info("No signals available.")


# ============================================================
# TRADE LOG
# ============================================================

st.markdown(
    '<div class="section-header">💹 Trade Log</div>',
    unsafe_allow_html=True
)

if trade_log:

    trade_df = pd.DataFrame(trade_log)

    st.dataframe(
        trade_df.head(100),
        use_container_width=True
    )

else:
    st.info("No trades executed.")


# ============================================================
# RISK ANALYSIS TABLE
# ============================================================

st.markdown(
    '<div class="section-header">⚠️ Risk Analysis</div>',
    unsafe_allow_html=True
)

risk_metrics = {
    "Metric": [
        "VaR 1D",
        "VaR 10D",
        "CVaR 1D",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Beta",
        "Alpha",
        "Max Drawdown",
        "Current Drawdown"
    ],

    "Value": [
        summary.get("var_1d_pct", 0),
        summary.get("var_10d_pct", 0),
        summary.get("cvar_1d_pct", 0),
        summary.get("sharpe_ratio", 0),
        summary.get("sortino_ratio", 0),
        summary.get("beta", 0),
        summary.get("alpha_annualised", 0),
        summary.get("max_drawdown_pct", 0),
        summary.get("current_drawdown_pct", 0)
    ]
}

risk_df = pd.DataFrame(risk_metrics)

st.dataframe(
    risk_df,
    use_container_width=True
)


# ============================================================
# AI ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-header">🤖 AI Financial Analysis</div>',
    unsafe_allow_html=True
)

if ai_path.exists():

    with open(ai_path, "r", encoding="utf-8") as f:
        ai_text = f.read()

    st.text_area(
        "Gemini AI Portfolio Commentary",
        ai_text,
        height=350
    )

else:
    st.warning(
        "AI analysis not found. Run main.py first."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Built using Streamlit, quantitative risk modeling, "
    "portfolio simulation, and AI-powered financial interpretation."
)
