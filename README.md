# Hedge Fund Risk Modeling & Semi-Automated Trading System

## Team Information

* **Team Name**: [Syntax Squad]
* **Year**: [2nd]
* **All-Female Team**: [No]

## Architecture Overview

#### Describe your approach here. Keep it short and clear.

```
- The system is designed as a modular hedge-fund-style quantitative trading and portfolio risk management platform capable of processing heterogeneous financial datasets from multiple domains. The ingestion layer loads equity market data, oil and commodity datasets, macroeconomic indicators, and multi-asset financial datasets using a flexible pandas-based ETL pipeline. Since the datasets contain varying schemas and structures, the ingestion engine dynamically standardizes column names, validates incoming schemas, reshapes wide-format datasets into analysis-ready structures, aligns timestamps, and performs dataset integrity verification before downstream analytics.

- The preprocessing layer focuses on improving robustness and reliability of financial time-series analysis. Missing values are handled using configurable median-based and KNN-based imputation techniques depending on the statistical characteristics of the dataset. Statistical outlier detection is performed using IQR filtering, rolling standard deviation analysis, and volatility-aware smoothing mechanisms to identify abnormal spikes and reduce noise in financial market behavior while preserving temporal continuity in the datasets.

- The feature engineering pipeline transforms raw financial data into quantitative predictive signals commonly used in institutional quantitative trading systems and machine-learning-assisted finance pipelines. Engineered features include rolling momentum indicators, rolling log returns, annualized volatility, moving averages, EMA crossover behavior, RSI-inspired trend indicators, oil-market z-score signals, cross-asset movement correlations, and macroeconomic sentiment features. These engineered features function similarly to classical machine learning input vectors used in Random Forests, XGBoost models, statistical alpha engines, and quantitative signal forecasting systems, making the architecture fully ML-ready while maintaining explainability.

- The core architecture flow of the system follows a structured institutional-style pipeline:
  
  Raw Financial Data
  → Ingestion & Schema Standardization
  → Data Cleaning & Imputation
  → Outlier Detection & Smoothing
  → Quantitative Feature Engineering
  → Weighted Signal Generation
  → Portfolio Constraint Validation
  → Trade Execution Simulation
  → Risk Modeling & Portfolio Analytics
  → Dashboard Visualization & AI Interpretation

- The signal generation engine uses a weighted multi-factor quantitative strategy to generate explainable BUY, SELL, and HOLD signals. Multiple engineered indicators including momentum scores, volatility behavior, macroeconomic sentiment, oil-market dynamics, and trend-following features are combined into a unified weighted score framework. Trading decisions are executed only when aggregate confidence exceeds configurable thresholds, reducing noisy signals and improving explainability. This semi-automated architecture enables modular integration of future supervised ML forecasting models without changing the overall system design.

- The portfolio execution simulator incorporates realistic institutional trading constraints to emulate real-world hedge fund operations. The execution layer models transaction costs, slippage effects, insufficient-capital handling, maximum portfolio exposure constraints, volatility-scaled position sizing, dynamic capital allocation, and periodic rebalancing logic. These constraints ensure that generated strategies remain realistic and risk-aware rather than purely theoretical.

- The risk modeling engine integrates multiple institutional-grade portfolio risk analytics directly into the trading pipeline. Key metrics computed by the platform include:
  
  • Historical Value at Risk (VaR)
  • Conditional Value at Risk (CVaR)
  • Maximum Drawdown
  • Current Drawdown
  • Sharpe Ratio
  • Sortino Ratio
  • Portfolio Volatility
  • Beta and Alpha Estimation
  • Annualized Return
  • NAV Tracking
  • Trade Distribution Analytics
  • Portfolio Exposure Monitoring

  These metrics are continuously evaluated to estimate downside risk, portfolio instability, risk-adjusted performance, and market sensitivity. Risk scores directly influence trade filtering and portfolio allocation logic to dynamically reduce exposure during unstable market conditions.

- The dashboard and reporting layer provide transparent and explainable portfolio analytics for stakeholders. The frontend visualizes NAV progression, cumulative portfolio returns, signal distributions, executed trade logs, risk exposure, drawdown curves, portfolio volatility, and portfolio performance metrics including Sharpe ratio and VaR estimates. The reporting engine generates structured JSON reports and interactive dashboard visualizations suitable for real-time analysis and institutional-style monitoring workflows.

- To improve explainability and stakeholder accessibility, the platform additionally integrates an AI-powered financial interpretation layer using Gemini-based large language models. This module automatically generates natural-language explanations of trading behavior, portfolio risk exposure, market conditions, feature-engineering rationale, signal quality, and overall strategy performance. The combination of quantitative finance analytics with LLM-powered explainability creates a transparent and interpretable hedge-fund-style decision support platform for both technical and non-technical stakeholders.

- Tech Stack Used:
  
  • Programming Language: Python
  • Data Processing & ETL: pandas, NumPy
  • Statistical Analysis & Risk Modeling: SciPy
  • Machine Learning Utilities: scikit-learn
  • Missing Value Imputation: KNNImputer
  • Quantitative Feature Engineering: pandas rolling analytics
  • AI Explainability Layer: Google Gemini API
  • Frontend Dashboard: Streamlit
  • Data Visualization: Streamlit Charts
  • Portfolio Simulation Engine: Custom Python Modules
  • Configuration & Environment Handling: python-dotenv
  • Report Generation: JSON-based analytics pipeline
  • Development Environment: VS Code / IntelliJ IDEA
```

Installation & Execution
Follow the steps below to run the project locally.
1. Clone the repository and navigate into the project directory.

2. Create a Python virtual environment:
   
   Windows:
   python -m venv venv

   Linux / Mac:
   python3 -m venv venv

3. Activate the virtual environment:
   
   Windows:
   venv\\Scripts\\activate

   Linux / Mac:
   source venv/bin/activate

4. Install all required dependencies:
   
   pip install -r requirements.txt

5. Configure environment variables:
   
   Create a .env file in the root project directory and add:
   
   GEMINI_API_KEY=your_api_key_here

6. Place datasets inside the data/ directory:
   
   • equity.csv
   • macro.csv
   • oil.csv
   • multiasset.csv

7. Run the main quantitative trading pipeline:
   
   python main.py

   This generates:
   • output/report.json
   • output/ai_analysis.txt

8. Launch the Streamlit dashboard frontend:
   
   streamlit run app.py

9. Open the dashboard in browser:
   
   http://localhost:8501

10. The dashboard provides:
   
   • Portfolio performance analytics
   • NAV progression
   • Risk metrics visualization
   • Trade execution logs
   • Signal analysis
   • AI-generated portfolio interpretation
