"""
Flexible Market Data Ingestion Pipeline
Hackathon-friendly loader that adapts heterogeneous datasets
without requiring strict institutional schemas.

Supports:
- equity
- macro
- oil
- multiasset

Falls back gracefully when columns differ.
"""

import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict


# -------------------------------------------------------------------
# Flexible Column Aliases
# -------------------------------------------------------------------

COLUMN_ALIASES = {
    "Date": "date",
    "DATE": "date",

    "Price": "close",
    "price": "close",

    "Adj Close": "close",
    "Close": "close",

    "Volume": "volume",

    "Symbol": "ticker",
    "Ticker": "ticker",

    "Contract": "contract",
}


# -------------------------------------------------------------------
# Flexible Minimal Schemas
# -------------------------------------------------------------------

MIN_REQUIRED = {
    "equity": ["date", "close"],
    "oil": ["date", "close"],
    "macro": ["date"],
    "multiasset": ["date"],
}


# -------------------------------------------------------------------
# Loader
# -------------------------------------------------------------------

class DataLoader:

    def __init__(self, data_paths: Dict[str, str]):
        self.data_paths = data_paths

    # ---------------------------------------------------------------
    # PUBLIC
    # ---------------------------------------------------------------

    def load_all(self) -> Dict[str, pd.DataFrame]:

        results = {}

        with ThreadPoolExecutor(max_workers=4) as pool:

            futures = {
                pool.submit(self._load_one, name, path): name
                for name, path in self.data_paths.items()
            }

            for future in as_completed(futures):

                name = futures[future]

                try:
                    df = future.result()

                    results[name] = df

                    print(f"    ✓ {name}: {len(df)} rows loaded")

                except Exception as e:

                    print(f"    ✗ {name}: load error – {e}")

        return results

    # ---------------------------------------------------------------
    # PRIVATE
    # ---------------------------------------------------------------

    def _load_one(self, name: str, path: str) -> pd.DataFrame:

        file = Path(path)

        if not file.exists():
            raise FileNotFoundError(f"{path} not found")

        df = pd.read_csv(file)

        # Normalize columns
        df.columns = [
            COLUMN_ALIASES.get(col, col.lower().strip())
            for col in df.columns
        ]

        # Validate minimum schema
        required = MIN_REQUIRED[name]

        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # Parse dates
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.dropna(subset=["date"])

        df = df.sort_values("date")

        # -----------------------------------------------------------
        # Dataset-specific fixes
        # -----------------------------------------------------------

        # EQUITY
        if name == "equity":

            if "ticker" not in df.columns:
                df["ticker"] = "EQUITY"

            if "volume" not in df.columns:
                df["volume"] = 0

        # OIL
        elif name == "oil":

            if "contract" not in df.columns:
                df["contract"] = "WTI"

            if "volume" not in df.columns:
                df["volume"] = 0

        # MACRO
        elif name == "macro":

            # Convert wide macro table into long format
            # Example:
            # date inflation rates sentiment
            # →
            # date indicator value

            non_date_cols = [
                c for c in df.columns
                if c != "date"
            ]

            if "indicator" not in df.columns:

                df = df.melt(
                    id_vars="date",
                    value_vars=non_date_cols,
                    var_name="indicator",
                    value_name="value"
                )

        # MULTIASSET
        elif name == "multiasset":

            # Convert wide multiasset data into long format

            asset_cols = [
                c for c in df.columns
                if c != "date"
            ]

            if "ticker" not in df.columns:

                df = df.melt(
                    id_vars="date",
                    value_vars=asset_cols,
                    var_name="ticker",
                    value_name="close"
                )

                df["asset_class"] = "MULTI"

                df["volume"] = 0

        return df.reset_index(drop=True)