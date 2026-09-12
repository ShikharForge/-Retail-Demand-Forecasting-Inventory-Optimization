"""
src/data/preprocessor.py
=========================
Clean and standardise the raw Walmart dataset.

Design decisions:
- Sorting by (Store, Date) ensures all downstream time-series operations
  are chronologically correct.
- Store is cast to a pandas Categorical so LightGBM/XGBoost can handle
  it natively without one-hot expansion.
- No imputation is required for this dataset (no nulls), but the function
  is structured to be extended if needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply cleaning steps to the raw DataFrame.

    Steps
    -----
    1. Sort by Store, Date (required for correct lag/rolling computation).
    2. Reset index.
    3. Cast Store to Categorical (for LightGBM).
    4. Rename columns to snake_case for consistency.
    5. Add a `year_week` column (ISO year-week string) for labelling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from loader.load_raw_data().
    config : dict
        Loaded configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Cleaned, sorted DataFrame.
    """
    store_col = config["data"]["store_column"]
    date_col = config["data"]["date_column"]

    df = df.copy()

    # 1. Sort chronologically within each store
    df = df.sort_values([store_col, date_col]).reset_index(drop=True)

    # 2. Rename to snake_case
    rename_map = {
        "Store": "store",
        "Date": "date",
        "Weekly_Sales": "weekly_sales",
        "Holiday_Flag": "holiday_flag",
        "Temperature": "temperature",
        "Fuel_Price": "fuel_price",
        "CPI": "cpi",
        "Unemployment": "unemployment",
    }
    df = df.rename(columns=rename_map)

    # Update config keys to reflect new names (caller's responsibility to use
    # snake_case column names after this function)

    # 3. Cast store to Categorical (preserves integer ordering)
    df["store"] = df["store"].astype("category")

    # 4. Add year_week label (e.g. "2010-W06")
    df["year_week"] = df["date"].dt.strftime("%G-W%V")

    # 5. Add week_ending label for display
    df["week_ending"] = df["date"].dt.date

    logger.info(
        "Cleaned dataset: %d rows, %d columns, %d stores, date %s to %s",
        len(df),
        df.shape[1],
        df["store"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )

    return df


def save_processed(df: pd.DataFrame, config: dict) -> Path:
    """
    Save the processed DataFrame to Parquet.

    Returns
    -------
    Path
        Path to the saved file.
    """
    out_path = Path(config["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved processed data to {out_path}")
    return out_path


def load_processed(config: dict) -> pd.DataFrame:
    """Load the processed Parquet file."""
    path = Path(config["data"]["processed_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run the pipeline first."
        )
    df = pd.read_parquet(path)
    # Ensure store is Categorical after loading from parquet
    df["store"] = df["store"].astype("category")
    return df
