"""
src/data/preprocessing.py
=========================
Clean, standardise, and persist the Walmart dataset for downstream modeling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Apply standard cleaning steps to the raw DataFrame.

    Steps:
    1. Sort by Store, Date chronologically.
    2. Reset index.
    3. Standardise columns to snake_case.
    4. Cast Store to Categorical (for LightGBM).
    5. Add ISO year_week and week_ending labels.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from ingestion.load_raw_data().
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

    # 1. Chronological sorting
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

    # 3. Categorical store
    df["store"] = df["store"].astype("category")

    # 4. Labels
    df["year_week"] = df["date"].dt.strftime("%G-W%V")
    df["week_ending"] = df["date"].dt.date

    logger.info(
        "Cleaned dataset: %d rows, %d columns, %d stores, %s to %s",
        len(df),
        df.shape[1],
        df["store"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


def save_processed(df: pd.DataFrame, config: dict[str, Any], path: str | Path | None = None) -> Path:
    """
    Save the processed DataFrame to Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    config : dict
        Configuration dictionary containing default processed_path.
    path : str or Path, optional
        Explicit path override.

    Returns
    -------
    Path
        Path where file was written.
    """
    out_path = Path(path or config["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved processed dataset (%d rows) to %s", len(df), out_path)
    return out_path
