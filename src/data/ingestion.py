"""
src/data/ingestion.py
=====================
Data loading, configuration parsing, and raw data validation for Walmart weekly sales dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load the raw Walmart CSV file.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with Date parsed to datetime.
    """
    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at: {raw_path.resolve()}")

    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)

    # Parse date column
    date_col = config["data"]["date_column"]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    return df


def validate_raw_data(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the raw DataFrame against expected schema and business invariants.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as loaded by load_raw_data.
    config : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Summary dictionary with validation findings.
    """
    expected_cols = [
        "Store", "Date", "Weekly_Sales", "Holiday_Flag",
        "Temperature", "Fuel_Price", "CPI", "Unemployment"
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw data: {missing_cols}")

    null_counts = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())
    duplicate_store_date = int(df.duplicated(subset=["Store", "Date"]).sum())
    store_count = int(df["Store"].nunique())
    date_range = (str(df["Date"].min().date()), str(df["Date"].max().date()))
    negative_sales_count = int((df["Weekly_Sales"] < 0).sum())
    zero_sales_count = int((df["Weekly_Sales"] == 0).sum())

    report = {
        "shape": df.shape,
        "null_counts": null_counts,
        "duplicate_rows": duplicate_rows,
        "duplicate_store_date": duplicate_store_date,
        "store_count": store_count,
        "date_range": date_range,
        "sales_stats": {
            "min": float(df["Weekly_Sales"].min()),
            "max": float(df["Weekly_Sales"].max()),
            "mean": float(df["Weekly_Sales"].mean()),
            "median": float(df["Weekly_Sales"].median()),
            "std": float(df["Weekly_Sales"].std()),
            "skewness": float(df["Weekly_Sales"].skew()),
            "negative_count": negative_sales_count,
            "zero_count": zero_sales_count,
        },
        "holiday_distribution": df["Holiday_Flag"].value_counts().to_dict(),
    }

    logger.info("Raw data validation completed successfully: %s", report["shape"])
    return report
