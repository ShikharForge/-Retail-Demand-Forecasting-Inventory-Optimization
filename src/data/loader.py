"""
src/data/loader.py
==================
Data loading and raw validation for the Walmart weekly sales dataset.

Design decisions:
- Centralises all I/O so the rest of the codebase never touches raw paths directly.
- Validates schema and raises informative errors rather than silent failures.
- Returns a copy of the DataFrame to avoid mutation surprises.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict) -> pd.DataFrame:
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

    Raises
    ------
    FileNotFoundError
        If the raw data file does not exist.
    ValueError
        If expected columns are missing.
    """
    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at: {raw_path.resolve()}")

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)

    # Parse date column
    date_col = config["data"]["date_column"]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

    # Validate expected columns
    expected = [
        config["data"]["store_column"],
        config["data"]["date_column"],
        config["data"]["target_column"],
        config["data"]["holiday_column"],
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df.copy()


def validate_raw_data(df: pd.DataFrame, config: dict) -> dict:
    """
    Run data quality checks and return a structured report.

    Returns
    -------
    dict
        Quality report with keys: shape, dtypes, nulls, duplicates,
        sales_stats, date_range, store_count, weeks_per_store.
    """
    target = config["data"]["target_column"]
    store_col = config["data"]["store_column"]
    date_col = config["data"]["date_column"]

    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_store_date": int(df.duplicated(subset=[store_col, date_col]).sum()),
        "date_range": {
            "min": str(df[date_col].min().date()),
            "max": str(df[date_col].max().date()),
        },
        "store_count": int(df[store_col].nunique()),
        "weeks_per_store": df.groupby(store_col)[date_col].count().to_dict(),
        "sales_stats": {
            "min": float(df[target].min()),
            "max": float(df[target].max()),
            "mean": float(df[target].mean()),
            "median": float(df[target].median()),
            "std": float(df[target].std()),
            "negative_count": int((df[target] < 0).sum()),
            "zero_count": int((df[target] == 0).sum()),
            "skewness": float(df[target].skew()),
        },
        "holiday_distribution": df[config["data"]["holiday_column"]].value_counts().to_dict(),
    }

    # Log any quality issues
    if report["null_counts"] and any(v > 0 for v in report["null_counts"].values()):
        logger.warning("NULL values detected: %s", report["null_counts"])
    if report["duplicate_rows"] > 0:
        logger.warning("Duplicate rows detected: %d", report["duplicate_rows"])
    if report["sales_stats"]["negative_count"] > 0:
        logger.warning("Negative sales detected: %d rows", report["sales_stats"]["negative_count"])

    return report
