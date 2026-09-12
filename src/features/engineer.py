"""
src/features/engineer.py
=========================
Leakage-safe time-series feature engineering for the Walmart dataset.

CRITICAL DESIGN PRINCIPLE — NO FUTURE LEAKAGE
----------------------------------------------
All lag and rolling features are computed per-store, after the DataFrame
has been sorted chronologically. The shift(n) operation ensures that at
prediction time t, the feature value contains information only from
time t-n or earlier.

Rolling windows use .shift(1) before .rolling() so that the window
at time t contains only [t-window, ..., t-1] — never t itself.

The function `verify_no_leakage()` provides an automated sanity check.

Feature categories
------------------
1. Calendar features   — deterministic, always available before the week
2. Holiday features    — derived from the Holiday_Flag column
3. Lag features        — past sales, shifted by n weeks
4. Rolling features    — rolling statistics of past sales
5. Economic covariates — Temperature, Fuel_Price, CPI, Unemployment
   NOTE: These are same-week values in this dataset. In a real deployment
   they would be lagged. This is documented as a modelling assumption.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calendar Features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar features derived from the date column."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon, 4=Fri
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    # Cyclic encoding of week_of_year (captures periodicity better than raw int)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# ---------------------------------------------------------------------------
# Holiday Features
# ---------------------------------------------------------------------------

def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose the binary Holiday_Flag into specific holiday types.

    The Walmart dataset identifies four holiday events:
      - Super Bowl:   Feb (week containing 2nd Friday of Feb)
      - Labour Day:   Sep (week containing 1st Monday of Sep)
      - Thanksgiving: Nov (4th Thursday)
      - Christmas:    Dec 25 / New Year Dec 31

    We infer the type from month because the flag is 1 for exactly these weeks.
    """
    df = df.copy()
    # Binary flags for specific holidays (derived from month + flag)
    df["is_superbowl"] = ((df["holiday_flag"] == 1) & (df["month"] == 2)).astype(int)
    df["is_laborday"] = ((df["holiday_flag"] == 1) & (df["month"] == 9)).astype(int)
    df["is_thanksgiving"] = ((df["holiday_flag"] == 1) & (df["month"] == 11)).astype(int)
    df["is_christmas"] = (
        (df["holiday_flag"] == 1) & (df["month"].isin([12, 1]))
    ).astype(int)
    # Weeks before major holidays (pre-holiday shopping effect)
    df["weeks_to_thanksgiving"] = df.apply(
        lambda r: _weeks_to_event(r["date"], r["year"], 11), axis=1
    ).clip(0, 4)
    df["weeks_to_christmas"] = df.apply(
        lambda r: _weeks_to_event(r["date"], r["year"], 12, day=25), axis=1
    ).clip(0, 4)
    return df


def _weeks_to_event(date: pd.Timestamp, year: int, month: int, day: int = 1) -> int:
    """Return how many weeks before a target month/day event. Returns 99 if past."""
    try:
        target = pd.Timestamp(year=year, month=month, day=day)
        delta = (target - date).days / 7
        return int(delta) if delta >= 0 else 99
    except Exception:
        return 99


# ---------------------------------------------------------------------------
# Lag Features (per store — CHRONOLOGICALLY SAFE)
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """
    Add sales lag features per store.

    Each lag_n feature at row t contains the sales value from t-n weeks ago.
    shift(n) within a group guarantees this — the first n rows of each store
    will be NaN, which is expected and handled later.
    """
    df = df.copy()
    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = df.groupby("store", observed=True)["weekly_sales"].shift(lag)
        logger.debug(f"Added {col_name}: {df[col_name].isna().sum()} NaN rows")
    return df


# ---------------------------------------------------------------------------
# Rolling Features (per store — CHRONOLOGICALLY SAFE)
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """
    Add rolling mean, std, and max features per store.

    Critical: we apply shift(1) before rolling so that the window
    at time t is [t-window, ..., t-1], never including t.
    This is the correct leakage-prevention pattern for time series.
    """
    df = df.copy()
    for window in windows:
        shifted = df.groupby("store", observed=True)["weekly_sales"].shift(1)
        # Rolling on the shifted series: window [t-window-1, ..., t-1]
        # This is equivalent to a window that sees t-window to t-1 relative to original
        df[f"rolling_mean_{window}"] = (
            shifted.groupby(df["store"], observed=True)
            .transform(lambda x: x.rolling(window, min_periods=max(1, window // 2)).mean())
        )
        df[f"rolling_std_{window}"] = (
            shifted.groupby(df["store"], observed=True)
            .transform(lambda x: x.rolling(window, min_periods=max(1, window // 2)).std())
        )
        if window <= 4:
            df[f"rolling_max_{window}"] = (
                shifted.groupby(df["store"], observed=True)
                .transform(lambda x: x.rolling(window, min_periods=max(1, window // 2)).max())
            )
    return df


# ---------------------------------------------------------------------------
# Economic Covariate Features
# ---------------------------------------------------------------------------

def add_economic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived economic covariate features.

    ASSUMPTION NOTE:
    ----------------
    Temperature, Fuel_Price, CPI, and Unemployment are same-week values
    in this dataset. In a real forecasting deployment these would need to be
    lagged (or obtained from forecasts). We use them here as contextual
    covariates for pattern learning, with this assumption documented.
    """
    df = df.copy()
    # Week-over-week change in fuel price (per store)
    df["fuel_price_chg"] = df.groupby("store", observed=True)["fuel_price"].diff()
    # Week-over-week change in CPI
    df["cpi_chg"] = df.groupby("store", observed=True)["cpi"].diff()
    # Year-over-year change in unemployment (approx, shift 52)
    df["unemployment_yoy"] = df.groupby("store", observed=True)["unemployment"].diff(52)
    return df


# ---------------------------------------------------------------------------
# Master Feature Engineering Function
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from preprocessor.clean().
    config : dict
        Loaded configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame (original rows preserved, NaNs from
        lag/rolling at the start of each store's history).
    """
    lags = config["features"]["lags"]
    windows = config["features"]["rolling_windows"]

    logger.info("Building features: lags=%s, rolling windows=%s", lags, windows)

    df = add_calendar_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df, lags)
    df = add_rolling_features(df, windows)
    df = add_economic_features(df)

    # Log NaN summary
    nan_cols = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    if nan_cols:
        logger.info("NaN counts after feature engineering (expected for lags): %s", nan_cols)

    logger.info(
        "Feature engineering complete. Shape: %s. Feature columns: %d",
        df.shape,
        df.shape[1],
    )
    return df


# ---------------------------------------------------------------------------
# Leakage Verification
# ---------------------------------------------------------------------------

def verify_no_leakage(df: pd.DataFrame, store_id: int = 1) -> bool:
    """
    Automated sanity check: verify that lag/rolling features at time t
    do not contain information from time t or later.

    Method: for a single store, pick a row at index i and verify that
    lag_1 == weekly_sales at index i-1 (within that store).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.
    store_id : int
        Store to use for the check.

    Returns
    -------
    bool
        True if no leakage detected, raises AssertionError otherwise.
    """
    store_df = df[df["store"].astype(int) == store_id].copy().reset_index(drop=True)

    if "lag_1" not in store_df.columns:
        logger.warning("lag_1 not found — skipping leakage check")
        return True

    errors = []
    for i in range(1, min(20, len(store_df))):
        expected_lag1 = store_df.loc[i - 1, "weekly_sales"]
        actual_lag1 = store_df.loc[i, "lag_1"]
        if pd.notna(actual_lag1) and not np.isclose(expected_lag1, actual_lag1, rtol=1e-5):
            errors.append(
                f"Row {i}: lag_1={actual_lag1:.2f} but t-1 sales={expected_lag1:.2f}"
            )

    if errors:
        raise AssertionError(
            f"LEAKAGE DETECTED in lag_1 for store {store_id}:\n" + "\n".join(errors)
        )

    logger.info("Leakage check PASSED for store %d", store_id)
    return True
