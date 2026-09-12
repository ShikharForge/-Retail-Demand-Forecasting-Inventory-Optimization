"""
src/features/engineering.py
===========================
Leakage-safe time-series feature engineering for the Walmart retail dataset.
"""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar features derived from the date column."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon, 4=Fri
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose the binary Holiday_Flag into specific retail holiday indicators."""
    df = df.copy()
    df["is_superbowl"] = ((df["holiday_flag"] == 1) & (df["month"] == 2)).astype(int)
    df["is_laborday"] = ((df["holiday_flag"] == 1) & (df["month"] == 9)).astype(int)
    df["is_thanksgiving"] = ((df["holiday_flag"] == 1) & (df["month"] == 11)).astype(int)
    df["is_christmas"] = ((df["holiday_flag"] == 1) & (df["month"].isin([12, 1]))).astype(int)
    df["weeks_to_thanksgiving"] = df.apply(lambda r: _weeks_to_event(r["date"], r["year"], 11), axis=1).clip(0, 4)
    df["weeks_to_christmas"] = df.apply(lambda r: _weeks_to_event(r["date"], r["year"], 12, day=25), axis=1).clip(0, 4)
    return df


def _weeks_to_event(date: pd.Timestamp, year: int, month: int, day: int = 1) -> int:
    """Return weeks before event or 99 if passed."""
    try:
        target = pd.Timestamp(year=year, month=month, day=day)
        delta = (target - date).days / 7
        return int(delta) if delta >= 0 else 99
    except Exception:
        return 99


def add_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """Add per-store sales lag features."""
    df = df.copy()
    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = df.groupby("store", observed=True)["weekly_sales"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Add leakage-safe rolling statistics (mean, std, max) of past sales."""
    df = df.copy()
    for window in windows:
        shifted = df.groupby("store", observed=True)["weekly_sales"].shift(1)
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


def add_economic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived economic changes and trends."""
    df = df.copy()
    df["fuel_price_chg"] = df.groupby("store", observed=True)["fuel_price"].diff()
    df["cpi_chg"] = df.groupby("store", observed=True)["cpi"].diff()
    df["unemployment_yoy"] = df.groupby("store", observed=True)["unemployment"].diff(52)
    return df


def build_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Execute full feature engineering pipeline."""
    lags = config["features"]["lags"]
    windows = config["features"]["rolling_windows"]
    logger.info("Building features (lags=%s, windows=%s)", lags, windows)

    df = add_calendar_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df, lags)
    df = add_rolling_features(df, windows)
    df = add_economic_features(df)

    logger.info("Feature engineering complete: shape=%s", df.shape)
    return df


def verify_no_leakage(df: pd.DataFrame, store_id: int = 1) -> bool:
    """Verify lag features contain strictly historical values."""
    store_df = df[df["store"].astype(int) == store_id].copy().reset_index(drop=True)
    if "lag_1" not in store_df.columns:
        return True

    errors = []
    for i in range(1, min(20, len(store_df))):
        expected_lag1 = store_df.loc[i - 1, "weekly_sales"]
        actual_lag1 = store_df.loc[i, "lag_1"]
        if pd.notna(actual_lag1) and not np.isclose(expected_lag1, actual_lag1, rtol=1e-5):
            errors.append(f"Row {i}: lag_1={actual_lag1:.2f} != t-1 sales={expected_lag1:.2f}")

    if errors:
        raise AssertionError(f"LEAKAGE DETECTED in lag_1 for store {store_id}:\n" + "\n".join(errors))

    return True
