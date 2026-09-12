"""
tests/test_features.py
=======================
Unit tests for feature engineering correctness.

The most critical property: ZERO FUTURE LEAKAGE.
We verify that lag and rolling features at time t contain only
information from t-1 or earlier.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    add_lag_features,
    add_rolling_features,
    build_features,
    verify_no_leakage,
)


@pytest.fixture
def sample_df():
    """Create a minimal two-store DataFrame for testing."""
    dates = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    stores = [1, 2]
    rows = []
    for store in stores:
        for i, date in enumerate(dates):
            rows.append({
                "store": store,
                "date": date,
                "weekly_sales": float(1000 * store + i * 10),
                "holiday_flag": 1 if i % 10 == 0 else 0,
                "temperature": 70.0,
                "fuel_price": 2.5,
                "cpi": 200.0,
                "unemployment": 5.0,
            })
    df = pd.DataFrame(rows)
    df["store"] = df["store"].astype("category")
    df = df.sort_values(["store", "date"]).reset_index(drop=True)
    return df


class TestLagFeatures:
    def test_lag1_is_previous_week(self, sample_df):
        """lag_1 at row i must equal weekly_sales at row i-1 (within store)."""
        df = add_lag_features(sample_df, lags=[1])
        for store, gdf in df.groupby("store", observed=True):
            gdf = gdf.reset_index(drop=True)
            for i in range(1, len(gdf)):
                expected = gdf.loc[i - 1, "weekly_sales"]
                actual = gdf.loc[i, "lag_1"]
                assert np.isclose(expected, actual, rtol=1e-5), (
                    f"Store {store}, row {i}: lag_1={actual} != t-1 sales={expected}"
                )

    def test_lag1_first_row_is_nan(self, sample_df):
        """First row per store must have NaN for lag_1."""
        df = add_lag_features(sample_df, lags=[1])
        for store, gdf in df.groupby("store", observed=True):
            gdf = gdf.reset_index(drop=True)
            assert pd.isna(gdf.loc[0, "lag_1"]), (
                f"Store {store}: lag_1 at first row should be NaN"
            )

    def test_lag4_is_4weeks_ago(self, sample_df):
        """lag_4 at row i must equal weekly_sales at row i-4."""
        df = add_lag_features(sample_df, lags=[4])
        for store, gdf in df.groupby("store", observed=True):
            gdf = gdf.reset_index(drop=True)
            for i in range(4, min(15, len(gdf))):
                expected = gdf.loc[i - 4, "weekly_sales"]
                actual = gdf.loc[i, "lag_4"]
                assert np.isclose(expected, actual, rtol=1e-5)

    def test_no_crossstore_contamination(self, sample_df):
        """Lag features must not bleed across stores."""
        df = add_lag_features(sample_df, lags=[1])
        for store, gdf in df.groupby("store", observed=True):
            gdf = gdf.reset_index(drop=True)
            # First row lag should be NaN (not from the other store's last row)
            assert pd.isna(gdf.loc[0, "lag_1"]), (
                f"Store {store}: cross-store leakage in lag_1"
            )


class TestRollingFeatures:
    def test_rolling_does_not_include_current_row(self, sample_df):
        """
        rolling_mean_4 at row i must not include weekly_sales[i].
        Verify: rolling_mean_4[i] = mean(weekly_sales[i-4:i-1])
        (because we shift(1) before rolling).
        """
        df = add_rolling_features(sample_df, windows=[4])
        for store, gdf in df.groupby("store", observed=True):
            gdf = gdf.reset_index(drop=True)
            # Check at row 5: window should be rows [1,2,3,4] (after shift)
            if len(gdf) >= 6:
                row_idx = 5
                actual_rolling = gdf.loc[row_idx, "rolling_mean_4"]
                # Expected: mean of sales[1:5] = shifted window
                expected = gdf.loc[1:4, "weekly_sales"].mean()
                if pd.notna(actual_rolling):
                    assert np.isclose(expected, actual_rolling, rtol=1e-4), (
                        f"Store {store}: rolling_mean_4 at row 5 includes future data"
                    )

    def test_rolling_std_positive(self, sample_df):
        """Rolling std should be non-negative wherever defined."""
        df = add_rolling_features(sample_df, windows=[4])
        col = "rolling_std_4"
        if col in df.columns:
            assert (df[col].dropna() >= 0).all(), "Negative rolling std found"


class TestLeakageVerification:
    def test_verify_no_leakage_passes(self, sample_df):
        """verify_no_leakage should pass on correctly engineered data."""
        config = {
            "features": {"lags": [1, 4], "rolling_windows": [4]},
            "data": {}
        }
        df = build_features(sample_df, config)
        # Should not raise
        result = verify_no_leakage(df, store_id=1)
        assert result is True

    def test_verify_leakage_detects_error(self, sample_df):
        """verify_no_leakage should raise if lag_1 is wrong."""
        df = sample_df.copy()
        df = add_lag_features(df, lags=[1])
        # Deliberately corrupt lag_1 to equal current row sales (leakage)
        df["lag_1"] = df["weekly_sales"]
        with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
            verify_no_leakage(df, store_id=1)
