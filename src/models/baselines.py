"""
src/models/baselines.py
========================
Baseline forecasting models.

Why baselines first?
--------------------
A model that doesn't beat a simple baseline doesn't justify its complexity.
We implement two standard baselines common in retail forecasting:

1. Naive (Last Value)
   Forecast = last observed weekly sales.
   Simplest possible predictor. Sets the floor.

2. Seasonal Naive
   Forecast = sales from the same week 52 weeks ago.
   Very competitive for weekly retail data with strong annual seasonality.
   Often hard to beat with ML unless the dataset is rich enough.

Interview Q: "Why seasonal naive instead of just naive?"
A: Weekly retail demand has strong annual seasonality (Christmas, back-to-school).
   The same-week-last-year value captures this without any parameter fitting.
   LightGBM also uses lag_52, so if LightGBM doesn't beat seasonal naive,
   that tells us our ML features aren't adding value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveForecast:
    """
    Forecast = last observed value in the training set.

    For multi-step forecasts, we propagate the last value forward.
    """

    def __init__(self):
        self._last_values: dict[int, float] = {}

    def fit(self, df: pd.DataFrame) -> "NaiveForecast":
        """Record the last sales value per store from the training set."""
        for store, gdf in df.groupby("store", observed=True):
            self._last_values[int(store)] = float(gdf["weekly_sales"].iloc[-1])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return the last training value for each row's store."""
        return np.array(
            [self._last_values.get(int(s), np.nan) for s in df["store"]]
        )


class SeasonalNaiveForecast:
    """
    Forecast = sales from the same week 52 weeks ago (lag_52).

    If lag_52 is already computed in the feature set, we use it directly.
    Otherwise we look it up from the training history.
    """

    def __init__(self):
        # dict: (store, week_of_year) -> sales
        self._lookup: dict[tuple, float] = {}

    def fit(self, df: pd.DataFrame) -> "SeasonalNaiveForecast":
        """Build a lookup of store × week_of_year -> most recent sales."""
        if "week_of_year" not in df.columns:
            df = df.copy()
            df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        for _, row in df.iterrows():
            key = (int(row["store"]), int(row["week_of_year"]))
            self._lookup[key] = float(row["weekly_sales"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return the historical value for the matching store × week_of_year.

        If lag_52 is present in the DataFrame, we use it directly (faster
        and consistent with the feature engineering pipeline).
        """
        if "lag_52" in df.columns:
            return df["lag_52"].values

        if "week_of_year" not in df.columns:
            df = df.copy()
            df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

        preds = []
        for _, row in df.iterrows():
            key = (int(row["store"]), int(row["week_of_year"]))
            preds.append(self._lookup.get(key, np.nan))
        return np.array(preds)
