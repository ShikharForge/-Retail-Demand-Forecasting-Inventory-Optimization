"""
src/models/evaluation.py
========================
Evaluation metrics and validation tools for demand forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (sum(|A - F|) / sum(A))."""
    total_actual = np.sum(np.abs(actual))
    if total_actual == 0:
        return np.nan
    return float(np.sum(np.abs(actual - predicted)) / total_actual)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    if np.any(actual == 0):
        return np.nan
    return float(np.mean(np.abs((actual - predicted) / actual)))


def evaluate(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute multiple forecast evaluation metrics."""
    if metrics is None:
        metrics = ["mae", "rmse", "wape", "mape"]

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0:
        return {m: np.nan for m in metrics}

    fn_map = {"mae": mae, "rmse": rmse, "wape": wape, "mape": mape}
    return {m: fn_map[m](actual, predicted) for m in metrics if m in fn_map}


def evaluate_by_store(
    df: pd.DataFrame,
    actual_col: str = "weekly_sales",
    pred_col: str = "predicted",
    store_col: str = "store",
) -> pd.DataFrame:
    """Compute per-store evaluation metrics."""
    rows = []
    for store, gdf in df.groupby(store_col, observed=True):
        m = evaluate(gdf[actual_col].values, gdf[pred_col].values)
        m["store"] = store
        m["n_obs"] = len(gdf)
        rows.append(m)
    return pd.DataFrame(rows).set_index("store")
