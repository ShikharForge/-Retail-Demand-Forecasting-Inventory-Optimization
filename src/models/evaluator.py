"""
src/models/evaluator.py
========================
Evaluation metrics for demand forecasting.

Metrics implemented
-------------------
MAE  — Mean Absolute Error
      Interpretable in original units. Robust to outliers.

RMSE — Root Mean Squared Error
      Penalises large errors more than MAE. Useful when big misses
      are particularly costly (e.g., stockouts).

WAPE — Weighted Absolute Percentage Error
      = sum(|actual - predicted|) / sum(actual)
      Preferred over MAPE because it handles near-zero actuals correctly
      (denominator is total actual, not per-row actual).
      This is our PRIMARY metric.

MAPE — Mean Absolute Percentage Error
      Reported for completeness. Mathematically valid here because
      Walmart weekly sales are always > 0 (no zero-demand weeks).
      Caution: can be misleading for low-volume series.

Why not R²?
      R² is commonly misused in forecasting. A model that perfectly
      predicts the mean of the test set gets R²=0, and it can be
      negative. WAPE provides a cleaner percentage-error interpretation.
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
    """
    Weighted Absolute Percentage Error.

    WAPE = sum(|A - F|) / sum(A)

    Preferred over MAPE: the denominator is the total actual demand
    so extreme values in individual rows don't blow up the metric.
    """
    total_actual = np.sum(np.abs(actual))
    if total_actual == 0:
        return np.nan
    return float(np.sum(np.abs(actual - predicted)) / total_actual)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Only valid when all actual values are > 0 (verified for this dataset).
    Use WAPE as the primary metric.
    """
    if np.any(actual == 0):
        return np.nan  # Undefined for zero actuals
    return float(np.mean(np.abs((actual - predicted) / actual)))


def evaluate(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute multiple forecast evaluation metrics.

    Parameters
    ----------
    actual : array-like
        True sales values.
    predicted : array-like
        Model predictions.
    metrics : list of str, optional
        Which metrics to compute. Defaults to ['mae', 'rmse', 'wape', 'mape'].

    Returns
    -------
    dict
        {metric_name: value}
    """
    if metrics is None:
        metrics = ["mae", "rmse", "wape", "mape"]

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    metric_fns = {"mae": mae, "rmse": rmse, "wape": wape, "mape": mape}
    return {m: metric_fns[m](actual, predicted) for m in metrics if m in metric_fns}


def evaluate_by_store(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    store_col: str = "store",
) -> pd.DataFrame:
    """
    Compute metrics per store and overall.

    Returns
    -------
    pd.DataFrame
        Rows = individual stores (integer index), plus a final 'ALL' row.
        Columns = mae, rmse, wape, mape.
    """
    rows = []
    for store, gdf in df.groupby(store_col, observed=True):
        m = evaluate(gdf[actual_col].values, gdf[pred_col].values)
        m[store_col] = store
        rows.append(m)

    overall = evaluate(df[actual_col].values, df[pred_col].values)
    overall[store_col] = "ALL"
    rows.append(overall)

    result = pd.DataFrame(rows).set_index(store_col)
    return result.round(4)

