"""
src/models/forecasting.py
=========================
Generate point and quantile forecasts from trained models.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models.trainer import get_feature_cols

logger = logging.getLogger(__name__)


def predict_point(
    model: Any,
    df: pd.DataFrame,
    store_col: str = "store",
) -> np.ndarray:
    """Generate non-negative point forecasts from a LightGBM model."""
    feature_cols = get_feature_cols(df)
    X = df[feature_cols + [store_col]]
    preds = model.predict(X)
    return np.clip(preds, 0, None)


def predict_xgb(
    model: Any,
    df: pd.DataFrame,
    store_col: str = "store",
) -> np.ndarray:
    """Generate point forecasts from an XGBoost model."""
    feature_cols = get_feature_cols(df)
    df_enc = df.copy()
    df_enc["store_code"] = df_enc[store_col].cat.codes
    xgb_features = feature_cols + ["store_code"]
    preds = model.predict(df_enc[xgb_features])
    return np.clip(preds, 0, None)


def predict_quantiles(
    models: dict[str, Any],
    df: pd.DataFrame,
    store_col: str = "store",
) -> pd.DataFrame:
    """Generate P10, P50, P90 quantile forecasts ensuring monotonicity."""
    feature_cols = get_feature_cols(df)
    X = df[feature_cols + [store_col]]

    p10 = np.clip(models["q10"].predict(X), 0, None)
    p50 = np.clip(models["q50"].predict(X), 0, None)
    p90 = np.clip(models["q90"].predict(X), 0, None)

    # Monotonicity check: P10 <= P50 <= P90
    p50 = np.maximum(p50, p10)
    p90 = np.maximum(p90, p50)

    result = df[["store", "date"]].copy() if "date" in df.columns else pd.DataFrame(index=df.index)
    result["p10"] = p10
    result["p50"] = p50
    result["p90"] = p90
    return result


def build_forecast_table(
    df: pd.DataFrame,
    actual_col: str = "weekly_sales",
    p10: Optional[np.ndarray] = None,
    p50: Optional[np.ndarray] = None,
    p90: Optional[np.ndarray] = None,
    point: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Construct a consolidated forecast table with actuals and prediction intervals."""
    out = df[["store", "date"]].copy()
    if actual_col in df.columns:
        out["actual"] = df[actual_col].values
    if point is not None:
        out["forecast_point"] = point
    if p10 is not None:
        out["p10"] = p10
    if p50 is not None:
        out["p50"] = p50
    if p90 is not None:
        out["p90"] = p90
    if p10 is not None and p90 is not None:
        out["uncertainty_range"] = out["p90"] - out["p10"]
    return out
