"""
src/models/forecaster.py
=========================
Generate point and quantile forecasts from trained models.

Forecast types produced
------------------------
- Point forecast (P50)  : LightGBM regression model
- P10 lower bound       : LightGBM quantile model (alpha=0.10)
- P90 upper bound       : LightGBM quantile model (alpha=0.90)

Interpretation of quantiles
----------------------------
P10: Pessimistic scenario — 10% chance actual demand is below this.
     Use for conservative planning (e.g., minimum stock to have).

P50: Best estimate — median demand. Use as point forecast.

P90: Optimistic scenario — 90% chance actual demand is below this.
     Use for safety stock calculation (see inventory/optimizer.py).

Why quantile regression rather than prediction intervals?
----------------------------------------------------------
Conformal prediction intervals assume exchangeability — harder to guarantee
for time series. Quantile regression directly optimises the pinball loss,
producing calibrated quantile estimates. It's more interpretable and
directly usable for inventory decisions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models.trainer import get_feature_cols

logger = logging.getLogger(__name__)


def predict_point(
    model,
    df: pd.DataFrame,
    store_col: str = "store",
) -> np.ndarray:
    """
    Generate point forecasts from a LightGBM regression model.

    Parameters
    ----------
    model : lgb.LGBMRegressor
        Trained LightGBM point forecast model.
    df : pd.DataFrame
        Feature-engineered DataFrame (rows to predict).

    Returns
    -------
    np.ndarray
        Predicted weekly sales values.
    """
    feature_cols = get_feature_cols(df)
    X = df[feature_cols + [store_col]]
    preds = model.predict(X)
    # Clip to non-negative (sales cannot be negative)
    return np.clip(preds, 0, None)


def predict_xgb(
    model,
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
    models: dict[str, object],
    df: pd.DataFrame,
    store_col: str = "store",
) -> pd.DataFrame:
    """
    Generate P10, P50, P90 quantile forecasts.

    Parameters
    ----------
    models : dict
        Keys: 'q10', 'q50', 'q90' → trained LightGBM quantile models.
    df : pd.DataFrame
        Feature-engineered rows to predict.

    Returns
    -------
    pd.DataFrame
        Columns: store, date, p10, p50, p90
    """
    feature_cols = get_feature_cols(df)
    X = df[feature_cols + [store_col]]

    result = df[["store", "date"]].copy().reset_index(drop=True)
    for quantile_key, model in models.items():
        preds = np.clip(model.predict(X), 0, None)
        result[quantile_key] = preds

    # Enforce ordering: p10 <= p50 <= p90
    if all(k in result.columns for k in ["p10", "p50", "p90"]):
        result["p10"] = result[["p10", "p50"]].min(axis=1)
        result["p90"] = result[["p50", "p90"]].max(axis=1)

    return result


def build_forecast_table(
    df: pd.DataFrame,
    point_preds: np.ndarray,
    quantile_df: Optional[pd.DataFrame] = None,
    actual_col: str = "weekly_sales",
) -> pd.DataFrame:
    """
    Assemble a clean forecast output table.

    Returns
    -------
    pd.DataFrame
        Columns: store, date, actual (if available), forecast,
                 p10, p50, p90 (if quantile_df provided).
    """
    result = df[["store", "date"]].copy().reset_index(drop=True)
    result["forecast"] = point_preds

    if actual_col in df.columns:
        result["actual"] = df[actual_col].values

    if quantile_df is not None:
        for col in ["p10", "p50", "p90"]:
            if col in quantile_df.columns:
                result[col] = quantile_df[col].values

    return result
