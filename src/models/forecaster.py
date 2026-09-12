"""
src/models/forecaster.py
========================
Backwards-compatible alias for src.models.forecasting.
"""

from src.models.forecasting import (
    build_forecast_table,
    predict_point,
    predict_quantiles,
    predict_xgb,
)

__all__ = [
    "predict_point",
    "predict_xgb",
    "predict_quantiles",
    "build_forecast_table",
]
