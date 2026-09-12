"""
src/features/engineer.py
========================
Backwards-compatible alias for src.features.engineering.
"""

from src.features.engineering import (
    add_calendar_features,
    add_economic_features,
    add_holiday_features,
    add_lag_features,
    add_rolling_features,
    build_features,
    verify_no_leakage,
)

__all__ = [
    "add_calendar_features",
    "add_economic_features",
    "add_holiday_features",
    "add_lag_features",
    "add_rolling_features",
    "build_features",
    "verify_no_leakage",
]
