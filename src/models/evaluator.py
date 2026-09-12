"""
src/models/evaluator.py
=======================
Backwards-compatible alias for src.models.evaluation.
"""

from src.models.evaluation import evaluate, evaluate_by_store, mae, mape, rmse, wape

__all__ = ["mae", "rmse", "wape", "mape", "evaluate", "evaluate_by_store"]
