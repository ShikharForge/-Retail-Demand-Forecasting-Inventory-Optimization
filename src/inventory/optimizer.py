"""
src/inventory/optimizer.py
==========================
Backwards-compatible alias for src.inventory.optimization.
"""

from src.inventory.optimization import (
    InventoryDecision,
    InventoryParameters,
    compute_eoq,
    compute_lead_time_demand,
    compute_safety_stock_parametric,
    compute_safety_stock_quantile,
    compute_stockout_risk,
    optimise,
    z_score,
)

__all__ = [
    "InventoryParameters",
    "InventoryDecision",
    "z_score",
    "compute_lead_time_demand",
    "compute_safety_stock_parametric",
    "compute_safety_stock_quantile",
    "compute_eoq",
    "compute_stockout_risk",
    "optimise",
]
