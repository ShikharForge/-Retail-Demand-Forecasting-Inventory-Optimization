"""
src/inventory/__init__.py
=========================
Inventory optimization and scenario simulation exports.
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
from src.inventory.scenarios import (
    HOLIDAY_UPLIFT_FACTOR,
    ScenarioResult,
    all_scenarios,
    scenario_demand_drop,
    scenario_demand_surge,
    scenario_high_uncertainty,
    scenario_holiday_uplift,
    scenario_lead_time_doubles,
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
    "ScenarioResult",
    "HOLIDAY_UPLIFT_FACTOR",
    "scenario_demand_surge",
    "scenario_demand_drop",
    "scenario_lead_time_doubles",
    "scenario_high_uncertainty",
    "scenario_holiday_uplift",
    "all_scenarios",
]
