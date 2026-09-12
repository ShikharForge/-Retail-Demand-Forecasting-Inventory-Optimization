"""
src/scenarios/__init__.py
=========================
Backwards-compatible package alias for scenario simulation.
"""

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
    "HOLIDAY_UPLIFT_FACTOR",
    "ScenarioResult",
    "all_scenarios",
    "scenario_demand_surge",
    "scenario_demand_drop",
    "scenario_lead_time_doubles",
    "scenario_high_uncertainty",
    "scenario_holiday_uplift",
]
