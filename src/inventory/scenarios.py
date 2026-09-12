"""
src/inventory/scenarios.py
==========================
What-if scenario analysis for demand forecasts and inventory decisions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from src.inventory.optimization import InventoryDecision, InventoryParameters, optimise

HOLIDAY_UPLIFT_FACTOR = 1.078  # +7.8% historical holiday uplift


@dataclass
class ScenarioResult:
    """Side-by-side comparison of base vs scenario inventory decision."""
    scenario_name: str
    base: InventoryDecision
    scenario: InventoryDecision

    def delta(self) -> dict[str, Any]:
        """Absolute and relative changes for key metrics."""
        fields = ["safety_stock", "reorder_point", "recommended_order", "stockout_risk"]
        d = {}
        for f in fields:
            b_val = getattr(self.base, f)
            s_val = getattr(self.scenario, f)
            abs_chg = s_val - b_val
            pct_chg = (abs_chg / b_val * 100) if b_val != 0 else float("nan")
            d[f] = {
                "base": b_val,
                "scenario": s_val,
                "abs_change": round(abs_chg, 2),
                "pct_change": round(pct_chg, 2),
            }
        return d


def scenario_demand_surge(
    store: int,
    weekly_p50: float,
    weekly_p90: float,
    demand_std: float,
    params: InventoryParameters,
    surge_pct: float = 0.20,
) -> ScenarioResult:
    """Scenario A: Demand surge (+20%)."""
    base = optimise(store, weekly_p50, weekly_p90, demand_std, params)
    sc_p50 = weekly_p50 * (1 + surge_pct)
    sc_p90 = weekly_p90 * (1 + surge_pct)
    sc_std = demand_std * (1 + surge_pct)
    scenario = optimise(store, sc_p50, sc_p90, sc_std, params)
    return ScenarioResult(f"Demand Surge (+{int(surge_pct*100)}%)", base, scenario)


def scenario_demand_drop(
    store: int,
    weekly_p50: float,
    weekly_p90: float,
    demand_std: float,
    params: InventoryParameters,
    drop_pct: float = 0.20,
) -> ScenarioResult:
    """Scenario B: Demand drop (-20%)."""
    base = optimise(store, weekly_p50, weekly_p90, demand_std, params)
    sc_p50 = weekly_p50 * (1 - drop_pct)
    sc_p90 = weekly_p90 * (1 - drop_pct)
    sc_std = demand_std * (1 - drop_pct)
    scenario = optimise(store, sc_p50, sc_p90, sc_std, params)
    return ScenarioResult(f"Demand Drop (-{int(drop_pct*100)}%)", base, scenario)


def scenario_lead_time_doubles(
    store: int,
    weekly_p50: float,
    weekly_p90: float,
    demand_std: float,
    params: InventoryParameters,
) -> ScenarioResult:
    """Scenario C: Supply chain disruption (lead time doubles)."""
    base = optimise(store, weekly_p50, weekly_p90, demand_std, params)
    sc_params = copy.deepcopy(params)
    sc_params.lead_time_weeks = params.lead_time_weeks * 2
    scenario = optimise(store, weekly_p50, weekly_p90, demand_std, sc_params)
    return ScenarioResult(f"Lead Time Doubles ({sc_params.lead_time_weeks} wks)", base, scenario)


def scenario_high_uncertainty(
    store: int,
    weekly_p50: float,
    weekly_p90: float,
    demand_std: float,
    params: InventoryParameters,
    uncertainty_mult: float = 1.5,
) -> ScenarioResult:
    """Scenario D: Increased volatility / uncertainty."""
    base = optimise(store, weekly_p50, weekly_p90, demand_std, params)
    spread = weekly_p90 - weekly_p50
    sc_p90 = weekly_p50 + spread * uncertainty_mult
    sc_std = demand_std * uncertainty_mult
    scenario = optimise(store, weekly_p50, sc_p90, sc_std, params)
    return ScenarioResult("High Uncertainty (+50% volatility)", base, scenario)


def scenario_holiday_uplift(
    store: int,
    weekly_p50: float,
    weekly_p90: float,
    demand_std: float,
    params: InventoryParameters,
    uplift_factor: float = HOLIDAY_UPLIFT_FACTOR,
) -> ScenarioResult:
    """Scenario E: Historical holiday promotion."""
    base = optimise(store, weekly_p50, weekly_p90, demand_std, params)
    sc_p50 = weekly_p50 * uplift_factor
    sc_p90 = weekly_p90 * uplift_factor
    scenario = optimise(store, sc_p50, sc_p90, demand_std, params)
    pct = int((uplift_factor - 1) * 100)
    return ScenarioResult(f"Holiday Promotion (+{pct}%)", base, scenario)


def all_scenarios(
    store: int,
    weekly_p50: float,
    weekly_p90: float,
    demand_std: float,
    params: InventoryParameters,
) -> list[ScenarioResult]:
    """Run all 5 standard scenarios for side-by-side analysis."""
    return [
        scenario_demand_surge(store, weekly_p50, weekly_p90, demand_std, params),
        scenario_demand_drop(store, weekly_p50, weekly_p90, demand_std, params),
        scenario_lead_time_doubles(store, weekly_p50, weekly_p90, demand_std, params),
        scenario_high_uncertainty(store, weekly_p50, weekly_p90, demand_std, params),
        scenario_holiday_uplift(store, weekly_p50, weekly_p90, demand_std, params),
    ]
