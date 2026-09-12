"""
src/scenarios/scenario_engine.py
==================================
What-if scenario analysis for demand forecasts and inventory decisions.

Each scenario modifies forecast inputs and recomputes inventory decisions,
showing the downstream effect on safety stock, ROP, and stockout risk.

Scenarios implemented
----------------------
A. Demand Surge (+20%)
   Simulates a promotion, seasonal spike, or supply competitor exit.
   Effect: higher LTD, higher ROP, larger recommended order.

B. Demand Drop (-20%)
   Simulates post-promotion slump, recession, or product substitution.
   Effect: over-stock risk increases, lower ROP.

C. Lead Time Doubles
   Simulates supplier disruption, port delays, or geopolitical events.
   Effect: LTD doubles, safety stock grows (√2 factor), ROP rises sharply.

D. High Uncertainty
   Widens the P10-P90 interval by multiplying σ by 1.5.
   Effect: safety stock increases significantly.

E. Holiday Promotion
   Applies the historical holiday uplift factor observed in the data.
   Effect: forecast rises, ROP rises.

Each scenario returns a modified InventoryDecision for side-by-side comparison.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.inventory.optimizer import InventoryDecision, InventoryParameters, optimise

logger = logging.getLogger(__name__)

# Historical holiday uplift observed in the Walmart dataset
HOLIDAY_UPLIFT_FACTOR = 1.078  # +7.8% in holiday weeks


@dataclass
class ScenarioResult:
    """Side-by-side comparison of base vs scenario inventory decision."""
    scenario_name: str
    base: InventoryDecision
    scenario: InventoryDecision

    def delta(self) -> dict:
        """Absolute and relative changes for key metrics."""
        fields = ["safety_stock", "reorder_point", "recommended_order", "stockout_risk"]
        result = {}
        for f in fields:
            base_val = getattr(self.base, f)
            scen_val = getattr(self.scenario, f)
            delta_abs = scen_val - base_val
            delta_pct = (delta_abs / base_val * 100) if base_val != 0 else np.nan
            result[f] = {
                "base": round(base_val, 2),
                "scenario": round(scen_val, 2),
                "delta_abs": round(delta_abs, 2),
                "delta_pct": round(delta_pct, 2),
            }
        return result


def run_scenario(
    scenario_name: str,
    store: int,
    base_p50: float,
    base_p90: float,
    demand_std_weekly: float,
    params: InventoryParameters,
    demand_multiplier: float = 1.0,
    lead_time_multiplier: float = 1.0,
    uncertainty_multiplier: float = 1.0,
) -> ScenarioResult:
    """
    Run a single what-if scenario and return the comparison.

    Parameters
    ----------
    scenario_name : str
        Human-readable label.
    store : int
        Store identifier.
    base_p50, base_p90 : float
        Base-case median and 90th-percentile weekly forecasts.
    demand_std_weekly : float
        Base-case weekly demand std dev.
    params : InventoryParameters
        Base inventory parameters.
    demand_multiplier : float
        Scale factor for P50 and P90 (default 1.0 = no change).
    lead_time_multiplier : float
        Scale factor for lead time weeks.
    uncertainty_multiplier : float
        Scale factor for P90 spread and demand_std.

    Returns
    -------
    ScenarioResult
        Base vs scenario decision comparison.
    """
    # Base decision
    base_decision = optimise(store, base_p50, base_p90, demand_std_weekly, params)

    # Scenario: modify inputs
    scen_p50 = base_p50 * demand_multiplier
    scen_p90 = base_p90 * demand_multiplier
    # Widen uncertainty if requested
    scen_p90 = scen_p50 + (scen_p90 - scen_p50) * uncertainty_multiplier
    scen_std = demand_std_weekly * uncertainty_multiplier

    scen_params = copy.deepcopy(params)
    scen_params.lead_time_weeks = max(1, int(params.lead_time_weeks * lead_time_multiplier))

    scen_decision = optimise(store, scen_p50, scen_p90, scen_std, scen_params)

    return ScenarioResult(
        scenario_name=scenario_name,
        base=base_decision,
        scenario=scen_decision,
    )


def all_scenarios(
    store: int,
    base_p50: float,
    base_p90: float,
    demand_std_weekly: float,
    params: InventoryParameters,
    config: Optional[dict] = None,
) -> list[ScenarioResult]:
    """
    Run all predefined scenarios and return a list of ScenarioResults.

    Parameters follow the config/config.yaml scenario section.
    """
    if config is not None:
        surge = config["scenarios"]["demand_surge_pct"]
        drop = config["scenarios"]["demand_drop_pct"]
        lt_mult = config["scenarios"]["lead_time_multiplier"]
        unc_mult = config["scenarios"]["uncertainty_multiplier"]
    else:
        surge, drop, lt_mult, unc_mult = 0.20, 0.20, 2.0, 1.5

    scenarios = [
        run_scenario(
            f"Demand Surge (+{int(surge*100)}%)",
            store, base_p50, base_p90, demand_std_weekly, params,
            demand_multiplier=1 + surge,
        ),
        run_scenario(
            f"Demand Drop (-{int(drop*100)}%)",
            store, base_p50, base_p90, demand_std_weekly, params,
            demand_multiplier=1 - drop,
        ),
        run_scenario(
            f"Lead Time Doubles (×{lt_mult:.0f})",
            store, base_p50, base_p90, demand_std_weekly, params,
            lead_time_multiplier=lt_mult,
        ),
        run_scenario(
            f"High Uncertainty (σ × {unc_mult:.1f})",
            store, base_p50, base_p90, demand_std_weekly, params,
            uncertainty_multiplier=unc_mult,
        ),
        run_scenario(
            f"Holiday Promotion (+{int((HOLIDAY_UPLIFT_FACTOR-1)*100)}%)",
            store, base_p50, base_p90, demand_std_weekly, params,
            demand_multiplier=HOLIDAY_UPLIFT_FACTOR,
        ),
    ]
    return scenarios
