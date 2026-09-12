"""
src/inventory/optimization.py
=============================
Statistically grounded inventory policy calculations driven by demand forecasts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class InventoryParameters:
    """
    Configurable inventory planning parameters.
    """
    lead_time_weeks: int = 2
    service_level: float = 0.95
    ordering_cost_usd: float = 500.0
    holding_cost_pct: float = 0.25
    current_inventory_usd: float = 0.0
    weeks_per_year: int = 52

    def __post_init__(self):
        assert 0 < self.service_level < 1, "service_level must be between 0 and 1"
        assert self.lead_time_weeks > 0, "lead_time_weeks must be positive"
        assert self.ordering_cost_usd >= 0, "ordering_cost_usd cannot be negative"
        assert self.holding_cost_pct > 0, "holding_cost_pct must be positive"


@dataclass
class InventoryDecision:
    """Output decision metrics for a single store and horizon."""
    store: int
    lead_time_demand: float
    demand_std: float
    safety_stock: float
    reorder_point: float
    eoq: float
    current_inventory: float
    recommended_order: float
    stockout_risk: float
    service_level: float
    weeks_of_coverage: float
    params: InventoryParameters = field(repr=False)


def z_score(service_level: float) -> float:
    """Standard normal quantile Z for given service level."""
    return float(stats.norm.ppf(service_level))


def compute_lead_time_demand(weekly_forecast_p50: float, lead_time_weeks: int) -> float:
    """Expected demand during replenishment lead time."""
    return float(weekly_forecast_p50 * lead_time_weeks)


def compute_safety_stock_parametric(demand_std_weekly: float, lead_time_weeks: int, service_level: float) -> float:
    """Parametric safety stock: Z(SL) * sigma_weekly * sqrt(L)."""
    z = z_score(service_level)
    return float(z * demand_std_weekly * np.sqrt(lead_time_weeks))


def compute_safety_stock_quantile(weekly_p50: float, weekly_p90: float, lead_time_weeks: int) -> float:
    """Non-parametric safety stock using quantile spread: (P90 - P50) * sqrt(L)."""
    uncertainty = max(0.0, weekly_p90 - weekly_p50)
    return float(uncertainty * np.sqrt(lead_time_weeks))


def compute_eoq(annual_demand: float, ordering_cost: float, holding_cost_pct: float, avg_item_value: float = 1.0) -> float:
    """Economic Order Quantity (Wilson Formula): sqrt(2*D*S / H)."""
    h = holding_cost_pct * avg_item_value
    if h <= 0 or annual_demand <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * ordering_cost / h))


def compute_stockout_risk(reorder_point: float, lead_time_demand: float, demand_std_lead_time: float) -> float:
    """Probability of demand exceeding ROP during lead time."""
    if demand_std_lead_time <= 0:
        return 0.0 if reorder_point >= lead_time_demand else 1.0
    z = (reorder_point - lead_time_demand) / demand_std_lead_time
    return float(1 - stats.norm.cdf(z))


def optimise(
    store: int,
    weekly_p50: float,
    weekly_p90: float,
    demand_std_weekly: float,
    params: InventoryParameters,
    use_quantile_safety_stock: bool = True,
) -> InventoryDecision:
    """Compute optimal inventory policy and replenishment decision."""
    l = params.lead_time_weeks
    ltd = compute_lead_time_demand(weekly_p50, l)

    if use_quantile_safety_stock:
        ss = compute_safety_stock_quantile(weekly_p50, weekly_p90, l)
    else:
        ss = compute_safety_stock_parametric(demand_std_weekly, l, params.service_level)

    rop = ltd + ss
    annual_demand = weekly_p50 * params.weeks_per_year
    eoq = compute_eoq(annual_demand, params.ordering_cost_usd, params.holding_cost_pct)

    shortfall = max(0.0, rop - params.current_inventory_usd)
    recommended_order = max(0.0, shortfall + eoq) if shortfall > 0 else 0.0

    sigma_lt = demand_std_weekly * np.sqrt(l)
    stockout_risk = compute_stockout_risk(rop, ltd, sigma_lt)

    weeks_coverage = (
        params.current_inventory_usd / weekly_p50
        if weekly_p50 > 0
        else float("inf")
    )

    return InventoryDecision(
        store=store,
        lead_time_demand=round(ltd, 2),
        demand_std=round(demand_std_weekly, 2),
        safety_stock=round(ss, 2),
        reorder_point=round(rop, 2),
        eoq=round(eoq, 2),
        current_inventory=params.current_inventory_usd,
        recommended_order=round(recommended_order, 2),
        stockout_risk=round(stockout_risk, 4),
        service_level=params.service_level,
        weeks_of_coverage=round(weeks_coverage, 2),
        params=params,
    )
