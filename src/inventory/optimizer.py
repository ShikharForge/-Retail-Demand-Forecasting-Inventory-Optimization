"""
src/inventory/optimizer.py
===========================
Inventory policy calculations driven by demand forecasts.

IMPORTANT — DATA vs ASSUMPTIONS
---------------------------------
The Walmart dataset contains NO inventory data. All parameters below
(lead time, service level, ordering cost, holding cost) are MODELLING
ASSUMPTIONS that must be configured by the user. They are NOT derived
from historical data.

Inventory formulas implemented
--------------------------------
1. Lead-Time Demand
   The expected demand during the replenishment lead time.
   = P50_forecast_per_week × lead_time_weeks

2. Safety Stock
   Buffer stock to absorb demand uncertainty during lead time.
   = Z(service_level) × σ_demand × √lead_time_weeks
   where σ_demand = standard deviation of weekly demand (from forecasts or history)
   and Z is the standard normal quantile at the target service level.

   Alternative: use (P90 - P50) from quantile forecasts as a non-parametric
   uncertainty estimate. Both approaches are offered.

3. Reorder Point (ROP)
   The inventory level at which a new order should be placed.
   = Lead-Time Demand + Safety Stock

4. Economic Order Quantity (EOQ)
   The optimal order size that minimises total ordering + holding costs.
   = √(2 × D × S / H)
   where D = annual demand, S = ordering cost per order, H = holding cost per unit/year

   NOTE: EOQ is a stylised model with many simplifying assumptions
   (constant demand rate, fixed lead time, no stockouts allowed).
   In real retail it serves as a starting point, not a rigid rule.

5. Stockout Risk
   Probability that actual demand exceeds the ROP during lead time.
   = 1 - Φ((ROP - μ_lead) / σ_lead)
   where Φ is the standard normal CDF.

Service Level → Z-score Lookup
--------------------------------
95% service level → Z = 1.645
98% service level → Z = 2.054
99% service level → Z = 2.326

References
----------
- Silver, Pyke & Thomas (2017). Inventory and Production Management in Supply Chains.
- Chopra & Meindl (2016). Supply Chain Management.
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
    ASSUMPTION container — all fields are user-configurable, NOT from data.

    Attributes
    ----------
    lead_time_weeks : int
        Number of weeks between placing and receiving an order.
        ASSUMPTION: 2 weeks (typical grocery/retail).

    service_level : float
        Target probability of not stocking out (0-1).
        ASSUMPTION: 0.95 (95% in-stock rate).

    ordering_cost_usd : float
        Fixed cost per order placed (USD).
        ASSUMPTION: $500 per order.

    holding_cost_pct : float
        Annual holding cost as fraction of inventory value.
        ASSUMPTION: 25% per year (includes capital, storage, obsolescence).

    current_inventory_usd : float
        Current on-hand inventory (USD). User input — not in dataset.
        ASSUMPTION: must be provided by user.
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
        assert self.ordering_cost_usd >= 0
        assert self.holding_cost_pct > 0


@dataclass
class InventoryDecision:
    """Output of the inventory optimizer for one store × horizon."""
    store: int
    lead_time_demand: float        # Expected demand during lead time
    demand_std: float              # Std dev of weekly demand
    safety_stock: float            # Buffer stock
    reorder_point: float           # ROP = lead_time_demand + safety_stock
    eoq: float                     # Economic Order Quantity
    current_inventory: float       # Current on-hand (from params)
    recommended_order: float       # max(0, ROP - current_inventory + EOQ)
    stockout_risk: float           # Probability of stockout (0-1)
    service_level: float           # Target service level (from params)
    weeks_of_coverage: float       # current_inventory / avg_weekly_demand
    params: InventoryParameters = field(repr=False)


def z_score(service_level: float) -> float:
    """Return the standard normal Z-score for a given service level."""
    return float(stats.norm.ppf(service_level))


def compute_lead_time_demand(
    weekly_forecast_p50: float,
    lead_time_weeks: int,
) -> float:
    """
    Expected demand during the replenishment lead time.

    Parameters
    ----------
    weekly_forecast_p50 : float
        Median (P50) weekly sales forecast.
    lead_time_weeks : int
        Lead time in weeks.

    Returns
    -------
    float
        Expected total demand during lead time period.
    """
    return weekly_forecast_p50 * lead_time_weeks


def compute_safety_stock_parametric(
    demand_std_weekly: float,
    lead_time_weeks: int,
    service_level: float,
) -> float:
    """
    Safety stock using the parametric (Normal distribution) formula.

    Safety Stock = Z(SL) × σ_weekly × √L
    where L = lead time in weeks.

    Assumption: weekly demand follows a Normal distribution.
    This is a reasonable approximation for aggregate store-level demand.

    Parameters
    ----------
    demand_std_weekly : float
        Standard deviation of weekly demand (USD).
    lead_time_weeks : int
        Replenishment lead time (weeks).
    service_level : float
        Target service level (e.g. 0.95).

    Returns
    -------
    float
        Safety stock in USD.
    """
    z = z_score(service_level)
    return z * demand_std_weekly * np.sqrt(lead_time_weeks)


def compute_safety_stock_quantile(
    weekly_p50: float,
    weekly_p90: float,
    lead_time_weeks: int,
) -> float:
    """
    Non-parametric safety stock using quantile forecast uncertainty.

    Safety Stock = (P90 - P50) × √lead_time_weeks

    This is a distribution-free alternative: instead of assuming Normal demand,
    we use the spread between P50 and P90 as the uncertainty measure.
    Preferred when demand may be skewed.

    Parameters
    ----------
    weekly_p50, weekly_p90 : float
        Median and 90th-percentile weekly forecast.
    lead_time_weeks : int
        Lead time in weeks.

    Returns
    -------
    float
        Safety stock in USD.
    """
    uncertainty = max(0, weekly_p90 - weekly_p50)
    return uncertainty * np.sqrt(lead_time_weeks)


def compute_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_pct: float,
    avg_item_value: float = 1.0,
) -> float:
    """
    Economic Order Quantity (Wilson formula).

    EOQ = √(2DS / H)
    where:
      D = annual demand (units or USD)
      S = ordering cost per order
      H = holding cost per unit per year = holding_cost_pct × avg_item_value

    When working in USD (no unit price), set avg_item_value=1.0 and interpret
    EOQ as dollar value of each order.

    Returns
    -------
    float
        Optimal order quantity (USD).
    """
    H = holding_cost_pct * avg_item_value
    if H <= 0 or annual_demand <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * ordering_cost / H))


def compute_stockout_risk(
    reorder_point: float,
    lead_time_demand: float,
    demand_std_lead_time: float,
) -> float:
    """
    Probability of stocking out given current ROP.

    P(demand during LT > ROP) = 1 - Φ((ROP - μ_LT) / σ_LT)

    Parameters
    ----------
    reorder_point : float
        Current reorder point.
    lead_time_demand : float
        Expected demand during lead time (μ_LT).
    demand_std_lead_time : float
        Std dev of demand during lead time (σ_LT = σ_weekly × √L).

    Returns
    -------
    float
        Stockout probability in [0, 1].
    """
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
    """
    Compute the full inventory policy for a store.

    Parameters
    ----------
    store : int
        Store identifier.
    weekly_p50 : float
        Median weekly demand forecast (USD).
    weekly_p90 : float
        90th-percentile weekly demand forecast (USD).
    demand_std_weekly : float
        Historical std dev of weekly demand for this store (USD).
        Used in parametric safety stock and stockout risk.
    params : InventoryParameters
        Configurable business parameters (all assumptions).
    use_quantile_safety_stock : bool
        If True, use (P90-P50) for safety stock (non-parametric).
        If False, use Normal formula with demand_std_weekly.

    Returns
    -------
    InventoryDecision
        Full inventory decision object.
    """
    L = params.lead_time_weeks

    # 1. Lead-time demand
    ltd = compute_lead_time_demand(weekly_p50, L)

    # 2. Safety stock
    if use_quantile_safety_stock:
        ss = compute_safety_stock_quantile(weekly_p50, weekly_p90, L)
    else:
        ss = compute_safety_stock_parametric(demand_std_weekly, L, params.service_level)

    # 3. Reorder point
    rop = ltd + ss

    # 4. EOQ
    annual_demand = weekly_p50 * params.weeks_per_year
    eoq = compute_eoq(annual_demand, params.ordering_cost_usd, params.holding_cost_pct)

    # 5. Recommended order quantity
    shortfall = max(0, rop - params.current_inventory_usd)
    recommended_order = max(0, shortfall + eoq) if shortfall > 0 else 0.0

    # 6. Stockout risk
    sigma_lt = demand_std_weekly * np.sqrt(L)
    stockout_risk = compute_stockout_risk(rop, ltd, sigma_lt)

    # 7. Weeks of coverage
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
