"""
tests/test_inventory.py
========================
Unit tests for inventory optimisation formulas.

Tests verify mathematical correctness of:
- Safety stock (parametric and quantile-based)
- Reorder point
- EOQ
- Stockout risk
"""

import numpy as np
import pytest

from src.inventory.optimizer import (
    InventoryParameters,
    compute_eoq,
    compute_lead_time_demand,
    compute_safety_stock_parametric,
    compute_safety_stock_quantile,
    compute_stockout_risk,
    optimise,
    z_score,
)


class TestZScore:
    def test_95_service_level(self):
        z = z_score(0.95)
        assert abs(z - 1.6449) < 0.001

    def test_99_service_level(self):
        z = z_score(0.99)
        assert abs(z - 2.3263) < 0.001

    def test_50_service_level(self):
        z = z_score(0.50)
        assert abs(z - 0.0) < 0.001


class TestLeadTimeDemand:
    def test_basic(self):
        ltd = compute_lead_time_demand(weekly_forecast_p50=1000.0, lead_time_weeks=2)
        assert ltd == 2000.0

    def test_single_week(self):
        ltd = compute_lead_time_demand(weekly_forecast_p50=500.0, lead_time_weeks=1)
        assert ltd == 500.0


class TestSafetyStockParametric:
    def test_zero_std(self):
        ss = compute_safety_stock_parametric(
            demand_std_weekly=0, lead_time_weeks=2, service_level=0.95
        )
        assert ss == 0.0

    def test_95sl_2weeks(self):
        # SS = 1.6449 * 200 * sqrt(2) = 465.3
        ss = compute_safety_stock_parametric(200.0, lead_time_weeks=2, service_level=0.95)
        expected = 1.6449 * 200.0 * np.sqrt(2)
        assert abs(ss - expected) < 1.0

    def test_higher_sl_higher_ss(self):
        ss95 = compute_safety_stock_parametric(200.0, 2, 0.95)
        ss99 = compute_safety_stock_parametric(200.0, 2, 0.99)
        assert ss99 > ss95

    def test_longer_lt_higher_ss(self):
        ss2 = compute_safety_stock_parametric(200.0, 2, 0.95)
        ss4 = compute_safety_stock_parametric(200.0, 4, 0.95)
        assert ss4 > ss2


class TestSafetyStockQuantile:
    def test_basic(self):
        # SS = (P90 - P50) * sqrt(L) = (1200 - 1000) * sqrt(2) = 282.84
        ss = compute_safety_stock_quantile(
            weekly_p50=1000.0, weekly_p90=1200.0, lead_time_weeks=2
        )
        expected = 200.0 * np.sqrt(2)
        assert abs(ss - expected) < 0.01

    def test_no_uncertainty(self):
        ss = compute_safety_stock_quantile(
            weekly_p50=1000.0, weekly_p90=1000.0, lead_time_weeks=2
        )
        assert ss == 0.0

    def test_p90_less_than_p50_gives_zero(self):
        ss = compute_safety_stock_quantile(
            weekly_p50=1000.0, weekly_p90=900.0, lead_time_weeks=2
        )
        assert ss == 0.0


class TestEOQ:
    def test_classic_example(self):
        # Classic textbook: D=1000, S=10, H=2 → EOQ = sqrt(2*1000*10/2) = 100
        eoq = compute_eoq(annual_demand=1000, ordering_cost=10, holding_cost_pct=1.0, avg_item_value=2.0)
        assert abs(eoq - 100.0) < 0.1

    def test_zero_demand(self):
        eoq = compute_eoq(0, 500, 0.25)
        assert eoq == 0.0

    def test_eoq_scales_with_demand(self):
        eoq_low = compute_eoq(10000, 500, 0.25)
        eoq_high = compute_eoq(40000, 500, 0.25)
        # EOQ scales with sqrt(D), so doubling D gives sqrt(2) × EOQ
        assert abs(eoq_high / eoq_low - 2.0) < 0.01


class TestStockoutRisk:
    def test_rop_equals_ltd_gives_50pct(self):
        # If ROP = mean LTD and no uncertainty → 50% stockout risk
        risk = compute_stockout_risk(
            reorder_point=1000.0, lead_time_demand=1000.0, demand_std_lead_time=100.0
        )
        assert abs(risk - 0.5) < 0.01

    def test_high_rop_low_risk(self):
        risk = compute_stockout_risk(2000.0, 1000.0, 100.0)
        assert risk < 0.01

    def test_low_rop_high_risk(self):
        risk = compute_stockout_risk(0.0, 1000.0, 100.0)
        assert risk > 0.99

    def test_zero_std(self):
        risk = compute_stockout_risk(1500.0, 1000.0, 0.0)
        assert risk == 0.0  # ROP > LTD, no uncertainty → no stockout

    def test_zero_std_shortfall(self):
        risk = compute_stockout_risk(500.0, 1000.0, 0.0)
        assert risk == 1.0  # ROP < LTD, no uncertainty → certain stockout


class TestOptimise:
    def test_output_types(self):
        params = InventoryParameters(lead_time_weeks=2, service_level=0.95)
        decision = optimise(
            store=1,
            weekly_p50=1_000_000.0,
            weekly_p90=1_200_000.0,
            demand_std_weekly=100_000.0,
            params=params,
        )
        assert decision.safety_stock >= 0
        assert decision.reorder_point >= decision.lead_time_demand
        assert 0 <= decision.stockout_risk <= 1

    def test_reorder_point_formula(self):
        params = InventoryParameters(lead_time_weeks=2, service_level=0.95)
        decision = optimise(
            store=1,
            weekly_p50=1_000_000.0,
            weekly_p90=1_200_000.0,
            demand_std_weekly=100_000.0,
            params=params,
        )
        expected_rop = decision.lead_time_demand + decision.safety_stock
        assert abs(decision.reorder_point - expected_rop) < 1.0

    def test_invalid_params(self):
        with pytest.raises(AssertionError):
            InventoryParameters(service_level=1.5)
        with pytest.raises(AssertionError):
            InventoryParameters(lead_time_weeks=0)
