"""
tests/test_dashboard.py
=======================
Unit and dry-run tests for Streamlit dashboard pages.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.ingestion import load_config, load_raw_data
from src.data.preprocessing import clean
from src.features.engineering import build_features
from src.inventory.optimization import InventoryParameters, optimise
from src.inventory.scenarios import all_scenarios


@pytest.fixture(scope="module")
def project_data():
    config = load_config("config/config.yaml")
    df_raw = load_raw_data(config)
    df = build_features(clean(df_raw, config), config)
    df["store"] = df["store"].astype("category")
    return df, config


class TestDashboardCompilation:
    def test_all_pages_syntax(self):
        """Compile every dashboard script to ensure zero syntax errors."""
        dashboard_dir = Path("dashboard")
        py_files = list(dashboard_dir.rglob("*.py"))
        assert len(py_files) >= 7, f"Expected at least 7 dashboard files, found {len(py_files)}"

        for p in py_files:
            src = p.read_text(encoding="utf-8")
            compile(src, str(p), "exec")


class TestDashboardLogic:
    def test_executive_overview_kpis(self, project_data):
        df, _ = project_data
        total_sales = df["weekly_sales"].sum()
        avg_weekly = df.groupby("date", observed=True)["weekly_sales"].sum().mean()
        assert total_sales > 0
        assert avg_weekly > 0

    def test_inventory_optimization_computation(self, project_data):
        df, _ = project_data
        store_df = df[df["store"].astype(int) == 1]
        p50 = float(store_df["weekly_sales"].median())
        p90 = float(store_df["weekly_sales"].quantile(0.90))
        std = float(store_df["weekly_sales"].std())

        params = InventoryParameters(lead_time_weeks=2, service_level=0.95, current_inventory_usd=100000.0)
        dec = optimise(store=1, weekly_p50=p50, weekly_p90=p90, demand_std_weekly=std, params=params)

        assert dec.safety_stock > 0
        assert dec.reorder_point > dec.lead_time_demand
        assert 0.0 <= dec.stockout_risk <= 1.0

    def test_scenarios_computation(self, project_data):
        df, _ = project_data
        store_df = df[df["store"].astype(int) == 1]
        p50 = float(store_df["weekly_sales"].median())
        p90 = float(store_df["weekly_sales"].quantile(0.90))
        std = float(store_df["weekly_sales"].std())

        params = InventoryParameters(lead_time_weeks=2, service_level=0.95)
        scenarios = all_scenarios(store=1, weekly_p50=p50, weekly_p90=p90, demand_std=std, params=params)

        assert len(scenarios) == 5
        for sc in scenarios:
            delta = sc.delta()
            assert "safety_stock" in delta
            assert "reorder_point" in delta
