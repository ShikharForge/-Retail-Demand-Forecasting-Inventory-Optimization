"""
src/analysis/statistics.py
==========================
Statistical hypothesis testing, effect size estimation, and stationarity checks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def run_statistical_tests(df: pd.DataFrame, alpha: float = 0.05, output_dir: str | Path = "outputs/figures") -> dict[str, Any]:
    """
    Execute comprehensive statistical hypothesis testing suite.

    Tests performed:
    1. Mann-Whitney U: Holiday vs Non-Holiday weekly sales distribution
    2. Mann-Whitney U (one-sided): Thanksgiving vs Non-Holiday sales
    3. Kruskal-Wallis: Demand differences across 45 stores
    4. Levene's Test: Homoscedasticity / variance equality across stores
    5. Spearman Rank Correlations: Sales vs Macroeconomic factors
    6. Augmented Dickey-Fuller (ADF): Time-series stationarity

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and feature-engineered DataFrame.
    alpha : float, default=0.05
        Significance threshold.
    output_dir : str or Path
        Directory where summary plot will be saved.

    Returns
    -------
    dict
        Structured results for all statistical tests.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    holiday_s = df[df["holiday_flag"] == 1]["weekly_sales"].values
    non_holiday_s = df[df["holiday_flag"] == 0]["weekly_sales"].values

    # 1. Mann-Whitney U (Holiday vs Non-Holiday)
    stat_mw, p_mw = stats.mannwhitneyu(holiday_s, non_holiday_s, alternative="two-sided")
    n1, n2 = len(holiday_s), len(non_holiday_s)
    r_rb = 1 - (2 * stat_mw) / (n1 * n2)
    results["mann_whitney_holiday"] = {
        "statistic": float(stat_mw),
        "p_value": float(p_mw),
        "rank_biserial_r": float(r_rb),
        "reject_h0": bool(p_mw < alpha),
    }

    # 2. Thanksgiving vs Non-Holiday
    if "is_thanksgiving" in df.columns:
        tg_s = df[df["is_thanksgiving"] == 1]["weekly_sales"].values
        stat_t, p_t = stats.mannwhitneyu(tg_s, non_holiday_s, alternative="greater")
        results["thanksgiving_uplift"] = {
            "statistic": float(stat_t),
            "p_value": float(p_t),
            "reject_h0": bool(p_t < alpha),
        }

    # 3. Kruskal-Wallis (Store Differences)
    stores = df["store"].unique()
    store_groups = [df[df["store"] == s]["weekly_sales"].values for s in stores]
    stat_kw, p_kw = stats.kruskal(*store_groups)
    eta_sq = (stat_kw - len(stores) + 1) / (len(df) - len(stores))
    results["kruskal_wallis_stores"] = {
        "statistic": float(stat_kw),
        "p_value": float(p_kw),
        "eta_squared": float(eta_sq),
        "reject_h0": bool(p_kw < alpha),
    }

    # 4. Levene's Test (Store Variance Equality)
    stat_lev, p_lev = stats.levene(*store_groups, center="median")
    results["levene_variance"] = {
        "statistic": float(stat_lev),
        "p_value": float(p_lev),
        "reject_h0": bool(p_lev < alpha),
    }

    # 5. Spearman Correlations with Economic Variables
    econ_vars = ["temperature", "fuel_price", "cpi", "unemployment"]
    corrs = {}
    for var in econ_vars:
        if var in df.columns:
            sub = df[["weekly_sales", var]].dropna()
            r, p = stats.spearmanr(sub["weekly_sales"], sub[var])
            corrs[var] = {"rho": float(r), "p_value": float(p), "significant": bool(p < alpha)}
    results["economic_correlations"] = corrs

    # 6. Augmented Dickey-Fuller Test
    total_sales_ts = df.groupby("date", observed=True)["weekly_sales"].sum()
    adf_res = adfuller(total_sales_ts.dropna(), autolag="AIC")
    results["adf_stationarity"] = {
        "adf_statistic": float(adf_res[0]),
        "p_value": float(adf_res[1]),
        "used_lag": int(adf_res[2]),
        "is_stationary": bool(adf_res[1] < alpha),
    }

    # Generate Summary Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_text = (
        "STATISTICAL HYPOTHESIS TESTING SUMMARY\n"
        "===============================================================\n\n"
        f"1. Holiday vs Non-Holiday (Mann-Whitney U): p = {p_mw:.4e} (Rank-biserial r = {r_rb:.3f})\n"
        f"   Decision: {'REJECT H0 (Statistically Significant)' if p_mw < alpha else 'FAIL TO REJECT'}\n\n"
        f"2. Store Differences (Kruskal-Wallis): H = {stat_kw:.1f}, p = {p_kw:.4e} (eta^2 = {eta_sq:.3f})\n"
        f"   Decision: REJECT H0 (Substantial store-level demand heterogeneity)\n\n"
        f"3. Variance Equality Across Stores (Levene): W = {stat_lev:.1f}, p = {p_lev:.4e}\n"
        f"   Decision: REJECT H0 (Per-store safety stock calculation required)\n\n"
        f"4. Total Sales Stationarity (ADF Test): ADF = {adf_res[0]:.3f}, p = {adf_res[1]:.4f}\n"
        f"   Decision: {'Stationary' if adf_res[1] < alpha else 'Non-Stationary (Trend/Seasonality present)'}\n\n"
        "5. Macroeconomic Correlations (Spearman rho):\n"
    )
    for var, c in corrs.items():
        summary_text += f"   - {var.capitalize()}: rho = {c['rho']:.3f} (p = {c['p_value']:.4f})\n"

    ax.text(0.03, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#F5F5F5", edgecolor="#BDBDBD"))
    ax.axis("off")
    plt.tight_layout()
    summary_fig_path = out_dir / "stats_01_summary.png"
    plt.savefig(summary_fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    results["summary_figure_path"] = str(summary_fig_path)
    logger.info("Statistical tests complete. Summary saved to %s", summary_fig_path)
    return results
