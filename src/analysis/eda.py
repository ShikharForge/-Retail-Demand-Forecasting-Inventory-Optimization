"""
src/analysis/eda.py
===================
Exploratory Data Analysis routines and visualizations for Walmart retail sales.
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
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

PALETTE = "#2196F3"
HOLIDAY_COLOR = "#FF5722"


def run_eda(df: pd.DataFrame, output_dir: str | Path = "outputs/figures") -> list[Path]:
    """
    Generate all 8 core EDA charts and save them to output_dir.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with engineered features.
    output_dir : str or Path
        Directory where figures will be saved.

    Returns
    -------
    list of Path
        Paths to generated figure files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

    # CHART 1: Overall Trend
    p1 = out_dir / "eda_01_overall_trend.png"
    weekly_agg = df.groupby("date", observed=True)["weekly_sales"].sum().reset_index()
    holiday_weeks = df[df["holiday_flag"] == 1]["date"].unique()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(weekly_agg["date"], weekly_agg["weekly_sales"] / 1e6, color=PALETTE, linewidth=1.5, label="Total Weekly Sales")
    ax.fill_between(weekly_agg["date"], weekly_agg["weekly_sales"] / 1e6, alpha=0.1, color=PALETTE)
    for hw in holiday_weeks:
        ax.axvline(hw, color=HOLIDAY_COLOR, alpha=0.35, linewidth=1.2)
    if len(holiday_weeks) > 0:
        ax.axvline(holiday_weeks[0], color=HOLIDAY_COLOR, alpha=0.35, linewidth=1.2, label="Holiday week")
    ax.set_title("Overall Retail Demand Trend Across All 45 Stores", fontsize=12, style="italic")
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Sales ($M)")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}M"))
    plt.tight_layout()
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    generated_files.append(p1)

    # CHART 2: Store Distribution
    p2 = out_dir / "eda_02_store_distribution.png"
    store_order = (df.groupby("store", observed=True)["weekly_sales"].median().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(16, 6))
    bp_data = [df[df["store"] == s]["weekly_sales"].values / 1e6 for s in store_order]
    bp = ax.boxplot(bp_data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE)
        patch.set_alpha(0.7)
    ax.set_xticklabels([str(s) for s in store_order], fontsize=8, rotation=45)
    ax.set_xlabel("Store (sorted by median sales)")
    ax.set_ylabel("Weekly Sales ($M)")
    ax.set_title("Weekly Sales Distribution by Store")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}M"))
    plt.tight_layout()
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    generated_files.append(p2)

    # CHART 3: Holiday Impact
    p3 = out_dir / "eda_03_holiday_impact.png"
    holiday_sales = df[df["holiday_flag"] == 1]["weekly_sales"] / 1e6
    non_holiday_sales = df[df["holiday_flag"] == 0]["weekly_sales"] / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    parts = axes[0].violinplot([non_holiday_sales.values, holiday_sales.values], positions=[1, 2], showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#90CAF9", HOLIDAY_COLOR][i])
        pc.set_alpha(0.7)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["Non-Holiday", "Holiday"])
    axes[0].set_ylabel("Weekly Sales ($M)")
    axes[0].set_title("Sales Distribution by Holiday Status")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}M"))

    holiday_types = {
        "Super Bowl": df.get("is_superbowl", pd.Series(0, index=df.index)).mul(df["weekly_sales"]).sum() / max(1, df.get("is_superbowl", pd.Series(0)).sum()) / 1e6,
        "Labour Day": df.get("is_laborday", pd.Series(0, index=df.index)).mul(df["weekly_sales"]).sum() / max(1, df.get("is_laborday", pd.Series(0)).sum()) / 1e6,
        "Thanksgiving": df.get("is_thanksgiving", pd.Series(0, index=df.index)).mul(df["weekly_sales"]).sum() / max(1, df.get("is_thanksgiving", pd.Series(0)).sum()) / 1e6,
        "Christmas": df.get("is_christmas", pd.Series(0, index=df.index)).mul(df["weekly_sales"]).sum() / max(1, df.get("is_christmas", pd.Series(0)).sum()) / 1e6,
        "Non-Holiday": float(non_holiday_sales.mean()),
    }
    colors = [HOLIDAY_COLOR] * 4 + ["#90CAF9"]
    bars = axes[1].bar(holiday_types.keys(), holiday_types.values(), color=colors, alpha=0.8)
    axes[1].set_ylabel("Mean Weekly Sales ($M)")
    axes[1].set_title("Mean Sales by Event Type")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}M"))
    for bar, val in zip(bars, holiday_types.values()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"${val:.2f}M", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    generated_files.append(p3)

    # CHART 4: Seasonality
    p4 = out_dir / "eda_04_seasonality.png"
    if "week_of_year" in df.columns:
        seasonal = df.groupby("week_of_year", observed=True)["weekly_sales"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(seasonal["week_of_year"], seasonal["weekly_sales"] / 1e6, color=PALETTE, linewidth=2, marker="o", markersize=3)
        ax.fill_between(seasonal["week_of_year"], seasonal["weekly_sales"] / 1e6, alpha=0.1, color=PALETTE)
        ax.set_xlabel("Week of Year")
        ax.set_ylabel("Mean Weekly Sales ($M)")
        ax.set_title("Annual Demand Seasonality (All 45 Stores)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}M"))
        plt.tight_layout()
        plt.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close()
        generated_files.append(p4)

    # CHART 5: Economic Correlations
    p5 = out_dir / "eda_05_economic_correlations.png"
    econ_vars = [v for v in ["temperature", "fuel_price", "cpi", "unemployment"] if v in df.columns]
    if len(econ_vars) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()
        for i, var in enumerate(econ_vars):
            sub = df[["weekly_sales", var]].dropna()
            sample = sub.sample(min(1000, len(sub)), random_state=42)
            axes[i].scatter(sample[var], sample["weekly_sales"] / 1e6, alpha=0.3, s=8, color=PALETTE)
            z = np.polyfit(sample[var], sample["weekly_sales"] / 1e6, 1)
            p = np.poly1d(z)
            x_line = np.linspace(sample[var].min(), sample[var].max(), 100)
            axes[i].plot(x_line, p(x_line), "r-", linewidth=1.5, alpha=0.8)
            r, pval = spearmanr(sample[var], sample["weekly_sales"])
            axes[i].set_xlabel(var.replace("_", " ").title())
            axes[i].set_ylabel("Weekly Sales ($M)")
            axes[i].set_title(f"Spearman ρ = {r:.3f} (p={pval:.3f})")
            axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}M"))
        plt.tight_layout()
        plt.savefig(p5, dpi=150, bbox_inches="tight")
        plt.close()
        generated_files.append(p5)

    # CHART 6: Top 5 vs Bottom 5
    p6 = out_dir / "eda_06_top_bottom_stores.png"
    store_means = df.groupby("store", observed=True)["weekly_sales"].mean().reset_index().rename(columns={"weekly_sales": "mean_sales"})
    store_means = store_means.sort_values("mean_sales", ascending=False)
    top5 = store_means.head(5)
    bot5 = store_means.tail(5)
    highlight = pd.concat([top5, bot5])
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_bar = ["#1565C0"] * 5 + ["#B71C1C"] * 5
    ax.barh(highlight["store"].astype(str)[::-1], highlight["mean_sales"][::-1] / 1e6, color=colors_bar[::-1], alpha=0.85)
    ax.set_xlabel("Mean Weekly Sales ($M)")
    ax.set_title("Store Revenue Disparity: Top 5 vs Bottom 5 Stores")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}M"))
    plt.tight_layout()
    plt.savefig(p6, dpi=150, bbox_inches="tight")
    plt.close()
    generated_files.append(p6)

    # CHART 7: Demand Volatility (CV)
    p7 = out_dir / "eda_07_demand_volatility.png"
    store_stats = df.groupby("store", observed=True)["weekly_sales"].agg(["mean", "std"]).reset_index()
    store_stats["cv"] = store_stats["std"] / store_stats["mean"]
    fig, ax = plt.subplots(figsize=(14, 5))
    bar_colors = [HOLIDAY_COLOR if cv > store_stats["cv"].quantile(0.75) else PALETTE for cv in store_stats["cv"]]
    ax.bar(store_stats["store"].astype(str), store_stats["cv"] * 100, color=bar_colors, alpha=0.8)
    ax.axhline(store_stats["cv"].mean() * 100, color="black", linestyle="--", label=f"Mean CV = {store_stats['cv'].mean()*100:.1f}%")
    ax.set_xlabel("Store")
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title("Demand Volatility by Store (High CV requires higher safety stock)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(p7, dpi=150, bbox_inches="tight")
    plt.close()
    generated_files.append(p7)

    # CHART 8: Lag Correlations
    p8 = out_dir / "eda_08_lag_correlations.png"
    if "lag_1" in df.columns and "lag_52" in df.columns:
        sample = df[["weekly_sales", "lag_1", "lag_4", "lag_52"]].dropna()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        lag_pairs = [("lag_1", "Lag 1 (1 wk)"), ("lag_4", "Lag 4 (4 wks)"), ("lag_52", "Lag 52 (52 wks)")]
        for ax, (lag_col, lag_label) in zip(axes, lag_pairs):
            sample_plot = sample.sample(min(1000, len(sample)), random_state=42)
            ax.scatter(sample_plot[lag_col] / 1e6, sample_plot["weekly_sales"] / 1e6, alpha=0.25, s=8, color=PALETTE)
            r = sample[lag_col].corr(sample["weekly_sales"])
            z = np.polyfit(sample_plot[lag_col], sample_plot["weekly_sales"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sample_plot[lag_col].min(), sample_plot[lag_col].max(), 100)
            ax.plot(x_line / 1e6, p(x_line) / 1e6, "r-", linewidth=1.5)
            ax.set_xlabel(f"{lag_label} ($M)")
            ax.set_ylabel("Sales ($M)")
            ax.set_title(f"Pearson r = {r:.3f}")
        plt.tight_layout()
        plt.savefig(p8, dpi=150, bbox_inches="tight")
        plt.close()
        generated_files.append(p8)

    logger.info("Generated %d EDA figures in %s", len(generated_files), out_dir)
    return generated_files
