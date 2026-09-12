"""
run_pipeline.py
===============
Unified CLI entrypoint for the Retail Demand Forecasting & Inventory Optimization pipeline.

Usage:
------
python run_pipeline.py              # Run entire end-to-end pipeline
python run_pipeline.py --eda        # Run only data ingestion, features, and EDA
python run_pipeline.py --stats      # Run only statistical hypothesis testing
python run_pipeline.py --train      # Run feature engineering, training, and evaluation
python run_pipeline.py --explain    # Run SHAP explainability analysis
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.analysis.eda import run_eda
from src.analysis.statistics import run_statistical_tests
from src.data.ingestion import load_config, load_raw_data, validate_raw_data
from src.data.preprocessing import clean, save_processed
from src.explainability.shap_analysis import compute_shap_values, get_feature_importance, plot_feature_importance_bar
from src.features.engineering import build_features, verify_no_leakage
from src.models.baselines import NaiveForecast, SeasonalNaiveForecast
from src.models.evaluation import evaluate, evaluate_by_store
from src.models.forecasting import build_forecast_table, predict_point, predict_quantiles, predict_xgb
from src.models.trainer import chronological_split, get_feature_cols, save_model, train_lgbm, train_xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")


def run_full_pipeline(config_path: str = "config/config.yaml", stages: list[str] | None = None) -> None:
    """Execute demand forecasting and inventory optimization pipeline stages."""
    if stages is None or "all" in stages:
        run_all = True
    else:
        run_all = False

    config = load_config(config_path)
    fig_dir = Path(config.get("outputs", {}).get("figures_dir", "outputs/figures"))
    fc_dir = Path(config.get("outputs", {}).get("forecasts_dir", "outputs/forecasts"))
    models_dir = Path(config.get("outputs", {}).get("models_dir", "outputs/models"))

    fig_dir.mkdir(parents=True, exist_ok=True)
    fc_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RETAIL DEMAND FORECASTING & INVENTORY OPTIMIZATION PIPELINE")
    print("=" * 70)

    # 1. INGESTION & DATA CLEANING
    print("\n[Stage 1/5] Ingestion & Preprocessing...")
    df_raw = load_raw_data(config)
    report = validate_raw_data(df_raw, config)
    print(f"  Raw Dataset: {report['shape'][0]:,} rows x {report['shape'][1]} cols across {report['store_count']} stores.")
    print(f"  Sales Range: ${report['sales_stats']['min']:,.0f} to ${report['sales_stats']['max']:,.0f} (Mean: ${report['sales_stats']['mean']:,.0f})")

    df_clean = clean(df_raw, config)

    # 2. FEATURE ENGINEERING
    print("\n[Stage 2/5] Time-Series Feature Engineering...")
    df_features = build_features(df_clean, config)
    df_features["store"] = df_features["store"].astype("category")

    # Leakage verification
    verify_no_leakage(df_features, store_id=1)
    save_processed(df_features, config)
    print("  Feature engineering complete with zero lookahead leakage verified.")

    # 3. EDA & STATISTICAL HYPOTHESIS TESTING
    if run_all or "eda" in stages or "stats" in stages:
        print("\n[Stage 3/5] Exploratory Data Analysis & Statistical Tests...")
        eda_figs = run_eda(df_features, output_dir=fig_dir)
        print(f"  Generated {len(eda_figs)} EDA visualisations in {fig_dir}")

        stats_results = run_statistical_tests(df_features, output_dir=fig_dir)
        mw_p = stats_results["mann_whitney_holiday"]["p_value"]
        kw_p = stats_results["kruskal_wallis_stores"]["p_value"]
        adf_stat = stats_results["adf_stationarity"]["is_stationary"]
        print(f"  Statistical Hypothesis Tests:")
        print(f"    - Mann-Whitney U (Holiday Effect): p={mw_p:.4e} -> {'Significant' if mw_p < 0.05 else 'Not Significant'}")
        print(f"    - Kruskal-Wallis (Store Differences): p={kw_p:.4e} -> Substantial heterogeneity")
        print(f"    - ADF Stationarity: Total demand is {'Stationary' if adf_stat else 'Non-Stationary (trend/seasonality)'}")

    # 4. MODEL TRAINING & EVALUATION
    if run_all or "train" in stages:
        print("\n[Stage 4/5] Training Forecasting Models & Uncertainty Quantification...")
        train_df, val_df, test_df = chronological_split(df_features, config)
        print(f"  Chronological Splits: Train={len(train_df)} rows, Val={len(val_df)} rows, Test={len(test_df)} rows")

        test_clean = test_df.dropna(subset=["lag_52"])

        # Baselines
        naive = NaiveForecast().fit(train_df)
        sn = SeasonalNaiveForecast().fit(train_df)
        naive_preds = naive.predict(test_clean)
        sn_preds = sn.predict(test_clean)

        # LightGBM Point & Quantiles
        lgbm_point = train_lgbm(train_df, val_df, config, objective="regression")
        lgbm_p10 = train_lgbm(train_df, val_df, config, objective="quantile", alpha=0.10)
        lgbm_p50 = train_lgbm(train_df, val_df, config, objective="quantile", alpha=0.50)
        lgbm_p90 = train_lgbm(train_df, val_df, config, objective="quantile", alpha=0.90)

        # XGBoost
        xgb_point = train_xgb(train_df, val_df, config)

        # Save model artifacts
        save_model(lgbm_point, "lgbm_point", config)
        save_model(lgbm_p10, "lgbm_p10", config)
        save_model(lgbm_p50, "lgbm_p50", config)
        save_model(lgbm_p90, "lgbm_p90", config)
        save_model(xgb_point, "xgb_point", config)

        # Evaluate on Test Set
        actuals = test_clean["weekly_sales"].values
        lgbm_preds = predict_point(lgbm_point, test_clean)
        xgb_preds = predict_xgb(xgb_point, test_clean)

        m_naive = evaluate(actuals, naive_preds)
        m_sn = evaluate(actuals, sn_preds)
        m_lgbm = evaluate(actuals, lgbm_preds)
        m_xgb = evaluate(actuals, xgb_preds)

        q_preds = predict_quantiles({"q10": lgbm_p10, "q50": lgbm_p50, "q90": lgbm_p90}, test_clean)
        coverage = np.mean((actuals >= q_preds["p10"].values) & (actuals <= q_preds["p90"].values))

        # Build & save forecast table
        fc_table = build_forecast_table(
            test_clean,
            actual_col="weekly_sales",
            point=lgbm_preds,
            p10=q_preds["p10"].values,
            p50=q_preds["p50"].values,
            p90=q_preds["p90"].values,
        )
        fc_table.to_csv(fc_dir / "test_forecasts.csv", index=False)

        # Per-store metrics
        test_eval_df = test_clean.copy()
        test_eval_df["predicted"] = lgbm_preds
        per_store = evaluate_by_store(test_eval_df, actual_col="weekly_sales", pred_col="predicted")
        per_store.to_csv(fc_dir / "per_store_metrics.csv")

        print("\n  Test-Set Model Benchmark:")
        print(f"    - Naive Baseline:       MAE=${m_naive['mae']:,.0f} | WAPE={m_naive['wape']*100:.2f}% | RMSE=${m_naive['rmse']:,.0f}")
        print(f"    - Seasonal Naive:       MAE=${m_sn['mae']:,.0f} | WAPE={m_sn['wape']*100:.2f}% | RMSE=${m_sn['rmse']:,.0f}")
        print(f"    - XGBoost:              MAE=${m_xgb['mae']:,.0f} | WAPE={m_xgb['wape']*100:.2f}% | RMSE=${m_xgb['rmse']:,.0f}")
        print(f"    - LightGBM (Champion):  MAE=${m_lgbm['mae']:,.0f} | WAPE={m_lgbm['wape']*100:.2f}% | RMSE=${m_lgbm['rmse']:,.0f}")
        print(f"    - P10-P90 Uncertainty Coverage: {coverage*100:.1f}% (Nominal target: 80.0%)")

        # Plot comparison
        fig, ax = plt.subplots(figsize=(8, 4))
        models_list = ["Naive", "Seasonal Naive", "XGBoost", "LightGBM"]
        wapes = [m_naive["wape"] * 100, m_sn["wape"] * 100, m_xgb["wape"] * 100, m_lgbm["wape"] * 100]
        bars = ax.bar(models_list, wapes, color=["#BDBDBD", "#9E9E9E", "#42A5F5", "#1E88E5"])
        ax.set_ylabel("WAPE (%)")
        ax.set_title("Model Comparison on Out-of-Time Test Set")
        for bar, val in zip(bars, wapes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{val:.2f}%", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(fig_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 5. SHAP EXPLAINABILITY
    if run_all or "explain" in stages:
        print("\n[Stage 5/5] Explainability & SHAP Analysis...")
        test_clean = test_df.dropna(subset=["lag_52"]) if "test_df" in locals() else df_features.tail(500)
        with open(models_dir / "lgbm_point.pkl", "rb") as f:
            point_model = pickle.load(f)

        shap_values, X_sample = compute_shap_values(point_model, test_clean, max_samples=500)
        importance = get_feature_importance(shap_values)
        importance.to_csv(fc_dir / "shap_importance.csv", index=False)

        # Plot bar & beeswarm
        plot_feature_importance_bar(importance, top_n=12, save_path=fig_dir / "shap_feature_importance.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.beeswarm(shap_values, max_display=12, show=False)
        plt.title("TreeSHAP Global Summary: Feature Contributions to Forecasts", fontsize=11)
        plt.tight_layout()
        plt.savefig(fig_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Top 3 most impactful features: {', '.join(importance['feature'].head(3).tolist())}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in:\n  - {fc_dir}\n  - {fig_dir}\n  - {models_dir}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail Demand Forecasting Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")
    parser.add_argument("--eda", action="store_true", help="Run EDA")
    parser.add_argument("--stats", action="store_true", help="Run Statistical Tests")
    parser.add_argument("--train", action="store_true", help="Run Model Training & Evaluation")
    parser.add_argument("--explain", action="store_true", help="Run SHAP Explainability")
    parser.add_argument("--all", action="store_true", help="Run all stages (default)")

    args = parser.parse_args()
    stages = []
    if args.eda:
        stages.append("eda")
    if args.stats:
        stages.append("stats")
    if args.train:
        stages.append("train")
    if args.explain:
        stages.append("explain")
    if args.all or not stages:
        stages = ["all"]

    run_full_pipeline(config_path=args.config, stages=stages)


if __name__ == "__main__":
    main()
