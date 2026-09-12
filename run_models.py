"""
Phase 4 & 5: Train all forecasting models and generate quantile forecasts.
Runs on actual data and saves model artifacts + forecast CSVs.
"""
import sys, warnings, logging
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from src.data.loader import load_config, load_raw_data
from src.data.preprocessor import clean
from src.features.engineer import build_features
from src.models.baselines import NaiveForecast, SeasonalNaiveForecast
from src.models.trainer import (
    train_lgbm, train_xgb, chronological_split,
    get_feature_cols, save_model
)
from src.models.evaluator import evaluate, evaluate_by_store
from src.models.forecaster import predict_point, predict_xgb, predict_quantiles, build_forecast_table

config = load_config('config/config.yaml')
fig_dir = Path('outputs/figures')
fc_dir = Path('outputs/forecasts')
fig_dir.mkdir(parents=True, exist_ok=True)
fc_dir.mkdir(parents=True, exist_ok=True)

print("Loading and engineering features...")
df_raw = load_raw_data(config)
df = build_features(clean(df_raw, config), config)
df['store'] = df['store'].astype('category')

# ── Chronological Split ───────────────────────────────────────────────────
train_df, val_df, test_df = chronological_split(df, config)
print(f"Train: {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"Val:   {len(val_df)} rows ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
print(f"Test:  {len(test_df)} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

# ── Baselines ─────────────────────────────────────────────────────────────
print("\n--- Training Baselines ---")
naive = NaiveForecast().fit(train_df)
seasonal_naive = SeasonalNaiveForecast().fit(train_df)

test_clean = test_df.dropna(subset=['lag_52'])  # SN needs lag_52
naive_preds = naive.predict(test_clean)
sn_preds = seasonal_naive.predict(test_clean)

m_naive = evaluate(test_clean['weekly_sales'].values, naive_preds)
m_sn = evaluate(test_clean['weekly_sales'].values, sn_preds)
print(f"Naive          - MAE={m_naive['mae']:,.0f}  RMSE={m_naive['rmse']:,.0f}  WAPE={m_naive['wape']:.4f}  MAPE={m_naive['mape']:.4f}")
print(f"Seasonal Naive - MAE={m_sn['mae']:,.0f}  RMSE={m_sn['rmse']:,.0f}  WAPE={m_sn['wape']:.4f}  MAPE={m_sn['mape']:.4f}")

# ── LightGBM Point Forecast (P50) ────────────────────────────────────────
print("\n--- Training LightGBM (point forecast) ---")
lgbm_model = train_lgbm(train_df, val_df, config, objective='regression')
save_model(lgbm_model, 'lgbm_point', config)

lgbm_preds_test = predict_point(lgbm_model, test_clean)
m_lgbm = evaluate(test_clean['weekly_sales'].values, lgbm_preds_test)
print(f"LightGBM Point - MAE={m_lgbm['mae']:,.0f}  RMSE={m_lgbm['rmse']:,.0f}  WAPE={m_lgbm['wape']:.4f}  MAPE={m_lgbm['mape']:.4f}")

# ── XGBoost Point Forecast ────────────────────────────────────────────────
print("\n--- Training XGBoost (point forecast) ---")
xgb_model = train_xgb(train_df, val_df, config)
save_model(xgb_model, 'xgb_point', config)

xgb_preds_test = predict_xgb(xgb_model, test_clean)
m_xgb = evaluate(test_clean['weekly_sales'].values, xgb_preds_test)
print(f"XGBoost Point  - MAE={m_xgb['mae']:,.0f}  RMSE={m_xgb['rmse']:,.0f}  WAPE={m_xgb['wape']:.4f}  MAPE={m_xgb['mape']:.4f}")

# ── LightGBM Quantile Models (P10, P50, P90) ──────────────────────────────
print("\n--- Training LightGBM Quantile Models (P10, P50, P90) ---")
quantile_models = {}
for q, alpha in [('p10', 0.10), ('p50', 0.50), ('p90', 0.90)]:
    print(f"  Training quantile {q} (alpha={alpha})...")
    qmodel = train_lgbm(train_df, val_df, config, objective='quantile', alpha=alpha)
    save_model(qmodel, f'lgbm_{q}', config)
    quantile_models[q] = qmodel
    print(f"  {q} model trained and saved.")

# Generate quantile forecasts on test set
quantile_preds = predict_quantiles(quantile_models, test_clean)
# Quantile coverage check
actual = test_clean['weekly_sales'].values
p10 = quantile_preds['p10'].values
p90 = quantile_preds['p90'].values
coverage = np.mean((actual >= p10) & (actual <= p90))
print(f"\nQuantile interval [P10, P90] coverage on test: {coverage:.3f} (target ~0.80)")

# ── Model Comparison Table ────────────────────────────────────────────────
print("\n" + "="*65)
print("MODEL COMPARISON — TEST SET RESULTS")
print("="*65)
print(f"{'Model':<22} {'MAE':>12} {'RMSE':>12} {'WAPE':>8} {'MAPE':>8}")
print("-"*65)
results = {
    'Naive': m_naive,
    'Seasonal Naive': m_sn,
    'LightGBM': m_lgbm,
    'XGBoost': m_xgb,
}
for name, m in results.items():
    print(f"{name:<22} ${m['mae']:>11,.0f} ${m['rmse']:>11,.0f} {m['wape']:>7.4f} {m['mape']:>7.4f}")

best = min(results, key=lambda k: results[k]['wape'])
print(f"\n>>> Best model by WAPE: {best}")
print(f"    LightGBM vs Seasonal Naive improvement: {(m_sn['wape'] - m_lgbm['wape'])/m_sn['wape']*100:.1f}% WAPE reduction")

# ── Save Forecast Table ───────────────────────────────────────────────────
fc_table = build_forecast_table(test_clean.reset_index(drop=True),
                                lgbm_preds_test,
                                quantile_preds.reset_index(drop=True))
fc_table.to_csv(fc_dir / 'test_forecasts.csv', index=False)
print(f"\nForecast table saved: {len(fc_table)} rows")

# Per-store metrics
store_metrics = evaluate_by_store(
    fc_table.assign(actual=test_clean['weekly_sales'].values),
    'actual', 'forecast'
)
store_metrics.to_csv(fc_dir / 'per_store_metrics.csv')
print("Per-store metrics saved.")

# ── CHART: Actual vs Forecast for Store 1 and Store 20 ───────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
stores_to_plot = [1, 20]

for ax, sid in zip(axes, stores_to_plot):
    # Full history for context
    hist = df[df['store'].astype(int)==sid].copy()
    test_s = fc_table[fc_table['store'].astype(int)==sid]
    test_actual = test_clean[test_clean['store'].astype(int)==sid]

    ax.plot(hist['date'], hist['weekly_sales']/1e6, color='#BDBDBD',
            linewidth=0.8, label='Historical')
    ax.plot(test_actual['date'], test_actual['weekly_sales']/1e6,
            color='#1565C0', linewidth=2, label='Actual (test)')
    ax.plot(test_s['date'], test_s['forecast']/1e6,
            color='#FF5722', linewidth=2, linestyle='--', label='LightGBM Forecast')
    if 'p10' in test_s.columns and 'p90' in test_s.columns:
        ax.fill_between(test_s['date'],
                        test_s['p10']/1e6, test_s['p90']/1e6,
                        alpha=0.2, color='#FF5722', label='P10-P90 interval')
    # Mark train/test split
    ax.axvline(pd.Timestamp(config['splits']['val_end']), color='green',
               linestyle=':', linewidth=1.5, label='Test start')
    ax.set_title(f'Store {sid}: Historical Demand + Forecast with P10-P90 Uncertainty Band')
    ax.set_ylabel('Weekly Sales ($M)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}M'))
    ax.legend(loc='upper left', fontsize=9)
    store_wape = store_metrics.loc[str(sid), 'wape'] if str(sid) in store_metrics.index else float('nan')
    ax.text(0.99, 0.05, f'WAPE: {store_wape:.4f}',
            transform=ax.transAxes, ha='right', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Demand Forecast with Uncertainty Intervals (LightGBM Quantile Regression)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'forecast_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("Forecast chart saved.")

# ── CHART: Model Comparison Bar Chart ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
model_names = list(results.keys())
wapes = [results[m]['wape'] for m in model_names]
maes = [results[m]['mae']/1e3 for m in model_names]

colors = ['#BDBDBD', '#90CAF9', '#1565C0', '#0D47A1']
axes[0].bar(model_names, wapes, color=colors, alpha=0.85)
axes[0].set_title('WAPE by Model (lower is better)')
axes[0].set_ylabel('WAPE')
for i, (name, val) in enumerate(zip(model_names, wapes)):
    axes[0].text(i, val + 0.001, f'{val:.4f}', ha='center', fontsize=10)

axes[1].bar(model_names, maes, color=colors, alpha=0.85)
axes[1].set_title('MAE by Model (lower is better)')
axes[1].set_ylabel('MAE ($K)')
for i, (name, val) in enumerate(zip(model_names, maes)):
    axes[1].text(i, val + 0.5, f'${val:.1f}K', ha='center', fontsize=10)

plt.suptitle('Model Selection: Test-Set Performance Comparison', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Model comparison chart saved.")
print("\nPHASE 4 & 5 COMPLETE.")
