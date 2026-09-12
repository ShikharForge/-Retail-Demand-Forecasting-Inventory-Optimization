"""Phase 8: SHAP Explainability on the LightGBM point model."""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, shap, pickle
from pathlib import Path

from src.data.loader import load_config, load_raw_data
from src.data.preprocessor import clean
from src.features.engineer import build_features
from src.models.trainer import chronological_split, get_feature_cols
from src.explainability.shap_analysis import (
    compute_shap_values, get_feature_importance,
    plot_feature_importance_bar
)

config = load_config('config/config.yaml')
fig_dir = Path('outputs/figures')

df_raw = load_raw_data(config)
df = build_features(clean(df_raw, config), config)
df['store'] = df['store'].astype('category')
_, _, test_df = chronological_split(df, config)
test_clean = test_df.dropna(subset=['lag_52'])

with open('outputs/models/lgbm_point.pkl', 'rb') as f:
    model = pickle.load(f)

print("Computing SHAP values (this may take ~30 seconds)...")
shap_values, X = compute_shap_values(model, test_clean, max_samples=600)

# Feature importance table
importance = get_feature_importance(shap_values)
print("\nTop 15 features by mean |SHAP|:")
print(importance.head(15).to_string(index=False))

# Bar chart
fig = plot_feature_importance_bar(importance, top_n=15,
                                  save_path=fig_dir / 'shap_feature_importance.png')
plt.close()

# Beeswarm summary
print("\nGenerating SHAP beeswarm plot...")
fig2, ax = plt.subplots(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("SHAP Summary: Feature Contributions to Weekly Sales Forecast\n"
          "[NOTE: SHAP shows associations, not causal relationships]", fontsize=11)
plt.tight_layout()
plt.savefig(fig_dir / 'shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP beeswarm saved.")

# Waterfall for a single prediction (store 1, first test week)
store1_idx = X[X['store'].astype(int)==1].index
if len(store1_idx) > 0:
    row_pos = list(X.index).index(store1_idx[0])
    fig3, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[row_pos], show=False)
    plt.title("SHAP Waterfall: Why did the model predict this value?\n(Store 1, first test week)",
              fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / 'shap_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP waterfall saved.")

importance.to_csv('outputs/forecasts/shap_importance.csv', index=False)
print("\nPHASE 8 EXPLAINABILITY COMPLETE.")
