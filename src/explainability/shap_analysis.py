"""
src/explainability/shap_analysis.py
=====================================
SHAP-based model explainability for the LightGBM forecasting model.

IMPORTANT CAVEAT (to be documented in README and dashboard)
-----------------------------------------------------------
SHAP values quantify the CONTRIBUTION of each feature to a specific
prediction. They tell us which features are ASSOCIATED with higher or
lower demand forecasts.

SHAP does NOT prove causation. For example:
- High lag_52 → high forecast does NOT mean last year's sales CAUSE
  this year's sales.
- High holiday_flag → high forecast reflects a learned correlation,
  not a causal mechanism.

SHAP is a powerful explanation tool for model transparency and debugging,
NOT a substitute for causal analysis.

Global vs Local Explanations
------------------------------
Global (SHAP summary plot):
  Shows which features matter most ON AVERAGE across all predictions.
  Useful for: model validation, feature selection, stakeholder trust.

Local (SHAP waterfall plot):
  Shows why the model predicted a SPECIFIC value for a specific
  store-week combination.
  Useful for: debugging surprising predictions, explainability to users.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.trainer import get_feature_cols

logger = logging.getLogger(__name__)


def compute_shap_values(
    model,
    df: pd.DataFrame,
    store_col: str = "store",
    max_samples: int = 1000,
) -> tuple[shap.Explanation, np.ndarray]:
    """
    Compute SHAP values for a LightGBM model.

    Uses TreeExplainer (exact, not approximate) which is efficient for
    tree-based models and supports SHAP interaction values.

    Parameters
    ----------
    model : lgb.LGBMRegressor
        Trained LightGBM model.
    df : pd.DataFrame
        Feature-engineered DataFrame (preferably test set).
    max_samples : int
        Cap the number of rows to explain (for speed).

    Returns
    -------
    shap_values : shap.Explanation
    X : np.ndarray
        Feature matrix used for SHAP computation.
    """
    feature_cols = get_feature_cols(df)
    X = df[feature_cols + [store_col]].dropna()

    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    logger.info("SHAP values computed for %d samples", len(X))
    return shap_values, X


def get_feature_importance(shap_values: shap.Explanation) -> pd.DataFrame:
    """
    Compute global feature importance from SHAP values.

    Mean |SHAP| across all predictions → average impact on model output.

    Returns
    -------
    pd.DataFrame
        Sorted by importance descending. Columns: feature, mean_abs_shap.
    """
    feature_names = shap_values.feature_names
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


def plot_summary(
    shap_values: shap.Explanation,
    max_display: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    SHAP beeswarm summary plot — global feature importance with distribution.

    Each point = one prediction. Colour = feature value (red=high, blue=low).
    X-axis = SHAP value (impact on model output).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Summary: Feature Impact on Weekly Sales Forecast", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved SHAP summary to {save_path}")
    return fig


def plot_waterfall(
    shap_values: shap.Explanation,
    row_idx: int = 0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    SHAP waterfall plot for a single prediction.

    Shows how each feature pushes the prediction UP (red) or DOWN (blue)
    from the model's average prediction (base value).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[row_idx], show=False)
    plt.title(f"SHAP Explanation: Prediction Row {row_idx}", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved SHAP waterfall to {save_path}")
    return fig


def plot_feature_importance_bar(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart of top N features by mean |SHAP|."""
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#2196F3")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Top {top_n} Features by SHAP Importance", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
