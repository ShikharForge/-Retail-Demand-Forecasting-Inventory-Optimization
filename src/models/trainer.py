"""
src/models/trainer.py
======================
Training pipeline for LightGBM and XGBoost panel models.

Architecture: Global Panel Model
---------------------------------
Instead of fitting 45 separate per-store models (which would have only
~100 training rows each), we fit a SINGLE global model across all stores
with `store` as a categorical feature. This is called a "panel model".

Advantages:
- Cross-store learning: the model can borrow statistical strength from
  stores with similar characteristics.
- More training data per model: ~4,500 rows vs ~100.
- Simpler deployment: one model artifact, not 45.

Disadvantage:
- Assumes the relationship between features and demand is shared across
  stores (partially relaxed by the store categorical feature).

Chronological Split (NEVER random)
------------------------------------
Time-series data MUST NOT be split randomly. Doing so would allow the model
to see "future" validation data during training — a form of data leakage.
We use a strict chronological cut: train → validation → test.

Walk-Forward Cross-Validation
------------------------------
For robust generalisation estimates, we use expanding-window CV:
- Start with 52 weeks minimum training
- Predict next 4 weeks
- Expand training window by 4 weeks
- Repeat until end of validation period
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.evaluator import evaluate

logger = logging.getLogger(__name__)

# Columns that are not features (always excluded)
NON_FEATURE_COLS = {
    "weekly_sales", "date", "year_week", "week_ending",
    "store",  # will be handled separately as categorical
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of columns to use as model features."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def chronological_split(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / validation / test using chronological date cuts.

    Returns
    -------
    train_df, val_df, test_df
    """
    train_end = pd.Timestamp(config["splits"]["train_end"])
    val_end = pd.Timestamp(config["splits"]["val_end"])

    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[df["date"] > val_end].copy()

    logger.info(
        "Split sizes — Train: %d, Val: %d, Test: %d",
        len(train), len(val), len(test)
    )
    return train, val, test


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    objective: str = "regression",
    alpha: Optional[float] = None,
) -> lgb.LGBMRegressor:
    """
    Train a LightGBM panel model.

    Parameters
    ----------
    train_df, val_df : pd.DataFrame
        Train and validation sets with all feature columns.
    config : dict
        Configuration dictionary.
    objective : str
        'regression' for point forecast, 'quantile' for quantile regression.
    alpha : float, optional
        Quantile level (required if objective='quantile').

    Returns
    -------
    lgb.LGBMRegressor
        Trained model.
    """
    feature_cols = get_feature_cols(train_df)
    target = "weekly_sales"

    # Drop NaN rows (from lags) — only in training
    train_clean = train_df[feature_cols + [target, "store"]].dropna()
    val_clean = val_df[feature_cols + [target, "store"]].dropna()

    X_train = train_clean[feature_cols + ["store"]]
    y_train = train_clean[target]
    X_val = val_clean[feature_cols + ["store"]]
    y_val = val_clean[target]

    params = dict(config["models"]["lgbm"])
    params["objective"] = objective
    if objective == "quantile" and alpha is not None:
        params["alpha"] = alpha

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)],
        categorical_feature=["store"],
    )

    val_preds = model.predict(X_val)
    metrics = evaluate(y_val.values, val_preds)
    logger.info(
        "LightGBM (%s alpha=%s) val metrics: %s",
        objective, alpha, {k: round(v, 4) for k, v in metrics.items()}
    )
    return model


def train_xgb(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
) -> xgb.XGBRegressor:
    """
    Train an XGBoost panel model for comparison.

    XGBoost requires numerical encoding for categorical features.
    We use the store integer code.
    """
    feature_cols = get_feature_cols(train_df)
    target = "weekly_sales"

    train_clean = train_df[feature_cols + [target, "store"]].dropna()
    val_clean = val_df[feature_cols + [target, "store"]].dropna()

    # Encode store as integer
    train_clean = train_clean.copy()
    val_clean = val_clean.copy()
    train_clean["store_code"] = train_clean["store"].cat.codes
    val_clean["store_code"] = val_clean["store"].cat.codes

    xgb_features = feature_cols + ["store_code"]

    X_train = train_clean[xgb_features]
    y_train = train_clean[target]
    X_val = val_clean[xgb_features]
    y_val = val_clean[target]

    params = dict(config["models"]["xgb"])
    params["early_stopping_rounds"] = 50

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds = model.predict(X_val)
    metrics = evaluate(y_val.values, val_preds)
    logger.info(
        "XGBoost val metrics: %s", {k: round(v, 4) for k, v in metrics.items()}
    )
    return model


def save_model(model, name: str, config: dict) -> Path:
    """Serialise a model to the outputs/models directory."""
    out_dir = Path(config["outputs"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {path}")
    return path


def load_model(name: str, config: dict):
    """Load a serialised model."""
    path = Path(config["outputs"]["models_dir"]) / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
