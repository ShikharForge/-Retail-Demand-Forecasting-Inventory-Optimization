"""
tests/test_models.py
===================
Unit tests for forecasting models, baseline predictors, and quantile monotonicity.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import NaiveForecast, SeasonalNaiveForecast
from src.models.forecasting import build_forecast_table, predict_point, predict_quantiles


@pytest.fixture
def sample_timeseries():
    dates = pd.date_range("2020-01-03", periods=104, freq="W-FRI")
    df = pd.DataFrame({
        "store": [1] * 104,
        "date": dates,
        "weekly_sales": np.linspace(1000, 2000, 104) + np.sin(np.linspace(0, 4 * np.pi, 104)) * 200,
        "lag_1": np.roll(np.linspace(1000, 2000, 104), 1),
        "lag_52": np.roll(np.linspace(1000, 2000, 104), 52),
    })
    df.loc[0, "lag_1"] = np.nan
    df.loc[:51, "lag_52"] = np.nan
    df["store"] = df["store"].astype("category")
    return df


class TestBaselines:
    def test_naive_forecast(self, sample_timeseries):
        train = sample_timeseries.iloc[:80]
        test = sample_timeseries.iloc[80:]
        naive = NaiveForecast().fit(train)
        preds = naive.predict(test)
        assert len(preds) == len(test)
        assert not np.isnan(preds).any()
        # All predictions equal the last value of the training set
        last_train_val = train["weekly_sales"].iloc[-1]
        assert np.allclose(preds, last_train_val)

    def test_seasonal_naive_forecast(self, sample_timeseries):
        train = sample_timeseries.iloc[:80]
        test = sample_timeseries.iloc[80:]
        sn = SeasonalNaiveForecast().fit(train)
        preds = sn.predict(test)
        assert len(preds) == len(test)
        assert not np.isnan(preds).any()
        assert np.allclose(preds, test["lag_52"].values)


class DummyQuantileModel:
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def predict(self, X):
        return np.ones(len(X)) * self.multiplier


class TestQuantileMonotonicity:
    def test_quantile_monotonicity(self, sample_timeseries):
        models = {
            "q10": DummyQuantileModel(100),
            "q50": DummyQuantileModel(150),
            "q90": DummyQuantileModel(200),
        }
        test = sample_timeseries.tail(10)
        q_df = predict_quantiles(models, test)
        assert (q_df["p10"] <= q_df["p50"]).all()
        assert (q_df["p50"] <= q_df["p90"]).all()

    def test_forecast_table_construction(self, sample_timeseries):
        test = sample_timeseries.tail(10)
        fc_table = build_forecast_table(
            test,
            actual_col="weekly_sales",
            point=np.ones(10) * 1500,
            p10=np.ones(10) * 1200,
            p50=np.ones(10) * 1500,
            p90=np.ones(10) * 1800,
        )
        assert "actual" in fc_table.columns
        assert "forecast_point" in fc_table.columns
        assert "uncertainty_range" in fc_table.columns
        assert np.allclose(fc_table["uncertainty_range"], 600)
