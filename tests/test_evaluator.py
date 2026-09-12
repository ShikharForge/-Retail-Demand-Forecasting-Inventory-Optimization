"""
tests/test_evaluator.py
========================
Unit tests for forecasting evaluation metrics.
"""

import numpy as np
import pytest

from src.models.evaluator import evaluate, mae, mape, rmse, wape


class TestMAE:
    def test_perfect_forecast(self):
        assert mae(np.array([100, 200, 300]), np.array([100, 200, 300])) == 0.0

    def test_basic(self):
        actual = np.array([100, 200])
        pred = np.array([90, 220])
        # |10| + |20| / 2 = 15
        assert mae(actual, pred) == 15.0

    def test_symmetric(self):
        actual = np.array([100.0, 200.0])
        pred = np.array([120.0, 180.0])
        assert mae(actual, pred) == mae(pred, actual)


class TestRMSE:
    def test_perfect_forecast(self):
        assert rmse(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0

    def test_basic(self):
        actual = np.array([0.0, 0.0])
        pred = np.array([3.0, 4.0])
        # sqrt((9 + 16) / 2) = sqrt(12.5)
        assert abs(rmse(actual, pred) - np.sqrt(12.5)) < 1e-8

    def test_rmse_penalises_outliers_more_than_mae(self):
        actual = np.array([100.0, 100.0, 100.0])
        pred_even = np.array([90.0, 90.0, 90.0])
        pred_outlier = np.array([100.0, 100.0, 70.0])  # same total error
        # RMSE of outlier > RMSE of even (outlier penalised more)
        assert rmse(actual, pred_outlier) > rmse(actual, pred_even)


class TestWAPE:
    def test_perfect_forecast(self):
        assert wape(np.array([100, 200]), np.array([100, 200])) == 0.0

    def test_basic(self):
        actual = np.array([100.0, 200.0])
        pred = np.array([80.0, 220.0])
        # (|20| + |20|) / (100 + 200) = 40 / 300 = 0.1333
        expected = 40.0 / 300.0
        assert abs(wape(actual, pred) - expected) < 1e-8

    def test_zero_actual_sum(self):
        assert np.isnan(wape(np.array([0.0, 0.0]), np.array([1.0, 2.0])))


class TestMAPE:
    def test_perfect_forecast(self):
        assert mape(np.array([100.0, 200.0]), np.array([100.0, 200.0])) == 0.0

    def test_basic(self):
        actual = np.array([100.0, 200.0])
        pred = np.array([110.0, 180.0])
        # (|0.1| + |0.1|) / 2 = 0.1
        assert abs(mape(actual, pred) - 0.1) < 1e-8

    def test_returns_nan_for_zero_actual(self):
        result = mape(np.array([0.0, 100.0]), np.array([10.0, 110.0]))
        assert np.isnan(result)


class TestEvaluate:
    def test_returns_all_metrics(self):
        actual = np.array([100.0, 200.0, 300.0])
        pred = np.array([110.0, 190.0, 290.0])
        result = evaluate(actual, pred)
        assert set(result.keys()) == {"mae", "rmse", "wape", "mape"}

    def test_all_metrics_non_negative(self):
        actual = np.array([100.0, 200.0, 300.0])
        pred = np.array([110.0, 190.0, 310.0])
        result = evaluate(actual, pred)
        for k, v in result.items():
            if not np.isnan(v):
                assert v >= 0, f"Metric {k} is negative: {v}"

    def test_custom_metric_selection(self):
        actual = np.array([100.0, 200.0])
        pred = np.array([110.0, 190.0])
        result = evaluate(actual, pred, metrics=["mae", "rmse"])
        assert "wape" not in result
        assert "mae" in result
        assert "rmse" in result
