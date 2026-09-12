"""
tests/test_data.py
==================
Unit tests for data ingestion, schema validation, and preprocessing routines.
"""

import pandas as pd
import pytest

from src.data.ingestion import load_config, load_raw_data, validate_raw_data
from src.data.preprocessing import clean


@pytest.fixture
def config():
    return load_config("config/config.yaml")


@pytest.fixture
def raw_df(config):
    return load_raw_data(config)


class TestDataIngestion:
    def test_load_config(self, config):
        assert isinstance(config, dict)
        assert "data" in config
        assert "splits" in config
        assert "features" in config
        assert "models" in config

    def test_load_raw_data(self, raw_df):
        assert not raw_df.empty
        assert "Store" in raw_df.columns
        assert "Date" in raw_df.columns
        assert "Weekly_Sales" in raw_df.columns
        assert pd.api.types.is_datetime64_any_dtype(raw_df["Date"])

    def test_validate_raw_data(self, raw_df, config):
        report = validate_raw_data(raw_df, config)
        assert report["shape"][0] == 6435
        assert report["store_count"] == 45
        assert report["duplicate_rows"] == 0
        assert report["duplicate_store_date"] == 0
        assert sum(report["null_counts"].values()) == 0


class TestDataPreprocessing:
    def test_clean_structure(self, raw_df, config):
        df_clean = clean(raw_df, config)
        assert not df_clean.empty
        assert "store" in df_clean.columns
        assert "weekly_sales" in df_clean.columns
        assert "year_week" in df_clean.columns
        assert df_clean["store"].dtype.name == "category"

    def test_chronological_ordering(self, raw_df, config):
        df_clean = clean(raw_df, config)
        for _, gdf in df_clean.groupby("store", observed=True):
            dates = gdf["date"].tolist()
            assert dates == sorted(dates), "Rows must be strictly chronologically sorted within each store"
