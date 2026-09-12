"""
src/analysis/__init__.py
========================
Analysis modules: Exploratory Data Analysis (EDA) and Statistical Testing.
"""

from src.analysis.eda import run_eda
from src.analysis.statistics import run_statistical_tests

__all__ = ["run_eda", "run_statistical_tests"]
