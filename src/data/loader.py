"""
src/data/loader.py
==================
Backwards-compatible alias for src.data.ingestion.
"""

from src.data.ingestion import load_config, load_raw_data, validate_raw_data

__all__ = ["load_config", "load_raw_data", "validate_raw_data"]
