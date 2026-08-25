"""
tests/conftest.py - Shared pytest fixtures for the RFM-T / CLV / ML / Cohort engines.

Fixtures are session-scoped where the underlying computation is expensive (K-Means,
PCA) and the tests only read the result, never mutate it in place.
"""

import os
import sys

import pandas as pd
import pytest

# Make `src` importable when pytest is run from the repo root (rootdir-relative
# imports already work via the repo's own conftest discovery, but this keeps the
# test file runnable directly too, e.g. `python tests/test_pipeline.py`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rfm_engine import process_rfmt_pipeline
from src.clv_engine import estimate_btyd_clv
from src.ml_engine import preprocess_rfmt_features


@pytest.fixture(scope="session")
def raw_transactions_df() -> pd.DataFrame:
    """The bundled synthetic enterprise dataset (450 customers, 5,550 transactions)."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "ecommerce_transactions.csv")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def clean_and_rfmt(raw_transactions_df):
    """(clean_transactions_df, rfmt_scored_df) from the standard pipeline entry point."""
    return process_rfmt_pipeline(raw_transactions_df)


@pytest.fixture(scope="session")
def clean_tx_df(clean_and_rfmt):
    return clean_and_rfmt[0]


@pytest.fixture(scope="session")
def rfmt_df(clean_and_rfmt):
    return clean_and_rfmt[1]


@pytest.fixture(scope="session")
def clv_df(rfmt_df):
    return estimate_btyd_clv(rfmt_df, prediction_horizon_days=90, gross_margin=0.35)


@pytest.fixture(scope="session")
def scaled_features(clv_df):
    X_scaled, scaler, features = preprocess_rfmt_features(clv_df)
    return X_scaled, scaler, features
