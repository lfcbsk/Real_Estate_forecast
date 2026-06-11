"""Fixtures dung chung cho toan bo test suite.

Chay: pytest -v
Chi chay 1 file: pytest tests/test_drift.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def reference_df():
    """Du lieu 'reference' - phan phoi on dinh, dung lam baseline."""
    rng = np.random.default_rng(seed=42)
    n = 200
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "sector": np.tile(["A", "B"], n // 2),
        "feature_1": rng.normal(loc=0, scale=1, size=n),
        "feature_2": rng.normal(loc=10, scale=2, size=n),
        "log_amount_new_house_transactions": rng.normal(loc=5, scale=0.5, size=n),
    })


@pytest.fixture
def current_df_no_drift(reference_df):
    """Cung phan phoi voi reference -> khong nen phat hien drift."""
    rng = np.random.default_rng(seed=123)
    n = len(reference_df)
    return pd.DataFrame({
        "date": pd.date_range("2023-08-01", periods=n, freq="D"),
        "sector": np.tile(["A", "B"], n // 2),
        "feature_1": rng.normal(loc=0, scale=1, size=n),
        "feature_2": rng.normal(loc=10, scale=2, size=n),
        "log_amount_new_house_transactions": rng.normal(loc=5, scale=0.5, size=n),
    })


@pytest.fixture
def current_df_with_drift(reference_df):
    """Phan phoi lech han -> nen phat hien feature drift."""
    rng = np.random.default_rng(seed=456)
    n = len(reference_df)
    return pd.DataFrame({
        "date": pd.date_range("2023-08-01", periods=n, freq="D"),
        "sector": np.tile(["A", "B"], n // 2),
        "feature_1": rng.normal(loc=8, scale=1, size=n),
        "feature_2": rng.normal(loc=10, scale=2, size=n),
        "log_amount_new_house_transactions": rng.normal(loc=5, scale=0.5, size=n),
    })


@pytest.fixture
def full_df(reference_df, current_df_no_drift):
    """Ghep reference + current thanh 1 DataFrame full (giong train_data)."""
    return pd.concat([reference_df, current_df_no_drift], ignore_index=True)


@pytest.fixture
def full_df_with_drift(reference_df, current_df_with_drift):
    return pd.concat([reference_df, current_df_with_drift], ignore_index=True)


class FakeModelGoodFit:
    """Model gia: du doan gan dung target -> MAE thap o ca 2 nua.

    predict() tra ve shape (n, 1) de mo phong ONNX output that,
    nham test logic .reshape(-1).
    """
    features = ["feature_1", "feature_2"]

    def predict(self, X):
        n = len(X)
        rng = np.random.default_rng(seed=1)
        preds = 5 + rng.normal(0, 0.05, size=n)
        return preds.reshape(-1, 1)


class FakeModelDegrading:
    """Model gia: du doan tot o reference, te han o current -> concept drift."""
    features = ["feature_1", "feature_2"]

    def __init__(self):
        self._call_count = 0

    def predict(self, X):
        n = len(X)
        self._call_count += 1
        rng = np.random.default_rng(seed=self._call_count)
        if self._call_count == 1:
            preds = 5 + rng.normal(0, 0.05, size=n)
        else:
            preds = 5 + rng.normal(0, 0.05, size=n) + 10
        return preds.reshape(-1, 1)


@pytest.fixture
def fake_model_good_fit():
    return FakeModelGoodFit()


@pytest.fixture
def fake_model_degrading():
    return FakeModelDegrading()