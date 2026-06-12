"""Global fixtures dùng chung cho toàn bộ test suite."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="session")
def api_client():
    """FastAPI test client dùng chung cho các test API."""
    return TestClient(app)


@pytest.fixture(scope="session")
def minimal_sample_data():
    """Dữ liệu tối thiểu dùng chung nếu cần (ví dụ: cho test pipeline cơ bản)."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "sector": [1] * 10,
            "feature_1": np.random.rand(10),
        }
    )
