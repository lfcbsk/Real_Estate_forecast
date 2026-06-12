"""Test API endpoints."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_registry():
    """Mock model registry."""
    registry = MagicMock()
    registry.predict.side_effect = lambda X: np.full(len(X), 5.0)
    registry.zero_sectors = {52, 95}
    return registry


@pytest.fixture
def mock_train_data():
    """Mock training data."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100, freq="MS"),
            "sector": [1, 2, 3, 4] * 25,
            "amount_new_house_transactions": np.random.randint(100, 1000, 100),
            "log_amount_new_house_transactions": np.log1p(np.random.randint(100, 1000, 100)),
        }
    )


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "RealEstate Forecast API"
    assert "version" in data
    assert "docs" in data


@patch("src.api.routes.get_model_registry")
@patch("src.api.routes.get_train_data")
def test_forecast_success(mock_get_train_data, mock_get_registry, client, mock_model_registry, mock_train_data):
    """Test forecast endpoint thành công."""
    mock_get_registry.return_value = mock_model_registry
    mock_get_train_data.return_value = mock_train_data

    # Mock forecast_next_year
    with patch("src.api.routes.forecast_next_year") as mock_forecast:
        mock_forecast.return_value = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=12, freq="MS"),
                "sector": ["1"] * 12,
                "pred_amount": [
                    100,
                    110,
                    120,
                    130,
                    140,
                    150,
                    160,
                    170,
                    180,
                    190,
                    200,
                    210,
                ],
            }
        )

        response = client.post("/api/v1/forecast", json={"n_months": 12})

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert data["n_months"] == 12
        assert "predictions" in data


@patch("src.api.routes.get_model_registry")
@patch("src.api.routes.get_train_data")
def test_forecast_no_data(mock_get_train_data, mock_get_registry, client, mock_model_registry):
    """Test forecast endpoint khi không có data."""
    mock_get_registry.return_value = mock_model_registry
    mock_get_train_data.return_value = None

    response = client.post("/api/v1/forecast", json={"n_months": 12})

    assert response.status_code == 500
    assert "Training data not available" in response.json()["detail"]


@patch("src.api.routes.get_model_registry")
def test_predict_single_success(mock_get_registry, client, mock_model_registry):
    """Test predict single endpoint thành công."""
    mock_get_registry.return_value = mock_model_registry

    payload = {
        "features": {
            "lag_1": 500,
            "lag_2": 480,
            "rolling_mean_3": 490,
        }
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "predicted_value" in data
    assert "predicted_amount" in data


@patch("src.api.routes.get_model_registry")
def test_predict_single_invalid_features(mock_get_registry, client, mock_model_registry):
    """Test predict single với features không hợp lệ."""
    mock_get_registry.return_value = mock_model_registry
    mock_model_registry.predict.side_effect = Exception("Invalid features")

    payload = {"features": {"invalid_feature": "not_a_number"}}

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 400


def test_forecast_invalid_n_months(client):
    """Test forecast với n_months không hợp lệ."""
    # n_months < 1
    response = client.post("/api/v1/forecast", json={"n_months": 0})
    assert response.status_code == 422

    # n_months > 36
    response = client.post("/api/v1/forecast", json={"n_months": 50})
    assert response.status_code == 422


@patch("src.api.routes.get_model_registry")
@patch("src.api.routes.get_train_data")
def test_get_sectors(mock_get_train_data, mock_get_registry, client, mock_model_registry, mock_train_data):
    """Test get sectors endpoint."""
    mock_get_registry.return_value = mock_model_registry
    mock_get_train_data.return_value = mock_train_data

    response = client.get("/api/v1/sectors")

    assert response.status_code == 200
    data = response.json()

    assert "total_sectors" in data
    assert "zero_sectors_count" in data
    assert "active_sectors_count" in data
    assert "sectors" in data


@patch("src.api.routes.get_model_registry")
@patch("src.api.routes.get_train_data")
def test_get_drift_report(mock_get_train_data, mock_get_registry, client, mock_model_registry, mock_train_data):
    """Test get drift report endpoint."""
    mock_get_registry.return_value = mock_model_registry
    mock_get_train_data.return_value = mock_train_data

    # Mock detect_data_drift
    with patch("src.monitoring.detect_drift.detect_data_drift") as mock_drift:
        mock_drift.return_value = {
            "drift_detected": False,
            "drift_score": 0.05,
            "affected_features": [],
            "severity": "low",
            "recommendation": "Continue monitoring",
        }

        response = client.get("/api/v1/drift")

        assert response.status_code == 200
        data = response.json()

        assert "drift_detected" in data
        assert "severity" in data
        assert "recommendation" in data


def test_api_cors_headers(client):
    """Test CORS headers được set đúng."""
    response = client.options("/api/v1/health", headers={"Origin": "http://localhost:3000"})

    # CORS headers phải có
    assert "access-control-allow-origin" in response.headers
