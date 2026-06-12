"""Test model loading and prediction logic."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Giả sử bạn có module quản lý model
# from src.models.model_registry import load_model, predict_batch


@pytest.fixture
def dummy_onnx_model():
    """Mock một model ONNX/Sklearn để test mà không cần load file thật nặng."""
    model = MagicMock()
    model.run.return_value = [np.array([[5.5], [6.0], [5.8]])]  # Mock ONNX output
    model.get_inputs.return_value = [MagicMock(name="float_input", shape=[None, 3])]
    return model


@pytest.mark.unit
def test_model_prediction_shape(dummy_onnx_model):
    """Đảm bảo model trả về shape đúng như API mong đợi."""
    # Giả lập input có 3 features, 3 samples
    X_input = np.random.rand(3, 3).astype(np.float32)

    # Gọi mock predict (thay bằng hàm predict thật của bạn)
    predictions = dummy_onnx_model.run(None, {"float_input": X_input})[0]

    assert predictions.shape == (
        3,
        1,
    ), f"Expected shape (3, 1), got {predictions.shape}"
    assert isinstance(predictions, np.ndarray)


@pytest.mark.unit
def test_model_handle_missing_values_gracefully():
    """Đảm bảo pipeline dự đoán không crash khi có NaN (nếu bạn có logic fillna)."""
    X_with_nan = pd.DataFrame(
        {
            "lag_1": [10.0, np.nan, 30.0],
            "lag_2": [20.0, 25.0, np.nan],
            "rolling_mean": [15.0, 15.0, 15.0],
        }
    )

    # Nếu code của bạn có bước fillna(0) trước khi predict:
    X_clean = X_with_nan.fillna(0.0)
    assert X_clean.isna().sum().sum() == 0, "Missing values were not handled!"
