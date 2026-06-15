"""Tests for drift -> retrain -> registry orchestration."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.monitoring.detect_drift import detect_data_drift
from src.monitoring.log_report import evaluate_retrain_decision
from src.pipeline.orchestrator import RegistryGate, evaluate_for_registry, run_orchestration, should_retrain


@pytest.fixture
def reference_df():
    rng = np.random.default_rng(seed=42)
    n = 100
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "sector": np.tile([1, 2], n // 2),
            "feature_1": rng.normal(0, 1, n),
            "feature_2": rng.normal(10, 2, n),
            "log_amount_new_house_transactions": rng.normal(5, 0.5, n),
        }
    )


@pytest.fixture
def stable_current_df(reference_df):
    df = reference_df.copy()
    df["date"] = pd.date_range("2023-06-01", periods=len(reference_df), freq="D")
    return df


@pytest.fixture
def drifted_current_df(reference_df):
    rng = np.random.default_rng(seed=99)
    n = len(reference_df)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-06-01", periods=n, freq="D"),
            "feature_1": rng.normal(8, 1, n),
            "feature_2": rng.normal(10, 2, n),
            "log_amount_new_house_transactions": rng.normal(5, 0.5, n),
        }
    )


@pytest.mark.unit
def test_detect_data_drift_stable_combined(reference_df, stable_current_df):
    combined = pd.concat([reference_df, stable_current_df]).sort_values("date").reset_index(drop=True)
    split_ratio = len(reference_df) / len(combined)
    report = detect_data_drift(
        combined,
        target_col="log_amount_new_house_transactions",
        feature_cols=["feature_1", "feature_2"],
        split_ratio=split_ratio,
    )
    assert report["severity"] == "low"
    assert report["overall_drift_detected"] is False


@pytest.mark.unit
def test_detect_data_drift_drifted_combined(reference_df, drifted_current_df):
    combined = pd.concat([reference_df, drifted_current_df]).sort_values("date").reset_index(drop=True)
    split_ratio = len(reference_df) / len(combined)
    report = detect_data_drift(
        combined,
        target_col="log_amount_new_house_transactions",
        feature_cols=["feature_1", "feature_2"],
        split_ratio=split_ratio,
    )
    assert report["overall_drift_detected"] is True
    assert report["severity"] in ["medium", "high"]


@pytest.mark.unit
def test_should_retrain_high_severity():
    assert should_retrain({"severity": "high"}) is True


@pytest.mark.unit
def test_should_retrain_low_severity():
    assert should_retrain({"severity": "low"}) is False


@pytest.mark.unit
def test_evaluate_for_registry_pass():
    gate = RegistryGate(min_competition_score=0.5, min_r2=0.0, max_mape=100.0)
    eligible, messages = evaluate_for_registry(
        {"competition_score": 0.7, "r2": 0.8, "mape": 10.0},
        gate=gate,
    )
    assert eligible is True
    assert any("passes" in m.lower() for m in messages)


@pytest.mark.unit
def test_evaluate_for_registry_fail():
    gate = RegistryGate(min_competition_score=0.8)
    eligible, messages = evaluate_for_registry(
        {"competition_score": 0.5, "r2": 0.8, "mape": 10.0},
        gate=gate,
    )
    assert eligible is False
    assert len(messages) >= 1


@pytest.mark.unit
def test_evaluate_retrain_decision_medium_drift():
    report = {
        "severity": "medium",
        "summary": {
            "feature_drift_ratio": 0.3,
            "concept_drift_detected": False,
            "data_quality_issues_count": 0,
        },
        "details": {},
    }
    decision = evaluate_retrain_decision(report)
    assert decision["decision"] == "MONITOR"


@pytest.mark.unit
@patch("src.pipeline.orchestrator.load_production_model")
@patch("src.pipeline.orchestrator.prepare_drift_dataframe")
@patch("src.pipeline.orchestrator.detect_data_drift")
@patch("src.pipeline.orchestrator.load_data")
@patch("src.pipeline.orchestrator.load_reference_dataset")
@patch("src.pipeline.orchestrator.save_drift_report")
def test_run_orchestration_stops_on_low_severity(
    mock_save_report,
    mock_load_ref,
    mock_load_data,
    mock_detect,
    mock_prepare,
    mock_load_model,
    reference_df,
    stable_current_df,
):
    mock_load_data.return_value = (reference_df, stable_current_df)
    mock_load_ref.return_value = reference_df.copy()
    mock_load_model.return_value = MagicMock()
    mock_prepare.return_value = (reference_df, ["feature_1", "feature_2"], 0.5)
    mock_detect.return_value = {
        "severity": "low",
        "overall_drift_detected": False,
        "recommendation": "OK",
        "summary": {},
    }

    result = run_orchestration(tune=False, promote=False)

    assert result.should_retrain is False
    assert result.retrain_triggered is False
    assert result.registry_promoted is False
    mock_detect.assert_called_once()
    mock_save_report.assert_called_once()
