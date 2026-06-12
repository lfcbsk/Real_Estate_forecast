"""Tests for drift -> retrain -> registry orchestration."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.monitoring.log_report import evaluate_retrain_decision
from src.pipeline.orchestrator import (RegistryGate,
                                       detect_drift_against_reference,
                                       evaluate_for_registry,
                                       run_orchestration, should_retrain)


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
def test_detect_drift_against_reference_stable(reference_df, stable_current_df):
    report = detect_drift_against_reference(
        reference_df=reference_df,
        current_df=stable_current_df,
        feature_cols=["feature_1", "feature_2"],
        target_col="log_amount_new_house_transactions",
    )
    assert report["severity"] == "low"
    assert report["overall_drift_detected"] is False


@pytest.mark.unit
def test_detect_drift_against_reference_drifted(reference_df, drifted_current_df):
    report = detect_drift_against_reference(
        reference_df=reference_df,
        current_df=drifted_current_df,
        feature_cols=["feature_1", "feature_2"],
        target_col="log_amount_new_house_transactions",
    )
    assert report["overall_drift_detected"] is True
    assert report["severity"] in ["medium", "high"]


@pytest.mark.unit
def test_should_retrain_high_severity():
    report = {"severity": "high", "overall_drift_detected": True}
    assert should_retrain(report) is True


@pytest.mark.unit
def test_should_retrain_low_severity():
    report = {"severity": "low", "overall_drift_detected": False}
    assert should_retrain(report) is False


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
@patch("src.pipeline.orchestrator.compute_sector_stats")
@patch("src.pipeline.orchestrator.build_sector_profile")
@patch("src.pipeline.orchestrator.create_training_features")
@patch("src.pipeline.orchestrator.get_valid_features")
@patch("src.pipeline.orchestrator.ingest_run")
@patch("src.pipeline.orchestrator.load_reference_dataset")
@patch("src.pipeline.orchestrator.save_drift_report")
def test_run_orchestration_skips_retrain_when_stable(
    mock_save_report,
    mock_load_ref,
    mock_ingest,
    mock_get_features,
    mock_create_features,
    mock_build_profile,
    mock_compute_stats,
    reference_df,
    stable_current_df,
):
    mock_ingest.return_value = (reference_df, stable_current_df)
    mock_load_ref.return_value = reference_df.copy()
    mock_get_features.return_value = ["feature_1", "feature_2"]
    mock_create_features.side_effect = lambda df, **kwargs: df
    mock_compute_stats.return_value = {}
    mock_build_profile.return_value = {}

    result = run_orchestration(tune=False, promote=False)

    assert result.should_retrain is False
    assert result.retrain_triggered is False
    assert result.registry_promoted is False
    mock_save_report.assert_called_once()
