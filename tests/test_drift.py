"""Test drift detection pipeline."""
import pytest
import pandas as pd
import numpy as np

from src.monitoring.detect_drift import (
    calculate_psi,
    test_feature_drift_stats,
    check_data_quality_drift,
    detect_distribution_drift,
    page_hinkley_test,
    detect_concept_drift_comprehensive,
    detect_data_drift,
)

@pytest.fixture
def reference_df():
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
def current_df_with_drift(reference_df):
    rng = np.random.default_rng(seed=456)
    n = len(reference_df)
    df = reference_df.copy()
    df["date"] = pd.date_range("2023-08-01", periods=n, freq="D")
    df["feature_1"] = rng.normal(loc=8, scale=1, size=n)  # Drift!
    return df

class FakeModelGoodFit:
    features = ["feature_1", "feature_2"]
    def predict(self, X):
        n = len(X)
        rng = np.random.default_rng(seed=1)
        return (5 + rng.normal(0, 0.05, size=n)).reshape(-1, 1)

@pytest.fixture
def fake_model_good_fit():
    return FakeModelGoodFit()

# ================= TEST CASES =================

@pytest.mark.unit
def test_calculate_psi_no_drift(reference_df):
    psi = calculate_psi(reference_df["feature_1"].values,reference_df["feature_1"].values)
    assert psi < 0.05, f"PSI should be near 0 for identical data, got {psi}"

@pytest.mark.unit
def test_calculate_psi_high_drift(reference_df, current_df_with_drift):
    """PSI của 2 tập dữ liệu khác biệt lớn phải > 0.2."""
    psi = calculate_psi(reference_df["feature_1"].values,current_df_with_drift["feature_1"].values)
    assert psi > 0.2, f"PSI should be > 0.2 for drifted data, got {psi}"

@pytest.mark.unit
def test_calculate_psi_moderate_drift(reference_df):
    rng = np.random.default_rng(seed=999)
    drifted = rng.normal(loc=2, scale=1.5, size=len(reference_df))
    psi = calculate_psi(reference_df["feature_1"].values, drifted)
    assert 0.05 <= psi <= 0.6, f"PSI for moderate drift: {psi}"  

@pytest.mark.unit
def test_feature_drift_stats_no_drift(reference_df):
    result = test_feature_drift_stats(reference_df["feature_1"],reference_df["feature_1"])
    assert "ks_statistic" in result
    assert "ks_pvalue" in result
    assert "psi" in result
    
    assert result["ks_drift"] is False
    assert result["psi_drift"] is False

@pytest.mark.unit
def test_feature_drift_stats_with_drift(reference_df, current_df_with_drift):
    result = test_feature_drift_stats(reference_df["feature_1"],current_df_with_drift["feature_1"])
    assert result["ks_drift"] is True or result["psi_drift"] is True

@pytest.mark.unit
def test_feature_drift_stats_empty_data():
    s1 = pd.Series([np.nan, np.nan])
    s2 = pd.Series([1, 2])
    result = test_feature_drift_stats(s1, s2)
    
    assert "error" in result
    assert "Insufficient data" in result["error"]

@pytest.mark.unit
def test_check_data_quality_drift_no_issues(reference_df, current_df_no_drift):
    result = check_data_quality_drift(reference_df, current_df_no_drift)
    assert result["quality_drift_detected"] is False
    assert len(result["issues"]) == 0

@pytest.mark.unit
def test_check_data_quality_drift_missing_increase(reference_df):
    current_df = reference_df.copy()
    current_df.loc[:100, "feature_1"] = np.nan  
    result = check_data_quality_drift(reference_df, current_df)
    assert result["quality_drift_detected"] is True
    assert any("Missing rate" in issue for issue in result["issues"])

@pytest.mark.unit
def test_check_data_quality_drift_schema_mismatch(reference_df):
    current_df = reference_df.drop(columns=["feature_2"])
    result = check_data_quality_drift(reference_df, current_df)
    assert result["quality_drift_detected"] is True
    assert any("Schema mismatch" in issue for issue in result["issues"])

@pytest.mark.unit
def test_detect_distribution_drift_no_drift(reference_df):
    result = detect_distribution_drift(reference_df["feature_1"],reference_df["feature_1"],"Test Feature")
    assert result["drift_detected"] is False
    assert result["severity"] == "low"

@pytest.mark.unit
def test_detect_distribution_drift_high_drift(reference_df, current_df_with_drift):
    result = detect_distribution_drift(reference_df["feature_1"],current_df_with_drift["feature_1"],"Test Feature")
    assert result["drift_detected"] is True
    assert result["severity"] in ["medium", "high"]

@pytest.mark.unit
def test_page_hinkley_no_drift():
    errors = np.random.normal(0, 0.1, 100)
    result = page_hinkley_test(errors, lambda_ph=50.0)
    assert result["drift_detected"] is False

@pytest.mark.unit
def test_page_hinkley_sudden_drift():
    errors = np.concatenate([np.random.normal(0, 0.1, 50), np.random.normal(5, 0.1, 50) ])
    result = page_hinkley_test(errors, lambda_ph=50.0)
    assert result["drift_detected"] is True

@pytest.mark.unit
def test_page_hinkley_gradual_drift():
    errors = np.linspace(0, 10, 200)
    result = page_hinkley_test(errors, lambda_ph=50.0)
    assert result["drift_detected"] is True

@pytest.mark.unit
def test_detect_concept_drift_no_drift(reference_df, fake_model_good_fit):
    ref_X = reference_df[["feature_1", "feature_2"]]
    ref_y = reference_df["log_amount_new_house_transactions"]
    result = detect_concept_drift_comprehensive(ref_X, ref_y,ref_X, ref_y, fake_model_good_fit,mae_threshold=0.2)
    assert result["concept_drift_detected"] is False

@pytest.mark.unit
def test_detect_concept_drift_with_degradation(reference_df, current_df_with_drift, fake_model_degrading):
    ref_X = reference_df[["feature_1", "feature_2"]]
    ref_y = reference_df["log_amount_new_house_transactions"]
    
    curr_X = current_df_with_drift[["feature_1", "feature_2"]]
    curr_y = current_df_with_drift["log_amount_new_house_transactions"]
    
    result = detect_concept_drift_comprehensive( ref_X, ref_y,curr_X, curr_y,fake_model_degrading,mae_threshold=0.2)
    assert result["concept_drift_detected"] is True

@pytest.mark.unit
def test_detect_data_drift_no_drift(full_df, fake_model_good_fit):
    result = detect_data_drift(
        full_df,
        model=fake_model_good_fit,
        target_col="log_amount_new_house_transactions",
        feature_cols=["feature_1", "feature_2"],
        split_ratio=0.5
    )
    assert "overall_drift_detected" in result
    assert "severity" in result
    assert "summary" in result
    assert result["severity"] in ["low", "medium"]

@pytest.mark.unit
def test_detect_data_drift_with_drift(full_df_with_drift, fake_model_degrading):
    result = detect_data_drift(
        full_df_with_drift,
        model=fake_model_degrading,
        target_col="log_amount_new_house_transactions",
        feature_cols=["feature_1", "feature_2"],
        split_ratio=0.5
    )
    assert result["overall_drift_detected"] is True
    assert result["severity"] in ["medium", "high"]

@pytest.mark.unit
def test_detect_data_drift_output_structure(full_df, fake_model_good_fit):
    result = detect_data_drift(
        full_df,
        model=fake_model_good_fit,
        target_col="log_amount_new_house_transactions",
        feature_cols=["feature_1", "feature_2"]
    )
    
    required_keys = [
        "overall_drift_detected",
        "severity",
        "recommendation",
        "summary"
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    summary_keys = [
        "feature_drift_ratio",
        "data_quality_issues_count",
        "concept_drift_detected"
    ]
    
    for key in summary_keys:
        assert key in result["summary"], f"Missing summary key: {key}"