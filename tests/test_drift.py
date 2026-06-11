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


def test_calculate_psi_no_drift(reference_df):
    """PSI của 2 tập dữ liệu giống nhau phải rất nhỏ."""
    psi = calculate_psi(
        reference_df["feature_1"].values,
        reference_df["feature_1"].values
    )
    assert psi < 0.05, f"PSI should be near 0 for identical data, got {psi}"


def test_calculate_psi_high_drift(reference_df, current_df_with_drift):
    """PSI của 2 tập dữ liệu khác biệt lớn phải > 0.2."""
    psi = calculate_psi(
        reference_df["feature_1"].values,
        current_df_with_drift["feature_1"].values
    )
    assert psi > 0.2, f"PSI should be > 0.2 for drifted data, got {psi}"


def test_calculate_psi_moderate_drift(reference_df):
    """PSI của 2 tập dữ liệu khác biệt vừa phải."""
    rng = np.random.default_rng(seed=999)
    drifted = rng.normal(loc=2, scale=1.5, size=len(reference_df))  # Mean tăng nhẹ
    
    psi = calculate_psi(reference_df["feature_1"].values, drifted)
    
    # PSI nên nằm trong khoảng 0.1 - 0.2 (moderate drift)
    assert 0.05 <= psi <= 0.3, f"PSI for moderate drift: {psi}"


def test_feature_drift_stats_no_drift(reference_df):
    """Test feature drift stats khi không có drift."""
    result = test_feature_drift_stats(
        reference_df["feature_1"],
        reference_df["feature_1"]
    )
    
    assert "ks_statistic" in result
    assert "ks_pvalue" in result
    assert "psi" in result
    
    # Không có drift
    assert result["ks_drift"] is False
    assert result["psi_drift"] is False


def test_feature_drift_stats_with_drift(reference_df, current_df_with_drift):
    """Test feature drift stats khi có drift."""
    result = test_feature_drift_stats(
        reference_df["feature_1"],
        current_df_with_drift["feature_1"]
    )
    
    # Phải phát hiện drift
    assert result["ks_drift"] is True or result["psi_drift"] is True


def test_feature_drift_stats_empty_data():
    """Test xử lý khi dữ liệu rỗng."""
    s1 = pd.Series([np.nan, np.nan])
    s2 = pd.Series([1, 2])
    
    result = test_feature_drift_stats(s1, s2)
    
    assert "error" in result
    assert "Insufficient data" in result["error"]


def test_check_data_quality_drift_no_issues(reference_df, current_df_no_drift):
    """Test data quality khi không có vấn đề."""
    result = check_data_quality_drift(reference_df, current_df_no_drift)
    
    assert result["quality_drift_detected"] is False
    assert len(result["issues"]) == 0


def test_check_data_quality_drift_missing_increase(reference_df):
    """Test phát hiện khi missing values tăng đột biến."""
    current_df = reference_df.copy()
    current_df.loc[:100, "feature_1"] = np.nan  # Tăng missing rate
    
    result = check_data_quality_drift(reference_df, current_df)
    
    assert result["quality_drift_detected"] is True
    assert any("Missing rate" in issue for issue in result["issues"])


def test_check_data_quality_drift_schema_mismatch(reference_df):
    """Test phát hiện khi schema thay đổi."""
    current_df = reference_df.drop(columns=["feature_2"])
    
    result = check_data_quality_drift(reference_df, current_df)
    
    assert result["quality_drift_detected"] is True
    assert any("Schema mismatch" in issue for issue in result["issues"])


def test_detect_distribution_drift_no_drift(reference_df):
    """Test distribution drift khi không có drift."""
    result = detect_distribution_drift(
        reference_df["feature_1"],
        reference_df["feature_1"],
        "Test Feature"
    )
    
    assert result["drift_detected"] is False
    assert result["severity"] == "low"


def test_detect_distribution_drift_high_drift(reference_df, current_df_with_drift):
    """Test distribution drift khi có drift mạnh."""
    result = detect_distribution_drift(
        reference_df["feature_1"],
        current_df_with_drift["feature_1"],
        "Test Feature"
    )
    
    assert result["drift_detected"] is True
    assert result["severity"] in ["medium", "high"]


def test_page_hinkley_no_drift():
    """Test Page-Hinkley khi không có drift."""
    errors = np.random.normal(0, 0.1, 100)
    result = page_hinkley_test(errors, lambda_ph=50.0)
    
    assert result["drift_detected"] is False


def test_page_hinkley_sudden_drift():
    """Test Page-Hinkley phát hiện drift đột ngột."""
    # 50 giá trị đầu ổn định, 50 giá trị sau tăng đột ngột
    errors = np.concatenate([
        np.random.normal(0, 0.1, 50),
        np.random.normal(5, 0.1, 50)  # Mean tăng từ 0 lên 5
    ])
    
    result = page_hinkley_test(errors, lambda_ph=50.0)
    
    assert result["drift_detected"] is True


def test_page_hinkley_gradual_drift():
    """Test Page-Hinkley phát hiện drift dần dần."""
    # Lỗi tăng dần đều
    errors = np.linspace(0, 10, 200)
    
    result = page_hinkley_test(errors, lambda_ph=50.0)
    
    assert result["drift_detected"] is True


def test_detect_concept_drift_no_drift(reference_df, fake_model_good_fit):
    """Test concept drift khi model vẫn tốt."""
    ref_X = reference_df[["feature_1", "feature_2"]]
    ref_y = reference_df["log_amount_new_house_transactions"]
    
    result = detect_concept_drift_comprehensive(
        ref_X, ref_y,
        ref_X, ref_y,  # Cùng dữ liệu
        fake_model_good_fit,
        mae_threshold=0.2
    )
    
    # Không có concept drift vì model vẫn tốt
    assert result["concept_drift_detected"] is False


def test_detect_concept_drift_with_degradation(reference_df, current_df_with_drift, fake_model_degrading):
    """Test concept drift khi model suy giảm."""
    ref_X = reference_df[["feature_1", "feature_2"]]
    ref_y = reference_df["log_amount_new_house_transactions"]
    
    curr_X = current_df_with_drift[["feature_1", "feature_2"]]
    curr_y = current_df_with_drift["log_amount_new_house_transactions"]
    
    result = detect_concept_drift_comprehensive(
        ref_X, ref_y,
        curr_X, curr_y,
        fake_model_degrading,
        mae_threshold=0.2
    )
    
    # Phải phát hiện concept drift
    assert result["concept_drift_detected"] is True


def test_detect_data_drift_no_drift(full_df, fake_model_good_fit):
    """Test detect_data_drift khi không có drift."""
    result = detect_data_drift(
        full_df,
        model=fake_model_good_fit,
        target_col="log_amount_new_house_transactions",
        feature_cols=["feature_1", "feature_2"],
        split_ratio=0.5
    )
    
    # Kiểm tra cấu trúc output
    assert "overall_drift_detected" in result
    assert "severity" in result
    assert "summary" in result
    
    # Không có drift hoặc severity thấp
    assert result["severity"] in ["low", "medium"]


def test_detect_data_drift_with_drift(full_df_with_drift, fake_model_degrading):
    """Test detect_data_drift khi có drift."""
    result = detect_data_drift(
        full_df_with_drift,
        model=fake_model_degrading,
        target_col="log_amount_new_house_transactions",
        feature_cols=["feature_1", "feature_2"],
        split_ratio=0.5
    )
    
    # Phải phát hiện drift
    assert result["overall_drift_detected"] is True
    assert result["severity"] in ["medium", "high"]


def test_detect_data_drift_output_structure(full_df, fake_model_good_fit):
    """Test cấu trúc output của detect_data_drift."""
    result = detect_data_drift(
        full_df,
        model=fake_model_good_fit,
        target_col="log_amount_new_house_transactions",
        feature_cols=["feature_1", "feature_2"]
    )
    
    # Kiểm tra các keys bắt buộc
    required_keys = [
        "overall_drift_detected",
        "severity",
        "recommendation",
        "summary"
    ]
    
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    
    # Kiểm tra summary structure
    summary_keys = [
        "feature_drift_ratio",
        "data_quality_issues_count",
        "concept_drift_detected"
    ]
    
    for key in summary_keys:
        assert key in result["summary"], f"Missing summary key: {key}"