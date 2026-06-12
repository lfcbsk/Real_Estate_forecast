import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import anderson_ksamp, ks_2samp

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. UTILITIES & STATISTICAL TESTS
# ==============================================================================


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: slight change
    PSI >= 0.2: Large drift
    """
    min_val = min(np.min(expected), np.min(actual))
    max_val = max(np.max(expected), np.max(actual))
    epsilon = 1e-5

    expected_counts, bin_edges = np.histogram(
        expected, bins=buckets, range=(min_val, max_val)
    )
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = (expected_counts + epsilon) / (
        len(expected) + epsilon * buckets
    )
    actual_percents = (actual_counts + epsilon) / (len(actual) + epsilon * buckets)

    psi = np.sum(
        (actual_percents - expected_percents)
        * np.log(actual_percents / expected_percents)
    )
    return float(psi)


def analyze_feature_drift_stats(
    ref: pd.Series, curr: pd.Series, alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Run KS-test, Anderson-Darling và PSI for a feature.
    """
    ref_clean = ref.dropna().values
    curr_clean = curr.dropna().values

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return {"error": "Insufficient data after dropping NaNs"}

    # 1. Kolmogorov-Smirnov
    ks_stat, ks_pval = ks_2samp(ref_clean, curr_clean)

    # 2. Anderson-Darling
    try:
        res = anderson_ksamp([ref_clean, curr_clean])
        ad_stat, ad_pval = res.statistic, res.pvalue
    except Exception:
        ad_stat, ad_pval = np.nan, np.nan

    # 3. PSI
    psi_val = calculate_psi(ref_clean, curr_clean)

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "ks_drift": ks_pval < alpha,
        "ad_statistic": float(ad_stat) if not np.isnan(ad_stat) else None,
        "ad_pvalue": float(ad_pval) if not np.isnan(ad_pval) else None,
        "ad_drift": ad_pval < alpha if not np.isnan(ad_pval) else False,
        "psi": float(psi_val),
        "psi_drift": psi_val >= 0.2,
    }


# ==============================================================================
# 2. DATA QUALITY DRIFT
# ==============================================================================


def check_data_quality_drift(
    ref_df: pd.DataFrame, curr_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Check data quality (Missing rate, Schema).
    """
    issues = []

    # 1. Missing rate increase
    ref_missing = ref_df.isnull().mean()
    curr_missing = curr_df.isnull().mean()

    for col in ref_missing.index:
        if col not in curr_missing.index:
            continue
        if curr_missing[col] > ref_missing[col] + 0.05:
            issues.append(
                f"Missing rate in '{col}' increased from {ref_missing[col]:.2%} to {curr_missing[col]:.2%}"
            )

    # 2. New columns or missing columns
    if set(ref_df.columns) != set(curr_df.columns):
        issues.append("Schema mismatch: Columns added or removed.")

    return {"quality_drift_detected": len(issues) > 0, "issues": issues}


# ==============================================================================
# 3. LABEL & PREDICTION DRIFT
# ==============================================================================


def detect_distribution_drift(
    ref_series: pd.Series, curr_series: pd.Series, name: str, alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Wrapper check drift for Label or Prediction.
    """
    stats = analyze_feature_drift_stats(ref_series, curr_series, alpha)
    psi = stats.get("psi", 0)

    severity = "low"
    if psi >= 0.2 or stats.get("ks_drift"):
        severity = "high"
    elif psi >= 0.1:
        severity = "medium"

    return {
        "target": name,
        "drift_detected": severity != "low",
        "severity": severity,
        "psi": psi,
        "ks_pvalue": stats.get("ks_pvalue"),
    }


# ==============================================================================
# 4. CONCEPT DRIFT (Page-Hinkley & MAE Degradation)
# ==============================================================================


def page_hinkley_test(
    errors: np.ndarray, delta: float = 0.005, lambda_ph: float = 50.0
) -> Dict[str, Any]:
    """
    Page-Hinkley Test for Time series  (sequential).
    deetect gradual or sudden drift in errors from predictions.
    """
    n = len(errors)
    if n == 0:
        return {"drift_detected": False, "max_m": 0}

    mean_error = np.mean(errors)
    m_t = 0.0
    max_m = 0.0

    for x_t in errors:
        m_t = m_t + (x_t - mean_error - delta)
        if m_t < 0:
            m_t = 0.0
        if m_t > max_m:
            max_m = m_t

    drift_detected = bool(max_m > lambda_ph)

    return {
        "drift_detected": drift_detected,
        "max_m_statistic": float(max_m),
        "threshold": lambda_ph,
    }


def detect_concept_drift_comprehensive(
    ref_X: pd.DataFrame,
    ref_y: pd.Series,
    curr_X: pd.DataFrame,
    curr_y: pd.Series,
    model,
    mae_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    check concept drift with MAE and Page-Hinkley on residuals.
    """
    try:
        ref_pred = np.asarray(model.predict(ref_X.fillna(0))).reshape(-1)
        curr_pred = np.asarray(model.predict(curr_X.fillna(0))).reshape(-1)

        # 1. MAE Degradation
        baseline_mae = float(np.mean(np.abs(ref_y - ref_pred)))
        current_mae = float(np.mean(np.abs(curr_y - curr_pred)))
        mae_degradation = (
            (current_mae - baseline_mae) / baseline_mae if baseline_mae > 0 else 0.0
        )

        # 2. Page-Hinkley on current absolute errors
        curr_abs_errors = np.abs(curr_y - curr_pred)
        ph_result = page_hinkley_test(curr_abs_errors)

        concept_drift_flag = bool(
            (mae_degradation > mae_threshold) or ph_result["drift_detected"]
        )

        return {
            "concept_drift_detected": concept_drift_flag,
            "baseline_mae": baseline_mae,
            "current_mae": current_mae,
            "mae_degradation_pct": mae_degradation,
            "page_hinkley": ph_result,
        }
    except Exception as e:
        return {"error": str(e), "concept_drift_detected": False}


# ==============================================================================
# 5. MAIN ORCHESTRATOR
# ==============================================================================


def detect_data_drift(
    df: pd.DataFrame,
    model: Any = None,
    target_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    alpha: float = 0.05,
    split_ratio: float = 0.5,
    mae_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    call all drift.
    """
    # ---- 1) Split reference / current ----
    df_sorted = (
        df.sort_values("date").reset_index(drop=True)
        if "date" in df.columns
        else df.reset_index(drop=True)
    )
    split_idx = max(1, int(len(df_sorted) * split_ratio))

    ref_df = df_sorted.iloc[:split_idx].copy()
    curr_df = df_sorted.iloc[split_idx:].copy()

    # ---- 2) Data Quality Drift ----
    quality_report = check_data_quality_drift(ref_df, curr_df)

    # ---- 3) Feature Drift ----
    numeric_cols = (
        feature_cols
        if feature_cols
        else ref_df.select_dtypes(include=np.number).columns.tolist()
    )
    feature_report = {}
    drifted_features_count = 0

    for col in numeric_cols:
        if col in ref_df.columns and col in curr_df.columns:
            stats = analyze_feature_drift_stats(ref_df[col], curr_df[col], alpha)
            feature_report[col] = stats
            if stats.get("ks_drift") or stats.get("psi_drift"):
                drifted_features_count += 1

    drift_ratio = drifted_features_count / len(numeric_cols) if numeric_cols else 0.0

    # ---- 4) Label Drift ----
    label_report = None
    if target_col and target_col in df_sorted.columns:
        label_report = detect_distribution_drift(
            ref_df[target_col], curr_df[target_col], "Target/Label", alpha
        )

    # ---- 5) Prediction Drift ----
    pred_report = None
    if model is not None and feature_cols is not None:
        try:
            ref_pred = model.predict(ref_df[feature_cols].fillna(0))
            curr_pred = model.predict(curr_df[feature_cols].fillna(0))
            pred_report = detect_distribution_drift(
                pd.Series(ref_pred), pd.Series(curr_pred), "Model Predictions", alpha
            )
        except Exception:
            pred_report = {"error": "Failed to generate predictions"}

    # ---- 6) Concept Drift ----
    concept_report = None
    if (
        model is not None
        and target_col is not None
        and target_col in df_sorted.columns
        and feature_cols is not None
    ):
        concept_report = detect_concept_drift_comprehensive(
            ref_df[feature_cols],
            ref_df[target_col],
            curr_df[feature_cols],
            curr_df[target_col],
            model,
            mae_threshold,
        )

    # ---- 7) Aggregate Severity & Recommendation ----
    concept_flag = bool(concept_report and concept_report.get("concept_drift_detected"))
    quality_flag = quality_report["quality_drift_detected"]

    # Logic quyết định severity
    if concept_flag or drift_ratio > 0.5 or quality_flag:
        severity = "high"
        recommendation = (
            "URGENT: Retrain model immediately. Check data pipeline for quality issues."
        )
    elif drift_ratio > 0.2 or (label_report and label_report["severity"] == "medium"):
        severity = "medium"
        recommendation = "WARNING: Monitor closely. Prepare retraining pipeline. Investigate drifted features."
    else:
        severity = "low"
        recommendation = "OK: System is stable. Continue routine monitoring."

    return {
        "overall_drift_detected": severity != "low",
        "severity": severity,
        "recommendation": recommendation,
        "summary": {
            "feature_drift_ratio": round(drift_ratio, 3),
            "data_quality_issues_count": len(quality_report["issues"]),
            "concept_drift_detected": concept_flag,
            "data_quality": quality_report,
            "feature_drift": feature_report,
            "label_drift": label_report,
            "prediction_drift": pred_report,
            "concept_drift": concept_report,
        },
    }
