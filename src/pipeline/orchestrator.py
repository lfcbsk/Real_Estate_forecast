"""
Drift check -> retrain decision -> registry gate orchestration.

Run after new data arrives to decide whether to retrain and promote a model.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.model_config import ModelConfig
from src.models.retrain import retrain_model, save_artifacts, save_onnx_model
from src.monitoring.detect_drift import (
    analyze_feature_drift_stats,
    check_data_quality_drift,
    detect_concept_drift_comprehensive,
    detect_distribution_drift,
)
from src.monitoring.log_report import save_drift_report
from src.monitoring.reference import load_reference_dataset, save_reference_dataset
from src.pipeline.evaluation import evaluate_holdout
from src.pipeline.features import (
    build_sector_profile,
    build_zero_sector_mask,
    compute_sector_stats,
    create_training_features,
    get_valid_features,
)
from src.pipeline.ingest_preprocess import run as ingest_run
from src.utils.config import load_config

cfg = load_config()
TARGET = cfg["target"]["column"]
TARGET_TRANSFORM = cfg["target"]["transform"]
TARGET_LOG = f"log_{TARGET}" if TARGET_TRANSFORM == "log1p" else TARGET

ORCH_CFG = cfg.get("orchestration", {})
DRIFT_CFG = ORCH_CFG.get("drift", {})
REGISTRY_CFG = ORCH_CFG.get("registry", {})


@dataclass
class RegistryGate:
    min_competition_score: float = 0.55
    min_r2: float = 0.0
    max_mape: float = 100.0
    require_improvement_over_current: bool = False

    @classmethod
    def from_config(cls) -> "RegistryGate":
        return cls(
            min_competition_score=float(
                REGISTRY_CFG.get("min_competition_score", 0.55)
            ),
            min_r2=float(REGISTRY_CFG.get("min_r2", 0.0)),
            max_mape=float(REGISTRY_CFG.get("max_mape", 100.0)),
            require_improvement_over_current=bool(
                REGISTRY_CFG.get("require_improvement_over_current", False)
            ),
        )


@dataclass
class OrchestrationResult:
    drift_report: Dict[str, Any]
    should_retrain: bool
    retrain_triggered: bool
    evaluation: Optional[Dict[str, Any]] = None
    registry_eligible: bool = False
    registry_promoted: bool = False
    messages: List[str] = field(default_factory=list)


def detect_drift_against_reference(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    model: Any = None,
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    alpha: float = 0.05,
    mae_threshold: float = 0.2,
) -> Dict[str, Any]:
    """Compare incoming data against a saved reference baseline."""
    quality_report = check_data_quality_drift(reference_df, current_df)

    if feature_cols is None:
        feature_cols = [
            c
            for c in reference_df.select_dtypes(include="number").columns
            if c in current_df.columns and c not in {target_col}
        ]

    feature_report: Dict[str, Any] = {}
    drifted_features_count = 0
    for col in feature_cols:
        stats = analyze_feature_drift_stats(reference_df[col], current_df[col], alpha)
        feature_report[col] = stats
        if stats.get("ks_drift") or stats.get("psi_drift"):
            drifted_features_count += 1

    drift_ratio = drifted_features_count / len(feature_cols) if feature_cols else 0.0

    label_report = None
    if (
        target_col
        and target_col in reference_df.columns
        and target_col in current_df.columns
    ):
        label_report = detect_distribution_drift(
            reference_df[target_col], current_df[target_col], "Target/Label", alpha
        )

    concept_report = None
    if (
        model is not None
        and target_col
        and feature_cols
        and target_col in reference_df.columns
        and target_col in current_df.columns
    ):
        concept_report = detect_concept_drift_comprehensive(
            reference_df[feature_cols],
            reference_df[target_col],
            current_df[feature_cols],
            current_df[target_col],
            model,
            mae_threshold,
        )

    concept_flag = bool(concept_report and concept_report.get("concept_drift_detected"))
    quality_flag = quality_report["quality_drift_detected"]
    feature_ratio_threshold = float(DRIFT_CFG.get("feature_drift_ratio_threshold", 0.2))

    if concept_flag or drift_ratio > 0.5 or quality_flag:
        severity = "high"
        recommendation = (
            "URGENT: Retrain model immediately. Check data pipeline for quality issues."
        )
    elif drift_ratio > feature_ratio_threshold or (
        label_report and label_report["severity"] == "medium"
    ):
        severity = "medium"
        recommendation = "WARNING: Monitor closely. Prepare retraining pipeline."
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
            "concept_drift": concept_report,
        },
    }


def should_retrain(drift_report: Dict[str, Any]) -> bool:
    """Decide if drift severity warrants a retrain."""
    severity_levels = DRIFT_CFG.get("severity_for_retrain", ["medium", "high"])
    severity = drift_report.get("severity", "low")
    if severity in severity_levels:
        return True
    return bool(drift_report.get("overall_drift_detected"))


def evaluate_for_registry(
    metrics: Dict[str, Any],
    gate: Optional[RegistryGate] = None,
    current_metrics: Optional[Dict[str, Any]] = None,
) -> tuple[bool, List[str]]:
    """Check whether retrained model meets promotion criteria."""
    gate = gate or RegistryGate.from_config()
    messages: List[str] = []
    score = metrics.get("competition_score", 0.0)
    r2 = metrics.get("r2", 0.0)
    mape = metrics.get("mape", float("inf"))

    if score < gate.min_competition_score:
        messages.append(
            f"Competition score {score:.4f} below minimum {gate.min_competition_score}"
        )
    if r2 < gate.min_r2:
        messages.append(f"R2 {r2:.4f} below minimum {gate.min_r2}")
    if mape > gate.max_mape:
        messages.append(f"MAPE {mape:.4f} above maximum {gate.max_mape}")

    if gate.require_improvement_over_current and current_metrics:
        current_score = current_metrics.get("competition_score", 0.0)
        if score <= current_score:
            messages.append(
                f"New score {score:.4f} did not beat current {current_score:.4f}"
            )

    eligible = len(messages) == 0
    if eligible:
        messages.append("Model passes all registry gates")
    return eligible, messages


def promote_to_registry(
    artifacts: Dict[str, Any],
    zero_sectors: set,
    reference_df: pd.DataFrame,
    artifact_dir: str | Path = ModelConfig.ARTIFACT_DIR,
) -> Path:
    """Persist ONNX model, pickles, and refresh reference baseline."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    save_onnx_model(artifacts["model"], path=str(artifact_dir / "model.onnx"))
    save_artifacts(artifacts, save_dir=str(artifact_dir))

    with open(artifact_dir / "zero_sectors.pkl", "wb") as f:
        pickle.dump(zero_sectors, f)

    save_reference_dataset(reference_df)
    return artifact_dir


def run_orchestration(
    new_data: Optional[pd.DataFrame] = None,
    model: Any = None,
    reference_df: Optional[pd.DataFrame] = None,
    cat_params: Optional[Dict[str, Any]] = None,
    tune: bool = False,
    n_trials: int = 10,
    test_ratio: float = 0.2,
    promote: bool = True,
    current_metrics: Optional[Dict[str, Any]] = None,
) -> OrchestrationResult:
    """
    Full workflow:
    1. Load/combine data
    2. Drift check vs reference
    3. Retrain if needed
    4. Holdout evaluation + registry gate
    5. Promote artifacts when eligible
    """
    messages: List[str] = []

    if new_data is None:
        df_train, df_test = ingest_run(test_ratio=test_ratio, save_outputs=False)
        combined = pd.concat([df_train, df_test]).sort_values(["sector", "date"])
    else:
        df_train, df_test = ingest_run(test_ratio=test_ratio, save_outputs=False)
        combined = pd.concat([df_train, df_test, new_data]).drop_duplicates(
            subset=["date", "sector"], keep="last"
        )
        combined = combined.sort_values(["sector", "date"]).reset_index(drop=True)
        split_idx = int(len(combined) * (1 - test_ratio))
        df_train = combined.iloc[:split_idx].copy()
        df_test = combined.iloc[split_idx:].copy()

    reference_df = reference_df or load_reference_dataset()
    if reference_df is None:
        reference_df = df_train.copy()
        messages.append("No reference dataset found; using training split as baseline")

    sector_stats = compute_sector_stats(df_train, TARGET_LOG)
    sector_profile = build_sector_profile(df_train)
    featured_train = create_training_features(
        df_train,
        target_col=TARGET_LOG,
        sector_stats=sector_stats,
        sector_profile=sector_profile,
        keep_nan=False,
    )
    feature_cols = get_valid_features(featured_train)

    current_slice = (
        df_test if len(df_test) > 0 else combined.tail(max(1, len(combined) // 5))
    )
    featured_current = create_training_features(
        pd.concat([df_train, current_slice]).sort_values(["sector", "date"]),
        target_col=TARGET_LOG,
        sector_stats=sector_stats,
        sector_profile=sector_profile,
        keep_nan=False,
    )
    featured_current = featured_current[
        featured_current["date"].isin(current_slice["date"])
    ]

    drift_report = detect_drift_against_reference(
        reference_df=reference_df,
        current_df=featured_current,
        model=model,
        feature_cols=feature_cols,
        target_col=TARGET_LOG,
    )
    save_drift_report(drift_report)

    retrain_needed = should_retrain(drift_report)
    messages.append(f"Drift severity: {drift_report['severity']}")
    messages.append(f"Retrain recommended: {retrain_needed}")

    result = OrchestrationResult(
        drift_report=drift_report,
        should_retrain=retrain_needed,
        retrain_triggered=False,
        messages=messages,
    )

    if not retrain_needed:
        messages.append("Skipping retrain; model remains in registry")
        return result

    result.retrain_triggered = True

    if tune:
        from src.pipeline.training import run_pipeline

        pipeline_result = run_pipeline(df_train=df_train, tune=True, n_trials=n_trials)
        eval_metrics = pipeline_result.get("test_results", {})
        zero_sectors = pipeline_result.get("zero_sectors", set())
        messages.append("Full tuning pipeline completed")
        eligible, gate_messages = evaluate_for_registry(
            eval_metrics, current_metrics=current_metrics
        )
        result.evaluation = eval_metrics
        result.registry_eligible = eligible
        result.messages.extend(gate_messages)
        result.registry_promoted = eligible and promote
        if result.registry_promoted:
            result.messages.append("Model promoted via training pipeline")
        elif not eligible:
            result.messages.append("Model not promoted — registry gates not met")
        return result

    zero_sectors, _ = build_zero_sector_mask(df_train)
    artifacts = retrain_model(
        df_train, cat_params or {}, model_name="ORCHESTRATION RETRAIN"
    )
    eval_metrics = evaluate_holdout(
        model=artifacts["model"],
        train_df=df_train,
        test_df=df_test,
        zero_sectors=zero_sectors,
    )
    messages.append("Fast retrain completed (no Optuna tuning)")

    result.evaluation = eval_metrics
    eligible, gate_messages = evaluate_for_registry(
        eval_metrics, current_metrics=current_metrics
    )
    result.registry_eligible = eligible
    result.messages.extend(gate_messages)

    if eligible and promote:
        promote_to_registry(
            artifacts=artifacts,
            zero_sectors=zero_sectors,
            reference_df=df_train,
        )
        result.registry_promoted = True
        result.messages.append("Model promoted to registry")
    elif not eligible:
        result.messages.append("Model not promoted — registry gates not met")

    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run drift -> retrain -> registry orchestration"
    )
    parser.add_argument("--tune", choices=["true", "false"], default="false")
    parser.add_argument("--promote", choices=["true", "false"], default="true")
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()

    outcome = run_orchestration(
        tune=args.tune == "true",
        promote=args.promote == "true",
        n_trials=args.n_trials,
    )

    print(
        json.dumps(
            {
                "should_retrain": outcome.should_retrain,
                "retrain_triggered": outcome.retrain_triggered,
                "registry_eligible": outcome.registry_eligible,
                "registry_promoted": outcome.registry_promoted,
                "drift_severity": outcome.drift_report.get("severity"),
                "messages": outcome.messages,
            },
            indent=2,
        )
    )
