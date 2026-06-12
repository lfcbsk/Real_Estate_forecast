"""
Drift check -> retrain decision -> registry gate orchestration.

Flow (GitHub Action / CLI):
    load data -> load prod model -> detect_data_drift()
    -> severity low? stop : retrain -> evaluate -> registry gate -> promote
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.models.model_config import ModelConfig
from src.models.model_registry import ModelRegistry
from src.models.retrain import retrain_model, save_artifacts, save_onnx_model
from src.monitoring.detect_drift import detect_data_drift
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

REGISTRY_CFG = cfg.get("orchestration", {}).get("registry", {})


@dataclass
class RegistryGate:
    min_competition_score: float = 0.55
    min_r2: float = 0.0
    max_mape: float = 100.0
    require_improvement_over_current: bool = False

    @classmethod
    def from_config(cls) -> "RegistryGate":
        return cls(
            min_competition_score=float(REGISTRY_CFG.get("min_competition_score", 0.55)),
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


def load_data(
    new_data: Optional[pd.DataFrame] = None,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split training data from CSVs (optionally merge new rows)."""
    df_train, df_test = ingest_run(test_ratio=test_ratio, save_outputs=False)

    if new_data is None:
        return df_train, df_test

    combined = (
        pd.concat([df_train, df_test, new_data])
        .drop_duplicates(subset=["date", "sector"], keep="last")
        .sort_values(["sector", "date"])
        .reset_index(drop=True)
    )
    split_idx = int(len(combined) * (1 - test_ratio))
    return combined.iloc[:split_idx].copy(), combined.iloc[split_idx:].copy()


def load_production_model() -> Optional[ModelRegistry]:
    """Load the production ONNX model from artifacts/ (returns None if missing)."""
    if not ModelConfig.MODEL_PATH.exists():
        return None
    try:
        return ModelRegistry()
    except Exception:
        return None


def prepare_drift_dataframe(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], float]:
    """
    Build a featured dataset for detect_data_drift().

    Reference rows come first chronologically; split_ratio tells detect_data_drift
    where to cut between reference and current windows.
    """
    combined_raw = (
        pd.concat([reference_df, current_df])
        .drop_duplicates(subset=["date", "sector"], keep="last")
        .sort_values(["date", "sector"])
        .reset_index(drop=True)
    )

    sector_stats = compute_sector_stats(reference_df, TARGET_LOG)
    sector_profile = build_sector_profile(reference_df)
    featured = create_training_features(
        combined_raw,
        target_col=TARGET_LOG,
        sector_stats=sector_stats,
        sector_profile=sector_profile,
        keep_nan=False,
    )
    featured = featured.sort_values("date").reset_index(drop=True)
    feature_cols = get_valid_features(featured)

    split_ratio = len(reference_df) / len(featured) if len(featured) else 0.5
    split_ratio = min(max(split_ratio, 0.01), 0.99)
    return featured, feature_cols, split_ratio


def should_retrain(drift_report: Dict[str, Any]) -> bool:
    """Retrain when drift severity is not low."""
    return drift_report.get("severity", "low") != "low"


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
    """Persist ONNX model, pickles, and refresh artifacts/reference.parquet."""
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
    messages: List[str] = []

    # 1. Load data
    df_train, df_test = load_data(new_data=new_data, test_ratio=test_ratio)
    messages.append(f"Loaded data: train={len(df_train)}, test={len(df_test)}")

    # 2. Load / bootstrap reference baseline (artifacts/reference.parquet)
    reference_df = reference_df or load_reference_dataset()
    if reference_df is None:
        reference_df = df_train.copy()
        save_reference_dataset(reference_df)
        messages.append("No reference found; saved training split to artifacts/reference.parquet")
    else:
        messages.append(f"Loaded reference baseline ({len(reference_df)} rows)")

    # 3. Load production model
    prod_model = model or load_production_model()
    if prod_model is None:
        messages.append("No production model in artifacts/ — concept/prediction drift skipped")
    else:
        messages.append("Loaded production model from artifacts/")

    # 4. Drift detection via detect_data_drift()
    current_df = df_test if len(df_test) > 0 else df_train.tail(max(1, len(df_train) // 5))
    drift_df, feature_cols, split_ratio = prepare_drift_dataframe(reference_df, current_df)

    drift_report = detect_data_drift(
        drift_df,
        model=prod_model,
        target_col=TARGET_LOG,
        feature_cols=feature_cols,
        split_ratio=split_ratio,
    )
    save_drift_report(drift_report)

    severity = drift_report.get("severity", "low")
    retrain_needed = should_retrain(drift_report)
    messages.append(f"Drift severity: {severity}")
    messages.append(f"Retrain needed: {retrain_needed}")

    result = OrchestrationResult(
        drift_report=drift_report,
        should_retrain=retrain_needed,
        retrain_triggered=False,
        messages=messages,
    )

    # 5. severity == low → stop
    if severity == "low":
        messages.append("Severity is low — stopping without retrain")
        return result

    result.retrain_triggered = True

    # 6. Retrain
    if tune:
        from src.pipeline.training import run_pipeline

        pipeline_result = run_pipeline(df_train=df_train, tune=True, n_trials=n_trials)
        eval_metrics = pipeline_result.get("test_results", {})
        messages.append("Full tuning pipeline completed")

        eligible, gate_messages = evaluate_for_registry(
            eval_metrics, current_metrics=current_metrics
        )
        result.evaluation = eval_metrics
        result.registry_eligible = eligible
        result.messages.extend(gate_messages)

        if eligible and promote:
            save_reference_dataset(df_train)
            result.registry_promoted = True
            result.messages.append("Model promoted via training pipeline; reference.parquet updated")
        elif not eligible:
            result.messages.append("Model not promoted — registry gates not met")
        return result

    zero_sectors, _ = build_zero_sector_mask(df_train)
    artifacts = retrain_model(df_train, cat_params or {}, model_name="ORCHESTRATION RETRAIN")
    messages.append("Fast retrain completed (no Optuna tuning)")

    # 7. Evaluate
    eval_metrics = evaluate_holdout(
        model=artifacts["model"],
        train_df=df_train,
        test_df=df_test,
        zero_sectors=zero_sectors,
    )
    result.evaluation = eval_metrics

    # 8. Registry gate
    eligible, gate_messages = evaluate_for_registry(eval_metrics, current_metrics=current_metrics)
    result.registry_eligible = eligible
    result.messages.extend(gate_messages)

    # 9. Promote
    if eligible and promote:
        promote_to_registry(
            artifacts=artifacts,
            zero_sectors=zero_sectors,
            reference_df=df_train,
        )
        result.registry_promoted = True
        result.messages.append("Model promoted to artifacts/; reference.parquet updated")
    elif not eligible:
        result.messages.append("Model not promoted — registry gates not met")

    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run drift -> retrain -> registry orchestration")
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
