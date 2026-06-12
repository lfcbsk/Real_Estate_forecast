import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Cấu hình logging cơ bản (có thể điều chỉnh level thành INFO hoặc DEBUG)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_monitoring_report(
    report: Dict[str, Any],
    model_name: str = "default_model",
    model_version: str = "v1.0.0",
    output_dir: str = "reports",
    metadata: Dict[str, Any] = None,
) -> str:
    """
    Lưu báo cáo monitoring kèm theo metadata để dễ truy vết (traceability).
    """
    try:
        # 1. Tạo thư mục nếu chưa có
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 2. Tạo timestamp và filename có ý nghĩa
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drift_report_{model_name}_{model_version}_{timestamp}.json"
        file_path = Path(output_dir) / filename

        # 3. Inject metadata vào report để lưu trữ cùng lúc
        enriched_report = report.copy()
        enriched_report["_metadata"] = {
            "model_name": model_name,
            "model_version": model_version,
            "evaluated_at": datetime.now().isoformat(),
            "custom_metadata": metadata or {},
        }

        # 4. Ghi file với error handling
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(enriched_report, f, indent=4, default=str, ensure_ascii=False)

        logger.info(f"✅ Monitoring report saved successfully: {file_path}")
        return str(file_path)

    except PermissionError:
        logger.error(f"❌ Permission denied when trying to save report to {output_dir}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to save monitoring report: {str(e)}")
        raise


def save_drift_report(
    report: Dict[str, Any],
    output_dir: str = "reports",
    model_name: str = "catboost",
    model_version: str = "production",
) -> str:
    """Persist a drift report JSON (alias for monitoring reports)."""
    return save_monitoring_report(
        report=report,
        model_name=model_name,
        model_version=model_version,
        output_dir=output_dir,
    )


def evaluate_retrain_decision(comprehensive_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Đánh giá xem có cần retrain model hay không dựa trên báo cáo toàn diện.
    Trả về một dict chứa quyết định và LÝ DO (explainability).
    """
    summary = comprehensive_report.get("summary", {})
    details = comprehensive_report.get("details", {})

    severity = comprehensive_report.get("severity", "low").lower()
    drift_ratio = summary.get("feature_drift_ratio", 0.0)
    concept_drift_detected = summary.get("concept_drift_detected", False)
    quality_issues_count = summary.get("data_quality_issues_count", 0)

    # Lấy chi tiết MAE degradation nếu có
    concept_details = details.get("concept_drift", {})
    mae_degradation = (
        concept_details.get("mae_degradation_pct", 0.0)
        if isinstance(concept_details, dict)
        else 0.0
    )

    decision = "HOLD"  # Mặc định là giữ nguyên model
    reasons = []
    priority = "LOW"

    # --- RULE 1: Data Quality Check (Quan trọng nhất) ---
    if quality_issues_count > 0:
        decision = "BLOCKED"
        priority = "CRITICAL"
        reasons.append(
            f"Data quality issues detected ({quality_issues_count} issues). Fix data pipeline BEFORE retraining."
        )
        logger.warning("⚠️ Retrain BLOCKED due to data quality issues.")

    # --- RULE 2: Concept Drift (Hiệu suất giảm) ---
    elif concept_drift_detected or mae_degradation > 0.2:
        decision = "RETRAIN"
        priority = "HIGH"
        reasons.append(
            f"Concept drift detected. MAE degraded by {mae_degradation:.1%}."
        )

    # --- RULE 3: Severe Feature/Label Drift ---
    elif severity == "high" or drift_ratio > 0.5:
        decision = "RETRAIN"
        priority = "HIGH"
        reasons.append(
            f"High severity data drift detected ({drift_ratio:.1%} features affected)."
        )

    # --- RULE 4: Moderate Drift (Cảnh báo) ---
    elif severity == "medium" or drift_ratio > 0.2:
        decision = "MONITOR"
        priority = "MEDIUM"
        reasons.append(
            "Moderate drift detected. Prepare retraining pipeline, but no immediate action required."
        )

    # --- RULE 5: Stable ---
    else:
        reasons.append(
            "System is stable. No significant drift or performance degradation."
        )

    result = {
        "decision": decision,  # BLOCKED, RETRAIN, MONITOR, HOLD
        "priority": priority,  # CRITICAL, HIGH, MEDIUM, LOW
        "reasons": reasons,
        "metrics_snapshot": {
            "drift_ratio": drift_ratio,
            "mae_degradation": mae_degradation,
            "concept_drift": concept_drift_detected,
        },
    }

    # Log quyết định
    log_level = logging.WARNING if decision in ["BLOCKED", "RETRAIN"] else logging.INFO
    logger.log(
        log_level,
        f"Retrain Decision: {result['decision']} | Priority: {result['priority']} | Reasons: {'; '.join(reasons)}",
    )

    return result
