"""Prometheus metrics: API latency, errors, and system health gauges."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

from prometheus_client import Gauge, Info

from src.pipeline.data_store import RAW_FILE_SPECS, get_train_dir
from src.utils.config import load_config

logger = logging.getLogger(__name__)

cfg = load_config()
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Custom gauges (system / MLOps health) ─────────────────────────────────────
MODEL_LOADED = Gauge(
    "realestate_model_loaded",
    "1 if production ONNX model and artifacts are loadable",
)
TRAIN_DATA_OK = Gauge(
    "realestate_train_data_ok",
    "1 if all 3 raw training CSV files exist under data/train",
)
MLFLOW_DB_OK = Gauge(
    "realestate_mlflow_db_ok",
    "1 if MLflow SQLite tracking database is reachable",
)
ARTIFACTS_OK = Gauge(
    "realestate_artifacts_ok",
    "1 if required model artifacts exist on disk",
)
BUILD_INFO = Info("realestate_build", "Application build metadata")


def _mlflow_db_path() -> Path:
    uri = cfg.get("mlflow", {}).get("tracking_uri", "sqlite:///mlruns.db")
    if uri.startswith("sqlite:///"):
        rel = uri.replace("sqlite:///", "")
        path = Path(rel)
        if not path.is_absolute():
            path = PROJECT_ROOT / rel
        return path
    return PROJECT_ROOT / "mlruns.db"


def update_system_gauges() -> None:
    """Refresh custom Prometheus gauges (call periodically)."""
    from src.models.model_config import ModelConfig

    artifacts_ok = all(
        p.exists()
        for p in (
            ModelConfig.MODEL_PATH,
            ModelConfig.FEATURE_LIST_PATH,
            ModelConfig.ZERO_SECTOR_PATH,
        )
    )
    ARTIFACTS_OK.set(1 if artifacts_ok else 0)

    try:
        from src.models.model_registry import ModelRegistry

        ModelRegistry()
        MODEL_LOADED.set(1)
    except Exception:
        MODEL_LOADED.set(0)

    train_dir = get_train_dir()
    csv_ok = all((train_dir / RAW_FILE_SPECS[k]["filename"]).exists() for k in RAW_FILE_SPECS)
    TRAIN_DATA_OK.set(1 if csv_ok else 0)

    db_path = _mlflow_db_path()
    try:
        if db_path.exists():
            with sqlite3.connect(str(db_path), timeout=2) as conn:
                conn.execute("SELECT 1")
            MLFLOW_DB_OK.set(1)
        else:
            MLFLOW_DB_OK.set(0)
    except Exception:
        MLFLOW_DB_OK.set(0)


async def _gauge_refresh_loop(interval: int = 15) -> None:
    while True:
        try:
            update_system_gauges()
        except Exception as exc:
            logger.warning("Failed to update system gauges: %s", exc)
        await asyncio.sleep(interval)


def start_gauge_refresh_task() -> asyncio.Task:
    BUILD_INFO.info({"app": "realestate-forecast", "component": "api"})
    update_system_gauges()
    return asyncio.create_task(_gauge_refresh_loop())
