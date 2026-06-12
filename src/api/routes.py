"""API routes for RealEstate Forecast API."""

import io
import logging
from typing import  Optional
import pandas as pd
import mlflow
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from src.api.schemas import (
    DriftReport,
    ErrorResponse,
    ForecastData,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    MetricData,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    SectorInfo,
    SectorsResponse,
    UploadResponse,
)
from src.monitoring.detect_drift import detect_data_drift
from src.pipeline.predict import forecast_next_year
from src.utils.config import load_config

cfg = load_config()

TARGET = cfg["target"]["column"]
TARGET_TRANSFORM = cfg["target"]["transform"]

TARGET_LOG = f"log_{TARGET}" if TARGET_TRANSFORM == "log1p" else TARGET


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["forecast"])

# Global model registry cache
_model_registry = None
_train_data = None


def get_model_registry():
    """Lazy load model registry."""
    global _model_registry
    if _model_registry is None:
        from src.models.model_registry import ModelRegistry

        _model_registry = ModelRegistry()
    return _model_registry


def get_train_data():
    """Load training data for forecasting."""
    global _train_data
    if _train_data is None:
        from pathlib import Path

        train_dir = Path("data/train")
        if train_dir.exists():
            df_parts = []
            for file in train_dir.glob("*.csv"):
                df_parts.append(pd.read_csv(file))
            if df_parts:
                _train_data = pd.concat(df_parts, ignore_index=True)
                _train_data["date"] = pd.to_datetime(_train_data["date"])
    return _train_data


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    responses={
        200: {"description": "Successful forecast"},
        500: {"model": ErrorResponse},
    },
)
async def get_forecast(request: ForecastRequest):
    """
    Generate forecast for next n_months.

    Uses recursive forecasting to predict future values month by month.
    """
    try:
        registry = get_model_registry()
        train_data = get_train_data()

        if train_data is None or len(train_data) == 0:
            raise HTTPException(status_code=500, detail="Training data not available")

        df_forecast = forecast_next_year(
            df=train_data,
            model=registry,
            results={"zero_sectors": registry.zero_sectors},
            n_months=request.n_months,
        )

        # Lọc theo sectors nếu người dùng chỉ định
        if request.sectors:
            df_forecast = df_forecast[df_forecast["sector"].isin(request.sectors)]
            if df_forecast.empty:
                raise HTTPException(
                    status_code=400,
                    detail=f"No matching sectors found for: {request.sectors}",
                )

        predictions = []
        for _, row in df_forecast.iterrows():
            predictions.append(
                ForecastData(
                    date=row["date"],
                    sector=row["sector"],
                    pred_amount=int(row["pred_amount"]),
                )
            )

        return ForecastResponse(
            status="success",
            n_months=request.n_months,
            total_predictions=len(predictions),
            predictions=predictions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse},
    },
)
async def predict_single(request: PredictRequest):
    """
    Make single prediction with custom features.

    Accepts a feature dictionary and returns prediction.
    """
    try:
        registry = get_model_registry()

        features_dict = request.features.copy()

        if "sector" in features_dict:
            del features_dict["sector"]
        if "date" in features_dict:
            del features_dict["date"]

        import numpy as np

        X = pd.DataFrame([features_dict])
        pred_log = registry.predict(X)[0]
        pred_amount = int(np.expm1(pred_log))

        return PredictResponse(
            predicted_value=float(pred_log),
            predicted_amount=pred_amount,
            confidence=None,
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid features: {str(e)}")


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={200: {"description": "File processed successfully"}},
)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload CSV/Excel file for batch prediction.

    File must contain required feature columns.
    """
    try:
        contents = await file.read()

        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Use CSV or Excel."
            )

        if df.empty:
            raise HTTPException(status_code=400, detail="Empty file")

        registry = get_model_registry()
        import numpy as np

        feature_cols = [c for c in df.columns if c not in ["sector", "date", "target"]]
        X = df[feature_cols].fillna(0)

        preds_log = registry.predict(X)
        preds_amount = np.expm1(preds_log).astype(int)

        predictions = []
        for i, (_, row) in enumerate(df.iterrows()):
            pred_dict = row.to_dict()
            pred_dict["predicted_amount"] = int(preds_amount[i])
            pred_dict["predicted_log"] = float(preds_log[i])
            predictions.append(pred_dict)

        return UploadResponse(
            status="success",
            filename=file.filename,
            rows_processed=len(df),
            predictions=predictions,
            message=f"Successfully processed {len(df)} rows",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.get(
    "/sectors",
    response_model=SectorsResponse,
    tags=["Metadata"],
)
async def get_sectors():
    """Get list of all sectors with statistics."""
    try:
        registry = get_model_registry()
        train_data = get_train_data()

        zero_sectors = set(registry.zero_sectors)

        if train_data is not None:
            all_sectors = sorted(train_data["sector"].unique().tolist())
            sector_stats = (
                train_data.groupby("sector")["amount_new_house_transactions"]
                .mean()
                .to_dict()
            )
        else:
            all_sectors = []
            sector_stats = {}

        sectors_list = []
        for sector in all_sectors:
            sectors_list.append(
                SectorInfo(
                    sector_name=str(sector),
                    is_zero_sector=sector in zero_sectors,
                    historical_avg=float(sector_stats.get(sector, 0)),
                )
            )

        return SectorsResponse(
            total_sectors=len(all_sectors),
            zero_sectors_count=len(zero_sectors),
            active_sectors_count=len(all_sectors) - len(zero_sectors),
            sectors=sectors_list,
        )

    except Exception as e:
        logger.error(f"Sectors error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Monitoring"],
)
async def get_metrics(run_id: Optional[str] = Query(None, description="MLflow run ID")):
    """Get model metrics from MLflow."""
    try:
        mlflow.set_tracking_uri("sqlite:///mlruns.db")

        if run_id:
            run = mlflow.get_run(run_id)
            metrics_dict = run.data.metrics
        else:
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=["0"], order_by=["start_time DESC"], max_results=1
            )
            if not runs:
                return MetricsResponse(status="no_data", metrics=[])

            run = runs[0]
            metrics_dict = run.data.metrics

        metrics_list = []
        for name, value in metrics_dict.items():
            metrics_list.append(MetricData(name=name, value=value))

        return MetricsResponse(
            status="success",
            metrics=metrics_list,
            model_version=run.info.run_id[:8] if run else None,
        )

    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return MetricsResponse(status="error", metrics=[], model_version=None)


@router.get(
    "/drift",
    response_model=DriftReport,
    tags=["Monitoring"],
)
async def get_drift_report(
    reference_period: Optional[str] = Query(None, description="Reference period")
):
    """Get drift detection report."""
    try:

        train_data = get_train_data()
        if train_data is None:
            return DriftReport(
                drift_detected=False,
                drift_score=0.0,
                affected_features=[],
                severity="low",
                recommendation="No data available for drift detection",
            )

        registry = get_model_registry()

        result = detect_data_drift(
            train_data,
            model=registry,
            target_col=TARGET_LOG,
            feature_cols=registry.features,
        )
        summary = result["summary"]
        affected = [
            col
            for col, s in summary["feature_drift"].items()
            if s.get("ks_drift") or s.get("psi_drift")
        ]

        return DriftReport(
            drift_detected=result.get("overall_drift_detected", False),
            drift_score=float(summary.get("feature_drift_ratio", 0.0)),
            affected_features=affected,
            severity=result.get("severity", "low"),
            recommendation=result.get("recommendation", "Continue monitoring"),
        )

    except Exception as e:
        logger.error(f"Drift error: {str(e)}")
        return DriftReport(
            drift_detected=False,
            drift_score=0.0,
            affected_features=[],
            severity="low",
            recommendation=f"Error computing drift: {str(e)}",
        )
