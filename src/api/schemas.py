"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(default="1.0.0")


class ForecastRequest(BaseModel):
    """Request for forecast prediction."""

    n_months: int = Field(default=12, ge=1, le=36, description="Number of months to forecast")
    sectors: Optional[List[str]] = Field(None, description="Optional list of sectors to filter")


class ForecastData(BaseModel):
    """Single forecast data point."""

    date: datetime
    sector: str
    pred_amount: int


class ForecastResponse(BaseModel):
    """Forecast response with predictions."""

    status: str
    n_months: int
    total_predictions: int
    predictions: List[ForecastData]
    generated_at: datetime = Field(default_factory=datetime.now)


class PredictRequest(BaseModel):
    """Custom prediction request with input data."""

    features: Dict[str, Any] = Field(..., description="Feature dictionary for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "sector": "District 1",
                    "date": "2024-01-01",
                    "nearby_sectors": 5,
                    "pre_owned": 100,
                    "lag_1": 500,
                    "lag_2": 480,
                }
            }
        }


class PredictResponse(BaseModel):
    """Single prediction response."""

    predicted_value: float
    predicted_amount: int
    confidence: Optional[float] = None


class UploadResponse(BaseModel):
    """File upload response."""

    status: str
    filename: str
    rows_processed: int
    predictions: List[Dict[str, Any]]
    message: str


class SectorInfo(BaseModel):
    """Sector information."""

    sector_name: str | int
    is_zero_sector: bool
    historical_avg: Optional[float] = None


class SectorsResponse(BaseModel):
    """List of all sectors."""

    total_sectors: int
    zero_sectors_count: int
    active_sectors_count: int
    sectors: List[SectorInfo]


class MetricData(BaseModel):
    """Model metric data point."""

    name: str
    value: float
    step: Optional[int] = None
    timestamp: Optional[datetime] = None


class MetricsResponse(BaseModel):
    """Model metrics from MLflow."""

    status: str
    metrics: List[MetricData]
    model_version: Optional[str] = None


class DriftReport(BaseModel):
    """Drift detection report."""

    drift_detected: bool
    drift_score: float
    affected_features: List[str]
    severity: str = Field(..., description="low|medium|high")
    recommendation: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response."""

    status: str = "error"
    message: str
    detail: Optional[str] = None
