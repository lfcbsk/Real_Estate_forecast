"""FastAPI application for RealEstate Forecast API."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting RealEstate Forecast API...")

    try:
        from src.models.model_registry import ModelRegistry

        _ = ModelRegistry()
        logger.info("Model registry loaded successfully")
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")

    yield

    logger.info("Shutting down API...")


app = FastAPI(
    title="RealEstate Forecast API",
    description="""
## Real Estate Transaction Forecasting API

This API provides machine learning-powered forecasts for real estate transactions.

### Features:
- **Recursive Forecasting**: Multi-step ahead predictions using recursive strategy
- **ONNX Model Inference**: Fast predictions with optimized ONNX runtime
- **Drift Detection**: Monitor data drift for model reliability
- **MLflow Integration**: Track model metrics and versions

### Endpoints:
- `/api/v1/health` - Health check
- `/api/v1/forecast` - Generate n-month forecast
- `/api/v1/predict` - Single prediction with custom features
- `/api/v1/upload` - Batch prediction via file upload
- `/api/v1/sectors` - List all sectors
- `/api/v1/metrics` - Model metrics from MLflow
- `/api/v1/drift` - Drift detection report
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RealEstate Forecast API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
