import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.model_config import ModelConfig

# Cấu hình logger
logger = logging.getLogger(__name__)


def save_reference_dataset(
    df: pd.DataFrame,
    description: str = "Baseline reference dataset for drift detection",
) -> Path:
    """
    Lưu dataset tham chiếu (reference) dưới dạng Parquet.
    """
    if df.empty:
        logger.warning("Attempted to save an empty DataFrame as reference dataset.")

    path = Path(ModelConfig.REFERENCE_DATA_PATH)

    try:
        # Đảm bảo thư mục cha tồn tại
        path.parent.mkdir(parents=True, exist_ok=True)

        # Lưu parquet với engine pyarrow (chuẩn công nghiệp, xử lý schema tốt hơn)
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")

        logger.info(f" Saved reference dataset ({len(df)} rows, {len(df.columns)} cols) -> {path}")
        return path

    except Exception as e:
        logger.error(f"Failed to save reference dataset to {path}. Error: {str(e)}")
        raise


def load_reference_dataset() -> Optional[pd.DataFrame]:
    """
    Tải dataset tham chiếu. Trả về None nếu không tìm thấy file.
    """
    path = Path(ModelConfig.REFERENCE_DATA_PATH)

    if not path.exists():
        logger.warning(f"Reference dataset not found at {path}. Drift detection will use current data as baseline.")
        return None

    try:
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info(f"Loaded reference dataset ({len(df)} rows) from {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load reference dataset from {path}. Error: {str(e)}")
        return None


def save_reference_statistics(df: pd.DataFrame) -> Path:
    """
    Tính toán và lưu thống kê cơ bản cho TẤT CẢ các loại cột (Numeric & Categorical).
    """
    stats = {}

    for col in df.columns:
        col_data = df[col]
        null_count = int(col_data.isna().sum())
        total_count = len(col_data)

        # 1. Thống kê cho cột số (Numeric)
        if pd.api.types.is_numeric_dtype(col_data):
            stats[col] = {
                "dtype": str(col_data.dtype),
                "count": int(total_count - null_count),
                "null_count": null_count,
                "null_pct": (round(null_count / total_count, 4) if total_count > 0 else 0.0),
                "mean": float(col_data.mean()) if null_count < total_count else None,
                "std": (
                    float(col_data.std(ddof=0)) if null_count < total_count else None
                ),  # ddof=0: population std dev
                "min": float(col_data.min()) if null_count < total_count else None,
                "max": float(col_data.max()) if null_count < total_count else None,
                "median": (float(col_data.median()) if null_count < total_count else None),
            }

        # 2. Thống kê cho cột phân loại (Categorical / Object / String)
        elif (
            pd.api.types.is_categorical_dtype(col_data)
            or pd.api.types.is_object_dtype(col_data)
            or pd.api.types.is_string_dtype(col_data)
        ):
            # Đếm frequency, bỏ qua NaN
            value_counts = col_data.value_counts(dropna=True)
            top_val = value_counts.index[0] if not value_counts.empty else None
            top_freq = int(value_counts.iloc[0]) if not value_counts.empty else 0

            stats[col] = {
                "dtype": str(col_data.dtype),
                "count": int(total_count - null_count),
                "null_count": null_count,
                "null_pct": (round(null_count / total_count, 4) if total_count > 0 else 0.0),
                "unique_count": int(col_data.nunique()),
                "top_value": str(top_val) if top_val is not None else None,
                "top_freq": top_freq,
                "top_freq_pct": (
                    round(top_freq / (total_count - null_count), 4) if (total_count - null_count) > 0 else 0.0
                ),
            }
        else:
            # Fallback cho các kiểu dữ liệu lạ (datetime, bool, v.v.)
            stats[col] = {
                "dtype": str(col_data.dtype),
                "count": int(total_count - null_count),
                "null_count": null_count,
                "unique_count": int(col_data.nunique()),
            }

    path = Path(ModelConfig.ARTIFACT_DIR) / "reference_stats.json"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

        logger.info(f" Saved reference statistics for {len(stats)} columns -> {path}")
        return path
    except Exception as e:
        logger.error(f" Failed to save reference statistics to {path}. Error: {str(e)}")
        raise
