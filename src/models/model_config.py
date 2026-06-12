from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:

    ARTIFACT_DIR: Path = Path("artifacts")

    MODEL_PATH: Path = ARTIFACT_DIR / "model.onnx"

    FEATURE_LIST_PATH: Path = ARTIFACT_DIR / "feature_list.pkl"

    SECTOR_STATS_PATH: Path = ARTIFACT_DIR / "sector_stats.pkl"

    SECTOR_PROFILE_PATH: Path = ARTIFACT_DIR / "sector_profile.pkl"

    ZERO_SECTOR_PATH: Path = ARTIFACT_DIR / "zero_sectors.pkl"

    REFERENCE_DATA_PATH: Path = ARTIFACT_DIR / "reference.parquet"
