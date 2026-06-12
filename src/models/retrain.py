import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor
from src.pipeline.features import (
    compute_sector_stats, build_sector_profile,
    create_training_features, get_valid_features,
    apply_zero_sector_rule, build_zero_sector_mask,
)
from pathlib import Path
import pickle
from src.pipeline.ingest_preprocess import run as ingest_run
from src.pipeline.evaluation import competition_score, evaluate_regression, evaluate_holdout
from src.utils.config import load_config


cfg = load_config()

TARGET = cfg["target"]["column"]
TARGET_TRANSFORM = cfg["target"]["transform"]

TARGET_LOG = (
    f"log_{TARGET}"
    if TARGET_TRANSFORM == "log1p"
    else TARGET
)

N_SPLITS    = 5
N_TRIALS    = 50
RANDOM_SEED = 42

def retrain_model(
    df,
    cat_params,
    model_name="MODEL"
):
    print("\n" + "=" * 60)
    print(f"RETRAIN {model_name}")
    print("=" * 60)

    sector_stats = compute_sector_stats(
        df,
        TARGET_LOG
    )

    sector_profile = build_sector_profile(
        df
    )

    featured = create_training_features(
        df,
        target_col=TARGET_LOG,
        sector_stats=sector_stats,
        sector_profile=sector_profile,
        keep_nan=False
    )

    feats = get_valid_features(featured)

    X = featured[feats].fillna(0)
    y = featured[TARGET_LOG]

    from src.pipeline.training import build_catboost

    model = build_catboost(cat_params)

    model.fit(X, y)
    return {
        "model": model,
        "sector_stats": sector_stats,
        "sector_profile": sector_profile,
        "features": feats,
    }

def save_onnx_model(
    model,
    path="artifacts/model.onnx"
):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_model(
        path,
        format="onnx"
    )

    print(f"✓ Saved ONNX model: {path}")

def save_artifacts(
    artifacts,
    save_dir="artifacts"
):
    save_dir = Path(save_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(save_dir / "sector_stats.pkl", "wb") as f:
        pickle.dump(
            artifacts["sector_stats"],
            f
        )

    with open(save_dir / "sector_profile.pkl", "wb") as f:
        pickle.dump(
            artifacts["sector_profile"],
            f
        )

    with open(save_dir / "feature_list.pkl", "wb") as f:
        pickle.dump(
            artifacts["features"],
            f
        )

    print("✓ Artifacts saved")
