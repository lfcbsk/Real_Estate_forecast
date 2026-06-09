import pandas as pd
import numpy as np
from src.pipeline.features import (
    compute_sector_stats, build_sector_profile,
    create_training_features, get_valid_features,
    apply_zero_sector_rule, build_zero_sector_mask,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error, median_absolute_error
from src.utils.config import load_config


cfg = load_config()

TARGET = cfg["target"]["column"]
TARGET_TRANSFORM = cfg["target"]["transform"]

TARGET_LOG = (
    f"log_{TARGET}"
    if TARGET_TRANSFORM == "log1p"
    else TARGET
)

def competition_score(y_true, y_pred, eps=1e-12):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ape = np.abs(
        y_pred - y_true
    ) / np.maximum(y_true, eps)

    # Stage 1
    bad_rate = (ape > 1.0).mean()
    if bad_rate > 0.30:
        return 0.0
    
    # Stage 2
    good_mask = ape <= 1.0
    D = ape[good_mask]
    if len(D) == 0:
        return 0.0
    mape_D = D.mean()
    good_rate = good_mask.mean()
    score = 1.0 - (mape_D / good_rate)

    return score

def evaluate_regression(y_true, y_pred, verbose=True):
    """
    Regression metrics dashboard
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    # tránh chia cho 0
    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(
            np.abs(y_true[mask] - y_pred[mask])
            / y_true[mask]
        ) * 100
    else:
        mape = np.nan
    bias = np.mean(y_pred - y_true)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "medae": medae,
        "mape": mape,
        "r2": r2,
        "Explained Variance": evs,
        "Bias": bias,
    }

    return metrics

def evaluate_holdout(model, train_df, test_df, zero_sectors):
    print("\n" + "="*60)
    print("HOLDOUT TEST")
    print("="*60)

    sector_stats   = compute_sector_stats(train_df, TARGET_LOG)
    sector_profile = build_sector_profile(train_df)

    # Tạo features trực tiếp từ test_df, KHÔNG mask hay concat
    featured = create_training_features(
        test_df,
        target_col=TARGET_LOG,
        sector_stats=sector_stats,
        sector_profile=sector_profile,
        keep_nan=False
    )

    test_feat      = featured.sort_values(["sector", "date"]).reset_index(drop=True)
    test_df_sorted = test_df.sort_values(["sector", "date"]).reset_index(drop=True)

    assert (test_df_sorted["sector"].values == test_feat["sector"].values).all()
    assert (test_df_sorted["date"].values   == test_feat["date"].values).all()

    feats  = get_valid_features(test_feat)
    X_test = test_feat[feats].fillna(0)


    pred_log = np.clip(model.predict(X_test), 0, None)
    pred     = np.expm1(pred_log)
    pred     = apply_zero_sector_rule(pred, test_feat["sector"], zero_sectors)
    y_true   = np.expm1(test_df_sorted[TARGET_LOG].values)

    ape      = np.abs(pred - y_true) / np.maximum(y_true, 1e-12)
    bad_mask = ape > 1.0

    print(f"Bad rate: {bad_mask.mean():.2%}")
    print(f"Bad cases: {bad_mask.sum()} / {len(bad_mask)}")

    # Phân tích bad cases theo sector
    bad_df = test_df_sorted.copy()
    bad_df["pred"]        = pred
    bad_df["y_true"]      = y_true
    bad_df["ape"]         = ape
    bad_df["is_bad"]      = bad_mask
    bad_df["overpredict"] = pred > y_true

    sector_bad = (
        bad_df.groupby("sector")
        .agg(
            bad_count   = ("is_bad",      "sum"),
            total       = ("is_bad",      "count"),
            mean_actual = ("y_true",      "mean"),
            mean_pred   = ("pred",        "mean"),
            zero_rate   = ("y_true",      lambda x: (x == 0).mean()),
            overpredict = ("overpredict", "mean"),
        )
        .query("bad_count > 0")
        .sort_values("bad_count", ascending=False)
    )

    print("\nSector analysis (bad cases):")
    print(sector_bad.head(15).to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nZero actual trong bad cases: {(y_true[bad_mask] == 0).mean():.2%}")
    print(f"Zero pred trong bad cases:   {(pred[bad_mask] == 0).mean():.2%}")

    metrics = evaluate_regression(y_true, pred)
    score   = competition_score(y_true, pred)

    print(f"\nCompetition Score: {score:.4f}")
    print("\nRegression Metrics")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric.upper():<10}: {value:.4f}")

    return {"competition_score": score, **metrics}