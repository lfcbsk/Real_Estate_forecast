import pickle

import mlflow
import mlflow.catboost
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.models.retrain import retrain_model, save_artifacts, save_onnx_model
from src.monitoring.reference import save_reference_dataset, save_reference_statistics
from src.pipeline.evaluation import (
    competition_score,
    evaluate_holdout,
    evaluate_regression,
)
from src.pipeline.features import (
    apply_zero_sector_rule,
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

N_SPLITS = 5
N_TRIALS = 50
RANDOM_SEED = 42

MLFLOW_EXPERIMENT = "catboost_timeseries"


def build_catboost(params=None):
    default = {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "border_count": 128,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "verbose": 0,
        "random_seed": RANDOM_SEED,
        "loss_function": "MAE",
    }

    if params:
        merged = {**default, **params}
        if merged.get("bootstrap_type") == "Bernoulli":
            merged.pop("bagging_temperature", None)
        return CatBoostRegressor(**merged)

    return CatBoostRegressor(**default)


def timeseries_cv(df_all, model, n_splits=N_SPLITS, zero_sectors=None, verbose=True, mlflow_run=None):
    """
    Proper TimeSeriesSplit CV:
      - Feature engineering computed INSIDE each fold
      - sector_stats computed from train fold only (no leakage)
      - Zero-sector rule applied after prediction

    Returns
    -------
    fold_scores : list of competition scores
    oof_df      : DataFrame with columns [date, sector, actual, pred, fold]
    """
    fold_metrics = []
    dates = np.sort(df_all["date"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    oof_parts = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(dates)):
        train_dates = dates[train_idx]
        val_dates = dates[val_idx]

        raw_train = df_all[df_all["date"].isin(train_dates)]
        raw_val = df_all[df_all["date"].isin(val_dates)]

        # --- Sector stats from train only ---
        sector_stats = compute_sector_stats(raw_train, TARGET_LOG)
        sector_profile = build_sector_profile(raw_train)
        # --- Feature engineering (combine, compute, then split back) ---
        combined = pd.concat([raw_train, raw_val]).sort_values(["sector", "date"])
        featured = create_training_features(
            combined,
            TARGET_LOG,
            sector_stats=sector_stats,
            sector_profile=sector_profile,
            keep_nan=False,
        )
        fold_train = featured[featured["date"].isin(train_dates)]
        fold_val = featured[featured["date"].isin(val_dates)].copy()

        feats = get_valid_features(featured)

        X_train = fold_train[feats].fillna(0)
        y_train = fold_train[TARGET_LOG]
        X_val = fold_val[feats].fillna(0)
        y_val = fold_val[TARGET_LOG]

        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_val), 0, None)

        # --- Apply zero-sector rule ---
        if zero_sectors:
            pred = apply_zero_sector_rule(pred, fold_val["sector"], zero_sectors)

        pred_real = np.expm1(pred)
        y_real = np.expm1(y_val)
        metrics = evaluate_regression(y_real, pred_real)

        score = competition_score(y_real, pred_real)

        fold_scores.append(score)

        oof_part = fold_val[["date", "sector", TARGET_LOG]].copy()
        oof_part["pred"] = pred
        oof_part["fold"] = fold + 1
        oof_parts.append(oof_part)

        metrics["CompetitionScore"] = score
        metrics["Fold"] = fold + 1

        fold_metrics.append(metrics)
        if mlflow_run is not None:
            mlflow.log_metrics(
                {
                    f"fold{fold+1}_competition_score": score,
                    f"fold{fold+1}_mae": metrics["mae"],
                    f"fold{fold+1}_rmse": metrics["rmse"],
                    f"fold{fold+1}_r2": metrics["r2"],
                    f"fold{fold+1}_mape": metrics["mape"],
                },
                run_id=mlflow_run.info.run_id,
            )
    oof_df = pd.concat(oof_parts).reset_index(drop=True)
    metrics_df = pd.DataFrame(fold_metrics)

    if mlflow_run is not None:
        mlflow.log_metrics(
            {
                "cv_competition_score_mean": metrics_df["CompetitionScore"].mean(),
                "cv_competition_score_std": metrics_df["CompetitionScore"].std(),
                "cv_mae_mean": metrics_df["mae"].mean(),
                "cv_rmse_mean": metrics_df["rmse"].mean(),
                "cv_r2_mean": metrics_df["r2"].mean(),
                "cv_mape_mean": metrics_df["mape"].mean(),
            },
            run_id=mlflow_run.info.run_id,
        )
    if verbose:

        display_df = metrics_df[
            [
                "Fold",
                "CompetitionScore",
                "mae",
                "rmse",
                "r2",
                "mape",
            ]
        ].copy()

        mean_row = {
            "Fold": "MEAN",
            "CompetitionScore": display_df["CompetitionScore"].mean(),
            "mae": display_df["mae"].mean(),
            "rmse": display_df["rmse"].mean(),
            "r2": display_df["r2"].mean(),
            "mape": display_df["mape"].mean(),
        }

        std_row = {
            "Fold": "STD",
            "CompetitionScore": display_df["CompetitionScore"].std(),
            "mae": display_df["mae"].std(),
            "rmse": display_df["rmse"].std(),
            "r2": display_df["r2"].std(),
            "mape": display_df["mape"].std(),
        }

        summary_df = pd.concat([display_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

        print("\nFold Metrics")
        print("=" * 80)

        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("=" * 80)
    return fold_scores, oof_df


def catboost_objective(trial, df_all, zero_sectors, n_splits=N_SPLITS):

    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])

    params = {
        "iterations": trial.suggest_int("iterations", 1500, 5000),
        "depth": trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 15, 80),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 0.9),
        "border_count": trial.suggest_categorical("border_count", [128, 254]),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 3, 10),
        "bootstrap_type": bootstrap_type,
    }

    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 3.0)
    else:  # Bernoulli
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    with mlflow.start_run(
        run_name=f"optuna_trial_{trial.number}",
        nested=True,
    ) as child_run:
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        model = build_catboost(params)
        scores, _ = timeseries_cv(
            df_all,
            model,
            n_splits=n_splits,
            zero_sectors=zero_sectors,
            verbose=False,
            mlflow_run=child_run,
        )
        mean_score = np.mean(scores)
        mlflow.log_metric("mean_competition_score", mean_score)
    return np.mean(scores)


def tune_model(model_name, df_all, zero_sectors, n_trials=N_TRIALS, n_splits=N_SPLITS):
    """
    Run Optuna hyperparameter search for a given model.

    Returns
    -------
    best_params : dict
    study       : optuna.Study
    """
    assert model_name in ("catboost", "lgbm"), "model_name must be 'catboost' or 'lgbm'"

    print(f"\n{'='*60}")
    print(f"Optuna tuning: {model_name.upper()} | {n_trials} trials | {n_splits}-fold CV")
    print(f"{'='*60}")

    if model_name == "catboost":

        def objective(trial):
            return catboost_objective(
                trial,
                df_all,
                zero_sectors,
                n_splits,
            )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best score : {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params, study


def run_pipeline(df_train=None, tune=True, n_trials=N_TRIALS):
    """
    Full pipeline:
    1. Ingest & Preprocess (nếu df_train=None)
    2. Zero-sector rule
    3. Optuna tuning (CatBoost + LightGBM)
    4. Final CV for each model + ensemble
    5. Ensemble weight optimisation on OOF preds
    """
    # ── 0. Load data nếu chưa có ───────────────────────────────────────────
    if df_train is None:
        print("\n" + "=" * 80)
        print("STEP 0: Ingest & Preprocess")
        print("=" * 80)
        df_train, test_df = ingest_run(test_ratio=0.2, save_outputs=False)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="pipeline_run") as parent_run:
        mlflow.log_params(
            {
                "n_splits": N_SPLITS,
                "n_trials": n_trials,
                "random_seed": RANDOM_SEED,
                "tune": tune,
            }
        )
        # ── 1. Zero-sector rule ────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 1: Zero-Sector Rule")
        print("=" * 60)
        zero_sectors, sector_meta = build_zero_sector_mask(df_train)
        mlflow.log_param("n_zero_sectors", len(zero_sectors))
        # ── 2. Optuna tuning ───────────────────────────────────────
        if tune:
            print("\n" + "=" * 60)
            print("STEP 2: Optuna Tuning")
            print("=" * 60)
            cat_best_params, study = tune_model("catboost", df_train, zero_sectors, n_trials)
            mlflow.log_params({f"best_{k}": v for k, v in cat_best_params.items()})
            mlflow.log_metric("optuna_best_score", study.best_value)
        else:
            cat_best_params = {}

        # ── 3. Full CV with tuned models ───────────────────────────
        print("\n" + "=" * 60)
        print("STEP 3: Full TimeSeriesSplit CV (5-fold)")
        print("=" * 60)

        cat_model = build_catboost(cat_best_params)

        print("CatBoost:")
        cat_scores, cat_oof = timeseries_cv(df_train, cat_model, zero_sectors=zero_sectors, mlflow_run=parent_run)
        # ── Summary ────────────────────────────────────────────────
        cat_mean = np.mean(cat_scores)
        cat_std = np.std(cat_scores)

        mlflow.log_metrics(
            {
                "final_cat_score_mean": cat_mean,
                "final_cat_score_std": cat_std,
            }
        )

        print("\n" + "=" * 60)
        print("STEP 4: Retrain Final Model")
        print("=" * 60)

        test_artifacts = retrain_model(df_train, cat_best_params, model_name="TEST MODEL")

        final_model = test_artifacts["model"]

        print("\n" + "=" * 60)
        print("STEP 5: Holdout Evaluation")
        print("=" * 60)

        test_results = evaluate_holdout(
            model=final_model,
            train_df=df_train,
            test_df=test_df,
            zero_sectors=zero_sectors,
        )
        mlflow.log_metrics(
            {
                "test_competition_score": test_results["competition_score"],
                "test_mae": test_results["mae"],
                "test_rmse": test_results["rmse"],
                "test_r2": test_results["r2"],
                "test_mape": test_results["mape"],
            }
        )
        print("\n" + "=" * 60)
        print("STEP 6: Train Production Model")
        print("=" * 60)

        full_df = pd.concat([df_train, test_df]).sort_values(["sector", "date"]).reset_index(drop=True)

        production_artifacts = retrain_model(full_df, cat_best_params, model_name="PRODUCTION MODEL")

        production_model = production_artifacts["model"]
        mlflow.catboost.log_model(production_model, artifact_path="catboost_production_model")
        save_onnx_model(production_model)
        save_artifacts(production_artifacts)
        with open("artifacts/zero_sectors.pkl", "wb") as f:
            pickle.dump(zero_sectors, f)

        print("\n" + "=" * 60)
        print("STEP 7: Save Drift Reference Baseline")
        print("=" * 60)
        save_reference_dataset(df_train)
        save_reference_statistics(df_train)
        print("✓ Reference saved: artifacts/reference.parquet, artifacts/reference_stats.json")

    print("✓ ONNX model saved: artifacts/model.onnx")
    return {
        "best_model": "catboost",
        "final_model": production_model,
        "test_results": test_results,
        "zero_sectors": zero_sectors,
        "sector_meta": sector_meta,
        "cat_params": cat_best_params,
        "cat_scores": cat_scores,
        "cat_oof": cat_oof,
        "mlflow_run_id": parent_run.info.run_id,
    }


if __name__ == "__main__":

    results = run_pipeline(tune=True, n_trials=N_TRIALS)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)

    print(f"Best model: {results['best_model']}")

    print(f"MLflow Run ID: {results['mlflow_run_id']}")

    if "test_results" in results:
        print("\nHoldout Test Results")
        print("-" * 40)

        for metric, value in results["test_results"].items():
            print(f"{metric:<20}: {value:.4f}")
