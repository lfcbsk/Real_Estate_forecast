import numpy as np
import pandas as pd

from src.utils.config import load_config

cfg = load_config()

TARGET = cfg["target"]["column"]
TARGET_TRANSFORM = cfg["target"]["transform"]

TARGET_LOG = f"log_{TARGET}" if TARGET_TRANSFORM == "log1p" else TARGET

DROP_COLS = [
    "num_new_house_transactions",
    "area_new_house_transactions",
    "price_new_house_transactions",
    "amount_new_house_transactions",
    "area_per_unit_new_house_transactions",
    "total_price_per_unit_new_house_transactions",
    "num_new_house_available_for_sale",
    "area_new_house_available_for_sale",
    "period_new_house_sell_through",
    "transportation_station_dense",
    "education_dense",
    "medical_health_dense",
    "num_new_house_transactions_nearby_sectors",
    "area_new_house_transactions_nearby_sectors",
    "price_new_house_transactions_nearby_sectors",
    "amount_new_house_transactions_nearby_sectors",
    "area_per_unit_new_house_transactions_nearby_sectors",
    "total_price_per_unit_new_house_transactions_nearby_sectors",
    "num_new_house_available_for_sale_nearby_sectors",
    "area_new_house_available_for_sale_nearby_sectors",
    "period_new_house_sell_through_nearby_sectors",
    "area_pre_owned_house_transactions",
    "amount_pre_owned_house_transactions",
    "num_pre_owned_house_transactions",
    "price_pre_owned_house_transactions",
]

FEATURES = [
    "downtrend_signal",
    "month",
    "quarter",
    "year",
    "month_sin",
    "month_cos",
    "regime",
    "trend",
    "trend_months",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "momentum_1",
    "momentum_3",
    "momentum_pct_1",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_std_3",
    "rolling_std_6",
    "rolling_std_12",
    "vol_ratio_3_12",
    "expanding_mean",
    "trend_strength",
    "zero_rate_6",
    "zero_rate_12",
    "yoy_diff",
    "yoy_ratio",
    "rolling_max_12",
    "rolling_min_12",
    "spike_ratio",
    "volatility_ratio",
    "regime_trend",
    "regime_month_sin",
    "regime_month_cos",
    "sector_type",
    "nearby_supply_lag1",
    "nearby_sellthrough_lag1",
    "sector_mean_train",
    "sector_std_train",
    "sector_zero_rate_train",
    "sector_cv_train",
    "sellthrough_lag1",
    "nearby_price_lag1",
    "preowned_area_lag1",
]


def assign_regime(month):
    if month < pd.Timestamp("2020-01-01"):
        return 0
    elif month < pd.Timestamp("2020-07-01"):
        return 1
    elif month < pd.Timestamp("2022-01-01"):
        return 2
    else:
        return 3


def build_sector_profile(df_train_fold):
    profile = df_train_fold.groupby("sector")[TARGET_LOG].agg(["mean", "std"]).reset_index()
    profile["cv"] = profile["std"] / (profile["mean"] + 1)

    profile["zero_rate"] = df_train_fold.groupby("sector")[TARGET_LOG].apply(lambda x: (x == 0).mean()).values

    profile["sector_type"] = "normal"
    profile.loc[profile["zero_rate"] >= 0.85, "sector_type"] = "dead"
    profile.loc[(profile["zero_rate"] < 0.85) & (profile["cv"] >= 1.20), "sector_type"] = "spike"

    return dict(zip(profile["sector"], profile["sector_type"]))


def build_zero_sector_mask(df_train, zero_thresh=0.85, recent_n=6):
    """
    Identify sectors that should always be predicted as 0.

    Rules (any one triggers → predict 0):
      A) Historical zero-rate >= zero_thresh  (e.g. 85% of months = 0)
      B) Last `recent_n` months are ALL zero  (recently dormant)
      C) Hardcoded dead sectors (52, 95 from EDA)

    Returns
    -------
    zero_sectors : set of int
    sector_meta  : DataFrame with diagnostics per sector
    """
    hardcoded = {52, 95, 49, 74}
    sector_meta = (
        df_train.groupby("sector")[TARGET_LOG]
        .agg(
            total_obs="count",
            zero_count=lambda x: (x == 0).sum(),
            mean_val="mean",
            last_mean=lambda x: (x.iloc[-recent_n:].mean() if len(x) >= recent_n else x.mean()),
            last_zeros=lambda x: ((x.iloc[-recent_n:] == 0).all() if len(x) >= recent_n else (x == 0).all()),
        )
        .reset_index()
    )
    sector_meta["zero_rate"] = sector_meta["zero_count"] / sector_meta["total_obs"]
    sector_meta["rule_A"] = sector_meta["zero_rate"] >= zero_thresh
    sector_meta["rule_B"] = sector_meta["last_zeros"]
    sector_meta["rule_hard"] = sector_meta["sector"].isin(hardcoded)
    sector_meta["is_zero_sector"] = sector_meta["rule_A"] | sector_meta["rule_B"] | sector_meta["rule_hard"]

    zero_sectors = set(sector_meta.loc[sector_meta["is_zero_sector"], "sector"].tolist())

    n = len(zero_sectors)
    print(f"[Zero-Sector Rule] {n} sectors flagged as always-zero → predict 0")
    print(f"  Rule A (zero_rate >= {zero_thresh:.0%}): " f"{sector_meta['rule_A'].sum()} sectors")
    print(f"  Rule B (last {recent_n} months all zero): " f"{sector_meta['rule_B'].sum()} sectors")
    print(f"  Rule Hard (sector 52, 95, 49, 74): {sector_meta['rule_hard'].sum()} sectors")
    return zero_sectors, sector_meta


def apply_zero_sector_rule(predictions: np.ndarray, sectors: pd.Series, zero_sectors: set) -> np.ndarray:
    """Zero-out predictions for flagged sectors."""
    preds = predictions.copy()
    mask = sectors.isin(zero_sectors).values
    preds[mask] = 0.0
    return preds


def compute_sector_stats(df_train_fold, target_col):
    """
    Compute per-sector statistics from the training fold ONLY.
    Pass the result to create_training_features() as sector_stats.
    """
    stats = df_train_fold.groupby("sector")[target_col].agg(["mean", "std"])
    stats["zero_rate"] = df_train_fold.groupby("sector")[target_col].apply(lambda x: (x == 0).mean())
    stats["cv"] = stats["std"] / (stats["mean"] + 1e-9)
    return {
        "mean": stats["mean"].to_dict(),
        "std": stats["std"].to_dict(),
        "zero_rate": stats["zero_rate"].to_dict(),
        "cv": stats["cv"].to_dict(),
    }


def create_training_features(
    df_in,
    target_col,
    sector_stats=None,
    sector_profile=None,
    LAG_LIST=[1, 2, 3, 6, 12],
    rolling_windows=[3, 6, 12],
    keep_nan=True,
):
    df = df_in.copy()
    df = df.sort_values(["sector", "date"]).reset_index(drop=True)

    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    # Calendar
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Regime
    if "regime" not in df.columns:
        df["regime"] = df["date"].apply(assign_regime)

    # Trend
    df["trend"] = df.groupby("sector").cumcount()

    df["trend_months"] = (df["date"] - df.groupby("sector")["date"].transform("min")).dt.days / 30.0

    # Lags
    for lag in LAG_LIST:
        df[f"lag_{lag}"] = df.groupby("sector")[target_col].shift(lag)

    shifted_target = df.groupby("sector")[target_col].shift(1)

    lag13 = df.groupby("sector")[target_col].shift(13)

    # Momentum
    df["momentum_1"] = df["lag_1"] - df.groupby("sector")[target_col].shift(2)
    df["momentum_3"] = df["lag_1"] - df.groupby("sector")[target_col].shift(4)
    df["momentum_pct_1"] = (df["lag_1"] - df.groupby("sector")[target_col].shift(2)) / (
        df.groupby("sector")[target_col].shift(2).abs() + 1e-9
    )
    # Rolling
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = shifted_target.groupby(df["sector"]).transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = shifted_target.groupby(df["sector"]).transform(
            lambda x: x.rolling(w, min_periods=2).std()
        )

    # Volatility
    df["vol_ratio_3_12"] = df["rolling_std_3"] / (df["rolling_std_12"] + 1e-9)

    # Expanding
    df["expanding_mean"] = shifted_target.groupby(df["sector"]).transform(lambda x: x.expanding().mean())

    # Trend strength
    df["trend_strength"] = df["rolling_mean_3"] / (df["rolling_mean_12"] + 1)

    # Zero rate
    shifted_zero = shifted_target.eq(0)
    for w in [6, 12]:
        df[f"zero_rate_{w}"] = shifted_zero.groupby(df["sector"]).transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )

    # YoY
    df["yoy_diff"] = df["lag_1"] - lag13
    df["yoy_ratio"] = df["lag_1"] / (lag13 + 1)

    # Rolling max/min
    df["rolling_max_12"] = shifted_target.groupby(df["sector"]).transform(lambda x: x.rolling(12, min_periods=1).max())
    df["rolling_min_12"] = shifted_target.groupby(df["sector"]).transform(lambda x: x.rolling(12, min_periods=1).min())

    df["spike_ratio"] = df["rolling_max_12"] / (df["rolling_mean_12"] + 1)
    df["volatility_ratio"] = df["rolling_std_12"] / (df["rolling_mean_12"] + 1)
    df["regime_trend"] = df["regime"] * df["trend"]
    df["regime_month_sin"] = df["regime"] * df["month_sin"]
    df["regime_month_cos"] = df["regime"] * df["month_cos"]

    if sector_profile is not None:
        df["sector_type"] = df["sector"].map(sector_profile).fillna("normal")
    else:
        df["sector_type"] = "normal"

    type_map = {"dead": 0, "normal": 1, "spike": 2}

    df["sector_type"] = df["sector_type"].map(type_map)

    df["downtrend_signal"] = (df["rolling_mean_3"] < df["rolling_mean_12"] * 0.5).astype(int)

    # Train-fold sector stats
    if sector_stats is not None:
        df["sector_mean_train"] = df["sector"].map(sector_stats["mean"])
        df["sector_std_train"] = df["sector"].map(sector_stats["std"])
        df["sector_zero_rate_train"] = df["sector"].map(sector_stats["zero_rate"])
        df["sector_cv_train"] = df["sector"].map(sector_stats["cv"])

    # exogenous features (optional columns for minimal / partial datasets)
    if "num_new_house_available_for_sale_nearby_sectors" in df.columns:
        df["nearby_supply_lag1"] = df.groupby("sector")["num_new_house_available_for_sale_nearby_sectors"].shift(1)
    if "period_new_house_sell_through_nearby_sectors" in df.columns:
        df["nearby_sellthrough_lag1"] = df.groupby("sector")["period_new_house_sell_through_nearby_sectors"].shift(1)

    g = df.groupby("sector")
    if "period_new_house_sell_through" in df.columns:
        df["sellthrough_lag1"] = g["period_new_house_sell_through"].shift(1)
    if "price_new_house_transactions_nearby_sectors" in df.columns:
        df["nearby_price_lag1"] = g["price_new_house_transactions_nearby_sectors"].shift(1)
    if "area_pre_owned_house_transactions" in df.columns:
        df["preowned_area_lag1"] = g["area_pre_owned_house_transactions"].shift(1)

    # Fill NA
    if not keep_nan:
        feature_cols = [c for c in df.columns if c != target_col]

        df[feature_cols] = df.groupby("sector")[feature_cols].ffill().bfill().fillna(0)

    # ==================================================
    # Drop raw columns
    # ==================================================

    drop_cols = [c for c in DROP_COLS if c in df.columns]

    df.drop(columns=drop_cols, inplace=True)

    return df


def get_valid_features(df):
    """Return only FEATURES columns that actually exist in df."""
    return [f for f in FEATURES if f in df.columns]
