import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from features import (
    create_training_features,
    get_valid_features,
    compute_sector_stats,
    build_sector_profile,
    apply_zero_sector_rule,
)

from src.utils.config import load_config

cfg = load_config()

TARGET = cfg["target"]["column"]
TARGET_TRANSFORM = cfg["target"]["transform"]

TARGET_LOG = (
    f"log_{TARGET}"
    if TARGET_TRANSFORM == "log1p"
    else TARGET
)

def recursive_forecast(
    df_history,
    df_test,
    model,
    target_col="amount_new_house_transactions",
    sector_profile=None,
    sector_stats=None
):
    """
    Recursive multi-step forecasting for time series panel data.
    Predicts month-by-month (all sectors at once per month),
    so lag features of month T are always available for month T+1.

    df_history : full train data (must include target)
    df_test    : future rows (no target), sorted by date
    """

    # Sort test by date (then sector) so we process month by month
    test_months = sorted(df_test["date"].unique())

    # Working history — will grow as we predict each month
    history = df_history.copy()

    all_preds = []

    for month in test_months:
        print(f"  Predicting {month.strftime('%Y-%m')} ...")

        # Rows to predict this month (all sectors)
        month_rows = df_test[df_test["date"] == month].copy()
        month_rows[target_col] = np.nan   # ensure target is empty

        # Combine history + this month's rows → compute features
        combined = (
            pd.concat([history, month_rows], ignore_index=True)
            .sort_values(["sector", "date"])
            .reset_index(drop=True)
        )

        featured = create_training_features(
            combined,
            target_col=target_col,
            sector_stats=sector_stats,
            sector_profile=sector_profile,
            keep_nan=False,
        )

        # Extract only current month rows
        month_feat = featured[featured["date"] == month].copy()

        feats = get_valid_features(month_feat)
        X = month_feat[feats].fillna(0)

        preds = np.clip(model.predict(X), 0, None)

        # Fill predictions back into month_rows and append to history
        month_rows = month_rows.reset_index(drop=True)
        month_rows[target_col] = preds

        history = pd.concat([history, month_rows], ignore_index=True)

        all_preds.append(
            month_rows.assign(pred_log=preds)
        )

    result = pd.concat(all_preds, ignore_index=True)
    return result


def build_forecast_grid(df_train, n_months=12):
    """
    Tự động tạo future rows cho n_months tháng kế tiếp
    dựa hoàn toàn vào df_train — không cần file test nào.

    Các cột exogenous (nearby_sectors, pre_owned, POI...) được
    forward-fill từ giá trị cuối cùng của từng sector trong train.
    Model sẽ lag chúng thêm 1 tháng bên trong create_training_features,
    nên không có data leakage.

    Parameters
    ----------
    df_train  : DataFrame — full training data
    n_months  : int — số tháng cần forecast (default 12)

    Returns
    -------
    DataFrame với columns [date, sector, TARGET_LOG=NaN, ...exog_last_known]
    """
    last_date    = pd.Timestamp(df_train["date"].max())
    future_dates = pd.date_range(
        start=last_date + relativedelta(months=1),
        periods=n_months,
        freq="MS",
    )
    all_sectors = sorted(df_train["sector"].unique())

    grid = (
        pd.MultiIndex.from_product(
            [future_dates, all_sectors],
            names=["date", "sector"],
        )
        .to_frame(index=False)
    )

    skip = {TARGET, TARGET_LOG, "date", "sector"}
    exog_cols = [c for c in df_train.columns if c not in skip]
    if exog_cols:
        last_known = (
            df_train.sort_values("date")
            .groupby("sector")[exog_cols]
            .last()
            .reset_index()
        )
        grid = grid.merge(last_known, on="sector", how="left")

    grid[TARGET_LOG] = np.nan
    return grid

def forecast_next_year(df, model, results, n_months=12):
    """
    Forecast n_months
 
    Parameters
    ----------
    df  : DataFrame 
    model     : trained model
    results   : dict — output của run_pipeline(), cần có key "zero_sectors"
    n_months  : int — số tháng dự báo (default 12)
 
    """
    zero_sectors   = results["zero_sectors"]
    sector_stats   = compute_sector_stats(df, TARGET_LOG)
    sector_profile = build_sector_profile(df)
 
    df_future = build_forecast_grid(df, n_months=n_months)
 
    raw = recursive_forecast(
        df_history     = df,
        df_test        = df_future,
        model          = model,
        target_col     = TARGET_LOG,
        sector_profile = sector_profile,
        sector_stats   = sector_stats,
    )
 
    pred = np.expm1(np.clip(raw[TARGET_LOG].values, 0, None))
    pred = apply_zero_sector_rule(pred, raw["sector"], zero_sectors)
    pred = np.round(pred).astype(int)
 
    return pd.DataFrame({
        "date":         raw["date"].values,
        "sector":       raw["sector"].values,
        "pred_amount":  pred,
    })