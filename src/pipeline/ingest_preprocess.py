"""
ingest.py
---------
Load, merge and handle missing value for  4 files CSV input.

4 files input (web upload):
    1. new_house_transactions.csv       
    2. new_house_transactions_nearby.csv  
    3. pre_owned_house_transactions.csv  

Handling missing:
    - Target / transaction counts  → 0  (no transaction = 0)
    - Exogenous time-series        → ffill by sector,  then 0
    - Cột số còn lại               → ffill by sector then 0

Usage:
    from ingest import load_and_merge
    df = load_and_merge(
        path_main    = "new_house_transactions.csv",
        path_nearby  = "new_house_transactions_nearby.csv",
        path_pre     = "pre_owned_house_transactions.csv",
    )
"""

import pandas as pd
import numpy as np
from typing import Tuple
from src.utils.config import load_config

cfg = load_config()

TARGET = cfg["target"]["column"]
TARGET_TRANSFORM = cfg["target"]["transform"]

TARGET_LOG = (
    f"log_{TARGET}"
    if TARGET_TRANSFORM == "log1p"
    else TARGET
)
ALL_SECTORS = np.arange(1, 97)

# Cột transaction / target: missing = 0 giao dịch (không phải dữ liệu lỗi)
_ZERO_FILL_PATTERNS = [
    "amount_", "num_", "area_", "price_",
    "total_price_", "area_per_unit_",
]

# Cột exogenous time-series: ffill → bfill → 0
_EXOG_PATTERNS = [
    "period_new_house_sell_through",
    "period_new_house_sell_through_nearby_sectors",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_csv(path: str, date_col: str = "month") -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def _normalise_sector(df: pd.DataFrame) -> pd.DataFrame:
    """normalizing type of sector: 'sector 1' → 1 (int)."""
    if df["sector"].dtype == object:
        df["sector"] = df["sector"].str.split().str[-1].astype(int)
    else:
        df["sector"] = df["sector"].astype(int)
    return df


def _build_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    create full grid × sector (1-96).
    """
    all_months = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="MS",
    )

    grid = (
        pd.MultiIndex.from_product(
            [all_months, ALL_SECTORS],
            names=["date", "sector"],
        )
        .to_frame(index=False)
    )

    df_full = grid.merge(df, on=["date", "sector"], how="left").fillna(0)
    df_full[TARGET_LOG] = np.log1p(df_full[TARGET])
    return df_full


def _report_missing(df: pd.DataFrame, label: str) -> None:
    """ print missing value report."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print(f"[ingest] {label}: không có missing value")
    else:
        total_cells = len(df) * len(df.columns)
        pct = missing.sum() / total_cells * 100
        print(f"[ingest] {label}: {len(missing)} cột có missing "
              f"({missing.sum():,} cells, {pct:.2f}% tổng)")
        for col, cnt in missing.items():
            print(f"         • {col:<55} {cnt:>6,} ({cnt/len(df)*100:.1f}%)")


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:

    
    df = df.copy()

    zero_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in _ZERO_FILL_PATTERNS)
        and c != TARGET_LOG         
    ]
    df[zero_cols] = df[zero_cols].fillna(0)

    exog_cols = [
        c for c in df.columns
        if any(pat in c for pat in _EXOG_PATTERNS)
    ]
    if exog_cols:
        df[exog_cols] = (
            df.sort_values(["sector", "date"])
              .groupby("sector")[exog_cols]
              .transform(lambda x: x.ffill())
        )
        df[exog_cols] = df[exog_cols].fillna(0)

    remaining_num = [
        c for c in df.select_dtypes(include="number").columns
        if df[c].isnull().any()
        and c not in zero_cols
        and c not in exog_cols
        and c != TARGET_LOG
    ]
    if remaining_num:
        df[remaining_num] = (
            df.sort_values(["sector", "date"])
              .groupby("sector")[remaining_num]
              .transform(lambda x: x.ffill())
        )
        df[remaining_num] = df[remaining_num].fillna(0)

    return df


# ── Main public function ───────────────────────────────────────────────────────

def load_and_merge(
    path_main: str,
    path_nearby: str,
    path_pre: str,
    path_poi: str,
    build_grid: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load 4 CSV, merge và handle missing value int 1 DataFrame .

    Parameters
    ----------
    path_main    : path to new_house_transactions.csv
    path_nearby  : path to new_house_transactions_nearby.csv
    path_pre     : path to pre_owned_house_transactions.csv
    build_grid   : if True, create full grid month × sector
    verbose      : print missing value report

    Returns
    -------
    pd.DataFrame  sorted by [date, sector]
    """

    main_df   = _read_csv(path_main,   date_col="month")
    nearby_df = _read_csv(path_nearby, date_col="month")
    pre_df    = _read_csv(path_pre,    date_col="month")

    main_df   = _normalise_sector(main_df)
    nearby_df = _normalise_sector(nearby_df)
    pre_df    = _normalise_sector(pre_df)

    df = df.merge(nearby_df,      on=["month", "sector"], how="left")
    df = df.merge(pre_df,         on=["month", "sector"], how="left")

    df.rename(columns={"month": "date"}, inplace=True)

    if verbose:
        _report_missing(df, "Trước xử lý missing")

    df = _handle_missing(df)

    if verbose:
        _report_missing(df, "Sau xử lý missing")

    if build_grid:
        df = _build_full_grid(df)

    df = df.sort_values(["date", "sector"]).reset_index(drop=True)

    print(f"[ingest] Shape      : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"[ingest] Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"[ingest] Sectors    : {df['sector'].nunique()}")
    print(f"[ingest] Zero rate  : {(df[TARGET] == 0).mean():.1%}")

    return df


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "sector"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def split_train_test(
    df: pd.DataFrame,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    df = sort_data(df)

    unique_dates = sorted(df["date"].unique())
    split_idx    = int(len(unique_dates) * (1 - test_ratio))
    split_date   = unique_dates[split_idx]

    train_df = df[df["date"] < split_date].copy()
    test_df  = df[df["date"] >= split_date].copy()

    print(f"Split date : {split_date}")
    print(f"Train      : {train_df.shape} | {train_df['date'].min()} → {train_df['date'].max()}")
    print(f"Test       : {test_df.shape}  | {test_df['date'].min()} → {test_df['date'].max()}")
    print(f"Train ratio: {len(train_df)/len(df):.2%}")

    return train_df, test_df


