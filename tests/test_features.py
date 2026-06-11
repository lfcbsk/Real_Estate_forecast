"""Test feature engineering pipeline."""
import pytest
import pandas as pd
import numpy as np

from src.pipeline.features import (
    assign_regime,
    build_sector_profile,
    build_zero_sector_mask,
    apply_zero_sector_rule,
    compute_sector_stats,
    create_training_features,
    get_valid_features,
)


@pytest.fixture
def sample_train_df():
    np.random.seed(42)
    n_sectors = 5
    n_months = 24
    
    data = []
    for sector in range(1, n_sectors + 1):
        for i, date in enumerate(pd.date_range("2020-01-01", periods=n_months, freq="MS")):
            num_transactions = np.random.randint(10, 100)
            area_transactions = num_transactions * np.random.uniform(80, 150)
            price_transactions = np.random.uniform(1e6, 1e8)
            amount_transactions = num_transactions * price_transactions
            num_nearby = np.random.randint(20, 150)
            area_nearby = num_nearby * np.random.uniform(80, 150)
            price_nearby = np.random.uniform(1e6, 1e8)
            amount_nearby = num_nearby * price_nearby
            num_available = np.random.randint(50, 500)
            area_available = num_available * np.random.uniform(80, 150)
            num_pre = np.random.randint(50, 200)
            area_pre = num_pre * np.random.uniform(80, 150)
            price_pre = np.random.uniform(1e6, 1e8)
            amount_pre = num_pre * price_pre
            
            data.append({
                "date": date,
                "sector": sector,
                "num_new_house_transactions": num_transactions,
                "area_new_house_transactions": area_transactions,
                "price_new_house_transactions": price_transactions,
                "amount_new_house_transactions": amount_transactions,
                "area_per_unit_new_house_transactions": area_transactions / num_transactions if num_transactions > 0 else 0,
                "total_price_per_unit_new_house_transactions": price_transactions / num_transactions if num_transactions > 0 else 0,
                "num_new_house_available_for_sale": num_available,
                "area_new_house_available_for_sale": area_available,
                "period_new_house_sell_through": np.random.uniform(2, 6),
                "num_new_house_transactions_nearby_sectors": num_nearby,
                "area_new_house_transactions_nearby_sectors": area_nearby,
                "price_new_house_transactions_nearby_sectors": price_nearby,
                "amount_new_house_transactions_nearby_sectors": amount_nearby,
                "area_per_unit_new_house_transactions_nearby_sectors": area_nearby / num_nearby if num_nearby > 0 else 0,
                "total_price_per_unit_new_house_transactions_nearby_sectors": price_nearby / num_nearby if num_nearby > 0 else 0,
                "num_new_house_available_for_sale_nearby_sectors": np.random.randint(100, 1000),
                "area_new_house_available_for_sale_nearby_sectors": np.random.uniform(10000, 100000),
                "period_new_house_sell_through_nearby_sectors": np.random.uniform(2, 6),
                "area_pre_owned_house_transactions": area_pre,
                "amount_pre_owned_house_transactions": amount_pre,
                "num_pre_owned_house_transactions": num_pre,
                "price_pre_owned_house_transactions": price_pre,
                "log_amount_new_house_transactions": np.log1p(amount_transactions),
            })
    return pd.DataFrame(data)

@pytest.mark.unit
def test_assign_regime():
    """Test gán regime theo thời gian."""
    assert assign_regime(pd.Timestamp("2019-06-01")) == 0
    assert assign_regime(pd.Timestamp("2020-03-01")) == 1
    assert assign_regime(pd.Timestamp("2021-06-01")) == 2
    assert assign_regime(pd.Timestamp("2023-01-01")) == 3

@pytest.mark.unit
def test_build_sector_profile(sample_train_df):
    """Test build sector profile."""
    profile = build_sector_profile(sample_train_df)
    
    assert isinstance(profile, dict)
    assert len(profile) == sample_train_df["sector"].nunique()
    
    # Tất cả sector types phải hợp lệ
    valid_types = {"dead", "normal", "spike"}
    for sector_type in profile.values():
        assert sector_type in valid_types

@pytest.mark.unit
def test_build_zero_sector_mask():
    df = pd.DataFrame({
        "sector": [1] * 20 + [2] * 20 + [52] * 20,
        "log_amount_new_house_transactions": [0] * 18 + [100, 200] +  
                                             [0] * 5 + [100] * 15 +   
                                             [0] * 20,                
    })
    
    zero_sectors, sector_meta = build_zero_sector_mask(df, zero_thresh=0.85, recent_n=6)
    assert 1 in zero_sectors
    assert 52 in zero_sectors

@pytest.mark.unit
def test_apply_zero_sector_rule():
    """Test áp dụng zero sector rule."""
    predictions = np.array([100, 200, 300, 400, 500])
    sectors = pd.Series([1, 2, 3, 2, 1])
    zero_sectors = {1, 3}
    
    result = apply_zero_sector_rule(predictions, sectors, zero_sectors)
    
    # Predictions cho sector 1 và 3 phải = 0
    assert result[0] == 0  # sector 1
    assert result[1] == 200  # sector 2
    assert result[2] == 0  # sector 3
    assert result[3] == 400  # sector 2
    assert result[4] == 0  # sector 1

@pytest.mark.unit
def test_compute_sector_stats(sample_train_df):
    """Test tính toán thống kê theo sector."""
    stats = compute_sector_stats(sample_train_df, "log_amount_new_house_transactions")
    
    assert "mean" in stats
    assert "std" in stats
    assert "zero_rate" in stats
    assert "cv" in stats
    
    # Mỗi stat phải là dict với key là sector
    assert isinstance(stats["mean"], dict)
    assert len(stats["mean"]) == sample_train_df["sector"].nunique()

@pytest.mark.unit
def test_create_training_features_basic(sample_train_df):
    """Test tạo training features cơ bản."""
    sector_stats = compute_sector_stats(sample_train_df, "log_amount_new_house_transactions")
    sector_profile = build_sector_profile(sample_train_df)
    
    df_featured = create_training_features(
        sample_train_df,
        target_col="log_amount_new_house_transactions",
        sector_stats=sector_stats,
        sector_profile=sector_profile,
        keep_nan=False
    )
    
    # Kiểm tra các features quan trọng được tạo
    expected_features = [
    "downtrend_signal","month","quarter","year","month_sin","month_cos","regime",
    "trend","trend_months","lag_1","lag_2","lag_3","lag_6","lag_12",
    "momentum_1","momentum_3","momentum_pct_1","rolling_mean_3","rolling_mean_6",
    "rolling_mean_12","rolling_std_3","rolling_std_6","rolling_std_12","vol_ratio_3_12","expanding_mean",
    "trend_strength","zero_rate_6","zero_rate_12","yoy_diff","yoy_ratio",
    "rolling_max_12","rolling_min_12","spike_ratio","volatility_ratio",
    "regime_trend","regime_month_sin", "regime_month_cos","sector_type", "nearby_supply_lag1",
    "nearby_sellthrough_lag1", "sector_mean_train","sector_std_train","sector_zero_rate_train","sector_cv_train", "sellthrough_lag1",
    "nearby_price_lag1","preowned_area_lag1"]
    
    for feat in expected_features:
        assert feat in df_featured.columns, f"Feature '{feat}' not found"

@pytest.mark.unit
def test_create_training_features_no_nan_when_keep_nan_false(sample_train_df):
    """Test không có NaN khi keep_nan=False."""
    sector_stats = compute_sector_stats(sample_train_df, "log_amount_new_house_transactions")
    
    df_featured = create_training_features(
        sample_train_df,
        target_col="log_amount_new_house_transactions",
        sector_stats=sector_stats,
        keep_nan=False
    )
    
    # Các feature columns không được có NaN
    feature_cols = [c for c in df_featured.columns if c != "log_amount_new_house_transactions"]
    nan_count = df_featured[feature_cols].isna().sum().sum()
    
    assert nan_count == 0, f"Found {nan_count} NaN values in features"


@pytest.mark.unit
def test_cyclical_encoding():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
        "sector": [1] * 12,
        "target": np.random.rand(12),
    })
    
    df_featured = create_training_features(
        df,
        target_col="target",
        keep_nan=False
    )
    assert df_featured["month_sin"].between(-1, 1).all()
    assert df_featured["month_cos"].between(-1, 1).all()
