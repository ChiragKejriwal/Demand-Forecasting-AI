"""Test suite for Prophet, Linear Regression, Confidence Intervals, and Resampling."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.evaluation.comparison import compare_all_models
from src.models.model import DemandForecaster, ProphetForecaster
from src.preprocessing.preprocessing import (
    preprocess_data,
    resample_time_series,
    temporal_train_test_split,
)


def test_prophet_and_resampling():
    print("=" * 65)
    print("TEST 1: Resampling Time-Series (Daily, Weekly, Monthly)")
    print("=" * 65)
    dates = pd.date_range(start="2025-01-01", periods=60, freq="D")
    test_df = pd.DataFrame(
        {
            "date": dates,
            "product_id": ["P1001"] * 30 + ["P1002"] * 30,
            "units_sold": [10] * 60,
            "current_price": [50.0] * 60,
            "revenue": [500.0] * 60,
        }
    )

    weekly_df = resample_time_series(test_df, freq="W", date_col="date", group_col="product_id")
    assert len(weekly_df) > 0
    assert "units_sold" in weekly_df.columns
    print(f"Weekly resampling succeeded: {len(weekly_df)} weekly periods.")

    monthly_df = resample_time_series(test_df, freq="M", date_col="date", group_col="product_id")
    assert len(monthly_df) > 0
    print(f"Monthly resampling succeeded: {len(monthly_df)} monthly periods.")

    print("\n" + "=" * 65)
    print("TEST 2: ProphetForecaster with Uncertainty Intervals")
    print("=" * 65)
    prophet_dates = pd.date_range(start="2025-01-01", periods=50, freq="D")
    prophet_train = pd.DataFrame(
        {
            "date": prophet_dates[:40],
            "units_sold": np.random.poisson(15, 40),
            "current_price": [25.0] * 40,
        }
    )
    prophet_test = pd.DataFrame(
        {
            "date": prophet_dates[40:],
            "units_sold": np.random.poisson(15, 10),
            "current_price": [25.0] * 10,
        }
    )

    p_forecaster = ProphetForecaster(date_col="date", target_col="units_sold")
    p_forecaster.fit(prophet_train)
    intervals_df = p_forecaster.predict_with_intervals(prophet_test)

    assert len(intervals_df) == 10
    assert "yhat" in intervals_df.columns
    assert "yhat_lower" in intervals_df.columns
    assert "yhat_upper" in intervals_df.columns
    assert (intervals_df["yhat_lower"] <= intervals_df["yhat"]).all()
    assert (intervals_df["yhat"] <= intervals_df["yhat_upper"]).all()
    print("Prophet 95% uncertainty intervals successfully verified!")

    print("\n" + "=" * 65)
    print("TEST 3: DemandForecaster Linear Regression & Confidence Intervals")
    print("=" * 65)
    # Preprocess a realistic sample
    from src.data.loader import load_raw_data

    raw_df = load_raw_data()
    sample_df = raw_df[raw_df["product_id"] == "P1241"].copy()
    proc_df = preprocess_data(sample_df)
    train_part, test_part = temporal_train_test_split(proc_df, test_days=14)

    lr_model = DemandForecaster(model_type="linear_regression")
    lr_model.fit(train_part, target_col="units_sold")
    lr_intervals = lr_model.predict_with_intervals(test_part)

    assert len(lr_intervals) == len(test_part)
    assert (lr_intervals["yhat_lower"] <= lr_intervals["yhat"]).all()
    assert (lr_intervals["yhat"] <= lr_intervals["yhat_upper"]).all()
    print("Linear regression prediction intervals verified!")

    print("\n" + "=" * 65)
    print("TEST 4: Complete Multi-Model Leaderboard Benchmarking")
    print("=" * 65)
    leaderboard = compare_all_models(train_part, test_part, include_prophet=True)
    print("Leaderboard Table:")
    print(leaderboard.to_string(index=False))

    assert len(leaderboard) >= 4
    assert "MAPE (%)" in leaderboard.columns
    assert "RMSE" in leaderboard.columns
    assert "Model" in leaderboard.columns
    print("\n>>> ALL PROPHET, LINEAR REGRESSION, AND RESAMPLING CHECKS PASSED! <<<")


if __name__ == "__main__":
    test_prophet_and_resampling()
