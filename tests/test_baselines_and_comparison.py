"""Unit tests for baseline models and comparison module."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.data.loader import load_raw_data
from src.evaluation.comparison import compare_models
from src.models.baseline import MeanBaseline, SeasonalNaiveBaseline
from src.models.model import DemandForecaster
from src.preprocessing.preprocessing import (
    preprocess_data,
    temporal_train_test_split,
)


def test_baselines_and_comparison():
    print("=" * 60)
    print("1. Testing MeanBaseline")
    print("=" * 60)
    train_dummy = pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P2", "P2"],
            "units_sold": [10, 20, 30, 50],
        }
    )
    test_dummy = pd.DataFrame(
        {
            "product_id": ["P1", "P2", "P3"],  # P3 unseen
        }
    )
    mean_model = MeanBaseline(target_col="units_sold")
    mean_model.fit(train_dummy)
    preds = mean_model.predict(test_dummy)
    assert preds[0] == 15.0  # (10+20)/2
    assert preds[1] == 40.0  # (30+50)/2
    assert preds[2] == 27.5  # Global mean: (10+20+30+50)/4
    print("MeanBaseline test passed.")

    print("\n" + "=" * 60)
    print("2. Testing SeasonalNaiveBaseline")
    print("=" * 60)
    test_seasonal = pd.DataFrame(
        {
            "units_sold_lag_7": [12.0, np.nan, 25.0],
        }
    )
    seasonal_model = SeasonalNaiveBaseline(lag_col="units_sold_lag_7")
    s_preds = seasonal_model.predict(test_seasonal)
    assert s_preds[0] == 12.0
    assert s_preds[1] == 0.0  # NaN filled with 0
    assert s_preds[2] == 25.0
    print("SeasonalNaiveBaseline test passed.")

    print("\n" + "=" * 60)
    print("3. Testing compare_models Integration")
    print("=" * 60)
    df = load_raw_data()
    # Pick first 25 products with all 90 days of transactions
    test_skus = df["product_id"].unique()[:25]
    sample_df = df[df["product_id"].isin(test_skus)].copy()
    proc_df = preprocess_data(sample_df)
    train_df, test_df = temporal_train_test_split(proc_df, test_days=14)

    forecaster = DemandForecaster(
        model_type="xgboost",
        n_estimators=50,
        max_depth=5,
    )

    summary_df, rmse_improvement = compare_models(train_df, test_df, forecaster)
    print("Comparison Summary Table:")
    print(summary_df.to_string(index=False))
    print(f"\nRMSE Improvement vs Best Baseline: {rmse_improvement:+.2f}%")

    assert len(summary_df) == 3
    assert "MAE" in summary_df.columns
    assert "RMSE" in summary_df.columns
    assert "WAPE (%)" in summary_df.columns
    assert "R2 Score" in summary_df.columns
    assert rmse_improvement > 0, "ML model should improve over naive baseline!"
    print("\n>>> ALL BASELINE AND COMPARISON TESTS PASSED! <<<")


if __name__ == "__main__":
    test_baselines_and_comparison()
