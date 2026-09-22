"""Script to thoroughly measure and benchmark the accuracy of the trained models."""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.data.loader import load_raw_data
from src.models.elasticity import (
    PriceElasticityEstimator,
    evaluate_elasticity_model,
)
from src.models.model import DemandForecaster, evaluate_temporal_test
from src.preprocessing.preprocessing import (
    preprocess_data,
    temporal_train_test_split,
)


def evaluate_accuracy():
    print("=" * 65)
    print("1. LOADING & PREPROCESSING DATASET")
    print("=" * 65)
    df = load_raw_data()
    print(f"Loaded raw dataset: {len(df):,} records, {df['product_id'].nunique()} products.")

    t0 = time.time()
    df_proc = preprocess_data(df)
    print(f"Preprocessing completed in {time.time() - t0:.2f}s (Features: {df_proc.shape[1]}).")

    print("\n" + "=" * 65)
    print("2. STRICT TEMPORAL VALIDATION SPLIT (14-DAY HOLDOUT)")
    print("=" * 65)
    train_df, test_df = temporal_train_test_split(df_proc, test_days=14)
    train_start, train_end = train_df["date"].min().strftime("%Y-%m-%d"), train_df["date"].max().strftime("%Y-%m-%d")
    test_start, test_end = test_df["date"].min().strftime("%Y-%m-%d"), test_df["date"].max().strftime("%Y-%m-%d")
    print(f"Training Set:   {train_start} to {train_end} ({len(train_df):,} rows, {len(train_df)/len(df_proc)*100:.1f}%)")
    print(f"Temporal Test:  {test_start} to {test_end} ({len(test_df):,} rows, {len(test_df)/len(df_proc)*100:.1f}%)")

    # Benchmarks
    y_true = test_df["units_sold"].values
    mean_val = np.mean(train_df["units_sold"].values)
    lag1_val = test_df["units_sold_lag_1"].values

    mean_baseline_mae = np.mean(np.abs(y_true - mean_val))
    mean_baseline_rmse = np.sqrt(np.mean((y_true - mean_val)**2))
    lag1_baseline_mae = np.mean(np.abs(y_true - lag1_val))
    lag1_baseline_rmse = np.sqrt(np.mean((y_true - lag1_val)**2))

    print("\n" + "=" * 65)
    print("3. TRAINING & EVALUATING ML FORECASTER (XGBoost)")
    print("=" * 65)
    forecaster = DemandForecaster(
        model_type="xgboost",
        n_estimators=140,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
    )
    t1 = time.time()
    forecaster.fit(train_df, target_col="units_sold")
    fit_time = time.time() - t1
    print(f"XGBoost fit time: {fit_time:.2f}s on {len(train_df):,} rows.")

    ml_metrics, ml_preds = evaluate_temporal_test(forecaster, test_df, target_col="units_sold")

    print("\n" + "=" * 65)
    print("4. TRAINING & EVALUATING ECONOMETRIC MODEL (Log-Log Elasticity)")
    print("=" * 65)
    elasticity_model = PriceElasticityEstimator()
    t2 = time.time()
    elasticity_model.fit_by_category(train_df)
    print(f"Elasticity fit time: {time.time() - t2:.2f}s across {len(elasticity_model.category_models_)} categories.")
    el_metrics, el_preds = evaluate_elasticity_model(elasticity_model, test_df, target_col="units_sold")

    print("\n" + "=" * 65)
    print("5. ACCURACY BENCHMARK COMPARISON TABLE")
    print("=" * 65)

    comparison_data = {
        "Model / Strategy": [
            "Mean Baseline (Historical Average)",
            "Prior-Day Lag Baseline (Persistence)",
            "Econometric Log-Log Model (Elasticity)",
            "Machine Learning Forecaster (XGBoost)",
        ],
        "MAE (Units)": [
            f"{mean_baseline_mae:.4f}",
            f"{lag1_baseline_mae:.4f}",
            f"{el_metrics['mae']:.4f}",
            f"{ml_metrics['mae']:.4f}",
        ],
        "RMSE (Units)": [
            f"{mean_baseline_rmse:.4f}",
            f"{lag1_baseline_rmse:.4f}",
            f"{el_metrics['rmse']:.4f}",
            f"{ml_metrics['rmse']:.4f}",
        ],
        "WAPE (%)": [
            f"{(np.sum(np.abs(y_true - mean_val)) / np.sum(y_true)) * 100:.2f}%",
            f"{(np.sum(np.abs(y_true - lag1_val)) / np.sum(y_true)) * 100:.2f}%",
            f"{el_metrics['wape_pct']:.2f}%",
            f"{ml_metrics['wape_pct']:.2f}%",
        ],
        "R² Score": [
            "0.0000",
            f"{1 - (np.sum((y_true - lag1_val)**2) / np.sum((y_true - np.mean(y_true))**2)):.4f}",
            f"{el_metrics['r2_score']:.4f}",
            f"{ml_metrics['r2_score']:.4f}",
        ],
    }

    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))

    # Error distribution analysis
    errors = np.abs(ml_preds["units_sold"].values - ml_preds["predicted_units_sold"].values)
    within_1 = np.mean(errors <= 1.0) * 100
    within_2 = np.mean(errors <= 2.0) * 100
    within_3 = np.mean(errors <= 3.0) * 100

    print("\n" + "=" * 65)
    print("6. RESIDUAL ERROR ACCURACY BANDS (XGBoost)")
    print("=" * 65)
    print(f"Predictions within ±1 unit of actual demand:  {within_1:.2f}%")
    print(f"Predictions within ±2 units of actual demand: {within_2:.2f}%")
    print(f"Predictions within ±3 units of actual demand: {within_3:.2f}%")

    top_features = forecaster.get_feature_importances()
    print("\nTop 5 Most Important Predictive Drivers:")
    print(top_features.head(5).to_string(index=False))


if __name__ == "__main__":
    evaluate_accuracy()
