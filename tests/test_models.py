"""Verification test suite for DemandForecaster and PriceElasticityEstimator."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def run_model_verification():
    print("=" * 60)
    print("STEP 1: Ingestion and Preprocessing")
    print("=" * 60)
    df = load_raw_data()
    # Use a solid sample for fast unit-testing
    sample_df = df.sample(n=15000, random_state=42).copy() if len(df) > 15000 else df.copy()
    processed_df = preprocess_data(sample_df)

    train_df, test_df = temporal_train_test_split(processed_df, test_days=14)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    print("\n" + "=" * 60)
    print("STEP 2: Training & Evaluating ML DemandForecaster (XGBoost)")
    print("=" * 60)
    forecaster = DemandForecaster(
        model_type="xgboost",
        n_estimators=80,
        max_depth=5,
        learning_rate=0.1,
    )
    forecaster.fit(train_df, target_col="units_sold")

    ml_metrics, ml_results = evaluate_temporal_test(forecaster, test_df, target_col="units_sold")
    print(f"ML Metrics on Temporal Test Set: {ml_metrics}")
    assert "mae" in ml_metrics and "rmse" in ml_metrics
    assert ml_metrics["mae"] > 0
    assert ml_metrics["rmse"] > 0

    importances = forecaster.get_feature_importances()
    print("\nTop 5 Feature Importances:")
    print(importances.head(5).to_string())

    print("\n" + "=" * 60)
    print("STEP 3: Training & Evaluating PriceElasticityEstimator (Log-Log)")
    print("=" * 60)
    elasticity_model = PriceElasticityEstimator(
        category_col="category",
        price_col="current_price",
        competitor_price_col="competitor_price",
        target_col="units_sold",
    )
    elasticity_model.fit_by_category(train_df)

    print("\nCategory Elasticity Summary:")
    print(
        elasticity_model.elasticity_summary_[
            ["category", "sample_size", "own_price_elasticity", "cross_price_elasticity", "elasticity_regime"]
        ].to_string()
    )

    el_metrics, el_results = evaluate_elasticity_model(elasticity_model, test_df, target_col="units_sold")
    print(f"\nElasticity Model Metrics on Temporal Test Set: {el_metrics}")
    assert "mae" in el_metrics and "rmse" in el_metrics

    print("\n" + "=" * 60)
    print("STEP 4: Simulating Price-Response & Profit Curve")
    print("=" * 60)
    test_cat = elasticity_model.elasticity_summary_["category"].iloc[0]
    response_curve = elasticity_model.simulate_price_response_curve(
        category=test_cat,
        base_price=100.0,
        unit_cost=50.0,
        competitor_price=105.0,
        min_price_pct=0.75,
        max_price_pct=1.25,
        n_points=10,
    )
    print(f"Price Response Curve Sample for '{test_cat}':")
    print(response_curve[["price", "predicted_demand", "expected_revenue", "expected_profit", "profit_margin_pct"]].to_string())

    print("\n>>> ALL MODEL AND ELASTICITY VERIFICATION CHECKS PASSED! <<<")


if __name__ == "__main__":
    run_model_verification()
