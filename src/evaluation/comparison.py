"""Model comparison and benchmarking module to evaluate ML against baselines."""

import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
from src.models.baseline import MeanBaseline, SeasonalNaiveBaseline
from src.models.model import evaluate_forecast

logger = logging.getLogger(__name__)


def compare_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ml_forecaster: Any,
    target_col: str = "units_sold",
) -> Tuple[pd.DataFrame, float]:
    """Compare machine learning forecaster against MeanBaseline and SeasonalNaiveBaseline.

    Parameters
    ----------
    train_df : pd.DataFrame
        Historical training data.
    test_df : pd.DataFrame
        Temporal holdout test data.
    ml_forecaster : Any
        Machine learning model instance (e.g. DemandForecaster). Fitted if not already fitted.
    target_col : str, optional
        Target column name, by default "units_sold".

    Returns
    -------
    Tuple[pd.DataFrame, float]
        (comparison_summary_df, rmse_improvement_pct)
        - comparison_summary_df: Metrics table (MAE, RMSE, WAPE, R2) for all models.
        - rmse_improvement_pct: Percentage improvement of ML RMSE over the best baseline's RMSE.
    """
    if target_col not in test_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in test_df.")

    y_true = test_df[target_col].values

    logger.info("Evaluating MeanBaseline...")
    mean_model = MeanBaseline(target_col=target_col, product_col="product_id")
    mean_model.fit(train_df)
    y_pred_mean = mean_model.predict(test_df)
    metrics_mean = evaluate_forecast(y_true, y_pred_mean)

    logger.info("Evaluating SeasonalNaiveBaseline (7-day lag persistence)...")
    seasonal_model = SeasonalNaiveBaseline(lag_col=f"{target_col}_lag_7")
    y_pred_seasonal = seasonal_model.predict(test_df)
    metrics_seasonal = evaluate_forecast(y_true, y_pred_seasonal)

    logger.info("Evaluating Machine Learning Forecaster...")
    if hasattr(ml_forecaster, "pipeline") and ml_forecaster.pipeline is None:
        logger.info("ML forecaster not yet fitted. Fitting on train_df...")
        ml_forecaster.fit(train_df, target_col=target_col)

    y_pred_ml = ml_forecaster.predict(test_df)
    metrics_ml = evaluate_forecast(y_true, y_pred_ml)

    # Calculate percentage improvement over the best baseline
    best_baseline_rmse = min(metrics_mean["rmse"], metrics_seasonal["rmse"])
    best_baseline_name = "MeanBaseline" if metrics_mean["rmse"] <= metrics_seasonal["rmse"] else "SeasonalNaiveBaseline"

    rmse_improvement_pct = ((best_baseline_rmse - metrics_ml["rmse"]) / best_baseline_rmse) * 100.0

    logger.info(
        "Best Baseline: %s (RMSE: %.4f) | ML Forecaster (RMSE: %.4f) | Improvement: +%.2f%%",
        best_baseline_name,
        best_baseline_rmse,
        metrics_ml["rmse"],
        rmse_improvement_pct,
    )

    summary_rows = [
        {
            "Model": "MeanBaseline (Product Historical Mean)",
            "MAE": metrics_mean["mae"],
            "RMSE": metrics_mean["rmse"],
            "WAPE (%)": f"{metrics_mean['wape_pct']:.2f}%",
            "MAPE (%)": f"{metrics_mean['mape_pct']:.2f}%",
            "R2 Score": metrics_mean["r2_score"],
        },
        {
            "Model": "SeasonalNaiveBaseline (7-Day Lag Persistence)",
            "MAE": metrics_seasonal["mae"],
            "RMSE": metrics_seasonal["rmse"],
            "WAPE (%)": f"{metrics_seasonal['wape_pct']:.2f}%",
            "MAPE (%)": f"{metrics_seasonal['mape_pct']:.2f}%",
            "R2 Score": metrics_seasonal["r2_score"],
        },
        {
            "Model": f"ML Forecaster ({getattr(ml_forecaster, 'model_type', 'XGBoost').upper()})",
            "MAE": metrics_ml["mae"],
            "RMSE": metrics_ml["rmse"],
            "WAPE (%)": f"{metrics_ml['wape_pct']:.2f}%",
            "MAPE (%)": f"{metrics_ml['mape_pct']:.2f}%",
            "R2 Score": metrics_ml["r2_score"],
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, round(rmse_improvement_pct, 2)


def compare_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "units_sold",
    include_prophet: bool = True,
) -> pd.DataFrame:
    """Benchmark all available modeling techniques on the temporal holdout set.

    Evaluates:
    - MeanBaseline (historical SKU average)
    - SeasonalNaiveBaseline (7-day cyclical persistence)
    - Linear Regression
    - Random Forest
    - XGBoost
    - Facebook Prophet (trend + weekly seasonality + regressors)

    Returns
    -------
    pd.DataFrame
        Complete model leaderboard sorted by RMSE ascending.
    """
    from src.models.model import DemandForecaster, ProphetForecaster

    y_true = test_df[target_col].values
    records = []

    # 1. Mean Baseline
    mean_b = MeanBaseline(target_col=target_col)
    mean_b.fit(train_df)
    m_mean = evaluate_forecast(y_true, mean_b.predict(test_df))
    records.append({
        "Model": "MeanBaseline",
        "Family": "Heuristic",
        "MAE": m_mean["mae"],
        "RMSE": m_mean["rmse"],
        "WAPE (%)": f"{m_mean['wape_pct']:.2f}%",
        "MAPE (%)": f"{m_mean['mape_pct']:.2f}%",
        "R2 Score": m_mean["r2_score"],
    })

    # 2. Seasonal Naive Baseline
    lag_c = f"{target_col}_lag_7" if f"{target_col}_lag_7" in test_df.columns else "units_sold_lag_7"
    if lag_c in test_df.columns:
        s_base = SeasonalNaiveBaseline(lag_col=lag_c)
        m_seas = evaluate_forecast(y_true, s_base.predict(test_df))
        records.append({
            "Model": "SeasonalNaive (7-Day)",
            "Family": "Persistence",
            "MAE": m_seas["mae"],
            "RMSE": m_seas["rmse"],
            "WAPE (%)": f"{m_seas['wape_pct']:.2f}%",
            "MAPE (%)": f"{m_seas['mape_pct']:.2f}%",
            "R2 Score": m_seas["r2_score"],
        })

    # 3. Linear Regression
    try:
        lr = DemandForecaster(model_type="linear_regression")
        lr.fit(train_df, target_col=target_col)
        m_lr = evaluate_forecast(y_true, lr.predict(test_df))
        records.append({
            "Model": "Linear Regression",
            "Family": "Statistical",
            "MAE": m_lr["mae"],
            "RMSE": m_lr["rmse"],
            "WAPE (%)": f"{m_lr['wape_pct']:.2f}%",
            "MAPE (%)": f"{m_lr['mape_pct']:.2f}%",
            "R2 Score": m_lr["r2_score"],
        })
    except Exception as e:
        logger.warning("Linear regression evaluation skipped: %s", e)

    # 4. Random Forest
    try:
        rf = DemandForecaster(model_type="random_forest", n_estimators=60, max_depth=8)
        rf.fit(train_df, target_col=target_col)
        m_rf = evaluate_forecast(y_true, rf.predict(test_df))
        records.append({
            "Model": "Random Forest",
            "Family": "Tree Ensemble",
            "MAE": m_rf["mae"],
            "RMSE": m_rf["rmse"],
            "WAPE (%)": f"{m_rf['wape_pct']:.2f}%",
            "MAPE (%)": f"{m_rf['mape_pct']:.2f}%",
            "R2 Score": m_rf["r2_score"],
        })
    except Exception as e:
        logger.warning("Random forest evaluation skipped: %s", e)

    # 5. XGBoost
    try:
        xgb_m = DemandForecaster(model_type="xgboost", n_estimators=60, max_depth=5)
        xgb_m.fit(train_df, target_col=target_col)
        m_xgb = evaluate_forecast(y_true, xgb_m.predict(test_df))
        records.append({
            "Model": "XGBoost",
            "Family": "Gradient Boosting",
            "MAE": m_xgb["mae"],
            "RMSE": m_xgb["rmse"],
            "WAPE (%)": f"{m_xgb['wape_pct']:.2f}%",
            "MAPE (%)": f"{m_xgb['mape_pct']:.2f}%",
            "R2 Score": m_xgb["r2_score"],
        })
    except Exception as e:
        logger.warning("XGBoost evaluation skipped: %s", e)

    # 6. Prophet
    if include_prophet:
        try:
            date_col = "date" if "date" in train_df.columns else ("ds" if "ds" in train_df.columns else None)
            if date_col:
                pf = ProphetForecaster(date_col=date_col, target_col=target_col)
                pf.fit(train_df)
                m_pf = evaluate_forecast(y_true, pf.predict(test_df))
                records.append({
                    "Model": "Prophet",
                    "Family": "Additive Time-Series",
                    "MAE": m_pf["mae"],
                    "RMSE": m_pf["rmse"],
                    "WAPE (%)": f"{m_pf['wape_pct']:.2f}%",
                    "MAPE (%)": f"{m_pf['mape_pct']:.2f}%",
                    "R2 Score": m_pf["r2_score"],
                })
        except Exception as e:
            logger.warning("Prophet evaluation skipped: %s", e)

    leaderboard = pd.DataFrame(records).sort_values("RMSE").reset_index(drop=True)
    return leaderboard
