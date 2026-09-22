"""Time-series feature engineering, missing value handling, and temporal validation splits.

Note on Time-Series Splitting:
Standard k-fold cross-validation or random train_test_split must NEVER be used on
time-series demand data. Random splits shuffle future observations into the training
set, causing severe lookahead bias (data leakage). Instead, temporal splits enforce
that models are trained strictly on past data and evaluated on future horizons.
"""

import logging
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Extract calendar and cyclical time-series features from the date column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a datetime column.
    date_col : str, optional
        Name of the datetime column, by default "date".

    Returns
    -------
    pd.DataFrame
        DataFrame with calendar and cyclical features appended.
    """
    logger.info("Extracting calendar and cyclical time-series features...")
    df = df.copy()

    dt = df[date_col].dt
    df["day_of_week"] = dt.dayofweek
    df["day_of_month"] = dt.day
    df["month"] = dt.month
    df["quarter"] = dt.quarter
    df["is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical encodings for periodic temporal signals
    # Day of week: 0-6 (period = 7)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Month: 1-12 (period = 12)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    return df


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    group_col: str = "product_id",
    target_col: str = "units_sold",
    date_col: str = "date",
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Generate strictly backward-looking lag and rolling statistics for demand.

    CRITICAL FOR FORECASTING:
    All rolling windows are shifted by at least 1 step (shift(1)) so that the target
    demand at time t is NEVER included in features predicting time t.

    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted DataFrame.
    group_col : str, optional
        Grouping column (e.g. 'product_id'), by default "product_id".
    target_col : str, optional
        Demand target column to lag and roll, by default "units_sold".
    date_col : str, optional
        Date column used to guarantee sorting, by default "date".
    lags : Optional[List[int]], optional
        List of day lags to compute, by default [1, 7, 14].
    windows : Optional[List[int]], optional
        List of rolling window day spans, by default [7, 14, 28].

    Returns
    -------
    pd.DataFrame
        DataFrame with lag and rolling features.
    """
    if lags is None:
        lags = [1, 7, 14]
    if windows is None:
        windows = [7, 14, 28]

    logger.info(
        "Creating lag features %s and rolling features %s for '%s' grouped by '%s'...",
        lags,
        windows,
        target_col,
        group_col,
    )
    df = df.sort_values(by=[group_col, date_col]).copy()

    grouped = df.groupby(group_col)[target_col]

    # Backward-looking lags
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = grouped.shift(lag)

    # Backward-looking rolling averages and standard deviations (strictly shifted by 1)
    for w in windows:
        shifted = grouped.shift(1)
        rolling_obj = shifted.groupby(df[group_col]).rolling(window=w, min_periods=1)
        df[f"{target_col}_rolling_mean_{w}"] = rolling_obj.mean().reset_index(level=0, drop=True)
        df[f"{target_col}_rolling_std_{w}"] = rolling_obj.std().reset_index(level=0, drop=True)

    return df.sort_values(by=[date_col, group_col]).reset_index(drop=True)


def create_pricing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer competitive pricing, margins, and discount depth features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing pricing, cost, and competitor price columns.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with pricing interaction features.
    """
    logger.info("Computing pricing and competitor elasticity interaction features...")
    df = df.copy()

    if "current_price" in df.columns and "competitor_price" in df.columns:
        # Ratio of our price to competitor price (>1 means we are more expensive)
        df["price_ratio_competitor"] = df["current_price"] / (df["competitor_price"] + 1e-6)
        # Absolute price differential
        df["price_diff_competitor"] = df["current_price"] - df["competitor_price"]

    if "current_price" in df.columns and "unit_cost" in df.columns:
        # Absolute unit profit margin
        df["unit_margin"] = df["current_price"] - df["unit_cost"]
        # Unit profit margin percentage
        df["margin_pct"] = df["unit_margin"] / (df["current_price"] + 1e-6)

    if "base_price" in df.columns and "current_price" in df.columns:
        # Calculated discount percentage
        df["effective_discount_pct"] = (df["base_price"] - df["current_price"]) / (df["base_price"] + 1e-6)

    return df


def handle_missing_values(
    df: pd.DataFrame,
    group_col: str = "product_id",
    target_col: str = "units_sold",
    numeric_strategy: str = "median",
) -> pd.DataFrame:
    """Handle missing values produced by lag/rolling windows and sensor drops.

    Uses group-wise backfill/forward-fill first, followed by global numeric
    median imputation for initial warm-up periods.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    group_col : str, optional
        Column to group by for forward/backward filling, by default "product_id".
    numeric_strategy : str, optional
        Fallback imputation strategy for remaining numeric NaNs, by default "median".

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame without missing values in engineered features.
    """
    logger.info("Handling missing values (group-wise bfill/ffill + %s)...", numeric_strategy)
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Group-wise backfill then forward-fill for initial lag periods per product
    if group_col in df.columns:
        df[numeric_cols] = df.groupby(group_col)[numeric_cols].transform(
            lambda grp: grp.bfill().ffill()
        )

    # Any remaining NaNs (e.g. initial lag periods or constant series std) are filled
    target_fallback = df[target_col].median() if target_col in df.columns else 0.0
    if pd.isna(target_fallback):
        target_fallback = 0.0

    for col in numeric_cols:
        if df[col].isnull().any():
            if numeric_strategy == "median" and df[col].notna().any():
                med = df[col].median()
                fill_val = target_fallback if pd.isna(med) else med
            else:
                fill_val = target_fallback if numeric_strategy == "median" else 0.0
            df[col] = df[col].fillna(fill_val)

    # Categorical columns filled with 'Unknown' or 'None'
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("None")

    return df


def aggregate_to_grain(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction data to the product-date grain ['date', 'product_id'].

    Performs:
    - Sum: units_sold, revenue, inventory_level
    - Mean: current_price, base_price, discount_pct, unit_cost, competitor_price
    - Max: stockout_flag
    - First: category, brand, season

    Parameters
    ----------
    df : pd.DataFrame
        Input sales DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame grouped by ['date', 'product_id'].
    """
    logger.info("Aggregating sales data to ['date', 'product_id'] grain from %d rows...", len(df))

    agg_dict = {}

    # Sum aggregations
    for col in ["units_sold", "revenue", "inventory_level"]:
        if col in df.columns:
            agg_dict[col] = "sum"

    # Mean aggregations
    for col in ["current_price", "base_price", "discount_pct", "unit_cost", "competitor_price"]:
        if col in df.columns:
            agg_dict[col] = "mean"

    # Max aggregations
    if "stockout_flag" in df.columns:
        agg_dict["stockout_flag"] = "max"

    # First aggregations
    for col in ["category", "brand", "season"]:
        if col in df.columns:
            agg_dict[col] = "first"

    # Retain promotion_type if present
    if "promotion_type" in df.columns:
        agg_dict["promotion_type"] = "first"

    aggregated_df = (
        df.groupby(["date", "product_id"], as_index=False)
        .agg(agg_dict)
        .sort_values(by=["date", "product_id"])
        .reset_index(drop=True)
    )

    logger.info(
        "Aggregation complete. Output shape: %s (%d rows).",
        aggregated_df.shape,
        len(aggregated_df),
    )
    return aggregated_df


def preprocess_data(
    df: pd.DataFrame,
    date_col: str = "date",
    group_col: str = "product_id",
    target_col: str = "units_sold",
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Run the complete preprocessing and feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or augmented pricing DataFrame.
    date_col : str, optional
        Date column name, by default "date".
    group_col : str, optional
        Product/entity grouping column, by default "product_id".
    target_col : str, optional
        Target demand column, by default "units_sold".
    lags : Optional[List[int]], optional
        Lags to compute, by default [1, 7, 14].
    windows : Optional[List[int]], optional
        Rolling windows, by default [7, 14, 28].

    Returns
    -------
    pd.DataFrame
        Fully engineered, model-ready feature matrix.
    """
    logger.info("Starting end-to-end preprocessing pipeline on %d rows...", len(df))
    # Step 1: Aggregate to product-date grain
    df_grain = aggregate_to_grain(df)

    df_features = create_calendar_features(df_grain, date_col=date_col)
    df_features = create_pricing_features(df_features)
    df_features = create_lag_and_rolling_features(
        df_features, group_col=group_col, target_col=target_col, date_col=date_col, lags=lags, windows=windows
    )
    df_clean = handle_missing_values(df_features, group_col=group_col, target_col=target_col)
    logger.info("Preprocessing complete. Output shape: %s", df_clean.shape)
    return df_clean


# -------------------------------------------------------------------------
# Temporal Validation Splitting (No Random Splits!)
# -------------------------------------------------------------------------

def temporal_train_test_split(
    df: pd.DataFrame,
    date_col: str = "date",
    test_days: int = 14,
    cutoff_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict chronological holdout split to prevent temporal data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed datetime column.
    date_col : str, optional
        Datetime column name, by default "date".
    test_days : int, optional
        Number of most recent days to allocate to the test holdout, by default 14.
    cutoff_date : Optional[str], optional
        Explicit cutoff string (e.g. '2026-03-18'). If provided, overrides test_days.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) partitioned strictly by date.
    """
    df_sorted = df.sort_values(by=date_col).copy()

    if cutoff_date is not None:
        split_point = pd.to_datetime(cutoff_date)
    else:
        max_date = df_sorted[date_col].max()
        split_point = max_date - pd.Timedelta(days=test_days)

    train_df = df_sorted[df_sorted[date_col] < split_point].copy()
    test_df = df_sorted[df_sorted[date_col] >= split_point].copy()

    train_min = train_df[date_col].min().strftime("%Y-%m-%d") if len(train_df) > 0 else "N/A"
    train_max = train_df[date_col].max().strftime("%Y-%m-%d") if len(train_df) > 0 else "N/A"
    test_min = test_df[date_col].min().strftime("%Y-%m-%d") if len(test_df) > 0 else "N/A"
    test_max = test_df[date_col].max().strftime("%Y-%m-%d") if len(test_df) > 0 else "N/A"

    logger.info(
        "Temporal Split Summary:\n"
        "  - Cutoff Date: %s\n"
        "  - Train Set: %s to %s (%d rows)\n"
        "  - Test Set:  %s to %s (%d rows)",
        split_point.strftime("%Y-%m-%d"),
        train_min,
        train_max,
        len(train_df),
        test_min,
        test_max,
        len(test_df),
    )

    return train_df, test_df


def temporal_expanding_window_split(
    df: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 4,
    test_window_days: int = 7,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]], None, None]:
    """Generate expanding-window (walk-forward) train/validation splits.

    In each fold, the training window expands chronologically, while validation
    is performed strictly on the subsequent non-overlapping time horizon.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with parsed datetime column.
    date_col : str, optional
        Name of datetime column, by default "date".
    n_splits : int, optional
        Number of cross-validation folds, by default 4.
    test_window_days : int, optional
        Length of validation window in days for each fold, by default 7.

    Yields
    ------
    Generator[Tuple[pd.DataFrame, pd.DataFrame, dict], None, None]
        Yields (train_fold, val_fold, fold_metadata) for each split.
    """
    df_sorted = df.sort_values(by=date_col).copy()
    max_date = df_sorted[date_col].max()

    # Total days reserved across folds
    total_val_days = n_splits * test_window_days
    first_split_cutoff = max_date - pd.Timedelta(days=total_val_days)

    logger.info(
        "Initializing %d-fold expanding window CV with %d-day evaluation horizons.",
        n_splits,
        test_window_days,
    )

    for fold in range(n_splits):
        val_start = first_split_cutoff + pd.Timedelta(days=fold * test_window_days)
        val_end = val_start + pd.Timedelta(days=test_window_days)

        train_fold = df_sorted[df_sorted[date_col] < val_start].copy()
        val_fold = df_sorted[
            (df_sorted[date_col] >= val_start) & (df_sorted[date_col] < val_end)
        ].copy()

        meta = {
            "fold": fold + 1,
            "train_start": train_fold[date_col].min().strftime("%Y-%m-%d"),
            "train_end": train_fold[date_col].max().strftime("%Y-%m-%d"),
            "val_start": val_fold[date_col].min().strftime("%Y-%m-%d"),
            "val_end": val_fold[date_col].max().strftime("%Y-%m-%d"),
            "train_rows": len(train_fold),
            "val_rows": len(val_fold),
        }

        logger.info(
            "Fold %d/%d: Train [%s to %s] (%d rows) -> Val [%s to %s] (%d rows)",
            fold + 1,
            n_splits,
            meta["train_start"],
            meta["train_end"],
            meta["train_rows"],
            meta["val_start"],
            meta["val_end"],
            meta["val_rows"],
        )

        yield train_fold, val_fold, meta


def resample_time_series(
    df: pd.DataFrame,
    freq: str = "W",
    date_col: str = "date",
    group_col: Optional[str] = "product_id",
) -> pd.DataFrame:
    """Resample transactional or daily sales time-series to Daily, Weekly, or Monthly grains.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a datetime column and sales metrics.
    freq : str, optional
        Target frequency: 'D' (Daily), 'W' (Weekly), or 'M' (Monthly), by default 'W'.
    date_col : str, optional
        Datetime column name, by default 'date'.
    group_col : Optional[str], optional
        Granularity grouping identifier (e.g. 'product_id'), by default 'product_id'.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with aggregated sales volumes and average price attributes.
    """
    df_clean = df.copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])

    freq_map = {
        "daily": "D",
        "d": "D",
        "weekly": "W-MON",
        "w": "W-MON",
        "monthly": "MS",
        "m": "MS",
    }
    target_freq = freq_map.get(str(freq).lower(), "W-MON")

    group_keys = [group_col] if group_col and group_col in df_clean.columns else []

    agg_rules: Dict[str, Any] = {}
    if "units_sold" in df_clean.columns:
        agg_rules["units_sold"] = "sum"
    if "revenue" in df_clean.columns:
        agg_rules["revenue"] = "sum"
    if "current_price" in df_clean.columns:
        agg_rules["current_price"] = "mean"
    if "base_price" in df_clean.columns:
        agg_rules["base_price"] = "median"
    if "competitor_price" in df_clean.columns:
        agg_rules["competitor_price"] = "mean"
    if "unit_cost" in df_clean.columns:
        agg_rules["unit_cost"] = "mean"
    if "inventory_level" in df_clean.columns:
        agg_rules["inventory_level"] = "last"
    if "category" in df_clean.columns:
        agg_rules["category"] = "first"

    if not group_keys:
        resampled = (
            df_clean.set_index(date_col)
            .resample(target_freq)
            .agg(agg_rules)
            .reset_index()
        )
    else:
        resampled = (
            df_clean.set_index(date_col)
            .groupby(group_keys)
            .resample(target_freq)
            .agg(agg_rules)
            .reset_index()
        )

    return resampled
