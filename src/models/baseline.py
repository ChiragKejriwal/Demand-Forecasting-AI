"""Baseline demand forecasting models for benchmarking machine learning estimators."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MeanBaseline:
    """Historical mean demand baseline per product with global mean fallback."""

    def __init__(
        self,
        target_col: str = "units_sold",
        product_col: str = "product_id",
    ) -> None:
        """Initialize MeanBaseline.

        Parameters
        ----------
        target_col : str, optional
            Target column to compute mean for, by default "units_sold".
        product_col : str, optional
            Product grouping column, by default "product_id".
        """
        self.target_col = target_col
        self.product_col = product_col
        self.product_means_: Dict[str, float] = {}
        self.global_mean_: float = 0.0
        self.is_fitted_: bool = False

    def fit(self, train_df: pd.DataFrame) -> "MeanBaseline":
        """Calculate historical mean demand per product and global mean.

        Parameters
        ----------
        train_df : pd.DataFrame
            Historical training data.

        Returns
        -------
        MeanBaseline
            Fitted baseline instance.
        """
        if self.target_col not in train_df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in training data.")

        self.global_mean_ = float(train_df[self.target_col].mean())
        if pd.isna(self.global_mean_):
            self.global_mean_ = 0.0

        if self.product_col in train_df.columns:
            self.product_means_ = (
                train_df.groupby(self.product_col)[self.target_col].mean().to_dict()
            )
        else:
            self.product_means_ = {}

        self.is_fitted_ = True
        logger.info(
            "MeanBaseline fitted across %d products (Global mean: %.2f).",
            len(self.product_means_),
            self.global_mean_,
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Map historical product means to the input dataframe with global fallback.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to predict demand for.

        Returns
        -------
        np.ndarray
            Array of predicted demand.
        """
        if not self.is_fitted_:
            raise RuntimeError("MeanBaseline must be fitted before predict().")

        if self.product_col in df.columns:
            preds = df[self.product_col].map(self.product_means_).fillna(self.global_mean_).values
        else:
            preds = np.full(len(df), self.global_mean_)

        return np.asarray(preds, dtype=float)


class SeasonalNaiveBaseline:
    """Seasonal persistence baseline returning 7-day prior demand (units_sold_lag_7)."""

    def __init__(self, lag_col: str = "units_sold_lag_7") -> None:
        """Initialize SeasonalNaiveBaseline.

        Parameters
        ----------
        lag_col : str, optional
            Column name containing 7-day lag, by default "units_sold_lag_7".
        """
        self.lag_col = lag_col

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return 7-day lagged demand values, filling any NaNs with 0.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the lag column.

        Returns
        -------
        np.ndarray
            Predicted demand based on 7-day seasonal persistence.
        """
        if self.lag_col in df.columns:
            preds = df[self.lag_col].fillna(0.0).values
        elif "units_sold" in df.columns:
            logger.warning("'%s' not found; defaulting to 0.", self.lag_col)
            preds = np.zeros(len(df))
        else:
            preds = np.zeros(len(df))

        return np.asarray(preds, dtype=float)
