"""Data validation module for sales and pricing data."""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "category",
    "region",
    "channel",
    "base_price",
    "current_price",
    "units_sold",
    "revenue",
]

TRUE_GRAIN_COLUMNS = ["product_id", "date", "region", "channel"]


class DataValidationError(Exception):
    """Exception raised when hard validation constraints are violated."""

    def __init__(self, message: str, errors: List[str]):
        super().__init__(message)
        self.errors = errors


def validate_sales_data(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> Tuple[bool, List[str], List[str]]:
    """Validate sales dataframe against integrity, domain, and grain rules.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    raise_on_error : bool, optional
        Whether to raise DataValidationError if hard fails are encountered, by default True.

    Returns
    -------
    Tuple[bool, List[str], List[str]]
        (is_valid, errors, warnings)

    Raises
    ------
    DataValidationError
        If raise_on_error is True and hard fails are detected.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # -------------------------------------------------------------
    # 1. Hard Fails
    # -------------------------------------------------------------
    # Missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Domain constraints on price and demand
    if "current_price" in df.columns:
        invalid_prices = (df["current_price"] <= 0).sum()
        if invalid_prices > 0:
            errors.append(f"Found {invalid_prices:,} rows with current_price <= 0.")

    if "units_sold" in df.columns:
        invalid_units = (df["units_sold"] < 0).sum()
        if invalid_units > 0:
            errors.append(f"Found {invalid_units:,} rows with units_sold < 0.")

    # -------------------------------------------------------------
    # 2. Soft Warnings
    # -------------------------------------------------------------
    # Null values
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].to_dict()
    if null_cols:
        warnings.append(f"Dataset contains null values across columns: {null_cols}")

    # Revenue calculation mismatch: current_price * units_sold != revenue (diff > 1.0)
    if all(col in df.columns for col in ["current_price", "units_sold", "revenue"]):
        calculated_rev = df["current_price"] * df["units_sold"]
        rev_diff = np.abs(calculated_rev - df["revenue"])
        mismatches = (rev_diff > 1.0).sum()
        if mismatches > 0:
            warnings.append(
                f"Found {mismatches:,} rows where (current_price * units_sold) differs from revenue by > 1.0."
            )

    # Duplicates at the true grain: product_id + date + region + channel
    grain_present = [col for col in TRUE_GRAIN_COLUMNS if col in df.columns]
    if len(grain_present) == len(TRUE_GRAIN_COLUMNS):
        dup_count = df.duplicated(subset=TRUE_GRAIN_COLUMNS).sum()
        if dup_count > 0:
            warnings.append(
                f"Found {dup_count:,} duplicate records at true grain ({' + '.join(TRUE_GRAIN_COLUMNS)})."
            )

    is_valid = len(errors) == 0

    if not is_valid:
        error_msg = f"Data validation failed with {len(errors)} error(s): " + " | ".join(errors)
        logger.error(error_msg)
        if raise_on_error:
            raise DataValidationError(error_msg, errors=errors)
    else:
        logger.info("Hard data validation passed successfully.")

    if warnings:
        logger.warning(
            "Data validation warnings (%d): %s",
            len(warnings),
            " | ".join(warnings),
        )

    return is_valid, errors, warnings
