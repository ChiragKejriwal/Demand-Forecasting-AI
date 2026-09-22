"""Unit tests for data validator and grain aggregation."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from src.data.loader import load_raw_data
from src.data.validator import DataValidationError, validate_sales_data
from src.preprocessing.preprocessing import aggregate_to_grain, preprocess_data


def test_validator_valid_data():
    df = load_raw_data()
    is_valid, errors, warnings = validate_sales_data(df)
    assert is_valid is True
    assert len(errors) == 0
    print("test_validator_valid_data: Passed")


def test_validator_missing_column():
    df = load_raw_data().head(10).drop(columns=["current_price"])
    try:
        validate_sales_data(df, raise_on_error=True)
        assert False, "Should have raised DataValidationError for missing column"
    except DataValidationError as e:
        assert any("current_price" in err for err in e.errors)
        print("test_validator_missing_column: Passed")


def test_validator_negative_price():
    df = load_raw_data().head(10).copy()
    df.loc[0, "current_price"] = -5.0
    try:
        validate_sales_data(df, raise_on_error=True)
        assert False, "Should have raised DataValidationError for negative price"
    except DataValidationError as e:
        assert any("current_price <= 0" in err for err in e.errors)
        print("test_validator_negative_price: Passed")


def test_validator_negative_units():
    df = load_raw_data().head(10).copy()
    df.loc[0, "units_sold"] = -1
    try:
        validate_sales_data(df, raise_on_error=True)
        assert False, "Should have raised DataValidationError for negative units_sold"
    except DataValidationError as e:
        assert any("units_sold < 0" in err for err in e.errors)
        print("test_validator_negative_units: Passed")


def test_validator_soft_warnings():
    df = load_raw_data().head(10).copy()
    # Create revenue mismatch
    df.loc[0, "revenue"] = 99999.0
    # Duplicate row at grain
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    is_valid, errors, warnings = validate_sales_data(df, raise_on_error=False)
    assert is_valid is True
    assert len(warnings) >= 2
    assert any("differs from revenue" in w for w in warnings)
    assert any("duplicate records" in w for w in warnings)
    print("test_validator_soft_warnings: Passed")


def test_aggregate_to_grain():
    df = load_raw_data().head(50).copy()
    agg_df = aggregate_to_grain(df)
    assert "date" in agg_df.columns
    assert "product_id" in agg_df.columns
    # Check uniqueness at grain
    assert agg_df.duplicated(subset=["date", "product_id"]).sum() == 0
    print("test_aggregate_to_grain: Passed")


def test_preprocess_data_integration():
    df = load_raw_data().head(500).copy()
    processed = preprocess_data(df)
    assert "date" in processed.columns
    assert "product_id" in processed.columns
    assert not processed.isnull().any().any()
    print("test_preprocess_data_integration: Passed")


def test_inspect_uploaded_df():
    from src.data.loader import inspect_uploaded_df

    # Test valid dataframe with synonyms and extra fields
    df_valid = pd.DataFrame({
        "order_date": ["2024-01-01"],
        "sku": ["SKU_1"],
        "qty": [10],
        "price": [25.0],
        "store_id": [101],
        "weather_temp": [75.2],
    })
    res = inspect_uploaded_df(df_valid)
    assert res["is_valid"] is True
    assert res["matched_mandatory"]["date"] == "order_date"
    assert res["matched_mandatory"]["units_sold"] == "qty"
    assert "store_id" in res["extra_columns"]
    assert "weather_temp" in res["extra_columns"]

    # Test missing mandatory date and units_sold
    df_invalid = pd.DataFrame({
        "sku": ["SKU_1"],
        "price": [25.0],
        "store_id": [101],
    })
    res_inv = inspect_uploaded_df(df_invalid)
    assert res_inv["is_valid"] is False
    assert "date" in res_inv["missing_mandatory"]
    assert "units_sold" in res_inv["missing_mandatory"]
    print("test_inspect_uploaded_df: Passed")


if __name__ == "__main__":
    test_validator_valid_data()
    test_validator_missing_column()
    test_validator_negative_price()
    test_validator_negative_units()
    test_validator_soft_warnings()
    test_aggregate_to_grain()
    test_preprocess_data_integration()
    test_inspect_uploaded_df()
    print("\n>>> ALL VALIDATION AND GRAIN AGGREGATION TESTS PASSED! <<<")
