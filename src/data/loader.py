"""Data loading and ingestion module for retail pricing and demand data."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from src.data.validator import DataValidationError, validate_sales_data

logger = logging.getLogger(__name__)

SYNONYMS: Dict[str, list] = {
    "date": ["date", "sales_date", "order_date", "datetime", "day", "timestamp", "time"],
    "product_id": ["product_id", "product", "sku", "item_id", "item", "product_name"],
    "units_sold": ["units_sold", "units", "quantity", "qty", "sales", "demand", "volume"],
    "current_price": ["current_price", "price", "unit_price", "selling_price", "sales_price"],
    "base_price": ["base_price", "original_price", "msrp", "list_price"],
    "unit_cost": ["unit_cost", "cost", "cogs", "acquisition_cost"],
    "competitor_price": ["competitor_price", "comp_price", "market_price", "competitor"],
    "category": ["category", "dept", "department", "product_category", "type", "segment"],
    "region": ["region", "store", "location", "territory", "city", "state"],
    "channel": ["channel", "sales_channel", "platform", "store_type"],
    "revenue": ["revenue", "total_sales", "amount", "total_revenue"],
    "inventory_level": ["inventory_level", "inventory", "stock", "on_hand"],
    "promotion_type": ["promotion_type", "promo", "discount_type"],
}

MANDATORY_COLUMNS = ["date", "product_id", "units_sold", "current_price"]

OPTIONAL_COLUMNS = [
    "base_price",
    "unit_cost",
    "competitor_price",
    "category",
    "region",
    "channel",
    "revenue",
    "inventory_level",
    "promotion_type",
]


def inspect_uploaded_df(df_raw: pd.DataFrame) -> Dict[str, Any]:
    """Inspect raw uploaded dataframe before normalization to categorize columns into mandatory, optional, and extra."""
    raw_cols = list(df_raw.columns)
    col_lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df_raw.columns}

    def _match_target_cols(targets):
        matched, missing = {}, []
        for col in targets:
            found = next((col_lookup[s] for s in SYNONYMS.get(col, [col]) if s in col_lookup), None)
            if found is not None:
                matched[col] = found
            else:
                missing.append(col)
        return matched, missing

    matched_mandatory, missing_mandatory = _match_target_cols(MANDATORY_COLUMNS)
    matched_optional, missing_optional = _match_target_cols(OPTIONAL_COLUMNS)

    accounted_cols = set(matched_mandatory.values()).union(set(matched_optional.values()))
    extra_columns = [c for c in raw_cols if c not in accounted_cols]

    is_valid = len(missing_mandatory) == 0
    errors = []
    if not is_valid:
        errors.append(f"Missing mandatory fields: {missing_mandatory}")

    # Domain constraints on price and demand if present
    if "current_price" in matched_mandatory:
        col_name = matched_mandatory["current_price"]
        num_prices = pd.to_numeric(df_raw[col_name], errors="coerce")
        if (num_prices <= 0).any():
            errors.append("Column 'current_price' contains non-positive or zero values.")
            is_valid = False

    if "units_sold" in matched_mandatory:
        col_name = matched_mandatory["units_sold"]
        num_units = pd.to_numeric(df_raw[col_name], errors="coerce")
        if (num_units < 0).any():
            errors.append("Column 'units_sold' contains negative values.")
            is_valid = False

    return {
        "raw_columns": raw_cols,
        "matched_mandatory": matched_mandatory,
        "missing_mandatory": missing_mandatory,
        "matched_optional": matched_optional,
        "missing_optional": missing_optional,
        "extra_columns": extra_columns,
        "is_valid": is_valid,
        "errors": errors,
    }


def normalize_and_impute_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names via synonym mapping and auto-impute standard retail attributes."""
    df = df.copy()

    # 1. Map common column synonyms
    col_lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    rename_map = {}
    for target, syn_list in SYNONYMS.items():
        if target not in df.columns:
            for s in syn_list:
                if s in col_lookup:
                    rename_map[col_lookup[s]] = target
                    break
    if rename_map:
        df = df.rename(columns=rename_map)

    # 2. Impute core demand and pricing features
    if "units_sold" not in df.columns:
        df["units_sold"] = 1.0
    else:
        df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(1.0).clip(lower=0.0)

    if "current_price" not in df.columns:
        df["current_price"] = 19.99
    else:
        df["current_price"] = pd.to_numeric(df["current_price"], errors="coerce").fillna(19.99).clip(lower=0.01)

    df["base_price"] = pd.to_numeric(df["base_price"], errors="coerce").fillna(df["current_price"]) if "base_price" in df.columns else df["current_price"]
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(df["current_price"] * df["units_sold"]) if "revenue" in df.columns else df["current_price"] * df["units_sold"]

    # 3. Impute categorical metadata
    df["product_id"] = df["product_id"].astype(str) if "product_id" in df.columns else "SKU_001"
    df["category"] = df["category"].astype(str) if "category" in df.columns else "General"
    df["region"] = df["region"].astype(str) if "region" in df.columns else "National"
    df["channel"] = df["channel"].astype(str) if "channel" in df.columns else "Online"
    df["promotion_type"] = df["promotion_type"].fillna("None").astype(str) if "promotion_type" in df.columns else "None"

    # 4. Impute benchmark cost, competitor, and inventory metrics
    default_cost = np.round(df["current_price"] * 0.55, 2)
    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(default_cost) if "unit_cost" in df.columns else default_cost

    default_comp = np.round(df["current_price"] * 1.02, 2)
    df["competitor_price"] = pd.to_numeric(df["competitor_price"], errors="coerce").fillna(default_comp) if "competitor_price" in df.columns else default_comp

    max_q = max(10.0, float(df["units_sold"].max()))
    df["inventory_level"] = pd.to_numeric(df["inventory_level"], errors="coerce").fillna(50.0) if "inventory_level" in df.columns else max_q * 3.0

    return df


def load_raw_data(
    file_path: Optional[Union[str, Path, Any]] = None,
    date_col: str = "date",
    sort_by: Optional[list] = None,
) -> pd.DataFrame:
    """Load pricing dataset from CSV or file buffer, validate schema, and sort chronologically."""
    if file_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        file_path = project_root / "data" / "retail_pricing_demand_augmented.csv"

    if hasattr(file_path, "read"):
        logger.info("Loading dataset from uploaded file buffer...")
        if hasattr(file_path, "seek"):
            file_path.seek(0)
        df = pd.read_csv(file_path)
    else:
        path_obj = Path(file_path)
        if not path_obj.exists():
            logger.error("Dataset not found at: %s", path_obj)
            raise FileNotFoundError(f"Dataset not found at: {path_obj}")
        logger.info("Loading dataset from: %s", path_obj)
        df = pd.read_csv(path_obj)

    logger.info("Raw data loaded with shape: %s", df.shape)

    # 1. Perform schema inspection on the raw uploaded data
    inspection = inspect_uploaded_df(df)
    if not inspection["is_valid"]:
        err = DataValidationError(
            f"Dataset schema validation failed: {', '.join(inspection['errors'])}",
            errors=inspection["errors"],
        )
        err.inspection = inspection
        raise err

    # 2. Normalize column names & impute missing standard columns
    df = normalize_and_impute_columns(df)
    df.attrs["schema_inspection"] = inspection

    # 3. Validate data integrity
    is_valid, errors, warnings = validate_sales_data(df, raise_on_error=True)
    for warning in warnings:
        logger.warning("[Data Validation Warning] %s", warning)

    # 4. Parse date column and sort chronologically
    if date_col not in df.columns:
        raise KeyError(f"Required date column '{date_col}' not found. Available: {list(df.columns)}")

    logger.info("Parsing column '%s' as datetime...", date_col)
    df[date_col] = pd.to_datetime(df[date_col])

    if sort_by is None:
        sort_by = [date_col, "product_id"] if "product_id" in df.columns else [date_col]

    logger.info("Sorting dataset chronologically by %s...", sort_by)
    df = df.sort_values(by=sort_by).reset_index(drop=True)

    date_min = df[date_col].min().strftime("%Y-%m-%d")
    date_max = df[date_col].max().strftime("%Y-%m-%d")
    logger.info("Dataset span: from %s to %s (%d records).", date_min, date_max, len(df))

    return df


def get_dataset_summary(df: pd.DataFrame, date_col: str = "date") -> Dict[str, Union[int, float, str, dict]]:
    """Generate high-level metadata and descriptive summary of the loaded dataset."""
    summary = {
        "num_records": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "date_min": df[date_col].min().strftime("%Y-%m-%d") if date_col in df.columns else None,
        "date_max": df[date_col].max().strftime("%Y-%m-%d") if date_col in df.columns else None,
        "num_products": df["product_id"].nunique() if "product_id" in df.columns else None,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "null_counts": df.isnull().sum().to_dict(),
    }
    logger.info(
        "Dataset summary: %d rows, %d products, %.2f MB",
        summary["num_records"],
        summary["num_products"] or 0,
        summary["memory_mb"],
    )
    return summary
