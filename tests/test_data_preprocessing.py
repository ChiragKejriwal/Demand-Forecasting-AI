"""Quick verification test for data loader and preprocessing pipeline."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import get_dataset_summary, load_raw_data
from src.preprocessing.preprocessing import (
    preprocess_data,
    temporal_expanding_window_split,
    temporal_train_test_split,
)


def run_tests():
    print("=== Testing Data Loader ===")
    df = load_raw_data()
    summary = get_dataset_summary(df)
    print(f"Loaded: {summary['num_records']} rows, {summary['num_products']} products.")
    print(f"Date range: {summary['date_min']} to {summary['date_max']}")

    print("\n=== Testing Preprocessing Pipeline ===")
    sample_df = df.head(5000).copy()
    processed_df = preprocess_data(sample_df)
    print(f"Original shape: {sample_df.shape} -> Processed shape: {processed_df.shape}")
    engineered_cols = [c for c in processed_df.columns if any(k in c for k in ["lag", "rolling", "price", "margin", "day", "sin"])]
    print(f"Engineered columns ({len(engineered_cols)}): {engineered_cols[:8]}...")
    assert not processed_df.isnull().any().any(), "Found unexpected NaNs in processed data!"

    print("\n=== Testing Temporal Train/Test Split ===")
    train_df, test_df = temporal_train_test_split(df, test_days=14)
    print(f"Train span: {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')} ({len(train_df)} rows)")
    print(f"Test span:  {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')} ({len(test_df)} rows)")
    assert train_df["date"].max() < test_df["date"].min(), "Temporal leakage detected! Train max >= Test min"

    print("\n=== Testing Expanding Window Walk-Forward Split ===")
    for train_fold, val_fold, meta in temporal_expanding_window_split(df, n_splits=3, test_window_days=7):
        print(f"Fold {meta['fold']}: Train [{meta['train_start']}..{meta['train_end']}] ({meta['train_rows']} rows) -> Val [{meta['val_start']}..{meta['val_end']}] ({meta['val_rows']} rows)")
        assert train_fold["date"].max() < val_fold["date"].min(), f"Temporal leakage detected in fold {meta['fold']}!"

    print("\n>>> ALL VALIDATION CHECKS PASSED PERFECTLY! <<<")


if __name__ == "__main__":
    run_tests()
