#!/usr/bin/env python3
"""Data Augmentation Script for Dynamic Pricing Under Demand Uncertainty.

This script augments the raw retail dataset with:
1. 'unit_cost': Uniformly distributed between 40% and 70% of 'base_price'.
2. 'competitor_price': Fluctuation uniformly distributed between -5% and +15% of 'base_price'
   (i.e., 95% to 115% of base_price).

Output: Saved to data/retail_pricing_demand_augmented.csv

Note: Includes optimized Pandas/NumPy vectorized implementation with an automatic
fallback to Python's standard `csv` module if dependencies are not yet installed.
"""

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _augment_with_pandas(
    input_path: Path, output_path: Path, random_seed: int
) -> None:
    import numpy as np
    import pandas as pd

    logger.info("Using Pandas/NumPy engine...")
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows across %d columns.", len(df), len(df.columns))

    if "base_price" not in df.columns:
        raise ValueError(f"'base_price' column missing from {input_path}")

    rng = np.random.default_rng(seed=random_seed)
    n_rows = len(df)

    cost_factors = rng.uniform(low=0.40, high=0.70, size=n_rows)
    df["unit_cost"] = np.round(df["base_price"] * cost_factors, 2)

    competitor_factors = rng.uniform(low=0.95, high=1.15, size=n_rows)
    df["competitor_price"] = np.round(df["base_price"] * competitor_factors, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved augmented dataset to %s", output_path)
    logger.info(
        "Augmented columns summary:\n%s",
        df[["base_price", "unit_cost", "competitor_price"]].describe().to_string(),
    )


def _augment_with_csv(
    input_path: Path, output_path: Path, random_seed: int
) -> None:
    logger.info("Using Python standard library CSV engine...")
    random.seed(random_seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, mode="r", encoding="utf-8", newline="") as fin, open(
        output_path, mode="w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if not fieldnames or "base_price" not in fieldnames:
            raise ValueError(f"'base_price' column missing from {input_path}")

        new_fieldnames = list(fieldnames) + ["unit_cost", "competitor_price"]
        writer = csv.DictWriter(fout, fieldnames=new_fieldnames)
        writer.writeheader()

        row_count = 0
        for row in reader:
            base_price = float(row["base_price"])
            # 40% to 70% of base_price
            unit_cost = round(base_price * random.uniform(0.40, 0.70), 2)
            # -5% to +15% of base_price (i.e. 0.95 to 1.15)
            competitor_price = round(base_price * random.uniform(0.95, 1.15), 2)

            row["unit_cost"] = f"{unit_cost:.2f}"
            row["competitor_price"] = f"{competitor_price:.2f}"
            writer.writerow(row)
            row_count += 1

    logger.info("Successfully processed and augmented %d rows.", row_count)
    logger.info("Saved augmented dataset to %s", output_path)


def augment_dataset(
    input_path: Path,
    output_path: Path,
    random_seed: int = 42,
) -> None:
    """Load raw pricing dataset, generate simulated cost & competitor prices, and export."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    start_time = time.time()
    logger.info("Starting dataset augmentation: %s -> %s", input_path, output_path)

    try:
        _augment_with_pandas(input_path, output_path, random_seed)
    except ImportError:
        _augment_with_csv(input_path, output_path, random_seed)

    elapsed = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Augmentation finished in %.2f seconds (Output size: %.2f MB).", elapsed, file_size_mb)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Augment raw retail pricing data with unit_cost and competitor_price."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=repo_root / "data" / "retail_pricing_demand_100k.csv",
        help="Path to input raw CSV file (default: data/retail_pricing_demand_100k.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=repo_root / "data" / "retail_pricing_demand_augmented.csv",
        help="Path to output augmented CSV file (default: data/retail_pricing_demand_augmented.csv)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    augment_dataset(input_path=args.input, output_path=args.output, random_seed=args.seed)


if __name__ == "__main__":
    main()
