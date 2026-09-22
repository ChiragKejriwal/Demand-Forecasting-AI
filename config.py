"""Configuration settings for Dynamic Pricing Under Demand Uncertainty project."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "retail_pricing_demand_augmented.csv"

# Global Model & Simulation Settings
RANDOM_SEED = 42
ASSUMED_ELASTICITY_FOR_DEMO = -1.18  # Empirical valid Electronics category elasticity benchmark
DEFAULT_PRICE_ELASTICITY_RANGE = (-3.5, -0.5)
OPTIMIZATION_SOLVER = "PULP_CBC_CMD"
