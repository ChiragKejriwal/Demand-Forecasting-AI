"""Comprehensive test suite for DynamicPricingOptimizer."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raw_data
from src.optimization.optimizer import DynamicPricingOptimizer


def run_optimizer_tests():
    optimizer = DynamicPricingOptimizer(default_max_price_change_pct=0.20)

    print("=" * 60)
    print("TEST 1: Standard Continuous Optimization (SciPy)")
    print("=" * 60)
    # Base: Price $100, Cost $50, Demand 20, Elasticity -1.8
    res = optimizer.optimize_price(
        base_price=100.0,
        unit_cost=50.0,
        base_demand=20.0,
        elasticity=-1.8,
        inventory_level=100.0,
        product_id="SKU-101",
        category="Electronics",
    )
    print(f"Base Price: $100.00 -> Recommended Price: ${res.recommended_price:.2f}")
    print(f"Base Profit: ${(100-50)*20:.2f} -> Expected Profit: ${res.expected_profit:.2f}")
    print(f"Expected Demand: {res.expected_demand:.2f} units, Margin: {res.profit_margin_pct:.1f}%")
    print(f"Active Constraints: {res.active_constraints}")

    # Verify Constraint 1: Bounds within +/- 20%
    assert 80.0 <= res.recommended_price <= 120.0, f"Price {res.recommended_price} violated +/- 20% bound!"
    # Verify Constraint 2: Margin strictly > unit_cost
    assert res.recommended_price > 50.0, "Price violated margin constraint!"
    # Verify profit maximization
    assert res.expected_profit >= (100.0 - 50.0) * 20.0, "Optimized profit did not meet or exceed baseline!"

    print("\n" + "=" * 60)
    print("TEST 2: Constraint Verification - Low Inventory / Stockout Prevention")
    print("=" * 60)
    # Inventory is constrained to only 10 units when base demand is 25
    res_inv = optimizer.optimize_price(
        base_price=100.0,
        unit_cost=40.0,
        base_demand=25.0,
        elasticity=-2.0,
        inventory_level=10.0,  # Low stock
        product_id="SKU-LOW-STOCK",
    )
    print(f"Low Stock Recommended Price: ${res_inv.recommended_price:.2f}")
    print(f"Expected Demand Sold: {res_inv.expected_demand:.2f} (Inventory Cap: {res_inv.inventory_level})")
    print(f"Active Constraints: {res_inv.active_constraints}")
    # Verify sold demand does not exceed inventory
    assert res_inv.expected_demand <= 10.0, "Predicted demand sold exceeded inventory level!"

    print("\n" + "=" * 60)
    print("TEST 3: Constraint Verification - Margin Floor vs Lower Bound")
    print("=" * 60)
    # Cost is $90 on $100 base price. Lower bound (-20%) is $80, but cost is $90!
    # The margin constraint must prevent price from going below cost!
    res_margin = optimizer.optimize_price(
        base_price=100.0,
        unit_cost=90.0,
        base_demand=30.0,
        elasticity=-3.0,  # Highly elastic, wanting to cut price
        inventory_level=100.0,
        product_id="SKU-HIGH-COST",
    )
    print(f"High Cost ($90) Recommended Price: ${res_margin.recommended_price:.2f}")
    assert res_margin.recommended_price > 90.0, f"Price ${res_margin.recommended_price} failed margin constraint!"
    print("Margin floor successfully protected profit margin from dropping below unit cost.")

    print("\n" + "=" * 60)
    print("TEST 4: Mixed-Integer Programming via PuLP")
    print("=" * 60)
    res_pulp = optimizer.optimize_discrete_pulp(
        base_price=100.0,
        unit_cost=50.0,
        base_demand=20.0,
        elasticity=-1.8,
        inventory_level=50.0,
        product_id="SKU-PULP-1",
    )
    print(f"PuLP Selected Price: ${res_pulp.recommended_price:.2f} ({res_pulp.price_change_pct:+.1f}%)")
    print(f"PuLP Expected Profit: ${res_pulp.expected_profit:.2f}, Status: {res_pulp.status}")
    assert res_pulp.recommended_price > 50.0
    assert 80.0 <= res_pulp.recommended_price <= 120.0

    print("\n" + "=" * 60)
    print("TEST 5: Batch Catalog Optimization on Augmented Dataset")
    print("=" * 60)
    df = load_raw_data()
    sample_catalog = df.head(100).copy()
    results_catalog = optimizer.optimize_catalog(
        sample_catalog,
        category_elasticity_map={"Sports": -1.4, "Shoes": -1.2, "Electronics": -1.8},
        default_elasticity=-1.3,
    )
    print(f"Optimized {len(results_catalog)} catalog items.")
    print("Sample Output Rows:")
    cols_to_show = [
        "product_id", "category", "base_price", "unit_cost",
        "recommended_price", "expected_profit", "profit_lift_dollars", "profit_lift_pct"
    ]
    print(results_catalog[cols_to_show].head(5).to_string())
    total_lift = results_catalog["profit_lift_dollars"].sum()
    print(f"\nTotal Projected Profit Lift on Sample Catalog: +${total_lift:,.2f}")
    assert total_lift >= 0, "Profit lift should be non-negative!"

    print("\n>>> ALL OPTIMIZATION AND CONSTRAINT TESTS PASSED! <<<")


if __name__ == "__main__":
    run_optimizer_tests()
