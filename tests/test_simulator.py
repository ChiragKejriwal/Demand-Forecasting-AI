"""Verification test suite for MonteCarloSimulator and downside risk metrics."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation.simulator import MonteCarloSimulator, UncertaintyParameters


def run_simulator_tests():
    simulator = MonteCarloSimulator()

    print("=" * 65)
    print("TEST 1: Monte Carlo Simulation on Recommended Optimal Price")
    print("=" * 65)
    # SKU setup: Base $100, Recommended $115, Cost $50, Demand 25, Inventory 50
    sim_res = simulator.simulate_price(
        recommended_price=115.0,
        base_price=100.0,
        base_demand=25.0,
        base_cost=50.0,
        own_elasticity=-1.5,
        cross_elasticity=0.4,
        inventory_level=50.0,
        params=UncertaintyParameters(
            cost_mean_shock=0.05,        # 5% cost surge
            competitor_mean_shock=-0.10, # 10% competitor price drop
            demand_shock_mean=0.0,
            n_simulations=5000,
            random_seed=42,
        ),
    )

    summary = sim_res.summary_table()
    print(summary.to_string(index=False))

    # Assertions on confidence intervals and downside risk
    assert sim_res.profit_ci_95[0] < sim_res.expected_profit < sim_res.profit_ci_95[1]
    assert sim_res.var_99 <= sim_res.var_95 <= sim_res.expected_profit
    assert sim_res.cvar_95 <= sim_res.var_95, "CVaR (Expected Shortfall) must be <= VaR"
    assert 0.0 <= sim_res.prob_negative_profit <= 1.0
    print("\nDownside risk and confidence interval consistency verified!")

    print("\n" + "=" * 65)
    print("TEST 2: Comparative Analysis (Baseline vs Recommended Price)")
    print("=" * 65)
    comparison_df = simulator.compare_prices(
        base_price=100.0,
        recommended_price=115.0,
        base_demand=25.0,
        base_cost=50.0,
        own_elasticity=-1.2,
        cross_elasticity=0.3,
        inventory_level=50.0,
    )
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 65)
    print("TEST 3: Stress Testing Across 5 Scenarios")
    print("=" * 65)
    stress_df = simulator.run_stress_scenarios(
        recommended_price=115.0,
        base_price=100.0,
        base_demand=25.0,
        base_cost=50.0,
        own_elasticity=-1.4,
        cross_elasticity=0.35,
        inventory_level=40.0,
    )
    print(stress_df.to_string(index=False))

    print("\n>>> ALL MONTE CARLO SIMULATION CHECKS PASSED! <<<")


if __name__ == "__main__":
    run_simulator_tests()
