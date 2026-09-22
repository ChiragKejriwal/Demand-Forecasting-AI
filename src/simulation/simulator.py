"""Monte Carlo simulation module for pricing decisions under demand uncertainty.

Injects stochastic variance into:
1. Competitor price actions (e.g., price cuts or aggressive discounting)
2. Cost fluctuations (e.g., supply chain shocks or wholesale inflation)
3. Latent demand shocks (market volatility and unobserved shocks)

Quantifies downside risk via:
- Value at Risk (VaR) at 95% and 99% confidence
- Conditional Value at Risk (CVaR / Expected Shortfall)
- Probability of Loss (Profit < 0)
- Probability of Stockout (Demand > Inventory)
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyParameters:
    """Configurable distribution parameters for stochastic Monte Carlo shocks."""

    # Cost shock: C_sim = C_base * (1 + N(mean, vol))
    cost_mean_shock: float = 0.0
    cost_volatility: float = 0.05

    # Competitor price shock: P_comp_sim = P_comp_base * (1 + N(mean, vol))
    competitor_mean_shock: float = 0.0
    competitor_volatility: float = 0.08

    # Multiplicative demand shock: Q_sim = Q_pred * exp(N(mean, vol))
    # Dynamically parameterized via ML holdout error (holdout RMSE / mean demand)
    demand_shock_mean: float = 0.0
    demand_volatility: Optional[float] = None

    # Simulation settings
    n_simulations: int = 5000
    random_seed: int = 42

    def get_demand_volatility(self, fallback: float = 0.15) -> float:
        """Return dynamic demand volatility, defaulting to fallback if not explicitly passed."""
        if self.demand_volatility is not None and self.demand_volatility > 0:
            return float(self.demand_volatility)
        return fallback


@dataclass
class SimulationResults:
    """Detailed summary and empirical distribution of Monte Carlo simulation runs."""

    price: float
    base_price: float
    unit_cost_base: float
    expected_profit: float
    median_profit: float
    std_profit: float
    profit_ci_90: Tuple[float, float]
    profit_ci_95: Tuple[float, float]
    profit_ci_99: Tuple[float, float]

    # Downside Risk Metrics
    var_95: float  # 5th percentile profit
    cvar_95: float  # Mean profit of the worst 5% outcomes
    var_99: float  # 1st percentile profit
    cvar_99: float  # Mean profit of the worst 1% outcomes
    prob_negative_profit: float
    prob_underperform_baseline: float
    prob_stockout: float

    # Demand & Revenue summary
    expected_demand: float
    demand_ci_95: Tuple[float, float]
    expected_revenue: float
    revenue_ci_95: Tuple[float, float]

    # Raw empirical samples for charting
    profit_samples: np.ndarray = field(repr=False)
    demand_samples: np.ndarray = field(repr=False)
    revenue_samples: np.ndarray = field(repr=False)
    cost_samples: np.ndarray = field(repr=False)

    def summary_table(self) -> pd.DataFrame:
        """Return formatted summary table of risk and return statistics."""
        data = {
            "Metric": [
                "Recommended Price",
                "Expected Profit (Mean)",
                "Median Profit",
                "Profit Std Dev",
                "95% Profit CI",
                "95% Value at Risk (VaR)",
                "95% Cond. VaR (CVaR / Expected Shortfall)",
                "99% Value at Risk (VaR)",
                "Probability of Loss (Profit < 0)",
                "Prob. Underperforming Baseline",
                "Probability of Stockout",
                "Expected Demand (Units)",
                "Expected Revenue",
            ],
            "Value": [
                f"${self.price:.2f}",
                f"${self.expected_profit:,.2f}",
                f"${self.median_profit:,.2f}",
                f"${self.std_profit:,.2f}",
                f"[${self.profit_ci_95[0]:,.2f}, ${self.profit_ci_95[1]:,.2f}]",
                f"${self.var_95:,.2f}",
                f"${self.cvar_95:,.2f}",
                f"${self.var_99:,.2f}",
                f"{self.prob_negative_profit * 100:.2f}%",
                f"{self.prob_underperform_baseline * 100:.2f}%",
                f"{self.prob_stockout * 100:.2f}%",
                f"{self.expected_demand:,.1f}",
                f"${self.expected_revenue:,.2f}",
            ],
        }
        return pd.DataFrame(data)


class MonteCarloSimulator:
    """Monte Carlo stress-testing simulator for pricing under demand, cost, and competitor uncertainty."""

    def __init__(self, default_params: Optional[UncertaintyParameters] = None) -> None:
        """Initialize simulator with default uncertainty parameters."""
        self.default_params = default_params or UncertaintyParameters()

    def simulate_price(
        self,
        recommended_price: float,
        base_price: float,
        base_demand: float,
        base_cost: float,
        own_elasticity: float = -1.18,
        cross_elasticity: float = 0.3,
        base_competitor_price: Optional[float] = None,
        inventory_level: Optional[float] = None,
        baseline_profit: Optional[float] = None,
        params: Optional[UncertaintyParameters] = None,
        demand_volatility: Optional[float] = None,
    ) -> SimulationResults:
        """Run Monte Carlo simulation for a given price point under stochastic shocks.

        Parameters
        ----------
        recommended_price : float
            Price to stress-test.
        base_price : float
            Reference baseline price.
        base_demand : float
            Baseline expected demand at base_price.
        base_cost : float
            Baseline unit production or acquisition cost.
        own_elasticity : float, optional
            Own-price elasticity of demand (PED, typically < 0), by default -1.18.
        cross_elasticity : float, optional
            Cross-price elasticity with competitor (XED, typically > 0 for substitutes), by default 0.3.
        base_competitor_price : Optional[float], optional
            Baseline competitor price, by default equal to base_price.
        inventory_level : Optional[float], optional
            Available stock constraint.
        baseline_profit : Optional[float], optional
            Profit under base price to calculate prob_underperform_baseline.
        params : Optional[UncertaintyParameters], optional
            Custom shock distribution parameters.
        demand_volatility : Optional[float], optional
            Dynamic demand volatility (e.g. holdout RMSE / mean demand). Overrides params if provided.

        Returns
        -------
        SimulationResults
            Comprehensive empirical distribution and downside risk metrics.
        """
        cfg = params or self.default_params
        if base_competitor_price is None:
            base_competitor_price = base_price

        if baseline_profit is None:
            base_sold = min(base_demand, inventory_level) if inventory_level is not None else base_demand
            baseline_profit = (base_price - base_cost) * base_sold

        rng = np.random.default_rng(seed=cfg.random_seed)
        n = cfg.n_simulations

        # -------------------------------------------------------------
        # 1. Generate Stochastic Shocks
        # -------------------------------------------------------------
        # Cost shock: C_sim = C_base * max(0.1, 1 + N(mean, vol))
        cost_shocks = rng.normal(loc=cfg.cost_mean_shock, scale=cfg.cost_volatility, size=n)
        sim_costs = np.maximum(base_cost * (1.0 + cost_shocks), base_cost * 0.1)

        # Competitor price shock: P_comp_sim = P_comp_base * max(0.1, 1 + N(mean, vol))
        comp_shocks = rng.normal(loc=cfg.competitor_mean_shock, scale=cfg.competitor_volatility, size=n)
        sim_comp_prices = np.maximum(base_competitor_price * (1.0 + comp_shocks), base_competitor_price * 0.2)

        # Demand shock: multiplicative log-normal shock exp(N(mean, vol))
        active_demand_vol = (
            demand_volatility
            if demand_volatility is not None and demand_volatility > 0
            else cfg.get_demand_volatility()
        )
        demand_shocks = rng.normal(loc=cfg.demand_shock_mean, scale=active_demand_vol, size=n)
        demand_multipliers = np.exp(demand_shocks)

        # -------------------------------------------------------------
        # 2. Compute Stochastic Demand via Econometric Response
        # -------------------------------------------------------------
        # Own-price ratio: (P / P_base)^own_elasticity
        own_price_ratio = max(recommended_price / base_price, 1e-4) ** own_elasticity

        # Cross-price ratio: (P_comp_sim / P_comp_base)^cross_elasticity
        cross_price_ratios = (sim_comp_prices / base_competitor_price) ** cross_elasticity

        # Realized demand
        sim_demands = base_demand * own_price_ratio * cross_price_ratios * demand_multipliers
        sim_demands = np.maximum(sim_demands, 0.0)

        # -------------------------------------------------------------
        # 3. Apply Inventory Constraint and Financial Calculation
        # -------------------------------------------------------------
        if inventory_level is not None and inventory_level > 0:
            sold_demands = np.minimum(sim_demands, inventory_level)
            stockout_flags = sim_demands > inventory_level
        else:
            sold_demands = sim_demands
            stockout_flags = np.zeros(n, dtype=bool)

        sim_revenues = recommended_price * sold_demands
        sim_profits = (recommended_price - sim_costs) * sold_demands

        # -------------------------------------------------------------
        # 4. Quantify Downside Risk & Confidence Intervals
        # -------------------------------------------------------------
        expected_profit = float(np.mean(sim_profits))
        median_profit = float(np.median(sim_profits))
        std_profit = float(np.std(sim_profits))

        # Confidence intervals
        ci_90 = (float(np.percentile(sim_profits, 5.0)), float(np.percentile(sim_profits, 95.0)))
        ci_95 = (float(np.percentile(sim_profits, 2.5)), float(np.percentile(sim_profits, 97.5)))
        ci_99 = (float(np.percentile(sim_profits, 0.5)), float(np.percentile(sim_profits, 99.5)))

        # Downside Risk: Value at Risk (VaR)
        # In risk management, 95% VaR is the 5th percentile profit outcome
        var_95 = float(np.percentile(sim_profits, 5.0))
        var_99 = float(np.percentile(sim_profits, 1.0))

        # Conditional VaR (Expected Shortfall): Mean of outcomes in the tail <= VaR
        tail_95 = sim_profits[sim_profits <= var_95]
        cvar_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95

        tail_99 = sim_profits[sim_profits <= var_99]
        cvar_99 = float(np.mean(tail_99)) if len(tail_99) > 0 else var_99

        # Probabilities
        prob_neg = float(np.mean(sim_profits < 0.0))
        prob_under_base = float(np.mean(sim_profits < baseline_profit))
        prob_stockout = float(np.mean(stockout_flags))

        # Demand & Revenue intervals
        expected_demand = float(np.mean(sold_demands))
        demand_ci_95 = (float(np.percentile(sold_demands, 2.5)), float(np.percentile(sold_demands, 97.5)))
        expected_revenue = float(np.mean(sim_revenues))
        revenue_ci_95 = (float(np.percentile(sim_revenues, 2.5)), float(np.percentile(sim_revenues, 97.5)))

        return SimulationResults(
            price=round(recommended_price, 2),
            base_price=round(base_price, 2),
            unit_cost_base=round(base_cost, 2),
            expected_profit=round(expected_profit, 2),
            median_profit=round(median_profit, 2),
            std_profit=round(std_profit, 2),
            profit_ci_90=(round(ci_90[0], 2), round(ci_90[1], 2)),
            profit_ci_95=(round(ci_95[0], 2), round(ci_95[1], 2)),
            profit_ci_99=(round(ci_99[0], 2), round(ci_99[1], 2)),
            var_95=round(var_95, 2),
            cvar_95=round(cvar_95, 2),
            var_99=round(var_99, 2),
            cvar_99=round(cvar_99, 2),
            prob_negative_profit=round(prob_neg, 4),
            prob_underperform_baseline=round(prob_under_base, 4),
            prob_stockout=round(prob_stockout, 4),
            expected_demand=round(expected_demand, 2),
            demand_ci_95=(round(demand_ci_95[0], 2), round(demand_ci_95[1], 2)),
            expected_revenue=round(expected_revenue, 2),
            revenue_ci_95=(round(revenue_ci_95[0], 2), round(revenue_ci_95[1], 2)),
            profit_samples=sim_profits,
            demand_samples=sold_demands,
            revenue_samples=sim_revenues,
            cost_samples=sim_costs,
        )

    def compare_prices(
        self,
        base_price: float,
        recommended_price: float,
        base_demand: float,
        base_cost: float,
        own_elasticity: float = -1.18,
        cross_elasticity: float = 0.3,
        inventory_level: Optional[float] = None,
        params: Optional[UncertaintyParameters] = None,
        demand_volatility: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run paired Monte Carlo simulations comparing the baseline price vs recommended price."""
        cfg = params or self.default_params

        sim_base = self.simulate_price(
            recommended_price=base_price,
            base_price=base_price,
            base_demand=base_demand,
            base_cost=base_cost,
            own_elasticity=own_elasticity,
            cross_elasticity=cross_elasticity,
            inventory_level=inventory_level,
            params=cfg,
            demand_volatility=demand_volatility,
        )

        sim_opt = self.simulate_price(
            recommended_price=recommended_price,
            base_price=base_price,
            base_demand=base_demand,
            base_cost=base_cost,
            own_elasticity=own_elasticity,
            cross_elasticity=cross_elasticity,
            inventory_level=inventory_level,
            baseline_profit=sim_base.expected_profit,
            params=cfg,
            demand_volatility=demand_volatility,
        )

        profit_diff = sim_opt.profit_samples - sim_base.profit_samples
        prob_opt_beats_base = float(np.mean(profit_diff > 0))

        comparison = pd.DataFrame(
            {
                "Strategy": ["Baseline Price", "Recommended Optimal Price", "Delta / Improvement"],
                "Price": [f"${base_price:.2f}", f"${recommended_price:.2f}", f"${recommended_price - base_price:+.2f}"],
                "Expected Profit": [
                    f"${sim_base.expected_profit:,.2f}",
                    f"${sim_opt.expected_profit:,.2f}",
                    f"${sim_opt.expected_profit - sim_base.expected_profit:+,.2f}",
                ],
                "Expected Demand": [
                    f"{sim_base.expected_demand:,.1f}",
                    f"{sim_opt.expected_demand:,.1f}",
                    f"{sim_opt.expected_demand - sim_base.expected_demand:+,.1f}",
                ],
                "95% Profit CI": [
                    f"[${sim_base.profit_ci_95[0]:,.2f}, ${sim_base.profit_ci_95[1]:,.2f}]",
                    f"[${sim_opt.profit_ci_95[0]:,.2f}, ${sim_opt.profit_ci_95[1]:,.2f}]",
                    "-",
                ],
                "95% VaR (Downside)": [
                    f"${sim_base.var_95:,.2f}",
                    f"${sim_opt.var_95:,.2f}",
                    f"${sim_opt.var_95 - sim_base.var_95:+,.2f}",
                ],
                "95% CVaR (Tail Risk)": [
                    f"${sim_base.cvar_95:,.2f}",
                    f"${sim_opt.cvar_95:,.2f}",
                    f"${sim_opt.cvar_95 - sim_base.cvar_95:+,.2f}",
                ],
                "Prob(Loss < 0)": [
                    f"{sim_base.prob_negative_profit * 100:.2f}%",
                    f"{sim_opt.prob_negative_profit * 100:.2f}%",
                    f"{(sim_opt.prob_negative_profit - sim_base.prob_negative_profit) * 100:+.2f}%",
                ],
                "Prob(Opt Beats Base)": ["-", f"{prob_opt_beats_base * 100:.1f}%", "-"],
            }
        )

        return comparison

    def run_stress_scenarios(
        self,
        recommended_price: float,
        base_price: float,
        base_demand: float,
        base_cost: float,
        own_elasticity: float = -1.18,
        cross_elasticity: float = 0.3,
        inventory_level: Optional[float] = None,
    ) -> pd.DataFrame:
        """Stress-test the recommended price across severe macroeconomic and competitor scenarios.

        Predefined Stress Scenarios:
        1. Base Volatility: Normal market variance.
        2. Competitor Price War: Competitor slashes price by 10% with high volatility.
        3. Cost Inflation Shock: Unit costs surge by +5% to +10%.
        4. Demand Slump / Recession: Demand experiences a -20% contraction.
        5. Severe Stagflation: High cost inflation (+10%), competitor price war (-10%), demand slump (-15%).
        """
        scenarios = {
            "1. Base Volatility": UncertaintyParameters(
                cost_mean_shock=0.0,
                cost_volatility=0.04,
                competitor_mean_shock=0.0,
                competitor_volatility=0.06,
                demand_shock_mean=0.0,
                demand_volatility=0.12,
            ),
            "2. Competitor Price War (-10%)": UncertaintyParameters(
                cost_mean_shock=0.0,
                cost_volatility=0.04,
                competitor_mean_shock=-0.10,  # 10% competitor price drop
                competitor_volatility=0.10,
                demand_shock_mean=0.0,
                demand_volatility=0.15,
            ),
            "3. Supply Cost Surge (+5%)": UncertaintyParameters(
                cost_mean_shock=0.05,  # 5% cost increase
                cost_volatility=0.06,
                competitor_mean_shock=0.0,
                competitor_volatility=0.06,
                demand_shock_mean=0.0,
                demand_volatility=0.12,
            ),
            "4. Demand Slump (-15%)": UncertaintyParameters(
                cost_mean_shock=0.0,
                cost_volatility=0.04,
                competitor_mean_shock=0.0,
                competitor_volatility=0.06,
                demand_shock_mean=-0.15,  # 15% negative demand shock
                demand_volatility=0.18,
            ),
            "5. Severe Stagflation": UncertaintyParameters(
                cost_mean_shock=0.10,  # 10% cost surge
                cost_volatility=0.08,
                competitor_mean_shock=-0.10,  # 10% competitor cut
                competitor_volatility=0.12,
                demand_shock_mean=-0.15,  # 15% demand slump
                demand_volatility=0.20,
            ),
        }

        results = []
        for name, p in scenarios.items():
            res = self.simulate_price(
                recommended_price=recommended_price,
                base_price=base_price,
                base_demand=base_demand,
                base_cost=base_cost,
                own_elasticity=own_elasticity,
                cross_elasticity=cross_elasticity,
                inventory_level=inventory_level,
                params=p,
            )
            results.append(
                {
                    "Scenario": name,
                    "Expected Profit": f"${res.expected_profit:,.2f}",
                    "Profit 95% CI": f"[${res.profit_ci_95[0]:,.2f}, ${res.profit_ci_95[1]:,.2f}]",
                    "95% VaR (Tail Risk)": f"${res.var_95:,.2f}",
                    "95% CVaR (Expected Shortfall)": f"${res.cvar_95:,.2f}",
                    "Expected Demand": f"{res.expected_demand:,.1f}",
                    "Prob(Loss < 0)": f"{res.prob_negative_profit * 100:.2f}%",
                    "Prob(Stockout)": f"{res.prob_stockout * 100:.2f}%",
                }
            )

        return pd.DataFrame(results)
