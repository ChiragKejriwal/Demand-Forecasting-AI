"""Dynamic price optimization module using SciPy and PuLP.

Formulates and solves the mathematical program:
    Maximize Expected Profit:
        Profit(P) = (P - unit_cost) * min(Predicted_Demand(P), inventory_level)

Subject to business constraints:
    1. Bound Constraint: (1 - max_price_change) * base_price <= P <= (1 + max_price_change) * base_price
    2. Margin Constraint: P >= unit_cost + min_margin
    3. Inventory Constraint: Predicted_Demand(P) <= inventory_level (or capped at available stock)
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default benchmark elasticity matching the empirical valid Electronics category
ASSUMED_ELASTICITY_FOR_DEMO: float = -1.18


@dataclass
class PriceOptimizationResult:
    """Container for price optimization outputs and constraint status."""

    product_id: Optional[str]
    category: Optional[str]
    base_price: float
    unit_cost: float
    recommended_price: float
    expected_demand: float
    expected_revenue: float
    expected_profit: float
    profit_margin_pct: float
    price_change_pct: float
    inventory_level: float
    status: str
    active_constraints: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DynamicPricingOptimizer:
    """Mathematical optimization engine for price recommendation under demand uncertainty."""

    def __init__(
        self,
        default_max_price_change_pct: float = 0.20,
        min_margin_dollars: float = 0.50,
        min_margin_pct: float = 0.05,
    ) -> None:
        """Initialize dynamic pricing optimizer.

        Parameters
        ----------
        default_max_price_change_pct : float, optional
            Maximum allowed price deviation from base price (e.g. 0.20 = +/-20%), by default 0.20.
        min_margin_dollars : float, optional
            Minimum required absolute margin above unit cost, by default $0.50.
        min_margin_pct : float, optional
            Minimum required percentage margin above unit cost, by default 5%.
        """
        self.max_price_change_pct = default_max_price_change_pct
        self.min_margin_dollars = min_margin_dollars
        self.min_margin_pct = min_margin_pct

    def _demand_function(
        self,
        price: float,
        base_price: float,
        base_demand: float,
        elasticity: float = ASSUMED_ELASTICITY_FOR_DEMO,
        demand_fn: Optional[Callable[[float], float]] = None,
    ) -> float:
        """Compute expected demand at candidate price using constant elasticity or custom function."""
        if demand_fn is not None:
            return float(max(0.0, demand_fn(price)))

        # Constant Elasticity of Demand (PED): Q(P) = Q_base * (P / P_base)^beta
        if base_price <= 0 or price <= 0:
            return 0.0

        # Enforce elasticity is negative or bounded
        ratio = max(price / base_price, 1e-4)
        pred_demand = base_demand * (ratio ** elasticity)
        return float(max(0.0, pred_demand))

    def optimize_price(
        self,
        base_price: float,
        unit_cost: float,
        base_demand: float,
        elasticity: float = ASSUMED_ELASTICITY_FOR_DEMO,
        inventory_level: Optional[float] = None,
        product_id: Optional[str] = None,
        category: Optional[str] = None,
        max_price_change_pct: Optional[float] = None,
        demand_fn: Optional[Callable[[float], float]] = None,
    ) -> PriceOptimizationResult:
        """Find the unconstrained or constrained continuous profit-maximizing price.

        Parameters
        ----------
        base_price : float
            Current standard catalog price.
        unit_cost : float
            Unit production or acquisition cost.
        base_demand : float
            Baseline expected demand at base_price.
        elasticity : float, optional
            Price elasticity of demand (typically negative, e.g. -1.5), by default -1.5.
        inventory_level : Optional[float], optional
            Available stock level. If provided, demand is capped at this stock level.
        product_id : Optional[str], optional
            Identifier for the product.
        category : Optional[str], optional
            Category name for the product.
        max_price_change_pct : Optional[float], optional
            Override for maximum allowed deviation (+/- %) from base_price.
        demand_fn : Optional[Callable[[float], float]], optional
            Custom callable that maps price -> predicted demand.

        Returns
        -------
        PriceOptimizationResult
            Optimal pricing prescription with expected demand and profit metrics.
        """
        change_pct = max_price_change_pct or self.max_price_change_pct

        # -------------------------------------------------------------
        # 1. Compute Constraint Bounds
        # -------------------------------------------------------------
        # Constraint 1: Price bound (+/- max_price_change_pct of base_price)
        p_lower_bound = base_price * (1.0 - change_pct)
        p_upper_bound = base_price * (1.0 + change_pct)

        # Constraint 2: Margin constraint (P >= unit_cost + min_margin)
        margin_floor = max(
            unit_cost + self.min_margin_dollars,
            unit_cost * (1.0 + self.min_margin_pct),
        )
        effective_min_price = max(p_lower_bound, margin_floor)

        active_constraints = []
        if effective_min_price > p_upper_bound:
            # If cost exceeds upper bound, set price to margin floor
            logger.warning(
                "Product %s: Margin floor ($%.2f) exceeds upper bound ($%.2f). Enforcing margin floor.",
                product_id or "Unknown",
                margin_floor,
                p_upper_bound,
            )
            p_upper_bound = effective_min_price
            active_constraints.append("margin_strictly_dominates")

        # -------------------------------------------------------------
        # 2. Objective Function: Maximize Expected Profit
        # -------------------------------------------------------------
        def negative_profit(p: float) -> float:
            demand = self._demand_function(p, base_price, base_demand, elasticity, demand_fn)

            # Constraint 3: Inventory constraint (Demand capped at stock level)
            if inventory_level is not None and inventory_level > 0:
                demand_sold = min(demand, inventory_level)
            else:
                demand_sold = demand

            unit_margin = p - unit_cost
            profit = unit_margin * demand_sold
            return -profit

        # -------------------------------------------------------------
        # 3. Solve via SciPy Bounded Scalar Minimization
        # -------------------------------------------------------------
        if effective_min_price >= p_upper_bound:
            optimal_price = effective_min_price
            status = "BOUND_FORCED"
        else:
            res = minimize_scalar(
                negative_profit,
                bounds=(effective_min_price, p_upper_bound),
                method="bounded",
                options={"xatol": 1e-3, "maxiter": 200},
            )
            optimal_price = float(res.x)
            status = "OPTIMAL" if res.success else "LOCAL_OPT"

        # Check which boundary constraints were reached
        if abs(optimal_price - effective_min_price) < 0.02:
            active_constraints.append("lower_bound_hit" if effective_min_price == p_lower_bound else "margin_floor_hit")
        if abs(optimal_price - p_upper_bound) < 0.02:
            active_constraints.append("upper_bound_hit")

        # Compute optimal metrics
        pred_demand = self._demand_function(optimal_price, base_price, base_demand, elasticity, demand_fn)
        if inventory_level is not None and inventory_level > 0:
            sold_demand = min(pred_demand, inventory_level)
            if pred_demand > inventory_level:
                active_constraints.append("inventory_stockout_capped")
        else:
            sold_demand = pred_demand

        expected_profit = (optimal_price - unit_cost) * sold_demand
        expected_revenue = optimal_price * sold_demand
        profit_margin_pct = ((optimal_price - unit_cost) / optimal_price) * 100.0 if optimal_price > 0 else 0.0
        price_change = ((optimal_price - base_price) / base_price) * 100.0

        return PriceOptimizationResult(
            product_id=product_id,
            category=category,
            base_price=round(base_price, 2),
            unit_cost=round(unit_cost, 2),
            recommended_price=round(optimal_price, 2),
            expected_demand=round(sold_demand, 2),
            expected_revenue=round(expected_revenue, 2),
            expected_profit=round(expected_profit, 2),
            profit_margin_pct=round(profit_margin_pct, 2),
            price_change_pct=round(price_change, 2),
            inventory_level=round(inventory_level, 2) if inventory_level is not None else 0.0,
            status=status,
            active_constraints=active_constraints,
        )

    def optimize_discrete_pulp(
        self,
        base_price: float,
        unit_cost: float,
        base_demand: float,
        elasticity: float = ASSUMED_ELASTICITY_FOR_DEMO,
        inventory_level: Optional[float] = None,
        candidate_price_pcts: Optional[List[float]] = None,
        product_id: Optional[str] = None,
    ) -> PriceOptimizationResult:
        """Solve discrete pricing selection problem via Mixed-Integer Programming (PuLP).

        Parameters
        ----------
        base_price : float
            Current catalog price.
        unit_cost : float
            Unit cost.
        base_demand : float
            Base expected demand.
        elasticity : float, optional
            Price elasticity of demand, by default -1.5.
        inventory_level : Optional[float], optional
            Stock constraint.
        candidate_price_pcts : Optional[List[float]], optional
            List of candidate price percentage shifts (e.g. [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]).
        product_id : Optional[str], optional
            Product ID.

        Returns
        -------
        PriceOptimizationResult
            Optimization result selecting the optimal discrete price point.
        """
        if not PULP_AVAILABLE:
            logger.warning("PuLP solver not available. Delegating to continuous SciPy optimizer.")
            return self.optimize_price(
                base_price=base_price,
                unit_cost=unit_cost,
                base_demand=base_demand,
                elasticity=elasticity,
                inventory_level=inventory_level,
                product_id=product_id,
            )

        if candidate_price_pcts is None:
            candidate_price_pcts = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]

        # Generate candidate prices and expected profits
        candidates = []
        for pct in candidate_price_pcts:
            p = round(base_price * (1.0 + pct), 2)
            # Constraint 2: Margin constraint
            if p <= unit_cost:
                continue

            demand = self._demand_function(p, base_price, base_demand, elasticity)
            sold = min(demand, inventory_level) if inventory_level is not None else demand
            profit = (p - unit_cost) * sold
            candidates.append((p, sold, profit))

        if not candidates:
            # Fallback if all candidates violate unit_cost
            p = max(base_price, unit_cost + 1.0)
            return self.optimize_price(base_price, unit_cost, base_demand, elasticity, inventory_level, product_id)

        prob = pulp.LpProblem(f"Discrete_Pricing_{product_id or 'item'}", pulp.LpMaximize)

        # Binary decision variables x_i = 1 if candidate i is selected
        x_vars = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(len(candidates))]

        # Objective: Maximize profit
        prob += pulp.lpSum([candidates[i][2] * x_vars[i] for i in range(len(candidates))])

        # Constraint: Exactly one price point must be chosen
        prob += pulp.lpSum(x_vars) == 1

        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        # Extract chosen candidate
        chosen_idx = 0
        for i, var in enumerate(x_vars):
            if pulp.value(var) is not None and pulp.value(var) > 0.5:
                chosen_idx = i
                break

        opt_price, opt_demand, opt_profit = candidates[chosen_idx]
        revenue = opt_price * opt_demand
        margin_pct = ((opt_price - unit_cost) / opt_price) * 100.0
        price_change = ((opt_price - base_price) / base_price) * 100.0

        return PriceOptimizationResult(
            product_id=product_id,
            category=None,
            base_price=round(base_price, 2),
            unit_cost=round(unit_cost, 2),
            recommended_price=round(opt_price, 2),
            expected_demand=round(opt_demand, 2),
            expected_revenue=round(revenue, 2),
            expected_profit=round(opt_profit, 2),
            profit_margin_pct=round(margin_pct, 2),
            price_change_pct=round(price_change, 2),
            inventory_level=round(inventory_level, 2) if inventory_level is not None else 0.0,
            status="PULP_OPTIMAL" if pulp.LpStatus[prob.status] == "Optimal" else "PULP_SUBOPTIMAL",
            active_constraints=["discrete_grid_selection"],
        )

    def optimize_catalog(
        self,
        df: pd.DataFrame,
        category_elasticity_map: Optional[Dict[str, float]] = None,
        default_elasticity: float = ASSUMED_ELASTICITY_FOR_DEMO,
    ) -> pd.DataFrame:
        """Run batch optimization across an entire catalog / DataFrame of products.

        Parameters
        ----------
        df : pd.DataFrame
            Catalog DataFrame containing base_price, unit_cost, inventory_level,
            and baseline demand or units_sold.
        category_elasticity_map : Optional[Dict[str, float]], optional
            Mapping of category -> own_price_elasticity, by default None.
        default_elasticity : float, optional
            Fallback elasticity when not found in mapping, by default -1.2.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with pricing recommendations, profit delta, and revenue delta.
        """
        logger.info("Optimizing prices across catalog of %d rows...", len(df))
        records = []

        demand_col = "units_sold" if "units_sold" in df.columns else "demand_index"
        cost_col = "unit_cost" if "unit_cost" in df.columns else "base_price"

        for _, row in df.iterrows():
            cat = str(row.get("category", "General"))
            p_id = str(row.get("product_id", "Unknown"))
            base_p = float(row.get("current_price", row.get("base_price", 50.0)))
            unit_c = float(row.get(cost_col, base_p * 0.6))
            base_q = float(row.get(demand_col, 10.0))
            inv = float(row.get("inventory_level", 999.0))

            # Retrieve category elasticity if available
            elasticity = default_elasticity
            if category_elasticity_map and cat in category_elasticity_map:
                elasticity = category_elasticity_map[cat]

            res = self.optimize_price(
                base_price=base_p,
                unit_cost=unit_c,
                base_demand=base_q,
                elasticity=elasticity,
                inventory_level=inv,
                product_id=p_id,
                category=cat,
            )

            # Baseline metrics for comparison
            base_sold = min(base_q, inv)
            baseline_profit = (base_p - unit_c) * base_sold
            baseline_revenue = base_p * base_sold
            profit_lift = res.expected_profit - baseline_profit
            profit_lift_pct = (profit_lift / (baseline_profit + 1e-6)) * 100.0

            rec_dict = res.to_dict()
            rec_dict["baseline_profit"] = round(baseline_profit, 2)
            rec_dict["baseline_revenue"] = round(baseline_revenue, 2)
            rec_dict["profit_lift_dollars"] = round(profit_lift, 2)
            rec_dict["profit_lift_pct"] = round(profit_lift_pct, 2)
            records.append(rec_dict)

        results_df = pd.DataFrame(records)
        total_lift = results_df["profit_lift_dollars"].sum()
        logger.info(
            "Catalog Optimization Complete. Projected Catalog Profit Lift: +$%.2f", total_lift
        )
        return results_df
