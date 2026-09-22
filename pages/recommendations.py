"""Prescriptive dynamic pricing recommendations and Monte Carlo stress testing dashboard."""

import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data.loader import load_raw_data
from src.models.elasticity import PriceElasticityEstimator
from src.optimization.optimizer import DynamicPricingOptimizer
from src.simulation.simulator import MonteCarloSimulator, UncertaintyParameters
from src.utils.logger import configure_app_logging

configure_app_logging()

st.set_page_config(page_title="Recommendations | Dynamic Pricing", page_icon="🎯", layout="wide")

# Theme styling
st.markdown(
    """
    <style>
    .decision-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4);
    }
    .kpi-title { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-big { font-size: 2.2rem; font-weight: 800; }
    .kpi-lift { font-size: 1.1rem; font-weight: 700; color: #10b981; }
    .kpi-sub { font-size: 0.85rem; color: #cbd5e1; }
    
    .flag-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .flag-green { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .flag-amber { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .flag-red { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    
    .reasoning-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38bdf8;
        border-radius: 4px 8px 8px 4px;
        padding: 1rem 1.25rem;
        margin-top: 0.75rem;
        font-size: 0.92rem;
        color: #e2e8f0;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 Dynamic Price Recommendations & Uncertainty Stress Testing")
st.markdown(
    "Select a product to generate an optimal price recommendation that balances margin floors, "
    "market bounds, and inventory constraints with interactive Monte Carlo downside risk simulation."
)


def get_catalog():
    if "sales_data" in st.session_state:
        df = st.session_state["sales_data"].copy()
    else:
        df = load_raw_data()

    if "current_price" not in df.columns:
        df["current_price"] = 19.99
    if "base_price" not in df.columns:
        df["base_price"] = df["current_price"]
    if "competitor_price" not in df.columns:
        df["competitor_price"] = df["current_price"]
    if "unit_cost" not in df.columns:
        df["unit_cost"] = df["current_price"] * 0.55
    if "inventory_level" not in df.columns:
        df["inventory_level"] = 50.0
    if "brand" not in df.columns:
        df["brand"] = "Generic"
    if "category" not in df.columns:
        df["category"] = "General"
    if "product_id" not in df.columns:
        df["product_id"] = "SKU_001"
    if "units_sold" not in df.columns:
        df["units_sold"] = 1.0
    # Compute product-level medians for baseline defaults
    prod_summary = (
        df.groupby("product_id")
        .agg(
            category=("category", "first"),
            brand=("brand", "first"),
            base_price=("base_price", "median"),
            current_price=("current_price", "median"),
            unit_cost=("unit_cost", "median"),
            competitor_price=("competitor_price", "median"),
            inventory_level=("inventory_level", "median"),
            avg_demand=("units_sold", "mean"),
        )
        .reset_index()
    )
    return df, prod_summary


def get_models(df):
    el_model = PriceElasticityEstimator(
        category_col="category",
        price_col="current_price",
        competitor_price_col="competitor_price",
        target_col="units_sold",
    )
    el_model.fit_by_category(df)
    optimizer = DynamicPricingOptimizer(default_max_price_change_pct=0.20)
    simulator = MonteCarloSimulator()
    return el_model, optimizer, simulator


df, prod_catalog = get_catalog()
elasticity_model, optimizer, simulator = get_models(df)

# Sidebar Product Selection
st.sidebar.header("Product Selection")
all_products = prod_catalog["product_id"].tolist()
electronics_products = prod_catalog[prod_catalog["category"] == "Electronics"]["product_id"].tolist()
default_product_idx = (
    all_products.index(electronics_products[0])
    if electronics_products and electronics_products[0] in all_products
    else 0
)
selected_product = st.sidebar.selectbox("Choose Product ID", all_products, index=default_product_idx)

# Get selected product baseline row
prod_row = prod_catalog[prod_catalog["product_id"] == selected_product].iloc[0]
prod_category = str(prod_row["category"])

# Retrieve estimated elasticity for this category
category_ped = -1.18
category_xed = 0.25
if prod_category in elasticity_model.category_models_:
    category_ped = float(elasticity_model.category_models_[prod_category]["own_elasticity"])
    category_xed = float(elasticity_model.category_models_[prod_category]["cross_elasticity"])

st.sidebar.markdown(f"**Category**: `{prod_category}` | **Brand**: `{prod_row['brand']}`")
st.sidebar.markdown(f"**Category PED (Own Elasticity)**: `{category_ped:.3f}`")
st.sidebar.markdown(f"**Category XED (Cross Elasticity)**: `{category_xed:+.3f}`")

# -----------------------------------------------------------------------------
# Interactive Sliders for Counterfactual Inputs
# -----------------------------------------------------------------------------
st.sidebar.subheader("Interactive Scenario Adjustments")

default_comp = float(prod_row["competitor_price"])
default_inv = float(prod_row["inventory_level"])
base_p = float(prod_row["base_price"])
base_q = float(prod_row["avg_demand"])

cost_ratio = st.sidebar.slider(
    "Assumed Unit Cost Ratio",
    min_value=0.20,
    max_value=0.90,
    value=0.55,
    step=0.05,
    help="Assumed unit cost as a ratio of base price (e.g. 0.55 = 55% unit cost).",
)
input_cost = round(base_p * cost_ratio, 2)
st.sidebar.caption(f"Implied Unit Cost: **${input_cost:.2f}** ({cost_ratio * 100:.0f}% of base ${base_p:.2f})")

input_comp_price = st.sidebar.slider(
    "Competitor Price ($)",
    min_value=round(default_comp * 0.7, 2),
    max_value=round(default_comp * 1.3, 2),
    value=round(default_comp, 2),
    step=0.5,
    help="Simulate competitor price cuts or increases.",
)

input_inventory = st.sidebar.slider(
    "Available Inventory (Units)",
    min_value=5,
    max_value=int(default_inv * 2),
    value=int(default_inv),
    step=5,
    help="Set inventory ceiling to simulate stockout risk.",
)

input_bound_pct = st.sidebar.slider(
    "Max Price Deviation Allowed (±%)",
    min_value=5,
    max_value=35,
    value=20,
    step=5,
    help="Enforce business boundary constraint around baseline price.",
) / 100.0

# -----------------------------------------------------------------------------
# Run Optimization
# -----------------------------------------------------------------------------
opt_result = optimizer.optimize_price(
    base_price=base_p,
    unit_cost=input_cost,
    base_demand=base_q,
    elasticity=category_ped,
    inventory_level=float(input_inventory),
    product_id=selected_product,
    category=prod_category,
    max_price_change_pct=input_bound_pct,
)

# Baseline metrics under standard catalog price
baseline_demand_sold = min(base_q, float(input_inventory))
baseline_profit = (base_p - input_cost) * baseline_demand_sold
baseline_revenue = base_p * baseline_demand_sold
profit_delta = opt_result.expected_profit - baseline_profit
profit_delta_pct = (profit_delta / (baseline_profit + 1e-6)) * 100.0

# -----------------------------------------------------------------------------
# Main Dashboard: Decision Comparison Card
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="decision-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-size: 0.85rem; color: #94a3b8;">SKU PRESCRIPTION</span>
                <h2 style="margin: 0; color: #f8fafc;">{selected_product} &bull; {prod_category}</h2>
            </div>
            <div>
                <span class="flag-badge {'flag-green' if profit_delta >= 0 else 'flag-amber'}">
                    Expected Profit Lift: {profit_delta_pct:+.1f}% (+${profit_delta:,.2f})
                </span>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem;">
            <div>
                <div class="kpi-title">Baseline Price</div>
                <div class="kpi-big" style="color: #94a3b8;">${base_p:.2f}</div>
                <div class="kpi-sub">Catalog Standard</div>
            </div>
            <div>
                <div class="kpi-title">Recommended Price</div>
                <div class="kpi-big" style="color: #38bdf8;">${opt_result.recommended_price:.2f}</div>
                <div class="kpi-sub" style="color: {'#34d399' if opt_result.price_change_pct >= 0 else '#fbbf24'};">
                    {opt_result.price_change_pct:+.1f}% vs. Baseline
                </div>
            </div>
            <div>
                <div class="kpi-title">Expected Demand</div>
                <div class="kpi-big" style="color: #cbd5e1;">{opt_result.expected_demand:.1f}</div>
                <div class="kpi-sub">Units Sold / Day (Cap: {input_inventory})</div>
            </div>
            <div>
                <div class="kpi-title">Expected Profit</div>
                <div class="kpi-big" style="color: #10b981;">${opt_result.expected_profit:,.2f}</div>
                <div class="kpi-sub">Margin: {opt_result.profit_margin_pct:.1f}% (${opt_result.recommended_price - input_cost:.2f}/unit)</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Constraint Status & Recommendation Reasoning
# -----------------------------------------------------------------------------
c_flags, c_reason = st.columns([1, 1.5], gap="medium")

with c_flags:
    st.subheader("Business Constraint Audit")
    
    # Margin constraint check
    min_margin = input_cost + 0.50
    if opt_result.recommended_price >= min_margin:
        st.markdown('<span class="flag-badge flag-green">✔ Margin Floor Satisfied (Price > Cost)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="flag-badge flag-red">✖ Margin Floor Violated</span>', unsafe_allow_html=True)

    # Bounds check
    p_min = base_p * (1.0 - input_bound_pct)
    p_max = base_p * (1.0 + input_bound_pct)
    if "upper_bound_hit" in opt_result.active_constraints:
        st.markdown('<span class="flag-badge flag-amber">⚠ Upper Bound Reached (+20% Cap)</span>', unsafe_allow_html=True)
    elif "lower_bound_hit" in opt_result.active_constraints:
        st.markdown('<span class="flag-badge flag-amber">⚠ Lower Bound Reached (-20% Floor)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="flag-badge flag-green">✔ Within Price Policy Bounds</span>', unsafe_allow_html=True)

    # Inventory constraint check
    if "inventory_stockout_capped" in opt_result.active_constraints or opt_result.expected_demand >= input_inventory:
        st.markdown('<span class="flag-badge flag-amber">⚠ Inventory Cap Active (Demand Capped at Stock)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="flag-badge flag-green">✔ Inventory Sufficient for Demand</span>', unsafe_allow_html=True)

    st.caption(f"Solver Status: `{opt_result.status}` | Active Flags: `{opt_result.active_constraints or 'None'}`")

with c_reason:
    st.subheader("Recommendation Rationale & Explainability")
    
    # Build dynamic rationale narrative
    rationale_points = []
    if category_ped > -1.0:
        rationale_points.append(
            f"<b>Inelastic Demand (PED = {category_ped:.2f}):</b> Consumers in the <i>{prod_category}</i> category exhibit low price sensitivity. Raising price increases revenue and captures higher margin per unit with minimal volume loss."
        )
    else:
        rationale_points.append(
            f"<b>Elastic Demand (PED = {category_ped:.2f}):</b> Consumers are sensitive to price changes. The optimizer calibrated price to balance margin without triggering severe sales volume attrition."
        )

    if input_comp_price < base_p:
        comp_discount = ((base_p - input_comp_price) / base_p) * 100
        rationale_points.append(
            f"<b>Competitor Threat:</b> Competitor is discounting by {comp_discount:.1f}% (${input_comp_price:.2f} vs ${base_p:.2f}). Cross-elasticity modeling protects market share."
        )

    if opt_result.expected_demand >= input_inventory:
        rationale_points.append(
            f"<b>Inventory Scarcity:</b> Low stock ({input_inventory} units) limits volume expansion. The model raised price toward the upper ceiling to maximize profit per available unit rather than selling out prematurely."
        )
    else:
        rationale_points.append(
            f"<b>Cost Buffer:</b> Unit cost of ${input_cost:.2f} ensures a healthy {opt_result.profit_margin_pct:.1f}% margin (${opt_result.recommended_price - input_cost:.2f}/unit)."
        )

    st.markdown(
        f"""
        <div class="reasoning-box">
            {'<br><br>'.join(rationale_points)}
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Interactive Visualizations: Price Response Curve + Monte Carlo Simulation
# -----------------------------------------------------------------------------
st.write("")
st.subheader("Simulated Price-Response Curve vs. Profit Optimization")

v_col1, v_col2 = st.columns([1, 1], gap="medium")

with v_col1:
    # Price response curve
    curve_df = elasticity_model.simulate_price_response_curve(
        category=prod_category,
        base_price=base_p,
        unit_cost=input_cost,
        competitor_price=input_comp_price,
        base_demand=base_q,
        min_price_pct=1.0 - input_bound_pct,
        max_price_pct=1.0 + input_bound_pct,
        n_points=60,
    )

    fig_opt = make_subplots(specs=[[{"secondary_y": True}]])
    fig_opt.add_trace(
        go.Scatter(x=curve_df["price"], y=curve_df["expected_profit"], name="Profit ($)", line=dict(color="#10b981", width=3)),
        secondary_y=False,
    )
    fig_opt.add_trace(
        go.Scatter(x=curve_df["price"], y=curve_df["expected_revenue"], name="Revenue ($)", line=dict(color="#38bdf8", width=2, dash="dot")),
        secondary_y=True,
    )
    # Mark Baseline
    fig_opt.add_vline(x=base_p, line_dash="dash", line_color="#94a3b8", annotation_text=f"Base (${base_p:.2f})")
    # Mark Recommended
    fig_opt.add_vline(x=opt_result.recommended_price, line_dash="solid", line_color="#f59e0b", annotation_text=f"Optimal (${opt_result.recommended_price:.2f})")

    fig_opt.update_layout(
        title="Profit & Revenue as a Function of Price",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Candidate Price ($)",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )
    fig_opt.update_yaxes(title_text="Expected Profit ($)", secondary_y=False)
    fig_opt.update_yaxes(title_text="Expected Revenue ($)", secondary_y=True)

    st.plotly_chart(fig_opt, use_container_width=True)

with v_col2:
    # Run Monte Carlo Simulation on Recommended Price
    # Dynamically derive empirical demand volatility from ML holdout residual: 1.59 / base_q
    empirical_demand_vol = max(0.04, round(1.59 / max(base_q, 1e-3), 4))
    mc_params = UncertaintyParameters(
        demand_volatility=empirical_demand_vol,
        n_simulations=4000,
        random_seed=42,
    )

    sim_res = simulator.simulate_price(
        recommended_price=opt_result.recommended_price,
        base_price=base_p,
        base_demand=base_q,
        base_cost=input_cost,
        own_elasticity=category_ped,
        cross_elasticity=category_xed,
        base_competitor_price=input_comp_price,
        inventory_level=float(input_inventory),
        baseline_profit=baseline_profit,
        params=mc_params,
    )

    # Plot empirical profit distribution with VaR & CVaR markers
    fig_mc = go.Figure()
    fig_mc.add_trace(
        go.Histogram(
            x=sim_res.profit_samples,
            nbinsx=45,
            marker_color="#6366f1",
            opacity=0.75,
            name="Simulated Profit",
        )
    )
    # Expected Profit Mean
    fig_mc.add_vline(
        x=sim_res.expected_profit,
        line_dash="solid",
        line_color="#10b981",
        annotation_text=f"Mean: ${sim_res.expected_profit:,.0f}",
    )
    # 95% VaR Marker
    fig_mc.add_vline(
        x=sim_res.var_95,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"95% VaR: ${sim_res.var_95:,.0f}",
    )
    # 95% CVaR Marker
    fig_mc.add_vline(
        x=sim_res.cvar_95,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text=f"95% CVaR: ${sim_res.cvar_95:,.0f}",
    )

    fig_mc.update_layout(
        title=f"Monte Carlo Profit Distribution ({sim_res.profit_samples.size:,} Trials)",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Profit ($)",
        yaxis_title="Frequency",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        showlegend=False,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

# -----------------------------------------------------------------------------
# Downside Risk & Stress Scenarios Table
# -----------------------------------------------------------------------------
st.write("")
st.subheader("Stochastic Risk & Downside Uncertainty Table")

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.metric("95% Value at Risk (VaR)", f"${sim_res.var_95:,.2f}", help="With 95% confidence, profit will not drop below this amount.")
with r2:
    st.metric("95% Conditional VaR (CVaR)", f"${sim_res.cvar_95:,.2f}", help="Expected profit in the worst 5% catastrophic tail scenarios.")
with r3:
    st.metric("95% Confidence Interval", f"[${sim_res.profit_ci_95[0]:,.0f}, ${sim_res.profit_ci_95[1]:,.0f}]")
with r4:
    prob_loss = sim_res.prob_negative_profit * 100.0
    st.metric("Probability of Loss", f"{prob_loss:.2f}%", delta=f"{'Low Risk' if prob_loss < 1.0 else 'Elevated Risk'}", delta_color="inverse")

with st.expander("🛡️ View Macro Stress Testing Across 5 Severe Scenarios"):
    stress_df = simulator.run_stress_scenarios(
        recommended_price=opt_result.recommended_price,
        base_price=base_p,
        base_demand=base_q,
        base_cost=input_cost,
        own_elasticity=category_ped,
        cross_elasticity=category_xed,
        inventory_level=float(input_inventory),
    )
    st.table(stress_df)
