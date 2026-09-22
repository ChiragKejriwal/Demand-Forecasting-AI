"""Analysis page for historical demand trends, price elasticity curves, and competitive pricing."""

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
from src.evaluation.comparison import compare_all_models, compare_models
from src.models.elasticity import PriceElasticityEstimator
from src.models.model import DemandForecaster, ProphetForecaster, evaluate_forecast
from src.preprocessing.preprocessing import (
    preprocess_data,
    resample_time_series,
    temporal_train_test_split,
)
from src.utils.logger import configure_app_logging

configure_app_logging()

st.set_page_config(page_title="Analysis | Dynamic Pricing", page_icon="🔬", layout="wide")

# Theme styling
st.markdown(
    """
    <style>
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-value { font-size: 1.75rem; font-weight: 700; color: #38bdf8; }
    .metric-sub { font-size: 0.85rem; color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Demand, Elasticity & Competitive Pricing Analysis")
st.markdown(
    "Deep-dive into historical sales trends, econometric price response curves, and multi-model AI forecasting with uncertainty intervals."
)


def load_and_prep_data():
    if "sales_data" in st.session_state:
        df = st.session_state["sales_data"].copy()
    else:
        df = load_raw_data()

    if "current_price" not in df.columns:
        df["current_price"] = 19.99
    if "competitor_price" not in df.columns:
        df["competitor_price"] = df["current_price"]
    if "unit_cost" not in df.columns:
        df["unit_cost"] = df["current_price"] * 0.55

    df["price_diff"] = df["current_price"] - df["competitor_price"]
    df["price_ratio"] = df["current_price"] / (df["competitor_price"] + 1e-6)
    df["markup_pct"] = ((df["current_price"] - df["unit_cost"]) / df["current_price"]) * 100.0
    return df


def get_fitted_elasticity_model(df_clean):
    model = PriceElasticityEstimator(
        category_col="category",
        price_col="current_price",
        competitor_price_col="competitor_price",
        target_col="units_sold",
    )
    model.fit_by_category(df_clean)
    return model


@st.cache_data(show_spinner="Evaluating ML forecaster against naive baselines...")
def get_model_benchmark():
    raw_df = load_raw_data()
    sample_skus = raw_df["product_id"].unique()[:30]
    sample_df = raw_df[raw_df["product_id"].isin(sample_skus)].copy()
    proc_df = preprocess_data(sample_df)
    train_df, test_df = temporal_train_test_split(proc_df, test_days=14)
    forecaster = DemandForecaster(model_type="xgboost", n_estimators=45, max_depth=5)
    summary_df, rmse_improvement = compare_models(train_df, test_df, forecaster)
    return summary_df, rmse_improvement


@st.cache_data(show_spinner="Benchmarking complete multi-model leaderboard (Linear Regression, Random Forest, XGBoost, Prophet)...")
def get_all_models_leaderboard():
    raw_df = load_raw_data()
    sample_skus = raw_df["product_id"].unique()[:30]
    sample_df = raw_df[raw_df["product_id"].isin(sample_skus)].copy()
    proc_df = preprocess_data(sample_df)
    train_df, test_df = temporal_train_test_split(proc_df, test_days=14)
    leaderboard_df = compare_all_models(train_df, test_df, include_prophet=True)
    return leaderboard_df


df = load_and_prep_data()
elasticity_model = get_fitted_elasticity_model(df)
elasticity_summary = elasticity_model.elasticity_summary_

# Sidebar Filters
st.sidebar.header("Filter & Segmentation")
categories = ["All"] + sorted(df["category"].unique().tolist())
selected_category = st.sidebar.selectbox("Select Product Category", categories, index=0)

filtered_df = df if selected_category == "All" else df[df["category"] == selected_category]

# KPI Summary Row
k1, k2, k3, k4 = st.columns(4)
with k1:
    avg_price = filtered_df["current_price"].mean()
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-sub">Average Selling Price</div>
            <div class="metric-value">${avg_price:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    avg_comp_price = filtered_df["competitor_price"].mean()
    price_gap = avg_price - avg_comp_price
    gap_color = "#34d399" if price_gap <= 0 else "#f87171"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-sub">Avg Competitor Price (Gap)</div>
            <div class="metric-value" style="color: {gap_color};">${avg_comp_price:.2f} ({price_gap:+.2f})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    avg_margin = filtered_df["markup_pct"].mean()
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-sub">Average Profit Margin</div>
            <div class="metric-value">{avg_margin:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    if selected_category != "All" and selected_category in elasticity_model.category_models_:
        ped = float(elasticity_model.category_models_[selected_category]["own_elasticity"])
        if ped > 0:
            regime = "Anomalous (Endogeneity)"
            ped_color = "#f87171"
        elif ped < -1.0:
            regime = "Elastic"
            ped_color = "#a78bfa"
        else:
            regime = "Inelastic"
            ped_color = "#38bdf8"
    else:
        ped = float(elasticity_summary["own_price_elasticity"].mean())
        if ped > 0:
            regime = "Anomalous (Endogeneity)"
            ped_color = "#f87171"
        else:
            regime = "Catalog Mean"
            ped_color = "#a78bfa"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-sub">Price Elasticity (PED)</div>
            <div class="metric-value" style="color: {ped_color};">{ped:.3f} <span style="font-size: 0.85rem; color: {'#f87171' if ped > 0 else '#94a3b8'};">({regime})</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Section 1: Historical Demand Trends & Seasonality
# -----------------------------------------------------------------------------
col_title, col_freq = st.columns([1.5, 1], gap="medium")
with col_title:
    st.subheader("1. Historical Demand Trends & Sales Velocity")
with col_freq:
    granularity = st.radio("Temporal Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True, index=0)

if granularity == "Weekly":
    time_agg = resample_time_series(filtered_df, freq="W", date_col="date", group_col=None)
    time_agg = time_agg.rename(columns={"units_sold": "total_units", "current_price": "avg_price", "competitor_price": "avg_comp_price"})
elif granularity == "Monthly":
    time_agg = resample_time_series(filtered_df, freq="M", date_col="date", group_col=None)
    time_agg = time_agg.rename(columns={"units_sold": "total_units", "current_price": "avg_price", "competitor_price": "avg_comp_price"})
else:
    time_agg = (
        filtered_df.groupby("date")
        .agg(
            total_units=("units_sold", "sum"),
            avg_price=("current_price", "mean"),
            avg_comp_price=("competitor_price", "mean"),
        )
        .reset_index()
    )

fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
fig_trend.add_trace(
    go.Scatter(
        x=time_agg["date"],
        y=time_agg["total_units"],
        mode="lines",
        name=f"Units Sold ({granularity})",
        line=dict(color="#38bdf8", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(56, 189, 248, 0.08)",
    ),
    secondary_y=False,
)
fig_trend.add_trace(
    go.Scatter(
        x=time_agg["date"],
        y=time_agg["avg_price"],
        mode="lines",
        name="Average Price ($)",
        line=dict(color="#f59e0b", width=2, dash="dash"),
    ),
    secondary_y=True,
)
fig_trend.add_trace(
    go.Scatter(
        x=time_agg["date"],
        y=time_agg["avg_comp_price"],
        mode="lines",
        name="Competitor Price ($)",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
    ),
    secondary_y=True,
)

fig_trend.update_layout(
    template="plotly_dark",
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    paper_bgcolor="rgba(15, 23, 42, 0.4)",
    plot_bgcolor="rgba(15, 23, 42, 0.4)",
)
fig_trend.update_yaxes(title_text=f"Total Units Sold ({granularity})", secondary_y=False, gridcolor="rgba(255,255,255,0.06)")
fig_trend.update_yaxes(title_text="Price ($)", secondary_y=True, gridcolor="rgba(255,255,255,0.06)")

st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------------------------------------------------------------
# Section: Model Validation vs. Naive Baselines
# -----------------------------------------------------------------------------
st.write("")
st.subheader("Model Validation & Multi-Model Leaderboard")
st.markdown(
    "Rigorous out-of-time temporal holdout evaluation comparing **Linear Regression, Random Forest, XGBoost, and Prophet** "
    "against naive persistence baselines."
)

tab_lead, tab_fast = st.tabs(["🏆 Complete Model Leaderboard (All 6 Models)", "⚡ ML vs Baseline Quick Benchmark"])

with tab_lead:
    try:
        all_board = get_all_models_leaderboard()
        st.dataframe(all_board, use_container_width=True, hide_index=True)
        st.caption("Sorted by holdout RMSE. Includes MAE, RMSE, WAPE (%), MAPE (%), and R² on 14-day holdout horizon.")
    except Exception as e_board:
        st.info(f"Model leaderboard evaluating in background: {e_board}")

with tab_fast:
    try:
        benchmark_df, rmse_imp = get_model_benchmark()
        b_col1, b_col2 = st.columns([3, 1], gap="medium")
        with b_col1:
            st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
        with b_col2:
            st.metric(
                label="ML Improvement (RMSE)",
                value=f"+{rmse_imp:.2f}%",
                delta="vs Best Baseline",
                help="Percentage reduction in RMSE achieved by ML Forecaster (XGBoost) relative to the best-performing naive baseline."
            )
            st.caption("Temporal holdout split: 14-day future test horizon with strictly backward-looking lag features.")
    except Exception as bench_err:
        st.info(f"Model benchmark preview unavailable: {bench_err}")

# -----------------------------------------------------------------------------
# Section 2: Price Elasticity of Demand (PED) & Response Curves
# -----------------------------------------------------------------------------
st.write("")
st.subheader("2. Econometric Price Elasticity of Demand (Log-Log Estimation)")

el_col1, el_col2 = st.columns([1.2, 1], gap="medium")

with el_col1:
    st.markdown("**Category Elasticity Comparison with 95% Confidence Intervals**")
    fig_ped = go.Figure()

    # Bar chart with error bars
    errors = elasticity_summary["ci_upper_95"] - elasticity_summary["own_price_elasticity"]
    colors = [
        "#10b981" if "Inelastic" in reg else "#ef4444" if "Elastic" in reg else "#818cf8"
        for reg in elasticity_summary["elasticity_regime"]
    ]

    fig_ped.add_trace(
        go.Bar(
            x=elasticity_summary["category"],
            y=elasticity_summary["own_price_elasticity"],
            error_y=dict(type="data", array=errors, visible=True, color="#cbd5e1"),
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>PED: %{y:.3f}<br>95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>",
            customdata=elasticity_summary[["ci_lower_95", "ci_upper_95"]].values,
        )
    )

    # Reference line at unitary elasticity -1.0
    fig_ped.add_hline(
        y=-1.0,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="Unitary Elasticity (PED = -1.0)",
        annotation_position="bottom right",
    )

    fig_ped.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Category",
        yaxis_title="Own-Price Elasticity (β)",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )
    st.plotly_chart(fig_ped, use_container_width=True)

with el_col2:
    st.markdown("**Simulated Price-Response Curve (Demand & Revenue)**")
    cat_for_curve = selected_category if selected_category != "All" else "Sports"
    base_p = float(filtered_df["base_price"].median())
    unit_c = float(filtered_df["unit_cost"].median())
    base_q = float(filtered_df["units_sold"].median())

    curve_df = elasticity_model.simulate_price_response_curve(
        category=cat_for_curve,
        base_price=base_p,
        unit_cost=unit_c,
        base_demand=base_q,
        n_points=40,
    )

    fig_curve = make_subplots(specs=[[{"secondary_y": True}]])
    fig_curve.add_trace(
        go.Scatter(
            x=curve_df["price"],
            y=curve_df["predicted_demand"],
            name="Demand Q(P)",
            line=dict(color="#38bdf8", width=2.5),
        ),
        secondary_y=False,
    )
    fig_curve.add_trace(
        go.Scatter(
            x=curve_df["price"],
            y=curve_df["expected_profit"],
            name="Profit ($)",
            line=dict(color="#10b981", width=2.5),
        ),
        secondary_y=True,
    )
    fig_curve.add_vline(x=base_p, line_dash="dot", line_color="#94a3b8", annotation_text="Base Price")

    fig_curve.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Candidate Price ($)",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )
    fig_curve.update_yaxes(title_text="Predicted Demand", secondary_y=False)
    fig_curve.update_yaxes(title_text="Expected Profit ($)", secondary_y=True)

    st.plotly_chart(fig_curve, use_container_width=True)

# -----------------------------------------------------------------------------
# Section 3: Baseline vs. Competitor Pricing Spread
# -----------------------------------------------------------------------------
st.write("")
st.subheader("3. Baseline vs. Competitor Pricing Distribution")

c1, c2 = st.columns([1, 1], gap="medium")

with c1:
    st.markdown("**Price Distribution Comparison by Category**")
    cat_price_sample = df.sample(n=min(5000, len(df)), random_state=42)
    fig_box = go.Figure()
    fig_box.add_trace(
        go.Box(
            x=cat_price_sample["category"],
            y=cat_price_sample["current_price"],
            name="Our Price",
            marker_color="#6366f1",
        )
    )
    fig_box.add_trace(
        go.Box(
            x=cat_price_sample["category"],
            y=cat_price_sample["competitor_price"],
            name="Competitor Price",
            marker_color="#ec4899",
        )
    )
    fig_box.update_layout(
        template="plotly_dark",
        boxmode="group",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        xaxis_title="Category",
        yaxis_title="Price ($)",
    )
    st.plotly_chart(fig_box, use_container_width=True)

with c2:
    st.markdown("**Price Differential vs. Sales Velocity**")
    sample_scatter = filtered_df.sample(n=min(1200, len(filtered_df)), random_state=42)
    fig_scatter = px.scatter(
        sample_scatter,
        x="price_diff",
        y="units_sold",
        color="category",
        size="revenue",
        hover_data=["product_id", "current_price", "competitor_price"],
        labels={"price_diff": "Price Differential (Our Price - Competitor)", "units_sold": "Units Sold"},
        template="plotly_dark",
        opacity=0.75,
    )
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="#cbd5e1", annotation_text="Price Parity")
    fig_scatter.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Econometric Elasticity Summary Table
with st.expander("📊 View Detailed Econometric Elasticity Table"):
    st.dataframe(
        elasticity_summary.style.format(
            {
                "own_price_elasticity": "{:.4f}",
                "std_error": "{:.4f}",
                "p_value": "{:.4f}",
                "ci_lower_95": "{:.4f}",
                "ci_upper_95": "{:.4f}",
                "r_squared": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# Section 4: Product-Wise Demand Forecasting with Confidence Intervals
# -----------------------------------------------------------------------------
st.write("")
st.subheader("4. Product-Wise Demand Forecasting & Uncertainty Intervals")
st.markdown(
    "Select any individual SKU and an AI forecasting architecture (**XGBoost**, **Prophet**, **Random Forest**, or **Linear Regression**) "
    "to generate future demand predictions equipped with 95% Confidence Intervals."
)

f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 1], gap="medium")
with f_col1:
    available_skus = sorted(filtered_df["product_id"].unique().tolist())
    target_sku = st.selectbox("Select Target Product ID", available_skus, index=0)
with f_col2:
    selected_model_type = st.selectbox(
        "Forecasting Model Architecture",
        ["XGBoost", "Prophet", "Random Forest", "Linear Regression"],
        index=0,
    )
with f_col3:
    forecast_horizon_days = st.slider("Holdout Horizon (Days)", min_value=7, max_value=28, value=14, step=7)

if st.button("🚀 Generate Forecast & Confidence Intervals", use_container_width=True):
    sku_data = df[df["product_id"] == target_sku].sort_values("date").copy()
    if len(sku_data) < 14:
        st.warning(f"Product {target_sku} has insufficient records ({len(sku_data)}) for modeling.")
    else:
        with st.spinner(f"Fitting {selected_model_type} on {target_sku} and computing 95% confidence intervals..."):
            proc_sku = preprocess_data(sku_data)
            split_idx = max(7, len(proc_sku) - forecast_horizon_days)
            train_part = proc_sku.iloc[:split_idx]
            test_part = proc_sku.iloc[split_idx:]

            m_key = selected_model_type.lower().replace(" ", "_")
            if m_key == "prophet":
                model_inst = ProphetForecaster(date_col="date", target_col="units_sold")
                model_inst.fit(train_part)
                forecast_res = model_inst.predict_with_intervals(test_part)
            else:
                model_inst = DemandForecaster(model_type=m_key)
                model_inst.fit(train_part, target_col="units_sold")
                forecast_res = model_inst.predict_with_intervals(test_part)

            # Build Plotly interactive forecast chart
            fig_fc = go.Figure()
            # Historical actuals
            fig_fc.add_trace(
                go.Scatter(
                    x=train_part["date"],
                    y=train_part["units_sold"],
                    name="Historical Sales (Train)",
                    line=dict(color="#64748b", width=1.5),
                )
            )
            # Actual holdout test
            fig_fc.add_trace(
                go.Scatter(
                    x=test_part["date"],
                    y=test_part["units_sold"],
                    name="Actual Sales (Holdout Test)",
                    line=dict(color="#38bdf8", width=2.5),
                )
            )
            # Upper Confidence Interval bound
            fig_fc.add_trace(
                go.Scatter(
                    x=forecast_res["ds"],
                    y=forecast_res["yhat_upper"],
                    name="95% CI Upper Bound",
                    line=dict(color="rgba(16, 185, 129, 0)", width=0),
                    showlegend=False,
                )
            )
            # Lower Confidence Interval bound with fill
            fig_fc.add_trace(
                go.Scatter(
                    x=forecast_res["ds"],
                    y=forecast_res["yhat_lower"],
                    name="95% Confidence Interval",
                    line=dict(color="rgba(16, 185, 129, 0)", width=0),
                    fill="tonexty",
                    fillcolor="rgba(16, 185, 129, 0.2)",
                )
            )
            # Forecast line
            fig_fc.add_trace(
                go.Scatter(
                    x=forecast_res["ds"],
                    y=forecast_res["yhat"],
                    name=f"Predicted ({selected_model_type})",
                    line=dict(color="#10b981", width=3, dash="dash"),
                )
            )

            fig_fc.update_layout(
                title=f"{selected_model_type} Forecast for SKU: {target_sku} (with 95% Confidence Intervals)",
                template="plotly_dark",
                height=420,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                xaxis_title="Date",
                yaxis_title="Units Sold",
                hovermode="x unified",
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Accuracy & summary metrics on test horizon
            eval_metrics = evaluate_forecast(test_part["units_sold"].values, forecast_res["yhat"].values)
            c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
            c_m1.metric("Holdout MAE", f"{eval_metrics['mae']:.2f}")
            c_m2.metric("Holdout RMSE", f"{eval_metrics['rmse']:.2f}")
            c_m3.metric("Holdout WAPE", f"{eval_metrics['wape_pct']:.1f}%")
            c_m4.metric("Holdout MAPE", f"{eval_metrics['mape_pct']:.1f}%")
            c_m5.metric("Holdout R²", f"{eval_metrics['r2_score']:.3f}")
