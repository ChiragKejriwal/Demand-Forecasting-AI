"""Executive performance dashboard for portfolio revenue, demand, and inventory health."""

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
import streamlit as st

from src.data.loader import load_raw_data
from src.utils.logger import configure_app_logging

configure_app_logging()

st.set_page_config(page_title="Dashboard | Dynamic Pricing", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .stat-val { font-size: 1.8rem; font-weight: 800; color: #f8fafc; }
    .stat-lbl { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Executive Operations & Performance Dashboard")
st.markdown("Monitor historical demand velocity, category revenue composition, and inventory health.")


def get_data():
    if "sales_data" in st.session_state:
        df = st.session_state["sales_data"].copy()
    else:
        df = load_raw_data()

    if "current_price" not in df.columns:
        df["current_price"] = 19.99
    if "unit_cost" not in df.columns:
        df["unit_cost"] = df["current_price"] * 0.55
    if "units_sold" not in df.columns:
        df["units_sold"] = 1.0
    if "revenue" not in df.columns:
        df["revenue"] = df["current_price"] * df["units_sold"]

    df["profit"] = (df["current_price"] - df["unit_cost"]) * df["units_sold"]
    df["margin_pct"] = ((df["current_price"] - df["unit_cost"]) / df["current_price"]) * 100
    return df


df = get_data()

# Top KPIs
k1, k2, k3, k4 = st.columns(4)
with k1:
    tot_rev = df["revenue"].sum()
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-lbl">Gross Revenue</div>
            <div class="stat-val" style="color: #38bdf8;">${tot_rev / 1e6:.2f}M</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    tot_profit = df["profit"].sum()
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-lbl">Gross Profit</div>
            <div class="stat-val" style="color: #34d399;">${tot_profit / 1e6:.2f}M</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    tot_units = df["units_sold"].sum()
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-lbl">Units Sold</div>
            <div class="stat-val">{tot_units:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    avg_margin = df["margin_pct"].mean()
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-lbl">Average Margin</div>
            <div class="stat-val" style="color: #a78bfa;">{avg_margin:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Section 1: Revenue by Category & Channel
c1, c2 = st.columns([1, 1], gap="medium")

with c1:
    cat_summary = (
        df.groupby("category")
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"), units=("units_sold", "sum"))
        .reset_index()
        .sort_values(by="revenue", ascending=False)
    )
    fig_cat = px.bar(
        cat_summary,
        x="category",
        y="revenue",
        color="profit",
        color_continuous_scale="Viridis",
        title="Revenue & Profit Contribution by Category",
        template="plotly_dark",
        labels={"revenue": "Revenue ($)", "profit": "Profit ($)"},
    )
    fig_cat.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with c2:
    channel_summary = df.groupby("channel")["revenue"].sum().reset_index()
    fig_channel = px.pie(
        channel_summary,
        names="channel",
        values="revenue",
        hole=0.45,
        title="Revenue by Sales Channel",
        template="plotly_dark",
        color_discrete_sequence=["#6366f1", "#38bdf8", "#ec4899", "#10b981"],
    )
    fig_channel.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
    )
    st.plotly_chart(fig_channel, use_container_width=True)

# Section 2: Inventory Health & Stockout Monitoring
st.subheader("Inventory Health & Stockout Risk Analysis")

i1, i2 = st.columns([1, 1], gap="medium")

with i1:
    inv_by_cat = df.groupby("category")[["inventory_level", "units_sold"]].mean().reset_index()
    fig_inv = go.Figure()
    fig_inv.add_trace(go.Bar(x=inv_by_cat["category"], y=inv_by_cat["inventory_level"], name="Avg Inventory", marker_color="#38bdf8"))
    fig_inv.add_trace(go.Bar(x=inv_by_cat["category"], y=inv_by_cat["units_sold"], name="Avg Daily Demand", marker_color="#f59e0b"))
    fig_inv.update_layout(
        title="Inventory Buffer vs. Daily Demand Velocity",
        template="plotly_dark",
        barmode="group",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )
    st.plotly_chart(fig_inv, use_container_width=True)

with i2:
    # Stockout incidents
    stockouts = df.groupby("category")["stockout_flag"].mean() * 100
    stockout_df = stockouts.reset_index().rename(columns={"stockout_flag": "stockout_rate_pct"})
    fig_stockout = px.bar(
        stockout_df,
        x="category",
        y="stockout_rate_pct",
        color="stockout_rate_pct",
        color_continuous_scale="Reds",
        title="Stockout Frequency Rate (% of days)",
        template="plotly_dark",
        labels={"stockout_rate_pct": "Stockout Rate (%)"},
    )
    fig_stockout.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )
    st.plotly_chart(fig_stockout, use_container_width=True)
