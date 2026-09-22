"""Main landing page for Dynamic Pricing Under Demand Uncertainty portfolio project."""

import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st
from src.data.loader import get_dataset_summary, inspect_uploaded_df, load_raw_data
from src.utils.logger import configure_app_logging

configure_app_logging()

st.set_page_config(
    page_title="Dynamic Pricing Under Demand Uncertainty",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics and modern typography
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-hero {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.75rem;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        line-height: 1.6;
        max-width: 900px;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 1.5rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .card-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        margin-bottom: 0.75rem;
    }

    .badge-blue { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-rose { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }
    .badge-indigo { background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); }
    .badge-cyan { background: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); }

    .pipeline-step {
        border-left: 3px solid #6366f1;
        padding-left: 1.25rem;
        margin-bottom: 1.5rem;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero Section
st.markdown(
    """
    <div class="main-hero">
        <div class="hero-title">📈 Dynamic Pricing Under Demand Uncertainty</div>
        <div class="hero-subtitle">
            An end-to-end Machine Learning and Mathematical Optimization decision-support system
            that estimates price elasticity of demand, formulates constrained profit-maximization programs,
            and stress-tests optimal prices using stochastic Monte Carlo risk analytics.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Methodological Warning Banner
st.warning(
    "⚠️ **METHODOLOGICAL WARNING: SYNTHETIC DATA & ENDOGENEITY**\n\n"
    "- **Synthetic Features & Demonstrative Metrics:** The `unit_cost` and `competitor_price` columns in this dataset "
    "are synthetically generated random noise created for end-to-end system architecture demonstration. Consequently, downstream "
    "financial profit, margin calculations, and cross-price elasticity (XED) values should be interpreted purely as proof-of-concept demonstrations.\n"
    "- **Price Endogeneity & Reverse Causation:** Positive price elasticity estimates observed across certain categories reflect "
    "observational price endogeneity (e.g., retailers systematically raising prices during high-demand peak seasons or promotional surges). "
    "These empirical correlations do not represent true causal downward-sloping demand curves."
)


# -------------------------------------------------------------
# Sample Template Modal Dialog
# -------------------------------------------------------------
@st.dialog("📥 Sample Sales CSV Template", width="large")
def show_schema_modal(inspection: Optional[Dict[str, Any]] = None, error_msg: Optional[str] = None):
    """Render an interactive modal displaying the sample CSV template and format requirements."""
    if error_msg or (inspection and not inspection.get("is_valid", True)):
        st.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 10px; padding: 1.1rem 1.25rem; margin-bottom: 1.25rem;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #f87171; margin-bottom: 0.35rem;">
                    ❌ Schema Validation Issue Detected
                </div>
                <div style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5;">
                    The uploaded CSV cannot be processed because one or more mandatory fields are missing or improperly formatted.
                    Please download and match the sample template below.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if inspection and inspection.get("missing_mandatory"):
            missing_html = " ".join([f"<span class='card-badge badge-rose'>Missing: {c}</span>" for c in inspection["missing_mandatory"]])
            st.markdown(f"**Missing Mandatory Columns:** {missing_html}", unsafe_allow_html=True)

        if inspection and inspection.get("raw_columns"):
            detected_html = " ".join([f"<span class='card-badge badge-blue'>{c}</span>" for c in inspection["raw_columns"]])
            st.markdown(f"**Detected Columns in Upload:** {detected_html}", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 10px; padding: 1.1rem 1.25rem; margin-bottom: 1.25rem;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #818cf8; margin-bottom: 0.35rem;">
                    📥 Pre-Formatted Sales Dataset Template
                </div>
                <div style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5;">
                    Download and populate this CSV template. Mandatory columns are <code>date</code>, <code>product_id</code>, <code>current_price</code>, and <code>units_sold</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Sample Dataset Preview")
    sample_df = pd.DataFrame([
        {"date": "2024-01-01", "product_id": "SKU_101", "category": "Electronics", "current_price": 49.99, "units_sold": 15, "base_price": 49.99, "unit_cost": 24.99, "competitor_price": 52.00, "region": "North", "channel": "Online"},
        {"date": "2024-01-02", "product_id": "SKU_101", "category": "Electronics", "current_price": 49.99, "units_sold": 18, "base_price": 49.99, "unit_cost": 24.99, "competitor_price": 52.00, "region": "North", "channel": "Online"},
        {"date": "2024-01-03", "product_id": "SKU_101", "category": "Electronics", "current_price": 44.99, "units_sold": 25, "base_price": 49.99, "unit_cost": 24.99, "competitor_price": 51.50, "region": "North", "channel": "Online"},
        {"date": "2024-01-01", "product_id": "SKU_202", "category": "Apparel", "current_price": 29.99, "units_sold": 30, "base_price": 29.99, "unit_cost": 12.50, "competitor_price": 31.00, "region": "South", "channel": "Retail"},
        {"date": "2024-01-02", "product_id": "SKU_202", "category": "Apparel", "current_price": 29.99, "units_sold": 28, "base_price": 29.99, "unit_cost": 12.50, "competitor_price": 31.00, "region": "South", "channel": "Retail"},
    ])
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

    csv_buffer = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Sample CSV Template (.csv)",
        data=csv_buffer,
        file_name="sample_sales_template.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )


# Sidebar Data Upload
st.sidebar.header("📁 Data Source Configuration")

if st.sidebar.button("📥 Download Sample CSV Template", use_container_width=True):
    show_schema_modal()

uploaded_file = st.sidebar.file_uploader(
    "Upload Custom Sales CSV",
    type=["csv"],
    help="Upload your transactional sales dataset matching the required schema.",
)

prev_file = st.session_state.get("_last_uploaded_filename", None)
curr_file = uploaded_file.name if uploaded_file is not None else None

if uploaded_file is not None:
    if prev_file != curr_file:
        st.session_state["_last_uploaded_filename"] = curr_file
        try:
            df_uploaded = load_raw_data(uploaded_file)
            st.session_state["sales_data"] = df_uploaded
            st.session_state["data_source_name"] = uploaded_file.name
            st.session_state.pop("upload_error_info", None)
            st.sidebar.success(f"Loaded: `{uploaded_file.name}` ({len(df_uploaded):,} records)")
        except Exception as upload_err:
            inspection = getattr(upload_err, "inspection", None)
            if inspection is None:
                try:
                    uploaded_file.seek(0)
                    df_preview = pd.read_csv(uploaded_file, nrows=10)
                    inspection = inspect_uploaded_df(df_preview)
                except Exception:
                    inspection = None
            err_info = {
                "msg": str(upload_err),
                "inspection": inspection,
                "name": uploaded_file.name,
            }
            st.session_state["upload_error_info"] = err_info
            show_schema_modal(inspection=inspection, error_msg=str(upload_err))
            if "sales_data" not in st.session_state:
                st.session_state["sales_data"] = load_raw_data()
                st.session_state["data_source_name"] = "Default Augmented Catalog"
    else:
        if "upload_error_info" in st.session_state:
            err_info = st.session_state["upload_error_info"]
            st.sidebar.error("❌ Schema Validation Alert")
            st.sidebar.caption(f"Mandatory fields missing in `{err_info['name']}`.")
            if st.sidebar.button("🔍 Open Schema & Fix Guide", type="primary", use_container_width=True):
                show_schema_modal(inspection=err_info["inspection"], error_msg=err_info["msg"])
        else:
            st.sidebar.success(f"Active: `{uploaded_file.name}`")
            inspection = getattr(st.session_state.get("sales_data"), "attrs", {}).get("schema_inspection", {})
            extra_cols = inspection.get("extra_columns", [])
            if extra_cols:
                with st.sidebar.expander(f"📦 Extra Fields Detected ({len(extra_cols)})", expanded=False):
                    st.markdown(f"**Custom columns preserved:** {', '.join([f'`{c}`' for c in extra_cols])}")
                    st.caption("These fields are preserved in memory and isolated from the ML forecasting feature vectors.")
else:
    st.session_state.pop("_last_uploaded_filename", None)
    st.session_state.pop("upload_error_info", None)
    if "sales_data" not in st.session_state:
        st.session_state["sales_data"] = load_raw_data()
        st.session_state["data_source_name"] = "Default Augmented Catalog"

df = st.session_state["sales_data"]
data_source_name = st.session_state.get("data_source_name", "Default Augmented Catalog")

if "upload_error_info" in st.session_state:
    err_info = st.session_state["upload_error_info"]
    st.error(
        f"⚠️ **Upload Notice:** The custom dataset `{err_info['name']}` failed schema validation. "
        "The application is currently displaying the fallback demo catalog."
    )
    if st.button("📋 Open Schema & Field Requirements Modal", type="primary"):
        show_schema_modal(inspection=err_info["inspection"], error_msg=err_info["msg"])

if data_source_name != "Default Augmented Catalog":
    st.info(f"📂 Active Dataset: **{data_source_name}** ({len(df):,} records)")
    if st.sidebar.button("Reset to Default Demo Catalog"):
        del st.session_state["sales_data"]
        del st.session_state["data_source_name"]
        st.session_state.pop("upload_error_info", None)
        st.rerun()

try:
    summary = get_dataset_summary(df)
    data_loaded = True
except Exception as e:
    st.error(f"Error analyzing dataset: {e}")
    data_loaded = False

if data_loaded:
    # High-level Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Dataset Records</div>
                <div class="metric-value">{summary['num_records']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Active SKUs</div>
                <div class="metric-value">{summary['num_products']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Categories</div>
                <div class="metric-value">{df['category'].nunique()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        avg_price = df["current_price"].mean()
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Average Price</div>
                <div class="metric-value">${avg_price:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        total_rev = df["revenue"].sum() / 1e6
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Historical Revenue</div>
                <div class="metric-value">${total_rev:.1f}M</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.subheader("System Architecture & Analytical Workflow")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown(
        """
        <div class="pipeline-step">
            <span class="card-badge badge-blue">STEP 1</span>
            <h4 style="margin: 0 0 0.5rem 0; color: #f1f5f9;">Temporal Feature Engineering & Split</h4>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
                Ingests augmented retail transactions, engineers cyclical calendar signals, price differentials,
                and backward-looking demand lags. Enforces strict temporal holdouts (no lookahead data leakage).
            </p>
        </div>

        <div class="pipeline-step">
            <span class="card-badge badge-purple">STEP 2</span>
            <h4 style="margin: 0 0 0.5rem 0; color: #f1f5f9;">Log-Log Econometric Elasticity</h4>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
                Calculates category-specific Own-Price Elasticity of Demand (PED) and Cross-Price Elasticity (XED)
                via structural OLS with robust standard errors, mapping consumer price sensitivity.
            </p>
        </div>

        <div class="pipeline-step">
            <span class="card-badge badge-emerald">STEP 3</span>
            <h4 style="margin: 0 0 0.5rem 0; color: #f1f5f9;">Constrained Mathematical Optimization</h4>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
                Formulates profit maximization programs solved via SciPy (continuous) and PuLP (mixed-integer).
                Enforces margin floors, price bounds (&plusmn;20%), and stockout prevention limits.
            </p>
        </div>

        <div class="pipeline-step">
            <span class="card-badge badge-amber">STEP 4</span>
            <h4 style="margin: 0 0 0.5rem 0; color: #f1f5f9;">Monte Carlo Stochastic Stress-Testing</h4>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
                Stress-tests recommendations across 5,000+ stochastic trials injecting competitor price cuts (-10%),
                wholesale cost spikes (+5%), and latent demand shocks. Quantifies 95% Value at Risk (VaR) and CVaR.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 1.25rem;">
            <h4 style="color: #38bdf8; margin-top: 0;">🔬 Page 1: Demand & Elasticity Analysis</h4>
            <p style="color: #cbd5e1; font-size: 0.92rem;">
                Explore category elasticity distributions, identify price-sensitive vs. inelastic product categories,
                and inspect empirical demand response curves and competitor pricing spreads using interactive Plotly charts.
            </p>
            <p style="color: #94a3b8; font-size: 0.85rem; font-style: italic;">
                Navigate to <b>Analysis</b> via the left sidebar to explore econometric curves.
            </p>
        </div>

        <div class="glass-card">
            <h4 style="color: #34d399; margin-top: 0;">🎯 Page 2: Prescriptive Pricing & Simulation</h4>
            <p style="color: #cbd5e1; font-size: 0.92rem;">
                Select any SKU to evaluate recommended pricing versus baseline rules. Adjust competitor pricing and
                unit cost sliders in real time to witness instant constraint propagation, profit lift, and Monte Carlo downside risk distributions.
            </p>
            <p style="color: #94a3b8; font-size: 0.85rem; font-style: italic;">
                Navigate to <b>Recommendations</b> via the left sidebar to run scenario stress tests.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption("Dynamic Pricing Under Demand Uncertainty | Developed with Streamlit, Plotly, PuLP, and SciPy")
