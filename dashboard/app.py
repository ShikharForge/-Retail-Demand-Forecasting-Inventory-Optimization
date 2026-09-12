"""
dashboard/app.py — Main Streamlit application entry point.
Retail Demand Forecasting & Inventory Optimization
"""
import streamlit as st

st.set_page_config(
    page_title="Retail Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 6px 0;
        color: white;
    }
    .metric-card .label { font-size: 0.8rem; color: #90CAF9; letter-spacing: 1px; text-transform: uppercase; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: white; }
    .metric-card .delta { font-size: 0.85rem; margin-top: 4px; }
    .section-header {
        background: linear-gradient(90deg, #0f3460, #16213e);
        padding: 10px 18px; border-radius: 8px; margin: 12px 0;
        color: white; font-weight: 600;
    }
    .assumption-box {
        background: #FFF3E0; border-left: 4px solid #FF9800;
        padding: 10px 14px; border-radius: 4px; margin: 8px 0; font-size: 0.9rem;
    }
    .insight-box {
        background: #E3F2FD; border-left: 4px solid #2196F3;
        padding: 10px 14px; border-radius: 4px; margin: 8px 0; font-size: 0.9rem;
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .stButton>button {
        background: linear-gradient(135deg, #0f3460, #1565C0);
        color: white; border: none; border-radius: 8px;
        padding: 8px 20px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Home page ────────────────────────────────────────────────────────────
st.title("📦 Retail Demand Forecasting & Inventory Optimization")
st.markdown("**An end-to-end decision-support system for retail supply chain management**")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class='metric-card'>
        <div class='label'>Dataset</div>
        <div class='value'>Walmart</div>
        <div class='delta'>45 stores · 143 weeks · 6,435 records</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class='metric-card'>
        <div class='label'>Primary Model</div>
        <div class='value'>LightGBM</div>
        <div class='delta'>Panel model · Quantile regression · WAPE 4.1%</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class='metric-card'>
        <div class='label'>Forecast Type</div>
        <div class='value'>P10 · P50 · P90</div>
        <div class='delta'>Uncertainty-aware · Inventory-ready</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Navigate using the sidebar →")
st.markdown("""
| Page | Purpose |
|---|---|
| 📊 **Executive Overview** | KPIs, sales trends, store rankings |
| 🔮 **Demand Forecast** | Historical + forecast + uncertainty band |
| 📦 **Inventory Optimization** | Safety stock, reorder point, order quantity |
| 🎲 **Scenario Analysis** | What-if: demand shocks, lead time changes |
| 📈 **Model Performance** | Actual vs predicted, metrics comparison |
| 🔍 **Explainability** | SHAP feature importance, prediction explanation |
""")
