"""dashboard/pages/03_Inventory_Optimization.py"""
import sys
sys.path.insert(0, '.')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.data.loader import load_config, load_raw_data
from src.data.preprocessor import clean
from src.features.engineer import build_features
from src.models.trainer import chronological_split
from src.inventory.optimizer import InventoryParameters, optimise, InventoryDecision

st.set_page_config(page_title="Inventory Optimization", page_icon="📦", layout="wide")

@st.cache_data
def load_data():
    config = load_config('config/config.yaml')
    df_raw = load_raw_data(config)
    df = build_features(clean(df_raw, config), config)
    df['store'] = df['store'].astype('category')
    fc_path = Path('outputs/forecasts/test_forecasts.csv')
    fc = pd.read_csv(fc_path, parse_dates=['date']) if fc_path.exists() else None
    return df, fc, config

df, fc, config = load_data()

st.title("📦 Inventory Optimization")
st.markdown("*Convert demand forecasts into statistically grounded inventory decisions*")

if fc is None:
    st.warning("⚠️ Forecast artifacts not found in `outputs/forecasts/`. Please run `python run_pipeline.py` to generate forecasts and model artifacts.")
    st.stop()

# ── Assumption Warning ─────────────────────────────────────────────────────
st.warning("""
**⚠️ MODELLING ASSUMPTIONS — Not from data**

The inventory parameters below (lead time, ordering cost, holding cost) are **assumptions**, not
observed values. The Walmart dataset contains no inventory records.
Adjust the sliders to reflect your actual business parameters.
""")

# ── Sidebar: Inventory Parameters ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Inventory Parameters")
    st.markdown("*All parameters are assumptions — adjust to your context*")

    stores = sorted(df['store'].astype(int).unique())
    store_id = st.selectbox("Store", stores, index=0)

    st.markdown("---")
    lead_time = st.slider("Lead Time (weeks)", 1, 8, 2,
                          help="Weeks between placing and receiving an order. ASSUMPTION.")
    service_level = st.slider("Service Level", 0.80, 0.99, 0.95, step=0.01,
                              help="Target probability of not stocking out. ASSUMPTION.")
    ordering_cost = st.number_input("Ordering Cost per Order ($)", 100, 5000, 500, step=50,
                                    help="Fixed cost of placing one order. ASSUMPTION.")
    holding_cost_pct = st.slider("Annual Holding Cost (%)", 0.10, 0.40, 0.25, step=0.01,
                                 help="% of inventory value per year. ASSUMPTION.")
    current_inv = st.number_input("Current Inventory ($)", 0, 10_000_000, 2_000_000, step=50000,
                                  help="Your current on-hand stock in USD. USER INPUT.")
    safety_method = st.radio("Safety Stock Method",
                             ["Quantile-based (P10–P90)", "Parametric (Normal dist.)"],
                             index=0)

# ── Compute for selected store ─────────────────────────────────────────────
fc_store = fc[fc['store'].astype(int)==store_id]
hist_store = df[df['store'].astype(int)==store_id]

if len(fc_store) > 0 and 'p50' in fc_store.columns and 'p90' in fc_store.columns:
    weekly_p50 = fc_store['p50'].mean()
    weekly_p90 = fc_store['p90'].mean()
else:
    weekly_p50 = hist_store['weekly_sales'].mean()
    weekly_p90 = weekly_p50 * 1.15

demand_std = hist_store['weekly_sales'].std()

params = InventoryParameters(
    lead_time_weeks=lead_time,
    service_level=service_level,
    ordering_cost_usd=ordering_cost,
    holding_cost_pct=holding_cost_pct,
    current_inventory_usd=current_inv,
)

use_q = (safety_method == "Quantile-based (P10–P90)")
decision = optimise(store_id, weekly_p50, weekly_p90, demand_std, params,
                    use_quantile_safety_stock=use_q)

# ── KPI Row ────────────────────────────────────────────────────────────────
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Lead-Time Demand", f"${decision.lead_time_demand/1e3:.1f}K",
            help="Expected demand during replenishment lead time")
col2.metric("Safety Stock", f"${decision.safety_stock/1e3:.1f}K",
            help="Buffer stock to protect against demand uncertainty")
col3.metric("Reorder Point", f"${decision.reorder_point/1e3:.1f}K",
            help="Order when inventory falls to this level")
col4.metric("EOQ", f"${decision.eoq/1e3:.1f}K",
            help="Economic Order Quantity — optimal order size")
col5.metric("Stockout Risk", f"{decision.stockout_risk*100:.1f}%",
            delta=f"Target: {(1-service_level)*100:.0f}%")

st.divider()

# ── Waterfall Chart ────────────────────────────────────────────────────────
col_a, col_b = st.columns([3, 2])

with col_a:
    st.subheader("Inventory Policy Breakdown")
    fig = go.Figure(go.Waterfall(
        name='Inventory Build-up',
        orientation='v',
        measure=['relative', 'relative', 'total', 'relative', 'total'],
        x=['Lead-Time\nDemand', 'Safety\nStock', 'Reorder\nPoint (ROP)', 'EOQ\n(order qty)', 'Total\nInventory Target'],
        y=[decision.lead_time_demand/1e3, decision.safety_stock/1e3,
           0, decision.eoq/1e3, 0],
        connector=dict(line=dict(color='#BDBDBD')),
        increasing=dict(marker=dict(color='#1565C0')),
        totals=dict(marker=dict(color='#FF5722')),
    ))
    fig.update_layout(height=380, template='plotly_white',
                      yaxis_title='Amount ($K)', yaxis_tickprefix='$', yaxis_ticksuffix='K')
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Current Inventory Status")
    inv_pct = min(current_inv / max(decision.reorder_point, 1), 2.0)
    color = '#B71C1C' if current_inv < decision.reorder_point else '#2E7D32'
    status = "⚠️ BELOW ROP — ORDER NOW" if current_inv < decision.reorder_point else "✅ ABOVE ROP — OK"

    fig2 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_inv/1e3,
        delta={'reference': decision.reorder_point/1e3, 'relative': False,
               'valueformat': '.0f', 'prefix': 'ROP: $', 'suffix': 'K'},
        number={'prefix': '$', 'suffix': 'K', 'valueformat': ',.0f'},
        title={'text': f"Current Inventory<br><span style='font-size:0.8em;color:{color}'>{status}</span>"},
        gauge={
            'axis': {'range': [0, max(current_inv, decision.reorder_point*2)/1e3]},
            'bar': {'color': color, 'thickness': 0.3},
            'steps': [
                {'range': [0, decision.reorder_point/1e3], 'color': '#FFEBEE'},
                {'range': [decision.reorder_point/1e3, (decision.reorder_point+decision.eoq)/1e3],
                 'color': '#E8F5E9'},
            ],
            'threshold': {
                'line': {'color': '#F44336', 'width': 3},
                'thickness': 0.75,
                'value': decision.reorder_point/1e3
            },
        }
    ))
    fig2.update_layout(height=320, template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

    weeks_cov = round(current_inv / max(weekly_p50, 1), 1)
    st.info(f"📅 **Weeks of coverage:** {weeks_cov} weeks at current demand rate")
    if decision.recommended_order > 0:
        st.error(f"📦 **Recommended order:** ${decision.recommended_order/1e3:,.1f}K")
    else:
        st.success(f"✅ No order needed — inventory above ROP")

# ── Full Inventory Report ──────────────────────────────────────────────────
with st.expander("📋 Full Inventory Report (with formula derivations)"):
    st.markdown(f"""
| Parameter | Value | Formula |
|---|---|---|
| Weekly Demand (P50) | ${weekly_p50:,.0f} | From LightGBM quantile forecast |
| Weekly Demand (P90) | ${weekly_p90:,.0f} | From LightGBM quantile forecast |
| Demand Std Dev | ${demand_std:,.0f} | From historical data |
| Lead Time | {lead_time} weeks | **ASSUMPTION** |
| Lead-Time Demand | ${decision.lead_time_demand:,.0f} | P50 × lead_time |
| Safety Stock | ${decision.safety_stock:,.0f} | (P90−P50) × √lead_time |
| **Reorder Point** | **${decision.reorder_point:,.0f}** | **LTD + Safety Stock** |
| Annual Demand | ${weekly_p50*52:,.0f} | P50 × 52 weeks |
| Ordering Cost | ${ordering_cost:,} | **ASSUMPTION** |
| Holding Cost | {holding_cost_pct*100:.0f}% p.a. | **ASSUMPTION** |
| **EOQ** | **${decision.eoq:,.0f}** | **√(2·D·S/H)** |
| Stockout Risk | {decision.stockout_risk*100:.2f}% | 1−Φ((ROP−LTD)/σ_LT) |
| Target Service Level | {service_level*100:.0f}% | **ASSUMPTION** |

**Reference:** Silver, Pyke & Thomas (2017). *Inventory and Production Management in Supply Chains*.
    """)
