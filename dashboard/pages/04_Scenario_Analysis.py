"""dashboard/pages/04_Scenario_Analysis.py"""
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
from src.inventory.optimizer import InventoryParameters, optimise
from src.scenarios.scenario_engine import all_scenarios, HOLIDAY_UPLIFT_FACTOR

st.set_page_config(page_title="Scenario Analysis", page_icon="🎲", layout="wide")

@st.cache_data
def load_data():
    config = load_config('config/config.yaml')
    df_raw = load_raw_data(config)
    df = build_features(clean(df_raw, config), config)
    df['store'] = df['store'].astype('category')
    fc = pd.read_csv('outputs/forecasts/test_forecasts.csv', parse_dates=['date'])
    return df, fc, config

df, fc, config = load_data()

st.title("🎲 What-If Scenario Analysis")
st.markdown("*Test how demand shocks and supply disruptions affect your inventory policy*")
st.divider()

with st.sidebar:
    st.header("⚙️ Base Parameters")
    store_id = st.selectbox("Store", sorted(df['store'].astype(int).unique()), index=0)
    lead_time = st.slider("Lead Time (weeks)", 1, 8, 2)
    service_level = st.slider("Service Level", 0.80, 0.99, 0.95, step=0.01)
    current_inv = st.number_input("Current Inventory ($)", 0, 10_000_000, 2_000_000, step=50000)

fc_store = fc[fc['store'].astype(int)==store_id]
hist_store = df[df['store'].astype(int)==store_id]
weekly_p50 = fc_store['p50'].mean() if 'p50' in fc_store.columns and len(fc_store) else hist_store['weekly_sales'].mean()
weekly_p90 = fc_store['p90'].mean() if 'p90' in fc_store.columns and len(fc_store) else weekly_p50 * 1.15
demand_std = hist_store['weekly_sales'].std()

params = InventoryParameters(lead_time_weeks=lead_time, service_level=service_level,
                              current_inventory_usd=current_inv)

scenario_results = all_scenarios(store_id, weekly_p50, weekly_p90, demand_std, params, config)

# ── Scenario Delta Table ────────────────────────────────────────────────────
st.subheader("Scenario Impact on Inventory Decisions")

rows = []
for res in scenario_results:
    d = res.delta()
    rows.append({
        'Scenario': res.scenario_name,
        'Safety Stock': f"${res.scenario.safety_stock/1e3:.1f}K",
        'SS Change': f"{d['safety_stock']['delta_pct']:+.1f}%",
        'Reorder Point': f"${res.scenario.reorder_point/1e3:.1f}K",
        'ROP Change': f"{d['reorder_point']['delta_pct']:+.1f}%",
        'Rec. Order': f"${res.scenario.recommended_order/1e3:.1f}K" if res.scenario.recommended_order > 0 else "None",
        'Stockout Risk': f"{res.scenario.stockout_risk*100:.1f}%",
    })

tbl = pd.DataFrame(rows)
# Baseline row
base = scenario_results[0].base
baseline_row = {
    'Scenario': '📍 BASE CASE',
    'Safety Stock': f"${base.safety_stock/1e3:.1f}K",
    'SS Change': '—',
    'Reorder Point': f"${base.reorder_point/1e3:.1f}K",
    'ROP Change': '—',
    'Rec. Order': f"${base.recommended_order/1e3:.1f}K" if base.recommended_order > 0 else "None",
    'Stockout Risk': f"{base.stockout_risk*100:.1f}%",
}
display = pd.concat([pd.DataFrame([baseline_row]), tbl], ignore_index=True)
st.dataframe(display, use_container_width=True)

st.divider()

# ── Visual: ROP by Scenario ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reorder Point by Scenario")
    scenario_names = ['Base Case'] + [r.scenario_name for r in scenario_results]
    rops = [base.reorder_point] + [r.scenario.reorder_point for r in scenario_results]
    ss_vals = [base.safety_stock] + [r.scenario.safety_stock for r in scenario_results]
    ltd_vals = [base.lead_time_demand] + [r.scenario.lead_time_demand for r in scenario_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Lead-Time Demand', x=scenario_names, y=[v/1e3 for v in ltd_vals],
                         marker_color='#90CAF9'))
    fig.add_trace(go.Bar(name='Safety Stock', x=scenario_names, y=[v/1e3 for v in ss_vals],
                         marker_color='#FF5722'))
    fig.update_layout(barmode='stack', height=380, template='plotly_white',
                      yaxis_title='Amount ($K)', yaxis_tickprefix='$', yaxis_ticksuffix='K',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Stockout Risk by Scenario")
    risks = [base.stockout_risk*100] + [r.scenario.stockout_risk*100 for r in scenario_results]
    colors = ['#2E7D32' if r < 10 else '#F9A825' if r < 20 else '#B71C1C' for r in risks]
    fig2 = go.Figure(go.Bar(
        x=scenario_names, y=risks,
        marker_color=colors,
        text=[f'{r:.1f}%' for r in risks],
        textposition='outside'
    ))
    fig2.add_hline(y=(1-service_level)*100, line=dict(color='red', dash='dot'),
                   annotation_text=f"Target: {(1-service_level)*100:.0f}%")
    fig2.update_layout(height=380, template='plotly_white',
                       yaxis_title='Stockout Risk (%)', yaxis_ticksuffix='%')
    st.plotly_chart(fig2, use_container_width=True)

# ── Scenario deep-dives ─────────────────────────────────────────────────────
with st.expander("📖 Scenario Definitions & Business Rationale"):
    st.markdown(f"""
| Scenario | What Changes | Business Rationale |
|---|---|---|
| Demand Surge (+20%) | P50, P90 ×1.20 | Promotion, competitor exit, viral product |
| Demand Drop (−20%) | P50, P90 ×0.80 | Post-promo slump, substitution, recession |
| Lead Time Doubles | Lead time ×2 | Supplier disruption, port delays |
| High Uncertainty (σ×1.5) | P90 spread ×1.5 | Forecasting harder (new product, volatile market) |
| Holiday Promotion (+{int((HOLIDAY_UPLIFT_FACTOR-1)*100)}%) | P50, P90 ×{HOLIDAY_UPLIFT_FACTOR:.3f} | Historical holiday uplift observed in data |

**Key insight:** Lead time disruption has the most dramatic effect on safety stock
(grows with √L), making supplier reliability a critical lever for inventory efficiency.
    """)
