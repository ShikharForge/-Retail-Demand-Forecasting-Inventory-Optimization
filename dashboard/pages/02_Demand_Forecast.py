"""dashboard/pages/02_Demand_Forecast.py"""
import sys
sys.path.insert(0, '.')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from src.data.loader import load_config, load_raw_data
from src.data.preprocessor import clean
from src.features.engineer import build_features
from src.models.trainer import chronological_split, get_feature_cols
import pickle

st.set_page_config(page_title="Demand Forecast", page_icon="🔮", layout="wide")

@st.cache_data
def load_all():
    config = load_config('config/config.yaml')
    df_raw = load_raw_data(config)
    df = build_features(clean(df_raw, config), config)
    df['store'] = df['store'].astype('category')
    _, _, test_df = chronological_split(df, config)
    fc_path = Path('outputs/forecasts/test_forecasts.csv')
    fc = pd.read_csv(fc_path, parse_dates=['date']) if fc_path.exists() else None
    return df, test_df, fc, config

df, test_df, fc, config = load_all()

@st.cache_resource
def load_models():
    models = {}
    for name in ['lgbm_point', 'lgbm_p10', 'lgbm_p50', 'lgbm_p90']:
        path = f'outputs/models/{name}.pkl'
        if Path(path).exists():
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    return models

models = load_models()

st.title("🔮 Demand Forecast")
st.markdown("*Historical sales + LightGBM forecast with P10–P90 uncertainty bands*")
st.divider()

if fc is None:
    st.warning("⚠️ Forecast artifacts not found in `outputs/forecasts/`. Please run `python run_pipeline.py` to generate forecasts and model artifacts.")
    st.stop()

# ── Sidebar Controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    stores = sorted(df['store'].astype(int).unique())
    store_id = st.selectbox("Select Store", stores, index=0)
    show_interval = st.checkbox("Show P10–P90 Uncertainty Band", value=True)
    show_holidays = st.checkbox("Highlight Holiday Weeks", value=True)

# ── Data for selected store ────────────────────────────────────────────────
hist_store = df[df['store'].astype(int)==store_id].sort_values('date')
fc_store = fc[fc['store'].astype(int)==store_id].sort_values('date')

# ── KPIs ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Store", f"Store {store_id}")
col2.metric("Avg Weekly Sales", f"${hist_store['weekly_sales'].mean()/1e6:.3f}M")
col3.metric("Demand Volatility (CV)", f"{hist_store['weekly_sales'].std()/hist_store['weekly_sales'].mean()*100:.1f}%")
if 'forecast' in fc_store.columns and len(fc_store) > 0:
    avg_forecast = fc_store['forecast'].mean()
    col4.metric("Avg Forecast (test period)", f"${avg_forecast/1e6:.3f}M")

st.divider()

# ── Main Forecast Chart ───────────────────────────────────────────────────
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=hist_store['date'], y=hist_store['weekly_sales']/1e6,
    name='Historical Sales', line=dict(color='#BDBDBD', width=1.5),
    hovertemplate='%{x|%d %b %Y}<br>Actual: $%{y:.3f}M<extra></extra>'
))

# Actual test period
test_store = test_df[test_df['store'].astype(int)==store_id].sort_values('date')
if len(test_store) > 0:
    fig.add_trace(go.Scatter(
        x=test_store['date'], y=test_store['weekly_sales']/1e6,
        name='Actual (test)', line=dict(color='#1565C0', width=2.5),
        hovertemplate='%{x|%d %b %Y}<br>Actual: $%{y:.3f}M<extra></extra>'
    ))

# Forecast
if len(fc_store) > 0:
    if show_interval and 'p10' in fc_store.columns and 'p90' in fc_store.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_store['date'], fc_store['date'][::-1]]),
            y=pd.concat([fc_store['p90']/1e6, fc_store['p10'][::-1]/1e6]),
            fill='toself', fillcolor='rgba(255,87,34,0.15)',
            line=dict(color='rgba(255,87,34,0)'),
            name='P10–P90 Uncertainty Band',
            hoverinfo='skip'
        ))

    fig.add_trace(go.Scatter(
        x=fc_store['date'], y=fc_store['forecast']/1e6,
        name='LightGBM Forecast (P50)', line=dict(color='#FF5722', width=2.5, dash='dash'),
        hovertemplate='%{x|%d %b %Y}<br>Forecast: $%{y:.3f}M<extra></extra>'
    ))

# Holiday markers
if show_holidays:
    holiday_dates = hist_store[hist_store['holiday_flag']==1]['date']
    for hd in holiday_dates:
        fig.add_vline(x=hd, line=dict(color='rgba(255,152,0,0.4)', width=1))

# Test split line
test_start = pd.Timestamp(config['splits']['val_end'])
fig.add_vline(x=test_start, line=dict(color='#4CAF50', width=1.5, dash='dot'),
              annotation_text="Test Start", annotation_position="top left")

fig.update_layout(
    height=500, template='plotly_white',
    title=f'Store {store_id}: Demand Forecast with Uncertainty Bands',
    yaxis_title='Weekly Sales ($M)', xaxis_title='Date',
    yaxis_tickprefix='$', yaxis_ticksuffix='M',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# ── Error Analysis ────────────────────────────────────────────────────────
if len(fc_store) > 0 and len(test_store) > 0:
    st.subheader("Forecast Error Analysis")
    merged = pd.merge(
        test_store[['date','weekly_sales']],
        fc_store[['date','forecast']],
        on='date', how='inner'
    )
    merged['error'] = merged['weekly_sales'] - merged['forecast']
    merged['abs_pct_error'] = abs(merged['error'] / merged['weekly_sales']) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"${abs(merged['error']).mean():,.0f}")
    col2.metric("WAPE", f"{abs(merged['error']).sum() / merged['weekly_sales'].sum():.4f}")
    col3.metric("Median Abs % Error", f"{merged['abs_pct_error'].median():.1f}%")

    # Residual plot
    fig_res = go.Figure()
    fig_res.add_trace(go.Bar(x=merged['date'], y=merged['error']/1e3,
                             marker_color=['#B71C1C' if e < 0 else '#1565C0' for e in merged['error']],
                             name='Residual (Actual - Forecast)'))
    fig_res.add_hline(y=0, line=dict(color='black', width=1))
    fig_res.update_layout(height=280, template='plotly_white',
                          title='Forecast Residuals (test period)',
                          yaxis_title='Error ($K)', xaxis_title='Date')
    st.plotly_chart(fig_res, use_container_width=True)

st.markdown("""
<div style='background:#E8F5E9;border-left:4px solid #4CAF50;padding:10px 14px;border-radius:4px;font-size:0.9rem;'>
<b>Interpretation guide:</b> The orange band shows the P10–P90 prediction interval.
In 80% of weeks we expect actual demand to fall within this band.
Wide bands = more uncertainty = larger safety stock needed.
</div>
""", unsafe_allow_html=True)
