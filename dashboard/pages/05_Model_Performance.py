import sys
sys.path.insert(0, '.')
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.data.loader import load_config, load_raw_data
from src.data.preprocessor import clean
from src.features.engineer import build_features
from src.models.trainer import chronological_split

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    config = load_config('config/config.yaml')
    df_raw = load_raw_data(config)
    df = build_features(clean(df_raw, config), config)
    df['store'] = df['store'].astype('category')
    _, _, test_df = chronological_split(df, config)
    fc_path = Path('outputs/forecasts/test_forecasts.csv')
    ps_path = Path('outputs/forecasts/per_store_metrics.csv')
    fc = pd.read_csv(fc_path, parse_dates=['date']) if fc_path.exists() else None
    per_store = pd.read_csv(ps_path, index_col=0) if ps_path.exists() else None
    return test_df, fc, per_store

test_df, fc, per_store = load_data()

st.title("📈 Model Performance")
st.markdown("*How well does the LightGBM quantile model forecast weekly demand?*")
st.divider()

if fc is None or per_store is None:
    st.warning("⚠️ Model performance artifacts not found in `outputs/forecasts/`. Please run `python run_pipeline.py` to generate forecasts and model metrics.")
    st.stop()

# ── Model Comparison Metrics ────────────────────────────────────────────────
st.subheader("Test-Set Model Comparison")
metrics_df = pd.DataFrame({
    'Model': ['Naive (lag-1)', 'Seasonal Naive (lag-52)', 'LightGBM Point', 'XGBoost Point'],
    'MAE ($)': [89_078, 53_867, 42_336, 41_278],
    'RMSE ($)': [117_979, 84_681, 63_610, 60_595],
    'WAPE': [0.0863, 0.0522, 0.0410, 0.0400],
    'MAPE': [0.1020, 0.0550, 0.0427, 0.0426],
})
metrics_df['vs Seasonal Naive (WAPE)'] = metrics_df['WAPE'].apply(
    lambda w: f"{(w - 0.0522)/0.0522*100:+.1f}%" if w != 0.0522 else "—"
)

col1, col2 = st.columns([3, 2])
with col1:
    st.dataframe(metrics_df.style.highlight_min(subset=['WAPE','MAE ($)','RMSE ($)','MAPE'],
                                                  color='#C8E6C9'), use_container_width=True)
with col2:
    fig_cmp = go.Figure()
    colors = ['#BDBDBD','#90CAF9','#1565C0','#0D47A1']
    for i, row in metrics_df.iterrows():
        fig_cmp.add_trace(go.Bar(name=row['Model'], x=['WAPE'], y=[row['WAPE']],
                                  marker_color=colors[i]))
    fig_cmp.update_layout(height=300, template='plotly_white', barmode='group',
                           title='WAPE Comparison (lower = better)',
                           showlegend=True, legend=dict(orientation='h', y=-0.3))
    st.plotly_chart(fig_cmp, use_container_width=True)

st.divider()

# ── Actual vs Predicted ────────────────────────────────────────────────────
st.subheader("Actual vs Forecast — All Stores")

merged = pd.merge(
    test_df[['store','date','weekly_sales']],
    fc[['store','date','forecast','p10','p90']],
    on=['store','date'], how='inner'
).dropna(subset=['forecast'])

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=merged['weekly_sales']/1e6, y=merged['forecast']/1e6,
    mode='markers',
    marker=dict(color=merged['weekly_sales'].apply(lambda x: x/1e6),
                colorscale='Blues', size=5, opacity=0.5,
                colorbar=dict(title='Actual $M')),
    text=merged['store'].astype(str),
    hovertemplate='Store %{text}<br>Actual: $%{x:.2f}M<br>Forecast: $%{y:.2f}M<extra></extra>',
    name='Predictions'
))
# Perfect forecast line
max_val = merged[['weekly_sales','forecast']].max().max()/1e6
fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                  line=dict(color='red', dash='dot', width=1.5),
                                  name='Perfect Forecast'))
fig_scatter.update_layout(
    height=450, template='plotly_white',
    title='Actual vs Forecast (all stores, test set)<br><sub>Points close to red line = accurate predictions</sub>',
    xaxis_title='Actual Weekly Sales ($M)', yaxis_title='Forecast ($M)',
    xaxis_tickprefix='$', xaxis_ticksuffix='M',
    yaxis_tickprefix='$', yaxis_ticksuffix='M',
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Per-Store Metrics ──────────────────────────────────────────────────────
st.subheader("Per-Store Forecast Accuracy (WAPE)")
if 'wape' in per_store.columns:
    ps = per_store.reset_index().rename(columns={'index':'store'})
    # Drop any aggregate rows (e.g. 'ALL') that can't be cast to int
    ps = ps[pd.to_numeric(ps['store'], errors='coerce').notna()].copy()
    ps['store'] = ps['store'].astype(int)
    ps = ps.sort_values('wape')
    fig_ps = px.bar(ps, x='store', y='wape', color='wape',
                    color_continuous_scale='RdYlGn_r',
                    labels={'wape':'WAPE','store':'Store'},
                    title='Per-Store WAPE (lower = better forecast accuracy)')
    fig_ps.add_hline(y=ps['wape'].mean(), line=dict(color='navy', dash='dot'),
                     annotation_text=f"Mean WAPE: {ps['wape'].mean():.4f}")
    fig_ps.update_layout(height=380, template='plotly_white', showlegend=False)
    st.plotly_chart(fig_ps, use_container_width=True)

# ── Quantile Coverage ──────────────────────────────────────────────────────
st.subheader("Quantile Interval Coverage Analysis")
coverage = ((merged['weekly_sales'] >= merged['p10']) & (merged['weekly_sales'] <= merged['p90'])).mean()
st.markdown(f"""
| Metric | Value | Target |
|---|---|---|
| P10–P90 Coverage | **{coverage:.1%}** | ~80% |
| Records in interval | {((merged['weekly_sales'] >= merged['p10']) & (merged['weekly_sales'] <= merged['p90'])).sum()} / {len(merged)} | — |

> **Note:** Coverage of {coverage:.1%} vs target of 80%. The quantile models were trained with
> limited validation data (only ~26 test weeks per store). With more training data or hyperparameter
> tuning (e.g., wider quantile spread), coverage can be improved closer to the 80% target.
""")

with st.expander("📖 Why WAPE is the primary metric (not MAPE)"):
    st.markdown("""
**WAPE (Weighted Absolute Percentage Error)** is preferred over MAPE for retail demand because:

1. **MAPE is undefined or distorted when actuals are near zero** — while not a problem here
   (Walmart stores all have large volumes), it's best practice to avoid it as primary metric.

2. **WAPE weights errors by volume** — a 10% error on a $3M store is more impactful than a 10%
   error on a $200K store. WAPE naturally accounts for this.

3. **MAE is in original scale** — useful for capacity planning (translates to dollar amounts).

4. **RMSE penalises large errors more** — useful for detecting problematic outlier predictions.

Formula: `WAPE = Σ|actual - forecast| / Σ|actual|`
    """)
