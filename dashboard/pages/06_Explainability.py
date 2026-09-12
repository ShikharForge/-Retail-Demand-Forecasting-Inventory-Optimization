import sys
sys.path.insert(0, '.')
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")

@st.cache_data
def load_importance():
    path = 'outputs/forecasts/shap_importance.csv'
    if Path(path).exists():
        return pd.read_csv(path)
    return None

importance = load_importance()

st.title("🔍 Model Explainability (SHAP)")
st.markdown("*Why does the model predict what it predicts?*")

st.warning("""
**⚠️ SHAP shows associations, NOT causation**

SHAP values quantify each feature's *contribution* to a prediction.
High `lag_52` pushing the prediction up means the model learned that
last year's sales are strongly associated with this year's — it does NOT
mean that historical sales *cause* current sales.
For causal analysis, a different methodology (e.g., difference-in-differences, 
causal forests) would be required.
""")

st.divider()

if importance is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Global Feature Importance (Mean |SHAP|)")
        top_n = st.slider("Show top N features", 5, len(importance), 15)
        top = importance.head(top_n)

        fig = go.Figure(go.Bar(
            y=top['feature'][::-1],
            x=top['mean_abs_shap'][::-1],
            orientation='h',
            marker=dict(
                color=top['mean_abs_shap'][::-1],
                colorscale='Blues',
                showscale=False
            ),
            text=[f'${v/1e3:.1f}K' for v in top['mean_abs_shap'][::-1]],
            textposition='outside',
        ))
        fig.update_layout(
            height=500, template='plotly_white',
            xaxis_title='Mean |SHAP Value| (average impact on output, $)',
            title='Feature Importance — averaged across all test predictions',
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Feature Interpretation")
        interpretations = {
            'lag_52': ('Same week last year', 'Dominant: yearly seasonality captured', '🥇'),
            'rolling_mean_4': ('4-week rolling avg', 'Recent demand trend', '🥈'),
            'lag_4': ('4 weeks ago', 'Short-term momentum', '🥉'),
            'store': ('Store identity', 'Store-level baseline demand', '📍'),
            'lag_1': ('1 week ago', 'Very recent sales level', '📊'),
            'lag_8': ('8 weeks ago', 'Medium-term trend', '📊'),
            'rolling_mean_8': ('8-week rolling avg', 'Smoothed demand level', '📊'),
            'unemployment': ('Unemployment rate', 'Macro-economic context', '🌍'),
            'month_sin': ('Month (cyclic)', 'Annual seasonality encoding', '📅'),
            'week_sin': ('Week (cyclic)', 'Weekly seasonality encoding', '📅'),
        }
        for _, row in importance.head(10).iterrows():
            feat = row['feature']
            if feat in interpretations:
                label, desc, icon = interpretations[feat]
                st.markdown(f"**{icon} `{feat}`** — {label}")
                st.caption(f"{desc} (SHAP: ${row['mean_abs_shap']/1e3:.1f}K)")

    # Show saved SHAP plots
    st.divider()
    st.subheader("SHAP Visualisations")
    tab1, tab2, tab3 = st.tabs(["📊 Beeswarm Summary", "📉 Feature Importance", "🎯 Single Prediction (Waterfall)"])

    with tab1:
        shap_path = Path('outputs/figures/shap_beeswarm.png')
        if shap_path.exists():
            st.image(str(shap_path), use_container_width=True)
            st.caption("Each dot = one prediction. Red = high feature value, Blue = low. X-axis = SHAP impact.")
        else:
            st.info("Run `python run_explainability.py` to generate SHAP plots.")

    with tab2:
        imp_path = Path('outputs/figures/shap_feature_importance.png')
        if imp_path.exists():
            st.image(str(imp_path), use_container_width=True)
        else:
            st.info("SHAP feature importance plot not found.")

    with tab3:
        wf_path = Path('outputs/figures/shap_waterfall.png')
        if wf_path.exists():
            st.image(str(wf_path), use_container_width=True)
            st.caption("Waterfall plot for Store 1, first test week. Shows how each feature pushes forecast up/down from model baseline.")
        else:
            st.info("SHAP waterfall not found.")

else:
    st.info("Run `python run_explainability.py` first to generate SHAP data.")

# ── Key insight box ────────────────────────────────────────────────────────
with st.expander("📖 SHAP Key Findings & Business Implications"):
    st.markdown("""
### What SHAP reveals about this model

**1. Lag-52 is the single most powerful feature** (mean SHAP: $217K)
- The same week from last year explains most of the model's predictions.
- Business implication: Year-over-year demand patterns are remarkably stable in retail.
- This means the model effectively learns store-specific seasonal profiles.

**2. Rolling means capture recent momentum** ($162K for 4-week rolling mean)
- The model uses recent trend to adjust predictions around the yearly baseline.
- When a store is trending up/down, the rolling features capture this drift.

**3. Store identity is a strong predictor** ($10K)
- Even after controlling for all other features, store identity matters.
- This validates our decision to use a global panel model with store as a feature
  rather than 45 separate models.

**4. Economic variables have modest impact** (unemployment: $5.5K)
- Temperature, CPI, and unemployment contribute, but far less than demand history.
- This is typical for established retail markets — macro factors explain little
  of the week-to-week variation once seasonality and trends are accounted for.

### Limitations
- SHAP does not replace domain knowledge.
- Features can be correlated (lag_52 and lag_4 both capture trend), making
  individual attributions partially redundant.
- SHAP computed on test data only — behaviour may differ on out-of-time samples.
    """)
