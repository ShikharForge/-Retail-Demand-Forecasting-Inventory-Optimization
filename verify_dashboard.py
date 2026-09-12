"""
Dry-run verification of all dashboard page logic.
Imports each page's data-loading and computation code outside Streamlit
to catch any runtime errors before the user sees them.
"""
import sys, pathlib, traceback
sys.path.insert(0, '.')

import pandas as pd
import numpy as np

from src.data.loader import load_config, load_raw_data
from src.data.preprocessor import clean
from src.features.engineer import build_features
from src.models.trainer import chronological_split
from src.inventory.optimizer import InventoryParameters, optimise
from src.scenarios.scenario_engine import all_scenarios

config = load_config('config/config.yaml')
df_raw = load_raw_data(config)
df = build_features(clean(df_raw, config), config)
df['store'] = df['store'].astype('category')
_, _, test_df = chronological_split(df, config)
fc = pd.read_csv('outputs/forecasts/test_forecasts.csv', parse_dates=['date'])
per_store = pd.read_csv('outputs/forecasts/per_store_metrics.csv', index_col=0)
shap_imp = pd.read_csv('outputs/forecasts/shap_importance.csv')

PASS = []
FAIL = []

def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()

# ── Page 1: Executive Overview ────────────────────────────────────────────
def page1():
    total = df['weekly_sales'].sum()
    avg_weekly = df.groupby('date', observed=True)['weekly_sales'].sum().mean()
    peak = df.groupby('date', observed=True)['weekly_sales'].sum().max()
    uplift = (df[df['holiday_flag']==1]['weekly_sales'].mean() /
              df[df['holiday_flag']==0]['weekly_sales'].mean() - 1) * 100
    assert total > 0
    assert avg_weekly > 0
    assert peak > avg_weekly
    assert 0 < uplift < 100
    # Events by type
    events = {
        'Thanksgiving': df[df['is_thanksgiving']==1]['weekly_sales'].mean(),
        'Christmas':    df[df['is_christmas']==1]['weekly_sales'].mean(),
    }
    assert events['Thanksgiving'] > events['Christmas']
    # Store stats
    store_stats = df.groupby('store', observed=True).agg(
        mean_sales=('weekly_sales','mean'),
        cv=('weekly_sales', lambda x: x.std()/x.mean())
    ).reset_index()
    assert len(store_stats) == 45

check("Page 1 — Executive Overview", page1)

# ── Page 2: Demand Forecast ───────────────────────────────────────────────
def page2():
    for store_id in [1, 20, 45]:
        hist = df[df['store'].astype(int)==store_id]
        fc_store = fc[fc['store'].astype(int)==store_id]
        assert len(hist) > 0, f"No history for store {store_id}"
        assert len(fc_store) > 0, f"No forecast for store {store_id}"
        assert 'p10' in fc_store.columns
        assert 'p90' in fc_store.columns
        assert 'forecast' in fc_store.columns
        # Check P10 < P50 < P90 on average
        assert fc_store['p10'].mean() <= fc_store['p50'].mean()
        assert fc_store['p50'].mean() <= fc_store['p90'].mean()
        # Error metrics
        merged = pd.merge(
            test_df[test_df['store'].astype(int)==store_id][['date','weekly_sales']],
            fc_store[['date','forecast']],
            on='date', how='inner'
        )
        if len(merged) > 0:
            error = abs(merged['weekly_sales'] - merged['forecast'])
            wape_val = error.sum() / merged['weekly_sales'].sum()
            assert 0 < wape_val < 1, f"Store {store_id} WAPE out of range: {wape_val}"

check("Page 2 — Demand Forecast", page2)

# ── Page 3: Inventory Optimization ───────────────────────────────────────
def page3():
    for store_id in [1, 20, 45]:
        fc_store = fc[fc['store'].astype(int)==store_id]
        hist = df[df['store'].astype(int)==store_id]
        p50 = fc_store['p50'].mean() if len(fc_store) else hist['weekly_sales'].mean()
        p90 = fc_store['p90'].mean() if len(fc_store) else p50 * 1.15
        std = hist['weekly_sales'].std()
        params = InventoryParameters(lead_time_weeks=2, service_level=0.95,
                                     current_inventory_usd=2_000_000)
        d = optimise(store_id, p50, p90, std, params, use_quantile_safety_stock=True)
        assert d.reorder_point >= d.lead_time_demand
        assert d.reorder_point >= d.safety_stock
        assert 0 <= d.stockout_risk <= 1
        assert d.eoq > 0
        assert d.weeks_of_coverage > 0
        # Parametric variant
        d2 = optimise(store_id, p50, p90, std, params, use_quantile_safety_stock=False)
        assert d2.reorder_point > 0

check("Page 3 — Inventory Optimization", page3)

# ── Page 4: Scenario Analysis ──────────────────────────────────────────────
def page4():
    store_id = 1
    fc_store = fc[fc['store'].astype(int)==store_id]
    hist = df[df['store'].astype(int)==store_id]
    p50 = fc_store['p50'].mean()
    p90 = fc_store['p90'].mean()
    std = hist['weekly_sales'].std()
    params = InventoryParameters(lead_time_weeks=2, service_level=0.95,
                                 current_inventory_usd=2_000_000)
    results = all_scenarios(store_id, p50, p90, std, params, config)
    assert len(results) == 5, f"Expected 5 scenarios, got {len(results)}"
    for r in results:
        assert r.scenario.safety_stock >= 0
        assert r.scenario.reorder_point >= 0
        delta = r.delta()
        assert 'safety_stock' in delta
        assert 'reorder_point' in delta
    # Demand surge should increase ROP
    surge = next(r for r in results if 'Surge' in r.scenario_name)
    assert surge.scenario.reorder_point > surge.base.reorder_point
    # Lead time doubling should increase SS (sqrt(2) factor)
    lt = next(r for r in results if 'Lead Time' in r.scenario_name)
    assert lt.scenario.safety_stock > lt.base.safety_stock

check("Page 4 — Scenario Analysis", page4)

# ── Page 5: Model Performance ─────────────────────────────────────────────
def page5():
    # Simulate the fixed store-cast logic
    ps = per_store.reset_index().rename(columns={'index':'store'})
    ps = ps[pd.to_numeric(ps['store'], errors='coerce').notna()].copy()
    ps['store'] = ps['store'].astype(int)
    assert len(ps) == 45, f"Expected 45 stores, got {len(ps)}"
    assert ps['store'].dtype == int or ps['store'].dtype == np.int64
    assert 'wape' in ps.columns
    assert ps['wape'].between(0, 1).all()
    # Actual vs forecast merge
    merged = pd.merge(
        test_df[['store','date','weekly_sales']],
        fc[['store','date','forecast','p10','p90']],
        on=['store','date'], how='inner'
    ).dropna(subset=['forecast'])
    assert len(merged) > 0
    # Coverage
    coverage = ((merged['weekly_sales'] >= merged['p10']) &
                (merged['weekly_sales'] <= merged['p90'])).mean()
    assert 0 < coverage <= 1, f"Coverage out of range: {coverage}"
    print(f"      Quantile coverage: {coverage:.3f}")

check("Page 5 — Model Performance", page5)

# ── Page 6: Explainability ────────────────────────────────────────────────
def page6():
    assert len(shap_imp) > 0
    assert 'feature' in shap_imp.columns
    assert 'mean_abs_shap' in shap_imp.columns
    assert shap_imp['mean_abs_shap'].iloc[0] >= shap_imp['mean_abs_shap'].iloc[-1]  # sorted desc
    # All SHAP figures exist
    import pathlib
    for fig in ['shap_beeswarm.png', 'shap_feature_importance.png', 'shap_waterfall.png']:
        p = pathlib.Path(f'outputs/figures/{fig}')
        assert p.exists(), f"Missing: {fig}"
        assert p.stat().st_size > 10_000, f"Suspiciously small file: {fig}"
    # use_column_width no longer present
    src = pathlib.Path('dashboard/pages/06_Explainability.py').read_text(encoding='utf-8')
    assert 'use_column_width' not in src, "Deprecated use_column_width still present"
    assert 'use_container_width' in src

check("Page 6 — Explainability", page6)

# ── Dashboard syntax ──────────────────────────────────────────────────────
def page_syntax():
    import pathlib
    errors = []
    for p in sorted(pathlib.Path('dashboard').rglob('*.py')):
        src = p.read_text(encoding='utf-8')
        try:
            compile(src, str(p), 'exec')
        except SyntaxError as e:
            errors.append(f'{p.name}: {e}')
    assert not errors, '\n'.join(errors)

check("All dashboard files — syntax", page_syntax)

# ── Summary ───────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"VERIFICATION SUMMARY")
print("=" * 55)
print(f"  PASS: {len(PASS)}/{len(PASS)+len(FAIL)}")
for p in PASS:
    print(f"    OK  {p}")
if FAIL:
    print(f"  FAIL: {len(FAIL)}")
    for f in FAIL:
        print(f"    !!  {f}")
else:
    print()
    print("  All checks passed. Dashboard is ready.")
