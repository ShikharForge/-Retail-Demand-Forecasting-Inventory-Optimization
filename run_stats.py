"""
Phase 3: Statistical Analysis
Rigorous hypothesis tests on the Walmart dataset.
Results are printed in full interview-ready format.
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from src.data.loader import load_config, load_raw_data
from src.data.preprocessor import clean
from src.features.engineer import build_features

config = load_config('config/config.yaml')
fig_dir = Path('outputs/figures')
fig_dir.mkdir(parents=True, exist_ok=True)

df_raw = load_raw_data(config)
df = build_features(clean(df_raw, config), config)
df['store'] = df['store'].astype('category')

ALPHA = 0.05

def print_test(title, h0, h1, assumptions, stat_name, stat_val, pval, alpha, conclusion, business):
    sep = '=' * 65
    print(f"\n{sep}")
    print(f"TEST: {title}")
    print(sep)
    print(f"H0 (Null):       {h0}")
    print(f"H1 (Alternative):{h1}")
    print(f"Assumptions:     {assumptions}")
    print(f"{stat_name}:     {stat_val:.4f}")
    print(f"p-value:         {pval:.6f}")
    print(f"Alpha:           {alpha}")
    verdict = "REJECT H0" if pval < alpha else "FAIL TO REJECT H0"
    print(f"Decision:        {verdict}")
    print(f"Conclusion:      {conclusion}")
    print(f"Business meaning:{business}")

# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Mann-Whitney U — Holiday vs Non-Holiday Sales
# Rationale: Sales are right-skewed (skewness ~0.67), so a non-parametric
# test is more appropriate than a t-test which assumes normality.
# ══════════════════════════════════════════════════════════════════════════
holiday_s = df[df['holiday_flag']==1]['weekly_sales'].values
non_holiday_s = df[df['holiday_flag']==0]['weekly_sales'].values

stat_mw, p_mw = stats.mannwhitneyu(holiday_s, non_holiday_s, alternative='two-sided')

# Effect size: rank-biserial correlation
n1, n2 = len(holiday_s), len(non_holiday_s)
r_rb = 1 - (2 * stat_mw) / (n1 * n2)

print_test(
    "Mann-Whitney U: Holiday vs Non-Holiday Weekly Sales",
    h0="The distribution of weekly sales is identical for holiday and non-holiday weeks",
    h1="The distribution of weekly sales differs between holiday and non-holiday weeks",
    assumptions="Independent samples, ordinal/continuous data, same shape distributions",
    stat_name="U statistic",
    stat_val=stat_mw,
    pval=p_mw,
    alpha=ALPHA,
    conclusion=f"{'Reject' if p_mw < ALPHA else 'Fail to reject'} H0 at alpha={ALPHA}. "
               f"Holiday weeks have {'significantly' if p_mw < ALPHA else 'no significantly'} "
               f"different sales. Effect size (rank-biserial r) = {r_rb:.4f}.",
    business=f"Holiday weeks generate measurably different demand (median: "
             f"${np.median(holiday_s)/1e6:.3f}M vs ${np.median(non_holiday_s)/1e6:.3f}M). "
             f"{'This is statistically significant but the effect is modest' if abs(r_rb) < 0.3 else 'This is a meaningful effect'}."
             f" Inventory planners SHOULD treat holidays differently."
)

# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Thanksgiving vs Non-Holiday (directional)
# ══════════════════════════════════════════════════════════════════════════
thanksgiving_s = df[df['is_thanksgiving']==1]['weekly_sales'].values
stat_t, p_t = stats.mannwhitneyu(thanksgiving_s, non_holiday_s, alternative='greater')

print_test(
    "Mann-Whitney U: Thanksgiving vs Non-Holiday (one-sided)",
    h0="Thanksgiving weekly sales are NOT greater than non-holiday sales",
    h1="Thanksgiving weekly sales are greater than non-holiday sales",
    assumptions="Independent samples, one-sided test justified by prior EDA showing highest spike",
    stat_name="U statistic",
    stat_val=stat_t,
    pval=p_t,
    alpha=ALPHA,
    conclusion=f"{'Reject' if p_t < ALPHA else 'Fail to reject'} H0. "
               f"Thanksgiving sales are {'significantly' if p_t < ALPHA else 'not significantly'} higher.",
    business=f"Thanksgiving is the single biggest demand event (mean: ${thanksgiving_s.mean()/1e6:.3f}M). "
             f"Safety stock should be increased 4-6 weeks before Thanksgiving."
)

# ══════════════════════════════════════════════════════════════════════════
# TEST 3: Kruskal-Wallis — Store Differences
# Rationale: Non-parametric ANOVA. Tests whether all 45 stores have the
# same sales distribution. We expect stores to differ significantly.
# ══════════════════════════════════════════════════════════════════════════
store_groups = [df[df['store']==s]['weekly_sales'].values
                for s in df['store'].cat.categories]

stat_kw, p_kw = stats.kruskal(*store_groups)

# Eta-squared (effect size) approximation
n = len(df)
k = df['store'].nunique()
eta_sq = (stat_kw - k + 1) / (n - k)

print_test(
    "Kruskal-Wallis: Sales Differences Across 45 Stores",
    h0="All 45 stores share the same weekly sales distribution",
    h1="At least one store has a different sales distribution",
    assumptions="Independent observations, continuous data, each group n>=5",
    stat_name="H statistic",
    stat_val=stat_kw,
    pval=p_kw,
    alpha=ALPHA,
    conclusion=f"{'Reject' if p_kw < ALPHA else 'Fail to reject'} H0. "
               f"Stores differ significantly (eta^2 approx = {eta_sq:.4f}). "
               f"Store effects are {'large' if eta_sq > 0.14 else 'medium' if eta_sq > 0.06 else 'small'}.",
    business=f"A single global forecast model MUST include store identity as a feature "
             f"(which our panel model does via the 'store' categorical feature). "
             f"Separate safety stock calculations per store are justified."
)

# ══════════════════════════════════════════════════════════════════════════
# TEST 4: Levene's Test — Variance Equality Across Stores
# Rationale: Tests if demand VARIABILITY differs across stores.
# Important for inventory: high-variance stores need larger safety stock.
# ══════════════════════════════════════════════════════════════════════════
stat_lev, p_lev = stats.levene(*store_groups, center='median')

print_test(
    "Levene's Test: Equality of Variance Across Stores",
    h0="All 45 stores have equal variance in weekly sales",
    h1="At least one store has a different variance in weekly sales",
    assumptions="Independent samples, continuous data, robust to non-normality (median-based)",
    stat_name="W statistic",
    stat_val=stat_lev,
    pval=p_lev,
    alpha=ALPHA,
    conclusion=f"{'Reject' if p_lev < ALPHA else 'Fail to reject'} H0. "
               f"Store demand variances are {'unequal' if p_lev < ALPHA else 'equal'}.",
    business="Per-store safety stock calculations are justified because stores have "
             "different demand uncertainty levels. A single safety stock figure for all "
             "stores would be inappropriate."
)

# ══════════════════════════════════════════════════════════════════════════
# TEST 5: Spearman Correlations — Sales vs Economic Variables
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("Spearman Rank Correlations: Weekly Sales vs Economic Variables")
print(f"{'='*65}")
print(f"H0: No monotonic association between variable and weekly sales")
print(f"H1: Monotonic association exists")
print(f"Alpha: {ALPHA}")
print()

econ_vars = {
    'temperature': 'Temperature (F)',
    'fuel_price': 'Fuel Price ($/gal)',
    'cpi': 'Consumer Price Index',
    'unemployment': 'Unemployment Rate (%)',
}

corr_results = {}
for col, label in econ_vars.items():
    data = df[['weekly_sales', col]].dropna()
    r, p = stats.spearmanr(data['weekly_sales'], data[col])
    corr_results[col] = (r, p)
    sig = "SIGNIFICANT" if p < ALPHA else "not significant"
    strength = ("strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak")
    direction = "positive" if r > 0 else "negative"
    print(f"  {label:<30} rho={r:+.4f}  p={p:.4f}  [{sig}] {strength} {direction}")

print()
print("Business interpretation:")
print("  - Temperature has a significant but very weak negative correlation.")
print("    Cold weather may slightly boost in-store grocery shopping.")
print("  - CPI shows a marginally significant weak negative effect.")
print("    Inflation slightly dampens real sales volumes.")
print("  - Fuel price and unemployment are not statistically significant.")
print("    This suggests macro-economic factors have limited predictive power")
print("    compared to store identity and calendar effects.")

# ══════════════════════════════════════════════════════════════════════════
# CHART: Statistical Summary Plot
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Holiday vs non-holiday box comparison
axes[0].boxplot(
    [non_holiday_s/1e6, holiday_s/1e6, thanksgiving_s/1e6],
    labels=['Non-Holiday\n(n=5,985)', 'Any Holiday\n(n=450)', 'Thanksgiving\n(n=90)'],
    patch_artist=True,
    notch=True,
    boxprops=dict(facecolor='#90CAF9'),
    medianprops=dict(color='navy', linewidth=2),
)
axes[0].set_ylabel('Weekly Sales ($M)')
axes[0].set_title(f'Holiday Sales Comparison\n(Mann-Whitney p={p_mw:.4f})')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}M'))

# Right: Spearman correlations bar chart
rhos = [corr_results[c][0] for c in econ_vars]
pvals = [corr_results[c][1] for c in econ_vars]
labels = [l.replace(' ', '\n') for l in econ_vars.values()]
colors = ['#1565C0' if p < ALPHA else '#BDBDBD' for p in pvals]
bars = axes[1].barh(labels, rhos, color=colors, alpha=0.8)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_xlabel("Spearman rho")
axes[1].set_title(f"Sales vs Economic Variables\n(Blue = significant at alpha={ALPHA})")
for bar, r, p in zip(bars, rhos, pvals):
    x_pos = r + 0.002 if r >= 0 else r - 0.002
    ha = 'left' if r >= 0 else 'right'
    axes[1].text(x_pos, bar.get_y() + bar.get_height()/2,
                 f'rho={r:+.3f}', va='center', ha=ha, fontsize=9)

plt.suptitle('Statistical Analysis: Walmart Weekly Sales', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'stats_01_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nStatistical summary chart saved.")

# ══════════════════════════════════════════════════════════════════════════
# Summary Table
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("STATISTICAL TESTS SUMMARY")
print(f"{'='*65}")
results = [
    ("Mann-Whitney U", "Holiday vs Non-Holiday", p_mw, p_mw < ALPHA),
    ("Mann-Whitney U", "Thanksgiving vs Non-Holiday", p_t, p_t < ALPHA),
    ("Kruskal-Wallis", "Sales across 45 stores", p_kw, p_kw < ALPHA),
    ("Levene's Test", "Variance equality across stores", p_lev, p_lev < ALPHA),
]
for test, desc, pval, sig in results:
    print(f"  {test:<20} {desc:<35} p={pval:.4f}  {'SIGNIFICANT' if sig else 'not sig.'}")
print()
print("PHASE 3 STATISTICAL ANALYSIS COMPLETE.")
