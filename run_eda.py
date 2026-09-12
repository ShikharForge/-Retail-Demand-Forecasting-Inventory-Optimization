"""
Phase 2: EDA — Exploratory Data Analysis
Generates purposeful visualisations from the Walmart weekly sales dataset.

Every chart answers a specific business question.
"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from src.data.loader import load_config, load_raw_data
from src.data.preprocessor import clean
from src.features.engineer import build_features

# ── Setup ────────────────────────────────────────────────────────────────────
config = load_config('config/config.yaml')
fig_dir = Path('outputs/figures')
fig_dir.mkdir(parents=True, exist_ok=True)

df_raw = load_raw_data(config)
df_clean = clean(df_raw, config)
df = build_features(df_clean, config)
df['store'] = df['store'].astype('category')
df['date'] = pd.to_datetime(df['date'])

PALETTE = '#2196F3'
HOLIDAY_COLOR = '#FF5722'
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#FAFAFA',
    'axes.grid': True, 'grid.alpha': 0.4, 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11,
})

print(f"Loaded: {df.shape[0]:,} rows, {df['store'].nunique()} stores")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1: Overall Weekly Sales Trend (all stores aggregated)
# Question: Is there a visible upward/downward trend? When are peaks?
# ═══════════════════════════════════════════════════════════════════════════
weekly_agg = df.groupby('date', observed=True)['weekly_sales'].sum().reset_index()
holiday_weeks = df[df['holiday_flag']==1]['date'].unique()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(weekly_agg['date'], weekly_agg['weekly_sales'] / 1e6,
        color=PALETTE, linewidth=1.5, label='Total Weekly Sales')
ax.fill_between(weekly_agg['date'], weekly_agg['weekly_sales'] / 1e6,
                alpha=0.1, color=PALETTE)
# Mark holiday weeks
for hw in holiday_weeks:
    ax.axvline(hw, color=HOLIDAY_COLOR, alpha=0.35, linewidth=1.2)
ax.axvline(holiday_weeks[0], color=HOLIDAY_COLOR, alpha=0.35,
           linewidth=1.2, label='Holiday week')
ax.set_title('Q: Does overall retail demand trend upward, and when are the peaks?',
             fontsize=12, style='italic', color='#555')
ax.set_xlabel('Week')
ax.set_ylabel('Total Sales (all 45 stores, $M)')
ax.legend(loc='upper left')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}M'))
# Annotate the biggest spike
max_row = weekly_agg.loc[weekly_agg['weekly_sales'].idxmax()]
ax.annotate(f"Peak: ${max_row['weekly_sales']/1e6:.1f}M\n{max_row['date'].strftime('%b %Y')}",
            xy=(max_row['date'], max_row['weekly_sales']/1e6),
            xytext=(30, -20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=9, color='black')
plt.tight_layout()
plt.savefig(fig_dir / 'eda_01_overall_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 1 saved: overall trend')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 2: Store Sales Distribution — Box Plot
# Question: How much do stores differ in size? Any outliers?
# ═══════════════════════════════════════════════════════════════════════════
store_order = (df.groupby('store', observed=True)['weekly_sales']
               .median().sort_values(ascending=False).index)

fig, ax = plt.subplots(figsize=(16, 6))
bp_data = [df[df['store']==s]['weekly_sales'].values / 1e6 for s in store_order]
bp = ax.boxplot(bp_data, patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch in bp['boxes']:
    patch.set_facecolor('#2196F3')
    patch.set_alpha(0.7)
ax.set_xticklabels([str(s) for s in store_order], fontsize=8, rotation=45)
ax.set_xlabel('Store (sorted by median sales)')
ax.set_ylabel('Weekly Sales ($M)')
ax.set_title('Q: How different are stores from each other? Which stores dominate?\n'
             'Store 20 is the largest; Stores 33, 36, 44 are smallest',
             fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}M'))
plt.tight_layout()
plt.savefig(fig_dir / 'eda_02_store_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 2 saved: store distribution')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 3: Holiday vs Non-Holiday Sales
# Question: Do holiday weeks actually drive meaningfully higher sales?
# ═══════════════════════════════════════════════════════════════════════════
holiday_sales = df[df['holiday_flag']==1]['weekly_sales'] / 1e6
non_holiday_sales = df[df['holiday_flag']==0]['weekly_sales'] / 1e6

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: violin plot
parts = axes[0].violinplot(
    [non_holiday_sales.values, holiday_sales.values],
    positions=[1, 2], showmedians=True, showextrema=True
)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#90CAF9', HOLIDAY_COLOR][i])
    pc.set_alpha(0.7)
axes[0].set_xticks([1, 2])
axes[0].set_xticklabels(['Non-Holiday', 'Holiday'])
axes[0].set_ylabel('Weekly Sales ($M)')
axes[0].set_title('Sales Distribution by Holiday Status')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}M'))

# Right: by holiday type
holiday_types = {
    'Super Bowl': df[df['is_superbowl']==1]['weekly_sales'].mean() / 1e6,
    'Labour Day': df[df['is_laborday']==1]['weekly_sales'].mean() / 1e6,
    'Thanksgiving': df[df['is_thanksgiving']==1]['weekly_sales'].mean() / 1e6,
    'Christmas': df[df['is_christmas']==1]['weekly_sales'].mean() / 1e6,
    'Non-Holiday': non_holiday_sales.mean(),
}
colors = [HOLIDAY_COLOR]*4 + ['#90CAF9']
bars = axes[1].bar(holiday_types.keys(), holiday_types.values(), color=colors, alpha=0.8)
axes[1].set_ylabel('Mean Weekly Sales ($M)')
axes[1].set_title('Mean Sales by Event Type')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.2f}M'))
for bar, val in zip(bars, holiday_types.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'${val:.2f}M', ha='center', va='bottom', fontsize=9)
plt.suptitle('Q: Which holiday events drive the biggest sales spikes?', 
             fontsize=11, style='italic', color='#555', y=1.02)
plt.tight_layout()
plt.savefig(fig_dir / 'eda_03_holiday_impact.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 3 saved: holiday impact')
for k, v in holiday_types.items():
    print(f'  {k}: ${v:.4f}M avg')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 4: Average Weekly Sales by Week-of-Year (Seasonality)
# Question: Is there a clear annual seasonal pattern?
# ═══════════════════════════════════════════════════════════════════════════
seasonal = df.groupby('week_of_year', observed=True)['weekly_sales'].mean().reset_index()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(seasonal['week_of_year'], seasonal['weekly_sales'] / 1e6,
        color=PALETTE, linewidth=2, marker='o', markersize=3)
ax.fill_between(seasonal['week_of_year'], seasonal['weekly_sales'] / 1e6,
                alpha=0.1, color=PALETTE)
# Annotate key periods
ax.axvspan(6, 7, alpha=0.2, color=HOLIDAY_COLOR, label='Super Bowl')
ax.axvspan(36, 37, alpha=0.2, color='green', label='Labour Day')
ax.axvspan(47, 48, alpha=0.2, color='purple', label='Thanksgiving')
ax.axvspan(52, 52.5, alpha=0.2, color='orange', label='Christmas')
ax.set_xlabel('Week of Year')
ax.set_ylabel('Mean Weekly Sales ($M) — all stores avg')
ax.set_title('Q: What does the annual demand cycle look like across all 45 stores?')
ax.legend(loc='upper left', fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.2f}M'))
plt.tight_layout()
plt.savefig(fig_dir / 'eda_04_seasonality.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 4 saved: seasonality')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 5: Sales Correlation with Economic Variables
# Question: Do macro-economic factors correlate with store sales?
# ═══════════════════════════════════════════════════════════════════════════
econ_vars = ['temperature', 'fuel_price', 'cpi', 'unemployment']
corr_data = {}
for var in econ_vars:
    corr_data[var] = df[['weekly_sales', var]].dropna()

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
for i, var in enumerate(econ_vars):
    data = corr_data[var].sample(min(1000, len(corr_data[var])), random_state=42)
    axes[i].scatter(data[var], data['weekly_sales']/1e6, alpha=0.3, s=8, color=PALETTE)
    # Trend line
    z = np.polyfit(data[var], data['weekly_sales']/1e6, 1)
    p = np.poly1d(z)
    x_line = np.linspace(data[var].min(), data[var].max(), 100)
    axes[i].plot(x_line, p(x_line), 'r-', linewidth=1.5, alpha=0.8)
    from scipy.stats import spearmanr
    r, pval = spearmanr(data[var], data['weekly_sales'])
    axes[i].set_xlabel(var.replace('_', ' ').title())
    axes[i].set_ylabel('Weekly Sales ($M)')
    axes[i].set_title(f'Spearman ρ = {r:.3f} (p={pval:.3f})')
    axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}M'))
    print(f'  {var}: Spearman r={r:.4f}, p={pval:.4f}')
plt.suptitle('Q: Do temperature, fuel price, CPI, or unemployment predict weekly sales?',
             fontsize=11, style='italic', color='#555')
plt.tight_layout()
plt.savefig(fig_dir / 'eda_05_economic_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart 5 saved: economic correlations')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 6: Top 5 vs Bottom 5 Stores
# Question: How extreme is the gap between largest and smallest stores?
# ═══════════════════════════════════════════════════════════════════════════
store_means = df.groupby('store', observed=True)['weekly_sales'].mean().reset_index()
store_means.columns = ['store', 'mean_sales']
store_means = store_means.sort_values('mean_sales', ascending=False)
top5 = store_means.head(5)
bot5 = store_means.tail(5)
highlight = pd.concat([top5, bot5])

fig, ax = plt.subplots(figsize=(10, 5))
colors_bar = ['#1565C0']*5 + ['#B71C1C']*5
ax.barh(highlight['store'].astype(str)[::-1],
        highlight['mean_sales'][::-1] / 1e6,
        color=colors_bar[::-1], alpha=0.85)
ax.set_xlabel('Mean Weekly Sales ($M)')
ax.set_title('Q: How large is the gap between highest and lowest-volume stores?\n'
             'Blue = Top 5, Red = Bottom 5')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}M'))
for i, (_, row) in enumerate(highlight[::-1].iterrows()):
    ax.text(row['mean_sales']/1e6 + 0.01, i,
            f"${row['mean_sales']/1e6:.2f}M", va='center', fontsize=9)
plt.tight_layout()
plt.savefig(fig_dir / 'eda_06_top_bottom_stores.png', dpi=150, bbox_inches='tight')
plt.close()
ratio = top5['mean_sales'].mean() / bot5['mean_sales'].mean()
print(f'Chart 6 saved: top/bottom stores. Top5/Bottom5 ratio: {ratio:.1f}x')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 7: Demand Volatility — Coefficient of Variation by Store
# Question: Which stores have the most unpredictable demand?
#           (High CV = harder to forecast, needs more safety stock)
# ═══════════════════════════════════════════════════════════════════════════
store_stats = df.groupby('store', observed=True)['weekly_sales'].agg(['mean','std']).reset_index()
store_stats['cv'] = store_stats['std'] / store_stats['mean']
store_stats = store_stats.sort_values('cv', ascending=False)

fig, ax = plt.subplots(figsize=(14, 5))
bar_colors = [HOLIDAY_COLOR if cv > store_stats['cv'].quantile(0.75) else PALETTE
              for cv in store_stats['cv']]
ax.bar(store_stats['store'].astype(str), store_stats['cv'] * 100, color=bar_colors, alpha=0.8)
ax.axhline(store_stats['cv'].mean() * 100, color='black', linestyle='--',
           label=f'Mean CV = {store_stats["cv"].mean()*100:.1f}%', linewidth=1.5)
ax.set_xlabel('Store')
ax.set_ylabel('Coefficient of Variation (%)')
ax.set_title('Q: Which stores have the most volatile demand?\n'
             'High CV → harder to forecast → needs larger safety stock')
ax.legend()
plt.tight_layout()
plt.savefig(fig_dir / 'eda_07_demand_volatility.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Chart 7 saved: volatility. Max CV: {store_stats["cv"].max()*100:.1f}% (Store {store_stats.iloc[0]["store"]})')

# ═══════════════════════════════════════════════════════════════════════════
# CHART 8: Lag Correlation (Autocorrelation insight)
# Question: How much does last week's sales predict this week's?
#           This justifies our lag features.
# ═══════════════════════════════════════════════════════════════════════════
if 'lag_1' in df.columns and 'lag_52' in df.columns:
    sample = df[['weekly_sales', 'lag_1', 'lag_4', 'lag_52']].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    lag_pairs = [('lag_1', 'Lag 1 (1 week ago)'),
                 ('lag_4', 'Lag 4 (4 weeks ago)'),
                 ('lag_52', 'Lag 52 (same week last year)')]
    for ax, (lag_col, lag_label) in zip(axes, lag_pairs):
        sample_plot = sample.sample(min(1000, len(sample)), random_state=42)
        ax.scatter(sample_plot[lag_col]/1e6, sample_plot['weekly_sales']/1e6,
                   alpha=0.25, s=8, color=PALETTE)
        r = sample[lag_col].corr(sample['weekly_sales'])
        z = np.polyfit(sample_plot[lag_col], sample_plot['weekly_sales'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sample_plot[lag_col].min(), sample_plot[lag_col].max(), 100)
        ax.plot(x_line/1e6, p(x_line)/1e6, 'r-', linewidth=1.5)
        ax.set_xlabel(f'{lag_label} ($M)')
        ax.set_ylabel('Current Week Sales ($M)')
        ax.set_title(f'r = {r:.3f}')
        print(f'  Pearson r({lag_col}, sales) = {r:.4f}')
    plt.suptitle('Q: How predictive are past sales of current sales? (justifies lag features)',
                 fontsize=11, style='italic', color='#555')
    plt.tight_layout()
    plt.savefig(fig_dir / 'eda_08_lag_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Chart 8 saved: lag correlations')

print('\n=== EDA COMPLETE — All figures saved to outputs/figures/ ===')
