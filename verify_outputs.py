"""Verify all output data files for structural correctness."""
import sys, pandas as pd
sys.path.insert(0, '.')

print("=== per_store_metrics.csv ===")
ps = pd.read_csv('outputs/forecasts/per_store_metrics.csv', index_col=0)
print(f"Shape: {ps.shape}")
print(f"Index dtype: {ps.index.dtype}")
print(f"First 5 index values: {list(ps.index[:5])}")
print(f"Last 3 index values:  {list(ps.index[-3:])}")
print(f"Columns: {list(ps.columns)}")
non_numeric = [i for i in ps.index if not str(i).lstrip('-').isdigit()]
print(f"Non-numeric index values: {non_numeric}")
print()

print("=== test_forecasts.csv ===")
fc = pd.read_csv('outputs/forecasts/test_forecasts.csv')
print(f"Shape: {fc.shape}")
print(f"Columns: {list(fc.columns)}")
store_col = fc['store']
print(f"Store dtype: {store_col.dtype}")
print(f"Unique stores: {sorted(store_col.unique())[:5]} ...")
print(f"Null forecasts: {fc['forecast'].isna().sum()}")
if 'p10' in fc.columns:
    print(f"Null p10: {fc['p10'].isna().sum()}")
    print(f"Null p90: {fc['p90'].isna().sum()}")
print()

print("=== shap_importance.csv ===")
si = pd.read_csv('outputs/forecasts/shap_importance.csv')
print(f"Shape: {si.shape}")
print(si.head(5).to_string(index=False))
print()

print("=== Output figures ===")
import pathlib
figs = sorted(pathlib.Path('outputs/figures').glob('*.png'))
for f in figs:
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name:<45} {size_kb:>5} KB")

print()
print("=== Model files ===")
for m in sorted(pathlib.Path('outputs/models').glob('*.pkl')):
    size_kb = m.stat().st_size // 1024
    print(f"  {m.name:<30} {size_kb:>5} KB")
