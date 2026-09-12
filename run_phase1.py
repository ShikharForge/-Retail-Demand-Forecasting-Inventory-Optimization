"""Phase 1 pipeline validation script."""
import sys, logging, copy
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from src.data.loader import load_config, load_raw_data, validate_raw_data
from src.data.preprocessor import clean, save_processed
from src.features.engineer import build_features, verify_no_leakage

config = load_config('config/config.yaml')
print('Config loaded.')

df_raw = load_raw_data(config)
print(f'Raw data: {df_raw.shape}')

report = validate_raw_data(df_raw, config)
print('=== DATA QUALITY REPORT ===')
print(f'Shape:           {report["shape"]}')
print(f'Null values:     {sum(report["null_counts"].values())}')
print(f'Duplicate rows:  {report["duplicate_rows"]}')
print(f'Dup Store+Date:  {report["duplicate_store_date"]}')
print(f'Date range:      {report["date_range"]}')
print(f'Stores:          {report["store_count"]}')
ss = report['sales_stats']
print(f'Sales min:       ${ss["min"]:,.0f}')
print(f'Sales max:       ${ss["max"]:,.0f}')
print(f'Sales mean:      ${ss["mean"]:,.0f}')
print(f'Sales skewness:  {ss["skewness"]:.3f}')
print(f'Negative sales:  {ss["negative_count"]}')
print(f'Zero sales:      {ss["zero_count"]}')
print(f'Holiday dist:    {report["holiday_distribution"]}')
print()

df_clean = clean(df_raw, config)
print(f'After clean: {df_clean.shape}')

df_feat = build_features(df_clean, config)
print(f'After features: {df_feat.shape}')
print(f'Feature columns ({df_feat.shape[1]}):')
for c in df_feat.columns:
    print(f'  {c}')

errors = 0
for store_id in range(1, 46):
    try:
        verify_no_leakage(df_feat, store_id=store_id)
    except AssertionError as e:
        print(f'LEAKAGE FAIL store {store_id}: {e}')
        errors += 1
print(f'Leakage check: {45-errors}/45 stores PASSED, {errors} FAILED')

feat_config = copy.deepcopy(config)
feat_config['data']['processed_path'] = 'data/processed/walmart_features.parquet'
save_processed(df_feat, feat_config)
print('Feature data saved.')

save_processed(df_clean, config)
print('Clean data saved.')
print('PHASE 1 COMPLETE.')
