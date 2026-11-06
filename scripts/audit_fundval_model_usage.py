import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/experiments/fundval_repair_research_20261026'
factors = json.loads((OUT / 'factors.json').read_text(encoding='utf-8'))
context = json.loads((OUT / 'context/coverage.json').read_text(encoding='utf-8'))
records = []
for path in sorted(OUT.glob('*.checkpoint.pkl')):
    model_path = path.with_name(path.name.replace('.checkpoint.pkl', '.txt'))
    fold, seed, arm = model_path.stem.split('_')
    if arm not in ['A0', 'A1', 'B']:
        continue
    keys = list(factors) + (context['feature_columns'] if arm == 'B' else [])
    model = lgb.Booster(model_str=model_path.read_text(encoding='utf-8'))
    split = model.feature_importance(importance_type='split')
    gain = model.feature_importance(importance_type='gain')
    if len(split) != len(keys):
        raise RuntimeError('模型维数与冻结因子清单不一致')
    for index, key in enumerate(keys):
        if key not in factors or not factors[key]['uses_fund_cache']:
            continue
        records.append({'fold': fold, 'seed': int(seed), 'arm': arm, 'factor': key,
                        'splits': int(split[index]), 'gain': float(gain[index]),
                        'gain_share': float(gain[index] / gain.sum()) if gain.sum() else 0.})
pd.DataFrame(records).to_csv(OUT / 'fundval_model_usage.csv', index=False, encoding='utf-8-sig')
frame = pd.DataFrame(records)
if not frame.empty:
    summary = frame.groupby('arm').agg(models=('factor', lambda values: len(values) // 9),
        total_fundval_splits=('splits', 'sum'), mean_gain_share_per_factor=('gain_share', 'mean'))
    print(summary.to_string())
