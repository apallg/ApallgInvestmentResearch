import gc
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('POLARS_SKIP_CPU_CHECK', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN

from core.data.fundamental import get_cache_identity, get_cache

BASE = ROOT / 'data/experiments/repair_group_research_20261025'
OUT = ROOT / 'data/experiments/fundval_repair_research_20261026'
OUT.mkdir(parents=True, exist_ok=True)
identity = get_cache_identity()
identity_path = OUT / 'input_identity.json'
if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8')) != identity:
    raise RuntimeError('修复实验输入缓存代号已改变')
identity_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
for name in ['factors.json', 'preflight.json', 'universe.json']:
    if not (OUT / name).exists():
        shutil.copy2(BASE / name, OUT / name)
(OUT / 'context').mkdir(exist_ok=True)
for name in ['context_features.parquet', 'coverage.json']:
    if not (OUT / 'context' / name).exists():
        shutil.copy2(BASE / 'context' / name, OUT / 'context' / name)
factors = json.loads((BASE / 'factors.json').read_text(encoding='utf-8'))
keys = [key for key, value in factors.items() if value['uses_fund_cache']]
from core.adapters.qlib.factor_resolver import resolve_factor_expression
for key in keys:
    expression, reason = resolve_factor_expression(key)
    if not expression:
        raise RuntimeError(f'修复后因子无法解析: {key}: {reason}')
    factors[key]['expression'] = expression
(OUT / 'factors.json').write_text(json.dumps(factors, ensure_ascii=False, indent=2), encoding='utf-8')
provider = ROOT / 'data/qlib_data_prewrite_20261025'
calendar = pd.DatetimeIndex(pd.to_datetime((provider / 'calendars/day.txt').read_text(encoding='utf-8').splitlines()))
instruments = {code: [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in intervals]
               for code, intervals in json.loads((BASE / 'universe.json').read_text(encoding='utf-8')).items()}
qlib.init(provider_uri=str(provider), region=REG_CN, kernels=2, joblib_backend='threading',
          expression_cache=None, dataset_cache=None)
sys.excepthook = sys.__excepthook__
from qlib.data import D
from core.adapters.qlib.custom_handler import _register_fundval

_register_fundval()
parts = []
for first, last in [('2019-01-01', '2020-12-31'), ('2021-01-01', '2022-12-31'),
                    ('2023-01-01', '2024-12-31'), ('2025-01-01', '2026-10-23')]:
    path = OUT / f'new_{first[:4]}.parquet'
    audit_path = OUT / f'verification_{first[:4]}.json'
    if path.exists() and audit_path.exists():
        parts.append(json.loads(audit_path.read_text(encoding='utf-8')))
        continue
    print(json.dumps({'stage': 'extract_repaired', 'first': first, 'last': last}), flush=True)
    repaired = D.features(instruments, [factors[key]['expression'] for key in keys], start_time=first, end_time=last, freq='day')
    repaired.columns = keys
    repaired = repaired.reorder_levels(['datetime', 'instrument']).sort_index().astype(np.float32)
    baseline = pd.read_parquet(BASE / f'new_{first[:4]}.parquet')
    if not repaired.index.equals(baseline.index):
        raise RuntimeError('修复前后股票日索引不一致')
    counts = {key: {'available': int(repaired[key].notna().sum()),
                    'nonzero': int((repaired[key].notna() & repaired[key].ne(0)).sum())} for key in keys}
    if any(value['nonzero'] == 0 for value in counts.values()):
        raise RuntimeError('修复后仍存在全零因子')
    checked = 0
    for instrument in sorted(set(repaired.index.get_level_values('instrument')))[::53]:
        stock = repaired.xs(instrument, level='instrument')
        bin_path = provider / 'features' / instrument / 'close.day.bin'
        start = int(np.fromfile(bin_path, dtype='<f4', count=1)[0])
        count = bin_path.stat().st_size // 4 - 1
        dates = calendar[start:start + count]
        cache = get_cache()
        for key in keys:
            expr = factors[key]['expression']
            field = 'real_fund_turnover_rate' if key == 'fund_turnover' else 'real_fund_total_mv'
            try:
                source = cache[field].xs(instrument[2:], level='code').reindex(dates)
            except KeyError:
                source = pd.Series(np.nan, index=dates)
            if key != 'fund_turnover':
                window = int(re.search(r',\s*(\d+)\)$', expr).group(1))
                source = source.diff().rolling(window, min_periods=1).std()
            expected = source.reindex(stock.index).astype(np.float32)
            np.testing.assert_allclose(stock[key], expected, rtol=1e-5, atol=1e-7, equal_nan=True)
            checked += len(stock)
    baseline.loc[:, keys] = repaired
    temporary = path.with_suffix('.tmp')
    baseline.to_parquet(temporary, compression='zstd')
    temporary.replace(path)
    shutil.copy2(BASE / f'old_{first[:4]}.parquet', OUT / f'old_{first[:4]}.parquet')
    record = {'first': first, 'rows': len(repaired), 'counts': counts, 'direct_reference_cells_checked': checked,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    audit_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    parts.append(record)
    print(json.dumps({'stage': 'verified', **record}), flush=True)
    del baseline, repaired
    gc.collect()
if get_cache_identity() != identity:
    raise RuntimeError('提取过程中基本面缓存发生变化')
(OUT / 'repair_verification.json').write_text(json.dumps({'passed': True, 'identity': identity,
    'scope': 'FundVal code/calendar/missing repair; fixed quote inputs and labels; not PIT certified',
    'parts': parts}, indent=2), encoding='utf-8')
for generation in ['old', 'new']:
    (OUT / f'{generation}_extraction_complete.json').write_text(json.dumps({'research_only': True}), encoding='utf-8')
audit = json.loads((BASE / 'input_audit.json').read_text(encoding='utf-8'))
audit['scope'] = 'FundVal v2 integration and cache missing-value repair versus frozen broken-feature baseline; not complete October repair'
(OUT / 'input_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
