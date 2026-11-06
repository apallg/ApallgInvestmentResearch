import gc
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from core.data.fundamental import get_cache, get_cache_identity

OUT = ROOT / 'data/experiments/fundval_repair_research_20261026'
provider = ROOT / 'data/qlib_data_prewrite_20261025'
identity = get_cache_identity()
cache = get_cache()
calendar = pd.DatetimeIndex(pd.to_datetime((provider / 'calendars/day.txt').read_text(encoding='utf-8').splitlines()))
factors = json.loads((OUT / 'factors.json').read_text(encoding='utf-8'))
keys = [key for key, value in factors.items() if value['uses_fund_cache']]
records = []
for year in [2019, 2021, 2023, 2025]:
    target = OUT / f'full_reference_{year}.json'
    if target.exists():
        record = json.loads(target.read_text(encoding='utf-8'))
        if record['identity'] != identity:
            raise RuntimeError('完整参考核对的缓存代号已改变')
        records.append(record)
        continue
    frame = pd.read_parquet(OUT / f'new_{year}.parquet', columns=keys).swaplevel().sort_index()
    checked = 0
    for instrument, part in frame.groupby(level='instrument', sort=False):
        part = part.droplevel('instrument')
        path = provider / 'features' / instrument / 'close.day.bin'
        if path.exists() and path.stat().st_size >= 4:
            start = int(np.fromfile(path, dtype='<f4', count=1)[0])
            dates = calendar[start:start + path.stat().st_size // 4 - 1]
        else:
            dates = pd.DatetimeIndex([])
        for key in keys:
            field = 'real_fund_turnover_rate' if key == 'fund_turnover' else 'real_fund_total_mv'
            try:
                values = cache[field].xs(instrument[2:], level='code').reindex(dates)
            except KeyError:
                values = pd.Series(np.nan, index=dates)
            if key != 'fund_turnover':
                window = int(re.search(r',\s*(\d+)\)$', factors[key]['expression']).group(1))
                values = values.diff().rolling(window, min_periods=1).std()
            expected = values.reindex(part.index).astype(np.float32)
            try:
                np.testing.assert_allclose(part[key], expected, rtol=1e-5, atol=1e-7, equal_nan=True)
            except AssertionError as exc:
                raise RuntimeError(f'逐值参考不一致: {year} {instrument} {key}') from exc
            checked += len(part)
    record = {'passed': True, 'year': year, 'rows': len(frame), 'cells_checked': checked,
              'identity': identity, 'rtol': 1e-5, 'atol': 1e-7}
    target.write_text(json.dumps(record, indent=2), encoding='utf-8')
    records.append(record)
    print(json.dumps(record), flush=True)
    del frame
    gc.collect()
if get_cache_identity() != identity:
    raise RuntimeError('参考核对过程中基本面缓存发生变化')
(OUT / 'full_reference_verification.json').write_text(json.dumps({'passed': True,
    'cells_checked': sum(record['cells_checked'] for record in records), 'parts': records}, indent=2), encoding='utf-8')
