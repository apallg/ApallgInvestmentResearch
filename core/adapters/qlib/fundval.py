import hashlib
import json
import pickle
import ast

import numpy as np
import pandas as pd
from qlib.data.dataset.processor import Processor
from qlib.data.ops import Rolling

from core.shared.fundamental_contract import (
    FUNDVAL_CONTRACT_VERSION, FundamentalDataError, fundamental_dates,
    fundamental_field_contract_sha256,
)


class FundVal(Rolling):
    def __init__(self, feature, field_id, cache_token=None):
        self._field_id = int(field_id)
        self._cache_token = cache_token
        super().__init__(feature, 1, 'mean')

    def __str__(self):
        from core.data.fundamental import get_cache_identity
        identity = get_cache_identity()
        token = f"v{FUNDVAL_CONTRACT_VERSION}:{identity['generation']}:{identity['field_contract_sha256'][:12]}"
        if self._cache_token is not None and self._cache_token != token:
            raise FundamentalDataError('FundVal表达式绑定了旧缓存代号')
        return f'FundVal({self.feature},{self._field_id},{token!r})'

    def _load_internal(self, instrument, start_index, end_index, *args):
        from qlib.data.data import Cal
        from core.data.fundamental import read_fundamental
        freq = args[0] if args else 'day'
        if freq != 'day':
            raise FundamentalDataError('FundVal仅支持日频数据')
        series = self.feature.load(instrument, start_index, end_index, *args)
        if series.empty:
            return pd.Series(dtype=float, index=series.index)
        dates = fundamental_dates(series.index, Cal.calendar(freq=freq))
        values = read_fundamental(self._field_id, instrument, dates)
        return pd.Series(values.to_numpy(), index=series.index)


def feature_contract(factor_keys):
    from core.adapters.qlib.factor_resolver import resolve_factor_expression
    expressions = {}
    for key in factor_keys or []:
        expr, reason = resolve_factor_expression(key)
        if not expr:
            raise FundamentalDataError(f'因子无法解析: {key}: {reason}')
        expressions[key] = expr
    requires_fundval = any('FundVal(' in expr for expr in expressions.values())
    contract = {'version': FUNDVAL_CONTRACT_VERSION, 'expressions': expressions,
                'requires_fundval': requires_fundval}
    if requires_fundval:
        contract['field_contract_sha256'] = fundamental_field_contract_sha256()
    return contract


def current_feature_contract(factor_keys):
    contract = feature_contract(factor_keys)
    if contract['requires_fundval']:
        from qlib.data.data import Cal
        from core.data.fundamental import get_cache_identity
        contract['fundamental_cache'] = get_cache_identity()
        calendar = pd.DatetimeIndex(Cal.calendar(freq='day'))
        contract['calendar_sha256'] = hashlib.sha256(calendar.asi8.tobytes()).hexdigest()
    return contract


def require_model_contract(artifacts, factor_keys):
    current = feature_contract(factor_keys)
    if not current['requires_fundval']:
        return
    path = artifacts / 'feature_contract.json'
    if not path.exists():
        raise FundamentalDataError('模型缺少FundVal输入契约，必须按修复后的特征重新训练')
    try:
        with path.open('rb') as handle:
            stored = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, ValueError):
        stored = path.read_text(encoding='utf-8')
    if isinstance(stored, str):
        stored = json.loads(stored)
    if not isinstance(stored, dict) or any(stored.get(key) != value for key, value in current.items()) or list(stored.get('expressions', {})) != list(current['expressions']):
        raise FundamentalDataError('模型与当前FundVal输入契约不兼容，必须重新训练')


def require_run_contract(artifacts, factor_keys=None):
    path = artifacts / 'factor_keys.json'
    keys = []
    if path.exists():
        try:
            with path.open('rb') as handle:
                raw = pickle.load(handle)
        except (pickle.UnpicklingError, EOFError, ValueError):
            raw = path.read_text(encoding='utf-8')
        info = json.loads(raw) if isinstance(raw, str) else raw
        keys = list(info.get('custom_factor_keys') or [])
    else:
        path = artifacts.parent / 'params/dataset.kwargs.handler.kwargs.factor_keys'
        if path.exists():
            try:
                keys = list(ast.literal_eval(path.read_text(encoding='utf-8')))
            except (ValueError, SyntaxError) as exc:
                raise FundamentalDataError('模型因子清单截断或无法还原') from exc
    if factor_keys is not None and keys and list(factor_keys) != keys:
        raise FundamentalDataError('预测因子清单与训练模型不一致')
    handler_path = artifacts.parent / 'params/dataset.kwargs.handler.class'
    if not keys and handler_path.exists() and 'WhiteBox' in handler_path.read_text(encoding='utf-8'):
        raise FundamentalDataError('白盒模型缺少完整训练因子清单，禁止猜测或替换因子')
    require_model_contract(artifacts, keys or factor_keys or [])
    return keys


def record_dataset_contract(dataset, task_config=None, label_horizon=None, label_mode=None, label_norm=None):
    from qlib.workflow import R
    contract = getattr(dataset.handler, '_feature_contract', None)
    if contract is not None:
        R.save_objects(**{'feature_contract.json': json.dumps(contract, ensure_ascii=False)})
        info = {'factor_set': 'WhiteBoxHandler', 'custom_factor_keys': list(contract['expressions'])}
        if task_config is not None:
            handler = task_config['dataset']['kwargs']['handler']['kwargs']
            info.update({'label_horizon': label_horizon, 'label_mode': label_mode, 'label_norm': label_norm,
                         'shared_processors': handler.get('shared_processors', []),
                         'infer_processors': handler.get('infer_processors', [])})
            for segment, period in task_config['dataset']['kwargs']['segments'].items():
                info[f'{segment}_start'], info[f'{segment}_end'] = period
        R.save_objects(**{'factor_keys.json': json.dumps(info, ensure_ascii=False)})
        report = {key: value for processor in dataset.handler.infer_processors
                  for key, value in getattr(processor, 'report', {}).items()}
        R.save_objects(**{'feature_quality.json': json.dumps(report, ensure_ascii=False)})


def require_dataset_contract(dataset, factor_keys):
    current = current_feature_contract(factor_keys)
    if not current['requires_fundval']:
        return
    stored = getattr(getattr(dataset, 'handler', None), '_feature_contract', None)
    if stored != current or list(stored.get('expressions', {})) != list(current['expressions']):
        raise FundamentalDataError('预建Dataset的基本面版本或日历与当前输入不一致')


class FundamentalQuality(Processor):
    def __init__(self, factor_keys, allow_extra=False):
        self.contract = feature_contract(factor_keys)
        self.report = {}
        self.allow_extra = allow_extra

    def __call__(self, frame):
        features = frame['feature']
        expected = list(self.contract['expressions'])
        actual = [key for key in features.columns if key in expected] if self.allow_extra else list(features.columns)
        if actual != expected:
            raise FundamentalDataError('请求因子清单与实际特征清单不一致')
        for key, expr in self.contract['expressions'].items():
            if 'FundVal(' not in expr:
                continue
            values = features[key].replace([np.inf, -np.inf], np.nan)
            self.report[key] = {'rows': len(values), 'available': int(values.notna().sum()),
                                'coverage': float(values.notna().mean()),
                                'unique': int(values.nunique()),
                                'zero_share': float(values.eq(0).mean())}
            if not values.notna().any():
                raise FundamentalDataError(f'基本面因子在请求区间全部缺失: {key}')
            if isinstance(values.index, pd.MultiIndex) and 'datetime' in values.index.names:
                coverage = values.notna().groupby(level='datetime').mean()
                self.report[key]['minimum_daily_coverage'] = float(coverage.min())
                missing_dates = coverage.index[coverage.eq(0)]
                if len(missing_dates):
                    raise FundamentalDataError(f'基本面因子存在整日缺失，禁止填零继续: {key}: {list(missing_dates[:3])}')
        return frame

    def readonly(self):
        return True

    def is_for_infer(self):
        return True
