import datetime
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.shared.fundamental_contract import (
    FundamentalDataError, fundamental_code, fundamental_dates, align_fundamental,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('value,expected', [
    ('sz000001', '000001'), ('SZ000001', '000001'), ('000001.SZ', '000001'),
    ('sh600000', '600000'), ('600000.SH', '600000'), ('600000', '600000'),
    ('bj920001', '920001'), ('430047.BJ', '430047'),
])
def test_security_codes_preserve_identity(value, expected):
    assert fundamental_code(value) == expected


@pytest.mark.parametrize('value', ['sh000001', 'sz600000', '000300.SH', '1', None, '600000.SZ', '../600000'])
def test_invalid_codes_and_index_stock_collisions_fail(value):
    with pytest.raises(FundamentalDataError):
        fundamental_code(value)


def test_calendar_mapping_uses_actual_truncated_indices():
    calendar = pd.to_datetime(['2024-02-08', '2024-02-19', '2024-02-20', '2024-02-21'])
    pd.testing.assert_index_equal(fundamental_dates(pd.Index([1, 3]), calendar), calendar.take([1, 3]))
    with pytest.raises(FundamentalDataError):
        fundamental_dates(pd.Index([4]), calendar)
    with pytest.raises(FundamentalDataError):
        fundamental_dates(pd.Index([1]))


def test_exact_daily_lookup_preserves_zero_and_missing_without_stale_fill():
    dates = pd.bdate_range('2024-01-01', periods=4)
    source = pd.Series([0., 2.], index=dates[[0, 2]])
    actual = align_fundamental(source, dates)
    np.testing.assert_allclose(actual, [0., np.nan, 2., np.nan], equal_nan=True)


def test_gross_margin_field_reads_the_explicit_single_quarter_feature(fund_cache):
    fund, dates = fund_cache
    actual = fund.read_fundamental(10, 'sz000001', dates)
    np.testing.assert_allclose(actual, [0.1, 0.2, 0.3, 0.4])


def test_fundamental_qlib_field_ids_match_shared_contract():
    from core.data.fundamental import FUND_FIELDS
    from core.shared.fundamental_contract import FUNDAMENTAL_QLIB_FIELDS

    assert {
        field_id: ("fundamental_data", column, label, category)
        for field_id, column, label, category in FUNDAMENTAL_QLIB_FIELDS
    } == {field_id: FUND_FIELDS[field_id] for field_id, *_ in FUNDAMENTAL_QLIB_FIELDS}


def test_fundamental_semantics_manifest_marks_uncertified_dimensions():
    from core.shared.fundamental_contract import FUNDAMENTAL_FIELD_SEMANTICS, fundamental_field_contract_sha256

    assert FUNDAMENTAL_FIELD_SEMANTICS["q_gsprofit_margin"]["period_basis"] == "single_quarter"
    assert FUNDAMENTAL_FIELD_SEMANTICS["q_gsprofit_margin"]["unit_basis"] == "unverified"
    assert FUNDAMENTAL_FIELD_SEMANTICS["roe"]["period_basis"] == "unverified"
    assert len(fundamental_field_contract_sha256()) == 64


@pytest.fixture
def fund_cache(monkeypatch):
    from core.data import fundamental as fund
    dates = pd.bdate_range('2024-01-01', periods=4)
    frame = pd.DataFrame({'code': ['000001'] * 4, 'trade_date': dates,
                          'real_fund_total_mv': [-3., -2., np.nan, 0.],
                          'real_fund_turnover_rate': [0.1, 0.2, np.nan, 0.],
                          'real_fund_q_gsprofit_margin': [0.1, 0.2, 0.3, 0.4]})
    for key in ['_cache', '_stock_cache', '_cache_loaded', '_cache_generation']:
        monkeypatch.setattr(fund, key, getattr(fund, key))
    fund._install_cache(frame, {'generation': 'test'})
    monkeypatch.setattr(fund, 'preload_cache', lambda **kwargs: True)
    return fund, dates


@pytest.mark.p0
def test_real_fundval_operator_uses_cache_and_restores_qlib_index(fund_cache, monkeypatch):
    from core.adapters.qlib.fundval import FundVal
    fund, dates = fund_cache
    monkeypatch.setattr('qlib.data.data.Cal', SimpleNamespace(calendar=lambda **kwargs: dates))

    class Feature:
        def load(self, *args):
            return pd.Series([10., 10., 10.], index=pd.RangeIndex(1, 4))

    actual = FundVal(Feature(), 1)._load_internal('sz000001', 0, 3, 'day')
    pd.testing.assert_index_equal(actual.index, pd.RangeIndex(1, 4))
    np.testing.assert_allclose(actual, [-2., np.nan, 0.], equal_nan=True)
    with pytest.raises(FundamentalDataError):
        FundVal(Feature(), 500)._load_internal('sz000001', 0, 3, 'day')
    with pytest.raises(FundamentalDataError):
        FundVal(Feature(), 1)._load_internal('sz000001', 0, 3, '1min')


def test_scalar_and_panel_evaluators_share_fundval_semantics(fund_cache):
    from core.domain.factors.expression import _Ev, _PanelEv
    fund, dates = fund_cache
    frame = pd.DataFrame({'close': np.ones(4)}, index=dates)
    scalar = _Ev(frame, code='sz000001')._eval_fundval([None, ('num', 1)])
    np.testing.assert_allclose(scalar, [-3., -2., np.nan, 0.], equal_nan=True)
    panel = _PanelEv({'close': frame[['close']]}, dates, ['sz000001'])._eval_fundval([None, ('num', 1)])
    np.testing.assert_allclose(np.asarray(panel).reshape(-1), scalar, equal_nan=True)


def test_loading_failure_does_not_mark_adapter_ready(monkeypatch):
    from core.adapters.qlib import custom_handler
    monkeypatch.setattr('core.data.fundamental.preload_cache', lambda: False)
    monkeypatch.setattr(custom_handler, '_FUND_CACHE_LOADED', False)
    with pytest.raises(FundamentalDataError):
        custom_handler._ensure_fund_cache()
    assert custom_handler._FUND_CACHE_LOADED is False


def test_source_missing_stays_missing_after_standardization():
    from core.data.fundamental import _compute_zscores
    frame = pd.DataFrame({'trade_date': [pd.Timestamp('2024-01-01')] * 12,
                          'turnover_rate': [0.] * 11 + [np.nan]})
    actual = _compute_zscores(frame, ['turnover_rate'])
    assert actual.turnover_rate_z.iloc[:11].eq(0).all()
    assert pd.isna(actual.turnover_rate_z.iloc[11])


def test_feature_quality_runs_before_fill_and_rejects_replaced_columns():
    from core.adapters.qlib.fundval import FundamentalQuality
    validator = FundamentalQuality(['fund_turnover'])
    frame = pd.concat({'feature': pd.DataFrame({'fund_turnover': [np.nan, np.nan]})}, axis=1)
    with pytest.raises(FundamentalDataError, match='全部缺失'):
        validator(frame)
    frame[('feature', 'fund_turnover')] = [0., 0.]
    assert validator(frame) is frame
    with pytest.raises(FundamentalDataError, match='清单'):
        validator(frame.rename(columns={'fund_turnover': 'CLOSE0'}))


def test_legacy_and_incompatible_models_cannot_use_corrected_features(tmp_path):
    from core.adapters.qlib.fundval import feature_contract, require_model_contract
    with pytest.raises(FundamentalDataError, match='缺少'):
        require_model_contract(tmp_path, ['fund_turnover'])
    info = feature_contract(['fund_turnover'])
    (tmp_path / 'feature_contract.json').write_text(json.dumps(info), encoding='utf-8')
    require_model_contract(tmp_path, ['fund_turnover'])
    info['field_contract_sha256'] = '0' * 64
    (tmp_path / 'feature_contract.json').write_text(json.dumps(info), encoding='utf-8')
    with pytest.raises(FundamentalDataError, match='不兼容'):
        require_model_contract(tmp_path, ['fund_turnover'])
    info['field_contract_sha256'] = feature_contract(['fund_turnover'])['field_contract_sha256']
    info['version'] = 4
    (tmp_path / 'feature_contract.json').write_text(json.dumps(info), encoding='utf-8')
    with pytest.raises(FundamentalDataError, match='不兼容'):
        require_model_contract(tmp_path, ['fund_turnover'])
    info['version'] = 1
    (tmp_path / 'feature_contract.json').write_bytes(pickle.dumps(info))
    with pytest.raises(FundamentalDataError, match='不兼容'):
        require_model_contract(tmp_path, ['fund_turnover'])
    require_model_contract(tmp_path, ['a158_macd'])


def test_dataset_cache_key_changes_with_fundamental_generation(monkeypatch):
    from core.adapters.qlib.config import _make_dataset_cache_key
    current = {'generation': 'one', 'field_contract_sha256': 'a' * 64}
    monkeypatch.setattr('core.adapters.qlib.fundval.current_feature_contract', lambda keys: dict(current))
    config = {'kwargs': {'handler': {'kwargs': {'factor_keys': ['fund_turnover']}}}}
    first = _make_dataset_cache_key(config)
    current['generation'] = 'two'
    assert first != _make_dataset_cache_key(config)


def test_prediction_cache_key_changes_with_feature_contract():
    from core.adapters.qlib.lstm_fusion import _prediction_cache_key
    args = ('exp', 'run', '2024-01-01', '2024-02-01', 'all', ['fund_turnover'], 20, .5, 'art', 'data')
    assert _prediction_cache_key(*args, feature_signature={'generation': 'one'}) != _prediction_cache_key(*args, feature_signature={'generation': 'two'})


def test_external_dataset_must_match_current_feature_generation(monkeypatch):
    from core.adapters.qlib.fundval import require_dataset_contract
    current = {'requires_fundval': True, 'generation': 'two'}
    monkeypatch.setattr('core.adapters.qlib.fundval.current_feature_contract', lambda keys: current)
    dataset = SimpleNamespace(handler=SimpleNamespace(_feature_contract={'requires_fundval': True, 'generation': 'one'}))
    with pytest.raises(FundamentalDataError, match='版本'):
        require_dataset_contract(dataset, ['fund_turnover'])


def test_future_reports_do_not_change_historical_daily_values():
    from core.data.fundamental import _forward_fill_fundamental, FUND_FIELDS
    dates = pd.bdate_range('2024-01-01', periods=5)
    reports = pd.DataFrame({'code': ['000001', '000001'],
                            'report_date': [pd.Timestamp('2023-09-30'), pd.Timestamp('2023-12-31')],
                            'ann_date': [pd.Timestamp('2023-10-20'), pd.Timestamp('2024-01-20')],
                            'source_observed_at': [pd.Timestamp('2023-10-21'), pd.Timestamp('2024-01-21')],
                            'avail_date': [dates[1], pd.Timestamp('2025-01-01')]})
    for table, column, *_ in FUND_FIELDS.values():
        if table == 'fundamental_data':
            reports[column] = [1., 999.]
    reports['grossprofit_margin'] = [1., 999.]
    original = _forward_fill_fundamental(reports, dates)
    reports.loc[1, 'roe'] = -1000
    reports.loc[1, 'q_gsprofit_margin'] = -1000
    changed = _forward_fill_fundamental(reports, dates)
    pd.testing.assert_frame_equal(original, changed)
    assert original.trade_date.min() == dates[1]


@pytest.mark.p0
def test_generation_write_is_atomic_and_preserves_missing(tmp_path, monkeypatch):
    from core.data import fundamental as fund
    from core.data import Database
    for key in ['_cache', '_stock_cache', '_cache_loaded', '_cache_generation', '_identity', '_identity_signature']:
        monkeypatch.setattr(fund, key, getattr(fund, key))
    monkeypatch.setattr(fund, 'CACHE_DIR', tmp_path)
    monkeypatch.setattr(Database, 'is_available', lambda: True)
    monkeypatch.setattr(Database, 'fetchall', lambda query, *args: [{'trade_date': pd.Timestamp('2024-01-02')}] if 'DISTINCT' in query else [{'first_date': pd.Timestamp('2024-01-02'), 'last_date': pd.Timestamp('2024-01-02')}])
    basic = pd.DataFrame({'code': ['000001', '600000'], 'trade_date': [pd.Timestamp('2024-01-02')] * 2,
                          'total_mv': [100., 200.], 'circ_mv': [90., 180.],
                          'pe': [10., np.nan], 'pb': [1., 2.], 'turnover_rate': [0., np.nan]})
    monkeypatch.setattr(fund, '_load_daily_basic', lambda *args: basic.copy())
    monkeypatch.setattr(fund, '_load_fundamental_data', lambda: pd.DataFrame())
    assert fund.preload_cache(force=True)
    first = fund.get_cache_identity()
    with (tmp_path / 'generations' / first['generation'] / 'fund_meta.pkl').open('rb') as handle:
        metadata = pickle.load(handle)
        assert metadata['contract_version'] == 6
        assert metadata['field_contract_sha256'] == first['field_contract_sha256']
        assert metadata['field_semantics']['q_gsprofit_margin']['period_basis'] == 'single_quarter'
    assert fund.read_fundamental(9, 'sz000001', pd.DatetimeIndex(['2024-01-02'])).iloc[0] == 0
    assert pd.isna(fund.read_fundamental(9, 'sh600000', pd.DatetimeIndex(['2024-01-02'])).iloc[0])
    monkeypatch.setattr(fund.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('publish failed')))
    assert not fund.preload_cache(force=True)
    assert fund.get_cache_identity() == first
    assert fund._cache_generation == first['generation']
    assert fund.preload_cache()
    assert fund._cache_generation == first['generation']


def test_legacy_fundval_cache_contract_is_rejected(tmp_path, monkeypatch):
    from core.data import fundamental as fund
    for key in ['_identity', '_identity_signature', '_cache_loaded', '_cache_generation']:
        monkeypatch.setattr(fund, key, getattr(fund, key))
    monkeypatch.setattr(fund, 'CACHE_DIR', tmp_path)
    legacy = tmp_path / 'generations' / 'legacy-v4'
    legacy.mkdir(parents=True)
    with (legacy / 'fund_meta.pkl').open('wb') as handle:
        pickle.dump({'contract_version': 4, 'generation': 'legacy-v4', 'data_sha256': '0' * 64}, handle)
    (tmp_path / 'current.json').write_text(json.dumps({'generation': 'legacy-v4'}), encoding='utf-8')

    with pytest.raises(FundamentalDataError, match='契约或代号不一致'):
        fund.get_cache_identity()
    assert fund.preload_cache() is False


def test_cache_without_current_field_contract_fingerprint_is_rejected(tmp_path, monkeypatch):
    from core.data import fundamental as fund
    for key in ['_identity', '_identity_signature', '_cache_loaded', '_cache_generation']:
        monkeypatch.setattr(fund, key, getattr(fund, key))
    monkeypatch.setattr(fund, 'CACHE_DIR', tmp_path)
    legacy = tmp_path / 'generations' / 'v5-without-field-contract'
    legacy.mkdir(parents=True)
    with (legacy / 'fund_meta.pkl').open('wb') as handle:
        pickle.dump({'contract_version': 5, 'generation': legacy.name, 'data_sha256': '0' * 64}, handle)
    (tmp_path / 'current.json').write_text(json.dumps({'generation': legacy.name}), encoding='utf-8')

    with pytest.raises(FundamentalDataError, match='契约或代号不一致'):
        fund.get_cache_identity()


def test_custom_handler_never_replaces_requested_features_on_failure(monkeypatch):
    from core.adapters.qlib import custom_handler
    from qlib.contrib.data.handler import Alpha158
    monkeypatch.setattr('core.adapters.qlib.fundval.current_feature_contract', lambda keys: {})
    calls = []
    def failed(*args, **kwargs):
        calls.append(1)
        raise RuntimeError('source load failed')
    monkeypatch.setattr(Alpha158, '__init__', failed)
    with pytest.raises(RuntimeError, match='source load failed'):
        custom_handler.WhiteBoxHandler(factor_keys=['fund_turnover'])
    assert len(calls) == 1


@pytest.mark.p0
def test_fundval_delta_declares_the_full_lag_for_shard_invariance():
    from core.adapters.qlib.factor_resolver import resolve_factor_expression
    from core.adapters.qlib.fundval import FundVal
    from qlib.data.base import Feature
    from qlib.data.ops import Ref, Std
    expr, reason = resolve_factor_expression('SP_n5')
    assert reason is None
    assert 'Delta(' not in expr and 'Ref(' in expr
    value = FundVal(Feature('close'), 1)
    assert Std(value - Ref(value, 1), 5).get_extended_window_size() == (5, 0)


def test_fundval_reregisters_after_qlib_clears_operator_registry(monkeypatch):
    from core.adapters.qlib import custom_handler
    registered = []
    monkeypatch.setattr(custom_handler, '_fundval_registered', True)
    monkeypatch.setattr(custom_handler, 'Operators', SimpleNamespace(register=registered.extend))
    custom_handler._register_fundval()
    assert len(registered) == 1 and registered[0].__name__ == 'FundVal'


@pytest.mark.p0
def test_stale_cache_cannot_hide_latest_day_missing_behind_valid_history():
    from core.adapters.qlib.fundval import FundamentalQuality
    index = pd.MultiIndex.from_product([pd.bdate_range('2024-01-01', periods=2), ['sz000001']], names=['datetime', 'instrument'])
    frame = pd.concat({'feature': pd.DataFrame({'fund_turnover': [1., np.nan]}, index=index)}, axis=1)
    with pytest.raises(FundamentalDataError, match='整日缺失'):
        FundamentalQuality(['fund_turnover'])(frame)


@pytest.mark.p0
def test_fundamental_loader_replays_immutable_versions_and_official_overrides(monkeypatch):
    from core.data import fundamental as fund

    rows = [
        {
            'code': '000001', 'report_date': datetime.date(2023, 12, 31),
            'ann_date': datetime.date(2024, 1, 3),
            'source_observed_at': datetime.datetime(2024, 1, 5, 18),
            'source_name': 'tushare_fina_indicator', 'version_kind': 'provider_observation',
            'payload_json': json.dumps({'roe': '5', 'netprofit_yoy': '10', 'or_yoy': '20',
                                        'debt_to_assets': '40', 'profit_dedt': '8'}),
        },
        {
            'code': '000001', 'report_date': datetime.date(2023, 12, 31),
            'ann_date': datetime.date(2024, 1, 3),
            'source_observed_at': datetime.datetime(2024, 1, 20, 18),
            'source_name': 'official_report', 'version_kind': 'official_field_reconciliation',
            'payload_json': json.dumps({'official_values': {'profit_dedt': '12'}}),
        },
        {
            'code': '000001', 'report_date': datetime.date(2023, 12, 31),
            'ann_date': datetime.date(2024, 1, 3),
            'source_observed_at': datetime.datetime(2024, 1, 25, 18),
            'source_name': 'tushare_fina_indicator', 'version_kind': 'provider_observation',
            'payload_json': json.dumps({'roe': '6', 'netprofit_yoy': '11', 'or_yoy': '21',
                                        'debt_to_assets': '39', 'profit_dedt': '9'}),
        },
    ]
    captured = {}
    monkeypatch.setattr('core.data.Database.is_available', lambda: True)

    def fake_fetchall(sql):
        captured['sql'] = sql
        return rows

    monkeypatch.setattr('core.data.Database.fetchall', fake_fetchall)

    result = fund._load_fundamental_data()

    assert 'FROM fundamental_data_versions' in captured['sql']
    assert 'FROM fundamental_data f' not in captured['sql']
    assert result['avail_date'].dt.strftime('%Y-%m-%d').tolist() == [
        '2024-01-06', '2024-01-21', '2024-01-26',
    ]
    assert result['roe'].tolist() == [5.0, 5.0, 6.0]
    assert result['profit_dedt'].tolist() == [8.0, 12.0, 12.0]


@pytest.mark.p0
def test_expression_cache_isolated_by_field_and_generation(fund_cache, monkeypatch):
    from core.adapters.qlib.fundval import FundVal
    fund, dates = fund_cache
    current = {'generation': 'one', 'field_contract_sha256': 'a' * 64}
    monkeypatch.setattr(fund, 'get_cache_identity', lambda: dict(current))
    monkeypatch.setattr('qlib.data.data.Cal', SimpleNamespace(calendar=lambda **kwargs: dates))
    class Feature:
        def __str__(self):
            return '$close'
        def load(self, *args):
            return pd.Series(1., index=pd.RangeIndex(1, 2))
    feature = Feature()
    size, turnover = FundVal(feature, 1), FundVal(feature, 9)
    assert str(size) != str(turnover)
    assert size.load('sz000001', 1, 1, 'day').iloc[0] == -2.
    assert turnover.load('sz000001', 1, 1, 'day').iloc[0] == .2
    first = str(turnover)
    current['generation'] = 'two'
    fund._cache['real_fund_turnover_rate'].iloc[1] = .7
    fund._stock_cache.clear()
    assert first != str(turnover)
    assert turnover.load('sz000001', 1, 1, 'day').iloc[0] == .7


def test_contract_failure_is_actionable_http_conflict():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.middleware import register_middleware
    app = FastAPI()
    register_middleware(app)
    @app.get('/probe')
    async def probe():
        raise FundamentalDataError('模型必须重新训练')
    response = TestClient(app).get('/probe')
    assert response.status_code == 409
    assert response.json()['code'] == 'fundamental_data_contract'
    assert response.json()['detail'] == '模型必须重新训练'


@pytest.mark.p0
def test_model_contract_preserves_feature_order(tmp_path):
    from core.adapters.qlib.fundval import feature_contract, require_model_contract
    keys = ['fund_turnover', 'SP_n5']
    (tmp_path / 'feature_contract.json').write_text(json.dumps(feature_contract(keys)), encoding='utf-8')
    require_model_contract(tmp_path, keys)
    with pytest.raises(FundamentalDataError, match='不兼容'):
        require_model_contract(tmp_path, keys[::-1])
