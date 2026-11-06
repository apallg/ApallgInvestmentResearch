import hashlib
import json
import re

import numpy as np
import pandas as pd


FUNDVAL_CONTRACT_VERSION = 6

FUNDAMENTAL_PROVIDER_FIELD_MAP = (
    ("roe", "roe"),
    ("netprofit_yoy", "net_profit_yoy"),
    ("or_yoy", "revenue_yoy"),
    ("debt_to_assets", "debt_to_assets"),
    ("eps", "eps"),
    ("grossprofit_margin", "grossprofit_margin"),
    ("q_gsprofit_margin", "q_gsprofit_margin"),
    ("roa", "roa"),
    ("ebit", "ebit"),
    ("netprofit_margin", "netprofit_margin"),
    ("ocf_to_debt", "ocf_to_debt"),
    ("roe_yoy", "roe_yoy"),
    ("assets_turn", "assets_turn"),
    ("quick_ratio", "quick_ratio"),
    ("profit_dedt", "profit_dedt"),
    ("diluted2_eps", "diluted2_eps"),
    ("bps", "bps"),
    ("cfps", "cfps"),
)

FUNDAMENTAL_SOURCE_FIELDS = tuple(source for source, _ in FUNDAMENTAL_PROVIDER_FIELD_MAP)

FUNDAMENTAL_CURRENT_PROJECTION = tuple(
    (source, canonical)
    for source, canonical in FUNDAMENTAL_PROVIDER_FIELD_MAP
    if source != "q_gsprofit_margin"
)

FUNDAMENTAL_QLIB_FIELDS = (
    (5, "roe", "ROE", "盈利"),
    (6, "net_profit_yoy", "净利润同比", "成长"),
    (7, "revenue_yoy", "营收同比", "成长"),
    (8, "debt_to_assets", "资产负债率", "质量"),
    (10, "q_gsprofit_margin", "单季度毛利率", "盈利"),
    (11, "roa", "总资产收益率", "盈利"),
    (12, "ebit", "EBIT", "盈利"),
    (13, "netprofit_margin", "净利率", "盈利"),
    (14, "ocf_to_debt", "现金流负债比", "质量"),
    (15, "roe_yoy", "ROE同比变动", "成长"),
    (16, "assets_turn", "总资产周转率", "效率"),
    (17, "quick_ratio", "速动比率", "偿债"),
    (18, "profit_dedt", "扣非净利润", "盈利"),
    (19, "diluted2_eps", "稀释EPS", "估值"),
    (20, "bps", "每股净资产", "估值"),
    (21, "cfps", "每股现金流", "现金流"),
)

FUNDAMENTAL_FIELD_SEMANTICS = {
    source: {
        "definition_status": "provider_documented",
        "period_basis": "single_quarter" if source == "q_gsprofit_margin" else "unverified",
        "unit_basis": "unverified",
        "entity_basis": "unverified",
        "source": "tushare.fina_indicator",
        "source_document": "https://tushare.pro/document/2?doc_id=79",
    }
    for source in FUNDAMENTAL_SOURCE_FIELDS
}


def fundamental_field_contract_sha256():
    contract = {
        "provider_field_map": FUNDAMENTAL_PROVIDER_FIELD_MAP,
        "current_projection": FUNDAMENTAL_CURRENT_PROJECTION,
        "qlib_fields": FUNDAMENTAL_QLIB_FIELDS,
        "semantics": FUNDAMENTAL_FIELD_SEMANTICS,
    }
    content = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class FundamentalDataError(RuntimeError):
    pass


def fundamental_code(value):
    text = str(value).strip().lower()
    match = re.fullmatch(r'(?:(sh|sz|bj))?(\d{6})(?:\.(sh|sz|bj))?', text)
    if not match:
        raise FundamentalDataError(f"非法证券代码: {value}")
    prefix, code, suffix = match.groups()
    exchange = 'sh' if code.startswith(('60', '68')) else 'sz' if code.startswith(('00', '30')) else 'bj' if code.startswith(('4', '8', '92')) else None
    if exchange is None or any(part and part != exchange for part in (prefix, suffix)):
        raise FundamentalDataError(f"证券代码与交易所不匹配或不是A股: {value}")
    return code


def fundamental_dates(index, calendar=None):
    if isinstance(index, pd.DatetimeIndex):
        if index.hasnans or index.tz is not None or not index.equals(index.normalize()):
            raise FundamentalDataError("基本面只接受无时区的有效日频日期")
        return index
    if not pd.api.types.is_integer_dtype(index.dtype) or calendar is None:
        raise FundamentalDataError("整数索引必须使用同一Qlib数据源交易日历映射")
    positions = index.to_numpy(dtype=np.int64)
    calendar = pd.DatetimeIndex(calendar)
    if not calendar.is_unique or not calendar.is_monotonic_increasing or calendar.hasnans:
        raise FundamentalDataError("Qlib交易日历无效")
    if len(positions) and ((positions < 0).any() or (positions >= len(calendar)).any()):
        raise FundamentalDataError("Qlib交易日索引越界")
    return calendar.take(positions)


def align_fundamental(series, dates):
    dates = fundamental_dates(dates)
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.has_duplicates:
        raise FundamentalDataError("基本面缓存日期索引无效或重复")
    return series.reindex(dates).replace([np.inf, -np.inf], np.nan)
