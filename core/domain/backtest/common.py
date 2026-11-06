import pandas as pd
from core.shared.market_rules import limit_pct_for_code


def _raw_daily_return(closes, adj, i):
    """不复权日收益率: 用后复权因子修正前复权价的除权跳空, 供涨跌停判定。

    涨跌停是交易所基于不复权价的规则, 前复权价在除权除息日无跳空会漏判。
    """
    prev_close = closes.iloc[i - 1]
    if prev_close <= 0:
        return None
    if adj is not None and pd.notna(adj.iloc[i]) and pd.notna(adj.iloc[i - 1]) and adj.iloc[i - 1] > 0:
        return float(closes.iloc[i]) / float(prev_close) * float(adj.iloc[i - 1]) / float(adj.iloc[i]) - 1.0
    return float(closes.iloc[i]) / float(prev_close) - 1.0


def _opening_ret(opens, closes, adj, i):
    """开盘价相对前收盘的涨幅 (一字开盘), 与 _raw_daily_return 同复权口径。"""
    open_px = float(opens.iloc[i])
    prev_close = float(closes.iloc[i - 1])
    if prev_close <= 0 or open_px <= 0 or pd.isna(open_px):
        return None
    ret = open_px / prev_close
    if adj is not None and pd.notna(adj.iloc[i]) and pd.notna(adj.iloc[i - 1]) and adj.iloc[i - 1] > 0:
        ret *= float(adj.iloc[i - 1]) / float(adj.iloc[i])
    return ret - 1


def is_open_limit(frame, index, code, is_st=False, direction="up"):
    if frame is None or index is None or index <= 0 or index >= len(frame):
        return False
    try:
        closes = frame["close"].astype(float)
        opens = frame["open"].astype(float) if "open" in frame.columns else None
        adj = frame["adj_factor"].astype(float) if "adj_factor" in frame.columns else None
        opening_return = (
            _opening_ret(opens, closes, adj, index)
            if opens is not None else _raw_daily_return(closes, adj, index)
        )
        if opening_return is None:
            return False
        threshold = limit_pct_for_code(code, is_st=is_st) * 0.98
        return opening_return >= threshold if direction == "up" else opening_return <= -threshold
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def precompute_limit_up(price_cache, limit_up_cache, st_codes=None, no_limit_dates=None,
                        st_codes_by_date=None):
    """预计算每只股票每日的涨停状态, O(n) 一次性完成.
    纯计算——不依赖 DB/文件/时间函数。
    st_codes: ST 股代码集合 (name 含 ST, 涨跌幅 5%)。
    判定基于当日开盘相对前收盘的涨幅 (一字开盘封板): 回测在次日开盘成交,
    只有开盘即涨停的股票买不到, 收盘涨停不影响开盘成交。无 open 列时回退收盘判定。
    """
    st_codes = st_codes or set()
    no_limit_dates = no_limit_dates or {}
    for code, df in price_cache.items():
        try:
            df_sorted = df.sort_values('trade_date').reset_index(drop=True)
            price_cache[code] = df_sorted
            trade_dates = pd.to_datetime(df_sorted['trade_date'])
            closes = df_sorted['close'].astype(float)
            code_str = str(code)
            limit_dates = {}
            pit_dates = sorted(st_codes_by_date or {})
            pit_index = -1
            for i in range(1, len(df_sorted)):
                prev_close = closes.iloc[i - 1]
                if prev_close <= 0:
                    continue
                date_key = trade_dates.iloc[i].strftime('%Y%m%d')
                while pit_index + 1 < len(pit_dates) and pit_dates[pit_index + 1] <= date_key:
                    pit_index += 1
                if date_key in no_limit_dates.get(code, no_limit_dates.get(code_str, set())):
                    continue
                is_st = (code_str in st_codes_by_date[pit_dates[pit_index]]
                         if pit_index >= 0 else code_str in st_codes)
                if is_open_limit(df_sorted, i, code_str, is_st, "up"):
                    limit_dates[date_key] = True
            if limit_dates:
                limit_up_cache[code] = limit_dates
        except Exception:
            pass


def precompute_limit_down(price_cache, limit_down_cache, st_codes=None, no_limit_dates=None,
                          st_codes_by_date=None):
    st_codes = st_codes or set()
    no_limit_dates = no_limit_dates or {}
    for code, df in price_cache.items():
        try:
            df_sorted = df.sort_values('trade_date').reset_index(drop=True)
            price_cache[code] = df_sorted
            trade_dates = pd.to_datetime(df_sorted['trade_date'])
            closes = df_sorted['close'].astype(float)
            code_str = str(code)
            limit_dates = {}
            pit_dates = sorted(st_codes_by_date or {})
            pit_index = -1
            for i in range(1, len(df_sorted)):
                prev_close = closes.iloc[i - 1]
                if prev_close <= 0:
                    continue
                date_key = trade_dates.iloc[i].strftime('%Y%m%d')
                while pit_index + 1 < len(pit_dates) and pit_dates[pit_index + 1] <= date_key:
                    pit_index += 1
                if date_key in no_limit_dates.get(code, no_limit_dates.get(code_str, set())):
                    continue
                is_st = (code_str in st_codes_by_date[pit_dates[pit_index]]
                         if pit_index >= 0 else code_str in st_codes)
                if is_open_limit(df_sorted, i, code_str, is_st, "down"):
                    limit_dates[date_key] = True
            if limit_dates:
                limit_down_cache[code] = limit_dates
        except Exception:
            pass


def lookup_price(df, date_str, col):
    """在内存 DataFrame 中查找指定日期的价格。

    date_str 支持 yyyy-mm-dd 和 yyyymmdd 两种格式。
    不依赖 DB/文件/时间——纯 DataFrame 查找。
    """
    try:
        date_str_compact = date_str.replace('-', '')[:8]
        trade_dates = df['trade_date'].astype(str)
        mask = (trade_dates == date_str_compact) | (trade_dates.str[:10] == date_str)
        row = df[mask]
        if len(row) > 0:
            return float(row.iloc[0][col])
    except Exception:
        pass
    return None


def norm_code(code):
    """归一化股票代码: sh600000→600000, sz000858→000858, 830799.BJ→830799"""
    code = str(code).strip().lower()
    if code.startswith('sh') or code.startswith('sz'):
        code = code[2:]
    if code.endswith('.bj'):
        code = code[:-3]
    return code


def get_benchmark_return_fast(price_map, date, all_dates, idx):
    """从预加载的价格映射中计算基准收益率。
    纯计算——不查 DB、不调 API、不依赖日期函数。
    """
    if idx == 0:
        return 0
    prev = all_dates[idx - 1].isoformat()[:10]
    curr = date.isoformat()[:10]
    p_prev = price_map.get(prev)
    p_curr = price_map.get(curr)
    if p_prev and p_curr and p_prev > 0:
        return p_curr / p_prev - 1
    return 0


def equal_weight_pool_return(price_cache, start_date, end_date, start_col="close", end_col="close", codes=None):
    returns = []
    eligible = 0
    selected = (price_cache or {}).items()
    if codes is not None:
        selected = ((code, (price_cache or {}).get(code)) for code in codes)
    for _, frame in selected:
        if frame is None:
            continue
        start_price = lookup_price(frame, start_date, start_col)
        if start_price is None or start_price <= 0:
            continue
        eligible += 1
        end_price = lookup_price(frame, end_date, end_col)
        if end_price is not None and end_price > 0:
            returns.append(end_price / start_price - 1.0)
    return {
        "return": sum(returns) / len(returns) if returns else 0.0,
        "eligible_count": eligible,
        "covered_count": len(returns),
        "coverage": len(returns) / eligible if eligible else 0.0,
    }


def initial_build(codes, scores, top_n, price_cache, date_str, sector_map,
                  allocator, limit_up_cache, buy_cost_rate,
                  score_key="score", log_label="", next_str=None):
    """统一初始建仓逻辑。纯 L3 函数——不依赖 DB/文件/时间函数。

    从 engine.py 和 screening.py 的 _initial_build 中提取 (~90% 重复)。

    Args:
        codes: 候选代码列表
        scores: {code: score} dict
        price_cache: {code: DataFrame} dict (已预加载的日K线)
        date_str: 建仓日期 yyyy-mm-dd
        sector_map: {code: industry_info} dict
        allocator: PositionAllocator 实例
        limit_up_cache: {code: {date_key: bool}} 预计算的涨停状态
        buy_cost_rate: 买入费率 (佣金+滑点)
        score_key: 传递给 allocator 的分数字段名 ("score" | "l0_score")
        log_label: 日志前缀 ("[DirBacktest]" | "[ScrBacktest]")
        next_str: 成交日 yyyy-mm-dd。给定则 T+1 开盘成交 (open[next_str]);
                  None 则同日收盘成交 (close[date_str], 原行为)。

    Returns:
        (valid_weights, entry_prices, cash, cooldown_set)
    """
    import logging
    logger = logging.getLogger(__name__)

    # 成交日: 传 next_str 时 T+1 开盘成交, 否则同日收盘成交 (向后兼容)
    if next_str is None:
        exec_date = date_str
        price_field = 'close'
    else:
        exec_date = next_str
        price_field = 'open'
    weights = allocator.allocate_weights_only(
        [{score_key: scores.get(c, 0), "code": c} for c in codes],
        {}, sector_map, {})
    entry_prices = {}
    valid_weights = {}
    skipped = 0
    for code, w in weights.items():
        df = price_cache.get(code)
        if df is not None:
            # 涨停过滤 (成交日 exec_date 的涨停状态)
            date_key = exec_date.replace('-', '')[:8]
            if limit_up_cache.get(code, {}).get(date_key, False):
                skipped += 1
                continue
            p = lookup_price(df, exec_date, price_field)
            if p is not None and p > 0:
                entry_prices[code] = p
                valid_weights[code] = w
    if skipped > 0:
        logger.info(f"{log_label} 初始建仓 {exec_date} 涨停跳过: {skipped} 只")

    total_w = sum(valid_weights.values())
    cost_adjusted = total_w * (1 + buy_cost_rate)
    cash = 1.0 - cost_adjusted if cost_adjusted < 1.0 else 0.0
    return valid_weights, entry_prices, cash, set()
