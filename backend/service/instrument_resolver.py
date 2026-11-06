"""Qlib 股票池解析服务 — 把 common.py 中的 instruments/代码前缀逻辑迁入 L2。"""
import logging
from pathlib import Path

from core.shared import config
from backend.service.qlib_access import qlib_read_locked

logger = logging.getLogger(__name__)


def get_qlib_data_dir():
    from core.adapters.qlib.generation import assert_no_active_historical_release

    assert_no_active_historical_release(config.QLIB_DATA_ROOT)
    return config.QLIB_DATA_DIR


@qlib_read_locked
def resolve_instruments(instruments: str):
    """解析组合股票池，如 csi300+csi500 → 合并去重后的股票列表。
    普通值 (csi300/all/a500) 原样返回，Qlib 原生支持。
    """
    if instruments in ("a500", "stable_history_500d"):
        return _stable_history_500d_instruments()
    if "+" in instruments:
        from qlib.data import D
        parts = instruments.split("+")
        combined = []
        for p in parts:
            p = p.strip()
            try:
                inst = D.list_instruments(D.instruments(p), as_list=True)
                combined.extend(inst)
            except Exception:
                logger.warning(f"[Train] 无法解析 instruments '{p}'，跳过")
        seen = set()
        result = []
        for c in combined:
            if c not in seen:
                seen.add(c)
                result.append(c)
        logger.info(f"[Train] 组合池 {instruments}: {len(result)} 只股票")
        return result
    return instruments


def _stable_history_500d_instruments(code_prefixes=None):
    from core.adapters.qlib.utils import mysql_to_qlib_code
    from core.data.repos.instruments import InstrumentRepo
    from core.domain.screening.exclusion import is_excluded

    ranges = InstrumentRepo.get_stable_history_500d_ranges()
    prefixes = tuple(str(prefix) for prefix in (code_prefixes or []))
    result = {}
    for code, date_range in ranges.items():
        if is_excluded(code)[0]:
            continue
        if prefixes and not str(code).startswith(prefixes):
            continue
        qlib_code = mysql_to_qlib_code(code)
        start, end = date_range
        result[qlib_code] = [
            start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start)[:10],
            end.strftime("%Y-%m-%d") if hasattr(end, "strftime") else str(end)[:10],
        ]
    return result


def _historical_quote_instruments(code_prefixes=None):
    from core.adapters.qlib.utils import mysql_to_qlib_code
    from core.data.repos.instruments import InstrumentRepo
    from core.domain.screening.exclusion import is_excluded

    ranges = InstrumentRepo.get_quote_instrument_ranges()
    prefixes = tuple(str(prefix) for prefix in (code_prefixes or []))
    result = {}
    for code, date_range in ranges.items():
        if is_excluded(code)[0]:
            continue
        if prefixes and not str(code).startswith(prefixes):
            continue
        start, end = date_range
        result[mysql_to_qlib_code(code)] = [
            start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start)[:10],
            end.strftime("%Y-%m-%d") if hasattr(end, "strftime") else str(end)[:10],
        ]
    return result


def filter_instrument_selection(instruments, code_prefixes):
    prefixes = tuple(str(prefix) for prefix in (code_prefixes or []))
    if isinstance(instruments, dict):
        return {
            code: date_range for code, date_range in instruments.items()
            if len(code) >= 3 and code[2:].startswith(prefixes)
        }
    if isinstance(instruments, list):
        return [code for code in instruments if len(code) >= 3 and code[2:].startswith(prefixes)]
    return build_filtered_instruments(instruments, code_prefixes)


def filter_st_stocks(df, code_col, name_col):
    """从指数成分股 DataFrame 中剔除 ST/*ST 股票"""
    codes = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, ""))
        if "ST" in name:
            continue
        codes.append(str(row[code_col]))
    return codes


@qlib_read_locked
def _load_local_qlib_instruments(pool, code_prefixes):
    path = Path(get_qlib_data_dir()) / "instruments" / f"{pool}.txt"
    if not path.exists():
        return []

    prefixes = tuple(str(p) for p in (code_prefixes or []))
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        code = line.split("\t", 1)[0].strip().lower()
        if not code:
            continue
        numeric = code[2:] if len(code) > 2 and code[:2] in ("sh", "sz", "bj") else code
        if prefixes and not numeric.startswith(prefixes):
            continue
        result.append(code)
    return result


@qlib_read_locked
def _load_local_qlib_instrument_ranges(pool, code_prefixes):
    path = Path(get_qlib_data_dir()) / "instruments" / f"{pool}.txt"
    if not path.exists():
        return {}

    prefixes = tuple(str(prefix) for prefix in (code_prefixes or []))
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        code = parts[0].strip().lower()
        numeric = code[2:] if len(code) > 2 and code[:2] in ("sh", "sz", "bj") else code
        if prefixes and not numeric.startswith(prefixes):
            continue
        result[code] = parts[1:3]
    return result


def get_all_stock_codes(fetcher):
    """获取全A股代码列表（已过滤 ST 和退市股）"""
    from core.data.sources.tushare import get_all_stock_codes
    try:
        info = get_all_stock_codes()
        return list(info)
    except Exception:
        from core.data import Database
        rows = Database.fetchall("SELECT DISTINCT code FROM daily_quotes ORDER BY code")
        return [r["code"] for r in rows]


def build_filtered_instruments(pool: str, code_prefixes: list):
    """根据股票池和代码前缀构建自定义 instruments 列表"""
    storage_pool = "a500" if pool == "stable_history_500d" else pool
    if pool in ("a500", "stable_history_500d"):
        return _stable_history_500d_instruments(code_prefixes)
    if pool == "all":
        if not config.MYSQL_ENABLED:
            return _load_local_qlib_instrument_ranges("all", code_prefixes)
        return _historical_quote_instruments(code_prefixes)
    from core.data.fetcher import DataFetcher
    fetcher = DataFetcher()

    if pool == "csi300":
        import akshare as ak
        df = ak.index_stock_cons_weight_csindex("000300")
        codes = filter_st_stocks(df, "成分券代码", "成分券名称")
    elif pool == "csi500":
        import akshare as ak
        df = ak.index_stock_cons_weight_csindex("000905")
        codes = filter_st_stocks(df, "成分券代码", "成分券名称")
    elif pool == "csi300+csi500":
        import akshare as ak
        df300 = ak.index_stock_cons_weight_csindex("000300")
        codes300 = set(filter_st_stocks(df300, "成分券代码", "成分券名称"))
        df500 = ak.index_stock_cons_weight_csindex("000905")
        codes500 = set(filter_st_stocks(df500, "成分券代码", "成分券名称"))
        codes = list(codes300 | codes500)
    elif pool == "small_cap":
        from core.data.sources.tushare import get_small_cap_codes
        codes = get_small_cap_codes()
    elif pool in ("a500", "stable_history_500d", "all"):
        if not config.MYSQL_ENABLED:
            local_codes = _load_local_qlib_instruments(storage_pool, code_prefixes)
            if local_codes:
                logger.info(f"[instruments] 使用本地 Qlib 股票池: {pool}, {len(local_codes)} 只")
                return local_codes
        all_codes = get_all_stock_codes(fetcher)
        if pool in ("a500", "stable_history_500d"):
            from core.data import Database
            codes = []
            batch_size = 500
            for bs in range(0, len(all_codes), batch_size):
                batch = all_codes[bs:bs + batch_size]
                ph = ",".join(["%s"] * len(batch))
                rows = Database.fetchall(
                    f"SELECT code, COUNT(*) as cnt FROM daily_quotes "
                    f"WHERE code IN ({ph}) GROUP BY code HAVING cnt >= 500",
                    tuple(batch),
                )
                codes.extend(r["code"] for r in rows)
        else:
            codes = all_codes
    else:
        codes = get_all_stock_codes(fetcher)

    if code_prefixes:
        prefixes = tuple(str(p) for p in code_prefixes)
        codes = [c for c in codes if c.startswith(prefixes)]

    from core.adapters.qlib.utils import mysql_to_qlib_code
    qlib_codes = [mysql_to_qlib_code(c) for c in codes]

    logger.info(f"[instruments] pool={pool} prefixes={code_prefixes} → {len(qlib_codes)} stocks")
    return qlib_codes if qlib_codes else storage_pool
