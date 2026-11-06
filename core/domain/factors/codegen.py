"""
因子代码生成器 v2 — 注册制 (不再修改源码文件)

替代 factor_codegen.py 的源码注入方式。
因子的评分函数注册到 FACTOR_REGISTRY，持久化到数据库 custom_factors 表。
重启后从数据库加载恢复。
"""

import os
import logging
from core.domain.factors.registry import get_pool

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))


def generate_scoring_function(key, label, logic_desc, dataflow="price_only", default_n=20):
    """生成评分函数代码字符串 (用于前端预览) — 与旧 API 兼容"""
    tmpl = {
        "price_only": (f"df, n={default_n}",
                       "import math as _math",
                       ["close","open","high","low","volume","amount","turnover"]),
        "price_with_db": (f"code, df, n={default_n}, preloaded_data=None",
                          "import math as _math",
                          ["close","open","high","low","volume","amount"]),
    }.get(dataflow, (f"df, n={default_n}", "import math as _math", ["close"]))
    func_params, imports, available_cols = tmpl

    code = f'''def _score_{key}({func_params}):
    """{label}: {logic_desc}"""
    try:
{imports}
        closes = df["close"].astype(float)
        if len(closes) < n:
            return 0.0
        ret = closes.iloc[-1] / closes.iloc[-n] - 1
        return round(_math.tanh(ret * 3), 4)
    except Exception:
        return 0.0'''
    return code, f"_score_{key}"


def register_factor_safe(key, label, func, category="custom",
                         signal_type="continuous", default_params=None,
                         data_safety="unknown", logic_desc="",
                         persist_to_db=True) -> bool:
    """注册因子到因子池 + 数据库持久化。"""
    if persist_to_db:
        _persist_to_db(key, label, category, signal_type, default_params, data_safety, logic_desc)
    get_pool().add_factor(
        key, label, category=category if category in ("white_box", "traditional", "hybrid") else "white_box",
        data_safety=data_safety or "unknown",
    )
    logger.info(f"[FactorCodeGen] 注册因子: {key} (category={category})")
    return True


CREATE_TABLE_SQL = (
    """CREATE TABLE IF NOT EXISTS custom_factors (
        factor_key VARCHAR(100) PRIMARY KEY,
        label VARCHAR(200), category VARCHAR(50),
        signal_type VARCHAR(20), params JSON,
        data_safety VARCHAR(20), logic_desc TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    )"""
)


def _ensure_table():
    try:
        from core.data import Database
        if Database.is_available():
            Database.execute(CREATE_TABLE_SQL)
    except Exception:
        pass


def _persist_to_db(key, label, category, signal_type, params, data_safety, logic_desc):
    import json
    try:
        from core.data import Database
        if not Database.is_available():
            return
        _ensure_table()
        Database.execute(
            """INSERT INTO custom_factors (factor_key, label, category, signal_type, params, data_safety, logic_desc)
               VALUES (%s,%s,%s,%s,%s,%s,%s)
               ON DUPLICATE KEY UPDATE label=VALUES(label), params=VALUES(params),
               data_safety=VALUES(data_safety), updated_at=NOW()""",
            (key, label, category, signal_type,
             json.dumps(params or {}, ensure_ascii=False),
             data_safety, logic_desc),
        )
    except Exception as e:
        logger.debug(f"[FactorCodeGen] 数据库持久化跳过: {e}")


def load_factors_from_db():
    try:
        from core.data import Database
        if not Database.is_available():
            return 0
        _ensure_table()
        rows = Database.fetchall("SELECT * FROM custom_factors")
        pool = get_pool()
        active_keys = set(pool.get_active_factors().keys())
        count = 0
        for row in rows:
            if row["factor_key"] not in active_keys:
                import json
                params = json.loads(row.get("params", "{}")) if isinstance(row.get("params"), str) else (row.get("params") or {})
                pool.add_factor(
                    key=row["factor_key"], label=row.get("label", row["factor_key"]),
                    category=row.get("category", "custom"),
                    params=params,
                    data_safety=row.get("data_safety") or "unknown",
                )
                count += 1
        if count > 0:
            logger.info(f"[FactorCodeGen] 从数据库恢复 {count} 个自定义因子")
        return count
    except Exception as e:
        logger.debug(f"[FactorCodeGen] 数据库恢复跳过: {e}")
        return 0


def unregister_factor_safe(key, remove_from_db=True) -> bool:
    """安全注销因子"""
    get_pool().remove_factor(key)
    if remove_from_db:
        try:
            from core.data import Database
            if Database.is_available():
                Database.execute("DELETE FROM custom_factors WHERE factor_key=%s", (key,))
        except Exception:
            pass
    logger.info(f"[FactorCodeGen] 注销因子: {key}")
    return True


