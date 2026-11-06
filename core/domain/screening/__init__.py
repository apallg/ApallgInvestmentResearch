"""
core.domain.screening — 选股领域 (L3 纯业务逻辑)
"""
# exclusion 零依赖, 可安全顶层导入
from core.domain.screening.exclusion import is_excluded, EXCLUDED_PREFIXES, B_SHARE_PREFIXES

# _screener 有重依赖 (DataFetcher, Qlib, etc.), 懒导入避免循环
def get_screener():
    from core.domain.screening._screener import get_screener as _get
    return _get()

def get_BuySideScreener():
    from core.domain.screening._screener import BuySideScreener
    return BuySideScreener
