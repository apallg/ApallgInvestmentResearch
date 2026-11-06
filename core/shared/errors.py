"""自定义异常 — 所有层通用的错误类型"""


class DataNotAvailableError(Exception):
    """数据不可用 (缓存未命中且源不可达)"""
    pass


class ConfigError(Exception):
    """配置错误"""
    pass


class UnsupportedExprError(Exception):
    """不支持的因子表达式"""
    pass
