import logging

from fastapi import HTTPException

from backend.service.instrument_lookup import resolve_current_stock_code

logger = logging.getLogger(__name__)


def require_current_stock_code(code: str):
    try:
        resolved = resolve_current_stock_code(code)
    except Exception as error:
        logger.warning("证券身份校验不可用: %s", type(error).__name__)
        raise HTTPException(503, "证券身份数据不可用，无法确认证券代码") from error
    if resolved is None:
        raise HTTPException(404, "证券代码不是当前已登记的A股证券")
    return resolved
