import logging

logger = logging.getLogger(__name__)


class FundamentalCacheService:
    @staticmethod
    def rebuild():
        from core.data.fundamental import preload_cache, get_cache

        if not preload_cache(force=True):
            raise RuntimeError("基本面缓存重建失败")
        cache = get_cache()
        result = {"fields": len(cache), "rows": len(next(iter(cache.values()))) if cache else 0}
        logger.info("[FundCache] 手动重建完成: %s字段, %s行", result["fields"], result["rows"])
        return result
