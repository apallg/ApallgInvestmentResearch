"""
FastAPI 中间件 — 统一错误处理 + 请求日志 + 数据源追踪
"""
import time
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api")


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        status = response.status_code
        if duration > 2.0 or status >= 400:
            logger.warning(f"{request.method} {request.url.path} → {status} ({duration:.2f}s)")
        elif duration > 0.5:
            logger.info(f"{request.method} {request.url.path} → {status} ({duration:.2f}s)")
        return response


async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, (KeyboardInterrupt, SystemExit, SystemError)):
        raise exc
    import traceback as tb_mod
    tb_str = tb_mod.format_exc()
    if isinstance(exc, (MemoryError, OSError)):
        level = "critical"
    else:
        level = "error"
    logger.error(f"未捕获异常 {request.method} {request.url.path}: {exc}\n{tb_str}")

    try:
        from core.shared.error_store import log_error
        from core.shared.infrastructure.thread_pool import thread_pool
        thread_pool.error.submit(
            lambda: log_error(
                level=level,
                source="middleware",
                method=request.method,
                path=str(request.url.path),
                exc_type=type(exc).__name__,
                exc_msg=str(exc),
                traceback=tb_str,
            )
        )
    except Exception:
        logger.exception("error_logs 写入失败 (middleware)")

    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误，请查看日志", "path": str(request.url.path)},
    )


def register_middleware(app):
    from core.shared.fundamental_contract import FundamentalDataError
    from core.shared.qlib_errors import QlibRestartRequired

    async def qlib_restart_required(request, exc):
        logger.warning('Qlib数据代等待重启 %s: %s', request.url.path, exc)
        return JSONResponse(status_code=503, content={
            'error': str(exc), 'detail': str(exc), 'code': 'qlib_restart_required',
            'restart_required': True, 'generation_id': exc.generation_id,
            'current_generation_id': exc.current_generation_id,
        })

    async def fundamental_error(request, exc):
        logger.warning('基本面输入契约拒绝 %s: %s', request.url.path, exc)
        return JSONResponse(status_code=409, content={'error': str(exc), 'detail': str(exc),
                                                     'code': 'fundamental_data_contract'})
    app.add_middleware(RequestLogMiddleware)
    app.add_exception_handler(QlibRestartRequired, qlib_restart_required)
    app.add_exception_handler(FundamentalDataError, fundamental_error)
    app.add_exception_handler(Exception, global_exception_handler)
    logger.info("[Middleware] 请求日志+全局异常处理已注册")
