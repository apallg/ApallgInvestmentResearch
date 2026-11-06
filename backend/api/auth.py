import os
import logging
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

TOKEN_EXPIRE_DAYS = 7


def _get_jwt_secret() -> str:
    return os.getenv("JWT_SECRET_KEY", "") or os.getenv("ADMIN_TOKEN", "")


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    expires_at: str


@router.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest):
    expected_user = os.getenv("LOGIN_USERNAME", "admin")
    expected_pass = os.getenv("LOGIN_PASSWORD")

    if not expected_pass:
        logger.error("LOGIN_PASSWORD 未配置, 拒绝登录")
        raise HTTPException(status_code=503, detail="登录口令未配置")

    if body.username != expected_user or body.password != expected_pass:
        raise HTTPException(status_code=401, detail="账号或密码错误")

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": body.username,
        "iat": now,
        "exp": expires_at,
    }
    secret = _get_jwt_secret()
    if not secret:
        logger.error("JWT_SECRET_KEY/ADMIN_TOKEN 未配置, 拒绝登录")
        raise HTTPException(status_code=503, detail="服务密钥未配置")
    token = jwt.encode(payload, secret, algorithm="HS256")

    logger.info(f"用户 {body.username} 登录成功")
    return LoginResponse(
        token=token,
        username=body.username,
        expires_at=expires_at.isoformat(),
    )


@router.get("/auth/verify")
def verify_token():
    return {"valid": True}


def decode_token(token: str) -> dict | None:
    """解析 JWT，失败返回 None"""
    secret = _get_jwt_secret()
    if not secret:
        return None
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def require_admin_actor(authorization: str | None) -> str:
    from backend.api.settings import _verify_admin

    if _verify_admin(authorization):
        return "admin-token"
    token = (authorization or "").removeprefix("Bearer ").strip()
    claims = decode_token(token) if token else None
    expected_user = os.getenv("LOGIN_USERNAME", "admin")
    if claims and claims.get("sub") == expected_user:
        return str(claims["sub"])
    raise HTTPException(status_code=403, detail="需要管理员权限")


# 白名单：不需要鉴权的路径
AUTH_WHITELIST = {
    "/api/auth/login",
    "/api/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}
