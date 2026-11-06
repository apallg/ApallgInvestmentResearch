"""
系统设置 API — 配置读取/更新/重载
统一前缀: /api/settings
"""
import os
import shutil
import tempfile
import json
import logging
from pathlib import Path

import anyio
from fastapi import APIRouter, Body, Header

router = APIRouter()
logger = logging.getLogger(__name__)

ENV_FILE = Path(__file__).parent.parent.parent / ".env"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


_env_write_lock = __import__('threading').Lock()


def _verify_admin(authorization: str | None) -> bool:
    if not authorization or not ADMIN_TOKEN:
        return False
    token = authorization.removeprefix("Bearer ").strip()
    if token == ADMIN_TOKEN:
        return True
    # 安全: 仅认 ADMIN_TOKEN, 不把任意登录 JWT 当管理员
    return False


def _safe_set_env_key(env_path, key, value, sensitive=False):
    """原子写入 .env：加锁→备份→写临时文件→原子替换"""
    with _env_write_lock:
        if not os.path.exists(env_path):
            raise FileNotFoundError(f".env 不存在: {env_path}")
        backup = env_path + ".bak"
        shutil.copy2(env_path, backup)
        logger.warning(f"[Settings] .env 备份: {backup}")

        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(env_path), suffix=".env")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tf:
                with open(env_path, 'r', encoding='utf-8') as f:
                    found = False
                    for line in f:
                        stripped = line.lstrip()
                        if stripped.startswith(key + '=') or stripped.startswith('#' + key + '='):
                            tf.write(f"{key}={value}\n")
                            found = True
                        else:
                            tf.write(line)
                    if not found:
                        tf.write(f"{key}={value}\n")
            os.replace(tmp, env_path)
            log_val = _mask_sensitive(value) if sensitive else value
            logger.info(f"[Settings] {key} 已更新为 {log_val}")
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def _safe_set_env_keys(env_path, kv_pairs):
    """批量原子写入多个 key=value 对 (单次读+写+替换)"""
    if not kv_pairs:
        return
    with _env_write_lock:
        if not os.path.exists(env_path):
            raise FileNotFoundError(f".env 不存在: {env_path}")
        backup = env_path + ".bak"
        shutil.copy2(env_path, backup)
        logger.warning(f"[Settings] .env 备份: {backup} (批量更新 {len(kv_pairs)} 项)")

        keys_set = {k for k, _ in kv_pairs.items()}
        updated = set()
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(env_path), suffix=".env")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tf:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        stripped = line.lstrip()
                        matched = None
                        for k in keys_set:
                            if stripped.startswith(k + '=') or stripped.startswith('#' + k + '='):
                                matched = k
                                break
                        if matched:
                            tf.write(f"{matched}={kv_pairs[matched]}\n")
                            updated.add(matched)
                        else:
                            tf.write(line)
                for k in keys_set - updated:
                    tf.write(f"{k}={kv_pairs[k]}\n")
            os.replace(tmp, env_path)
            for k in keys_set:
                logger.info(f"[Settings] {k} 已更新")
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def _mask_sensitive(value):
    s = str(value)
    if len(s) <= 8:
        return s[:2] + "***" + s[-2:] if len(s) >= 4 else "***"
    return s[:3] + "***" + s[-3:]


def _get_current_values():
    from core.shared import config
    from backend.config_metadata import CONFIG_SCHEMA

    result = {}
    for item in CONFIG_SCHEMA:
        key = item["key"]
        val = getattr(config, key, None)
        if val is None:
            val = os.getenv(item["env_key"], item["default"])
        result[key] = str(val) if not isinstance(val, (str, int, float, bool)) else val
    return result


@router.get("/settings")
async def get_settings():
    from backend.config_metadata import CONFIG_SCHEMA, GROUP_ORDER

    current = _get_current_values()
    groups_dict = {}

    for item in CONFIG_SCHEMA:
        group = item["group"]
        if group not in groups_dict:
            groups_dict[group] = {"name": group, "fields": []}

        key = item["key"]
        raw_value = current.get(key, item["default"])
        display_value = raw_value
        if item.get("sensitive") and raw_value:
            display_value = _mask_sensitive(str(raw_value))

        groups_dict[group]["fields"].append({
            "key": key,
            "label": item.get("label", key),
            "env_key": item["env_key"],
            "type": item["type"],
            "value": display_value,
            "default": item["default"],
            "description": item["description"],
            "options": item.get("options"),
            "sensitive": item.get("sensitive", False),
        })

    groups = [groups_dict[g] for g in GROUP_ORDER if g in groups_dict]

    return {"data": {"groups": groups}}


@router.put("/settings")
async def update_settings(updates: dict = Body(...), authorization: str | None = Header(None)):
    if not _verify_admin(authorization):
        from fastapi.responses import JSONResponse
        return JSONResponse({"data": {"errors": ["未授权: 缺少有效的 Admin Token"]}}, status_code=401)

    def _run():
        from backend.config_metadata import SCHEMA_BY_KEY
        from core.shared import config

        changed = []
        errors = []
        env_writes = {}  # 收集所有待写入的 env_key→env_val
        config_sets = {}  # 收集所有待热更新的 config_key→typed_val

        # 第一遍: 校验所有 key
        for k, v in updates.items():
            meta = SCHEMA_BY_KEY.get(k)
            if meta is None:
                errors.append(f"未知配置项: {k}")
                continue

            target_type = meta["type"]
            try:
                if target_type == "bool":
                    typed_val = str(v).lower() in ("true", "1", "yes", "on")
                    env_val = "true" if typed_val else "false"
                elif target_type == "int":
                    typed_val = int(v)
                    env_val = str(typed_val)
                elif target_type == "float":
                    typed_val = float(v)
                    env_val = str(typed_val)
                elif target_type == "json":
                    parsed = json.loads(v) if isinstance(v, str) else v
                    json.dumps(parsed)
                    typed_val = parsed
                    env_val = json.dumps(parsed, ensure_ascii=False)
                elif target_type in ("select",):
                    options = meta.get("options", [])
                    if options and v not in options:
                        errors.append(f"{k}: 值 '{v}' 不在可选项 {options} 中")
                        continue
                    typed_val = v
                    env_val = str(v)
                else:
                    typed_val = v
                    env_val = str(v)
            except (ValueError, TypeError):
                errors.append(f"{k}: 类型转换失败, 期望 {target_type}, 收到 '{v}'")
                continue
            except json.JSONDecodeError as e:
                errors.append(f"{k}: JSON 格式错误: {e}")
                continue

            if isinstance(env_val, str) and ("\n" in env_val or "\r" in env_val):
                errors.append(f"{k}: 值包含换行符, 已拒绝")
                continue

            env_writes[meta["env_key"]] = env_val
            config_sets[k] = typed_val
            changed.append(k)

        # 第二遍: 批量写入 .env (单次IO)
        if env_writes:
            try:
                _safe_set_env_keys(str(ENV_FILE), env_writes)
            except Exception as e:
                errors.append(f"批量写入 .env 失败: {e}")
                # 回滚 changed 列表, 因为 .env 未更新
                changed.clear()
                return {"changed": [], "errors": errors}

        # 第三遍: 热更新 config 模块
        for k, typed_val in config_sets.items():
            try:
                setattr(config, k, typed_val)
            except Exception as e:
                errors.append(f"{k}: 热更新失败: {e}")

        return {"changed": changed, "errors": errors}
    result = await anyio.to_thread.run_sync(_run)
    return {"data": result}


@router.post("/settings/reload")
async def reload_settings(authorization: str | None = Header(None)):
    if not _verify_admin(authorization):
        from fastapi.responses import JSONResponse
        return JSONResponse({"data": {"errors": ["未授权: 缺少有效的 Admin Token"]}}, status_code=401)
    def _run():
        from core.shared import config
        from dotenv import load_dotenv

        load_dotenv(str(ENV_FILE), override=True)

        from backend.config_metadata import CONFIG_SCHEMA

        reloaded = []
        for item in CONFIG_SCHEMA:
            key = item["key"]
            env_key = item["env_key"]
            raw = os.getenv(env_key, item["default"])
            target_type = item["type"]

            if target_type == "bool":
                val = raw.lower() in ("true", "1", "yes", "on")
            elif target_type == "int":
                val = int(raw)
            elif target_type == "float":
                val = float(raw)
            elif target_type == "json":
                try:
                    val = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    val = raw
            else:
                val = raw

            try:
                setattr(config, key, val)
                reloaded.append(key)
            except Exception:
                pass

        return {"reloaded": len(reloaded), "keys": reloaded}
    result = await anyio.to_thread.run_sync(_run)
    return {"data": result}
