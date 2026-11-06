import hashlib
import json
import logging
import os
import shutil
import errno
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from core.shared import config as app_config
from core.shared.persistence import now_iso, save_json_atomic
from core.shared.qlib_errors import QlibRestartRequired


logger = logging.getLogger(__name__)
_configured_qlib_root_at_import = Path(app_config.QLIB_DATA_ROOT).resolve()
_generation_pointer_thread_lock = threading.Lock()
_generation_pointer_local = threading.local()
_generation_read_local = threading.local()
_generation_stage_leases = {}
_generation_stage_leases_lock = threading.Lock()


@contextmanager
def _generation_pointer_lock(root):
    root = Path(root).resolve()
    if str(root) in getattr(_generation_read_local, "roots", set()):
        raise RuntimeError("持有Qlib读租约期间拒绝获取generation写锁")
    root_key = str(root)
    held_roots = getattr(_generation_pointer_local, "roots", set())
    if root_key in held_roots:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".generation-pointer.lock"
    with _generation_pointer_thread_lock:
        with lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            if os.name == "nt":
                import msvcrt

                while True:
                    handle.seek(0)
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as error:
                        if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        time.sleep(0.05)
                held_roots = set(held_roots)
                held_roots.add(root_key)
                _generation_pointer_local.roots = held_roots
                try:
                    yield
                finally:
                    held_roots.remove(root_key)
                    _generation_pointer_local.roots = held_roots
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                held_roots = set(held_roots)
                held_roots.add(root_key)
                _generation_pointer_local.roots = held_roots
                try:
                    yield
                finally:
                    held_roots.remove(root_key)
                    _generation_pointer_local.roots = held_roots
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _stage_lease_path(root, generation_id):
    return Path(root) / "generations" / ".stage-leases" / f"{generation_id}.lock"


def _try_acquire_stage_lease(root, generation_id):
    key = (str(Path(root).resolve()), generation_id)
    with _generation_stage_leases_lock:
        if key in _generation_stage_leases:
            return None
        path = _stage_lease_path(key[0], generation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return None
            raise
        _generation_stage_leases[key] = handle
    return handle


def _acquire_stage_lease(root, generation_id):
    key = (str(Path(root).resolve()), generation_id)
    with _generation_stage_leases_lock:
        if key in _generation_stage_leases:
            return
    if _try_acquire_stage_lease(root, generation_id) is None:
        raise RuntimeError(f"Qlib候选代正在被其他进程使用: {generation_id}")


def _release_stage_lease(root, generation_id):
    key = (str(Path(root).resolve()), generation_id)
    with _generation_stage_leases_lock:
        handle = _generation_stage_leases.pop(key, None)
    if handle is None:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _root(data_root=None):
    if data_root is not None:
        return Path(data_root).resolve()
    from core.shared import config
    return Path(config.QLIB_DATA_ROOT).resolve()


def _assert_provider_uri_matches_configured_data_dir(root):
    if root != _configured_qlib_root_at_import:
        return
    provider_uri = str(app_config.QLIB_PROVIDER_URI or "").strip()
    if provider_uri.startswith("file:"):
        provider_uri = provider_uri.removeprefix("file:")
    if not provider_uri or Path(provider_uri).resolve() != Path(app_config.QLIB_DATA_DIR).resolve():
        raise RuntimeError("QLIB_PROVIDER_URI 与受管Qlib数据目录不一致，拒绝读写")


@contextmanager
def qlib_generation_mutation_lock(data_root=None, require_legacy=False):
    root = _root(data_root)
    _assert_provider_uri_matches_configured_data_dir(root)
    if str(root) in getattr(_generation_read_local, "roots", set()):
        raise RuntimeError("持有Qlib读租约期间拒绝升级为generation写入")
    with _generation_pointer_lock(root):
        if require_legacy and (root / "current.json").exists():
            raise RuntimeError("Qlib已启用generation，拒绝直接修改旧数据目录")
        yield


@contextmanager
def qlib_generation_read_lock(data_root=None):
    root = _root(data_root)
    root_key = str(root)
    roots = getattr(_generation_read_local, "roots", set())
    if root_key in roots:
        assert_no_active_historical_release(root)
        yield
        return
    if root_key in getattr(_generation_pointer_local, "roots", set()):
        raise RuntimeError("Qlib generation写入期间拒绝开始读取")
    with _generation_pointer_lock(root):
        assert_no_active_historical_release(root)
        roots = set(roots)
        roots.add(root_key)
        _generation_read_local.roots = roots
        try:
            yield
        finally:
            roots.remove(root_key)
            _generation_read_local.roots = roots


def _pointer(root):
    path = root / "current.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    generation_id = value.get("generation_id")
    if not generation_id or Path(generation_id).name != generation_id:
        raise RuntimeError(f"Qlib当前代指针格式无效: {path}")
    generation_dir = root / "generations" / generation_id
    manifest_path = generation_dir / "generation.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Qlib当前代清单缺失: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (
        manifest.get("generation_id") != generation_id
        or manifest.get("status") != "ready"
        or value.get("manifest_sha256") != manifest_hash
        or not (generation_dir / "calendars" / "day.txt").is_file()
        or not (generation_dir / "instruments" / "all.txt").is_file()
        or not (generation_dir / "features").is_dir()
    ):
        raise RuntimeError(f"Qlib当前代清单无效: {manifest_path}")
    return value


def resolve_qlib_data_dir(data_root=None):
    root = _root(data_root)
    pointer = _pointer(root)
    return root / "generations" / pointer["generation_id"] if pointer else root


def has_qlib_generation_pointer(data_root=None):
    return (_root(data_root) / "current.json").is_file()


def _valid_release_marker_id(release_id):
    return (
        isinstance(release_id, str)
        and 1 <= len(release_id) <= 128
        and Path(release_id).name == release_id
        and release_id not in (".", "..")
        and release_id.replace("-", "").replace("_", "").isalnum()
    )


def mark_historical_release_active(release_id, manifest_path, data_root=None):
    root = _root(data_root)
    if not _valid_release_marker_id(release_id):
        raise ValueError("历史发布ID格式无效，拒绝登记活动发布")
    marker_dir = root.parent / "historical_repair" / "active_releases"
    marker_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic({
        "release_id": release_id,
        "manifest_path": str(Path(manifest_path).resolve()),
        "started_at": now_iso(),
    }, marker_dir / f"{release_id}.json")


def clear_historical_release_active(release_id, data_root=None):
    root = _root(data_root)
    if not _valid_release_marker_id(release_id):
        raise ValueError("历史发布ID格式无效，拒绝清理活动发布标记")
    marker = root.parent / "historical_repair" / "active_releases" / f"{release_id}.json"
    marker.unlink(missing_ok=True)


def assert_no_active_historical_release(data_root=None):
    root = _root(data_root)
    marker_dir = root.parent / "historical_repair" / "active_releases"
    if marker_dir.is_dir():
        for marker_path in sorted(marker_dir.glob("*.json")):
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                release_id = marker.get("release_id")
                manifest_path = Path(marker["manifest_path"])
                if marker_path.stem != release_id or not _valid_release_marker_id(release_id):
                    raise ValueError("活动发布标记身份无效")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("release_id") != release_id:
                    raise ValueError("活动发布清单身份不匹配")
                if manifest.get("status") in ("published", "rolled_back"):
                    continue
                raise RuntimeError(f"历史发布 {release_id} 状态为 {manifest.get('status') or 'unknown'}")
            except Exception as error:
                raise RuntimeError(f"Qlib被未完成历史发布阻断: {marker_path.name}: {error}") from error
    if root == _configured_qlib_root_at_import and (root / "current.json").is_file():
        pointer = _pointer(root)
        configured_dir = Path(app_config.QLIB_DATA_DIR).resolve()
        expected_dir = (root / "generations" / app_config.QLIB_DATA_GENERATION_ID).resolve() if app_config.QLIB_DATA_GENERATION_ID else root
        if (
            pointer["generation_id"] != app_config.QLIB_DATA_GENERATION_ID
            or configured_dir != expected_dir
        ):
            raise QlibRestartRequired(app_config.QLIB_DATA_GENERATION_ID, pointer["generation_id"])
    _assert_provider_uri_matches_configured_data_dir(root)


def _recover_interrupted_generation_stage_copies(root, generation_id=None):
    generations = root / "generations"
    if not generations.is_dir():
        return
    markers = [generations / f".stage-copy-intent-{generation_id}.json"] if generation_id else sorted(
        generations.glob(".stage-copy-intent-*.json")
    )
    for marker_path in markers:
        if not marker_path.is_file():
            continue
        marker_id = marker_path.name.removeprefix(".stage-copy-intent-").removesuffix(".json")
        if len(marker_id) != 32 or not marker_id.isalnum():
            raise RuntimeError(f"Qlib候选复制恢复清单ID无效: {marker_path.name}")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RuntimeError(f"Qlib候选复制恢复清单不可读: {marker_path.name}") from error
        copy_dir = generations / f".copying-{marker_id}"
        stage_dir = generations / f".staging-{marker_id}"
        status = marker.get("status")
        if (
            marker.get("generation_id") != marker_id
            or Path(marker.get("data_root", "")).resolve() != root
            or Path(marker.get("copy_dir", "")).resolve() != copy_dir.resolve()
            or Path(marker.get("stage_dir", "")).resolve() != stage_dir.resolve()
            or status not in ("copying", "ready", "publishing")
        ):
            raise RuntimeError(f"Qlib候选复制恢复清单身份不匹配: {marker_path.name}")
        current = _pointer(root)
        current_id = current.get("generation_id") if current else None
        if current_id == marker_id:
            if current.get("parent_generation_id") != marker.get("parent_generation_id"):
                raise RuntimeError(f"活动generation父代与候选复制恢复清单不匹配: {marker_path.name}")
            marker_path.unlink()
            _release_stage_lease(root, marker_id)
            continue
        if status != "copying":
            if (marker.get("owner") or {}).get("type") == "historical_release":
                continue
            lease = _try_acquire_stage_lease(root, marker_id)
            if lease is None:
                continue
        if current_id != marker.get("parent_generation_id"):
            raise RuntimeError(f"Qlib父代已变化，保留候选复制现场待恢复: {marker_path.name}")
        final_dir = generations / marker_id
        candidates = (copy_dir, stage_dir, final_dir) if status != "copying" else (copy_dir, stage_dir)
        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                raise RuntimeError(f"Qlib候选复制恢复目标类型异常，保留现场: {candidate}")
            if candidate == final_dir:
                manifest_path = candidate / "generation.json"
                if not manifest_path.is_file():
                    raise RuntimeError(f"Qlib未发布候选代清单缺失，保留现场: {candidate}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("generation_id") != marker_id or manifest.get("parent_generation_id") != marker.get("parent_generation_id"):
                    raise RuntimeError(f"Qlib未发布候选代身份不匹配，保留现场: {candidate}")
            shutil.rmtree(candidate)
        marker_path.unlink()
        _release_stage_lease(root, marker_id)


def stage_qlib_generation(source_dir=None, data_root=None, generation_id=None, expected_parent_generation_id=None, lifecycle_lease=False, owner=None):
    root = _root(data_root)
    with _generation_pointer_lock(root):
        _recover_interrupted_generation_stage_copies(root)
        parent = _pointer(root)
        parent_generation_id = parent.get("generation_id") if parent else None
        if expected_parent_generation_id is not None and parent_generation_id != expected_parent_generation_id:
            raise RuntimeError("Qlib当前代已变化，拒绝基于过期版本创建候选代")
        source = Path(source_dir).resolve() if source_dir is not None else resolve_qlib_data_dir(root)
        generations = root / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        generation_id = generation_id or uuid.uuid4().hex
        if not isinstance(generation_id, str) or not generation_id.isalnum() or len(generation_id) != 32:
            raise ValueError("Qlib候选代ID格式无效")
        stage_dir = generations / f".staging-{generation_id}"
        copy_dir = generations / f".copying-{generation_id}"
        intent_path = generations / f".stage-copy-intent-{generation_id}.json"
        if stage_dir.exists() or copy_dir.exists() or intent_path.exists():
            raise RuntimeError(f"Qlib候选暂存身份已存在: {generation_id}")
        if lifecycle_lease:
            _acquire_stage_lease(root, generation_id)
        try:
            save_json_atomic({
                "generation_id": generation_id,
                "data_root": str(root),
                "parent_generation_id": parent_generation_id,
                "copy_dir": str(copy_dir),
                "stage_dir": str(stage_dir),
                "status": "copying",
                "lifecycle_lease": bool(lifecycle_lease),
                "owner": owner or {"type": "generic"},
            }, intent_path)
            shutil.copytree(
                source,
                copy_dir,
                ignore=shutil.ignore_patterns(
                    "dataset_cache", "generation.json", "generations", "current.json",
                    ".generation-pointer.lock", ".stage-copy-intent-*.json",
                ),
            )
            os.replace(copy_dir, stage_dir)
            if lifecycle_lease:
                save_json_atomic({
                    "generation_id": generation_id,
                    "data_root": str(root),
                    "parent_generation_id": parent_generation_id,
                    "copy_dir": str(copy_dir),
                    "stage_dir": str(stage_dir),
                    "status": "ready",
                    "lifecycle_lease": True,
                    "owner": owner or {"type": "generic"},
                }, intent_path)
            else:
                intent_path.unlink()
        except Exception as error:
            cleanup_error = None
            try:
                shutil.rmtree(copy_dir, ignore_errors=True)
                shutil.rmtree(stage_dir, ignore_errors=True)
                if not copy_dir.exists() and not stage_dir.exists():
                    intent_path.unlink(missing_ok=True)
            except Exception as cleanup_exception:
                cleanup_error = cleanup_exception
            finally:
                _release_stage_lease(root, generation_id)
            if cleanup_error is not None:
                raise RuntimeError("Qlib候选复制失败且清理未完成，保留恢复清单待下次恢复") from error
            if copy_dir.exists() or stage_dir.exists():
                raise RuntimeError("Qlib候选复制失败且残留目录未清理，保留恢复清单待下次恢复") from error
            raise
        return {
            "generation_id": generation_id,
            "stage_dir": stage_dir,
            "data_root": root,
            "parent_generation_id": parent_generation_id,
            "lifecycle_lease": bool(lifecycle_lease),
            "owner": owner or {"type": "generic"},
        }


def discard_qlib_generation_stage(stage):
    root = _root(stage["data_root"])
    generation_id = stage.get("generation_id")
    stage_dir = Path(stage["stage_dir"]).resolve()
    generations = (root / "generations").resolve()
    if not isinstance(generation_id, str) or len(generation_id) != 32 or not generation_id.isalnum():
        raise ValueError("Qlib候选代ID格式无效，拒绝清理")
    if stage_dir.name != f".staging-{generation_id}" or not stage_dir.is_relative_to(generations):
        raise RuntimeError("Qlib候选暂存目录身份无效，拒绝清理")
    with _generation_pointer_lock(root):
        if stage.get("lifecycle_lease"):
            _acquire_stage_lease(root, generation_id)
        _recover_interrupted_generation_stage_copies(root, generation_id)
        current = _pointer(root)
        if stage_dir.exists() and current and current.get("generation_id") == generation_id:
            raise RuntimeError("Qlib候选代已成为活动代，拒绝清理")
        final_dir = generations / generation_id
        intent_path = root / "generations" / f".stage-copy-intent-{generation_id}.json"
        intent = json.loads(intent_path.read_text(encoding="utf-8")) if intent_path.is_file() else {}
        candidates = [stage_dir]
        if intent.get("status") == "publishing" and (stage.get("owner") or {}).get("type") != "historical_release":
            candidates.append(final_dir)
        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                raise RuntimeError(f"Qlib候选清理目标类型异常，拒绝清理: {candidate}")
            if candidate == final_dir:
                manifest_path = candidate / "generation.json"
                if not manifest_path.is_file():
                    raise RuntimeError(f"Qlib未发布候选代清单缺失，拒绝清理: {candidate}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("generation_id") != generation_id
                    or manifest.get("parent_generation_id") != stage.get("parent_generation_id")
                ):
                    raise RuntimeError(f"Qlib未发布候选代身份不匹配，拒绝清理: {candidate}")
            shutil.rmtree(candidate)
        intent_path.unlink(missing_ok=True)
        _release_stage_lease(root, generation_id)


def publish_qlib_generation(stage, metadata=None, expected_generation_id=None):
    root = _root(stage["data_root"])
    with _generation_pointer_lock(root):
        return _publish_qlib_generation(stage, metadata, expected_generation_id)


def _publish_qlib_generation(stage, metadata=None, expected_generation_id=None):
    root = _root(stage["data_root"])
    generation_id = stage["generation_id"]
    if stage.get("lifecycle_lease"):
        intent_path = root / "generations" / f".stage-copy-intent-{generation_id}.json"
        if not intent_path.is_file():
            raise RuntimeError("Qlib候选生命周期恢复清单缺失，拒绝发布")
        try:
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RuntimeError("Qlib候选生命周期恢复清单不可读，拒绝发布") from error
        stage_dir = Path(stage["stage_dir"]).resolve()
        expected_owner = stage.get("owner") or {"type": "generic"}
        if (
            intent.get("generation_id") != generation_id
            or Path(intent.get("data_root", "")).resolve() != root
            or Path(intent.get("stage_dir", "")).resolve() != stage_dir
            or intent.get("parent_generation_id") != stage.get("parent_generation_id")
            or intent.get("lifecycle_lease") is not True
            or intent.get("owner") != expected_owner
            or intent.get("status") not in ("ready", "publishing")
        ):
            raise RuntimeError("Qlib候选生命周期恢复清单身份不匹配，拒绝发布")
        _acquire_stage_lease(root, generation_id)
        intent["status"] = "publishing"
        save_json_atomic(intent, intent_path)
    stage_dir = Path(stage["stage_dir"]).resolve()
    generations = (root / "generations").resolve()
    if not stage_dir.is_relative_to(generations) or not stage_dir.is_dir():
        raise RuntimeError("Qlib候选目录不在受管代际目录内")
    for required in ("calendars/day.txt", "instruments/all.txt", "features"):
        if not (stage_dir / required).exists():
            raise RuntimeError(f"Qlib候选缺少必要数据: {required}")
    current = _pointer(root)
    current_id = current.get("generation_id") if current else None
    expected_id = stage.get("parent_generation_id") if expected_generation_id is None else expected_generation_id
    if current_id != expected_id:
        raise RuntimeError("Qlib当前代已变化，拒绝基于过期版本发布")
    if generation_id != stage_dir.name.removeprefix(".staging-"):
        raise RuntimeError("Qlib候选代ID与临时目录不匹配")
    final_dir = generations / generation_id
    if final_dir.exists():
        raise RuntimeError(f"Qlib目标代已存在: {final_dir}")
    manifest = {
        "schema_version": 1,
        "generation_id": generation_id,
        "status": "ready",
        "created_at": now_iso(),
        "parent_generation_id": current_id,
        "metadata": metadata or {},
    }
    save_json_atomic(manifest, stage_dir / "generation.json")
    replace_delays = (0.1, 0.25, 0.5, 1, 2, 4)
    for attempt, delay in enumerate(replace_delays):
        try:
            os.replace(stage_dir, final_dir)
            break
        except PermissionError as error:
            if final_dir.exists() and not stage_dir.exists():
                break
            if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32) or attempt == len(replace_delays) - 1:
                raise
            time.sleep(delay)
    manifest_hash = hashlib.sha256((final_dir / "generation.json").read_bytes()).hexdigest()
    pointer = {
        "generation_id": generation_id,
        "manifest_sha256": manifest_hash,
        "parent_generation_id": current_id,
    }
    if _pointer(root) != current:
        raise RuntimeError("Qlib当前代在候选晋升期间被外部修改，拒绝覆盖并保留现场")
    try:
        save_json_atomic(pointer, root / "current.json")
    except Exception:
        if _pointer(root) != pointer:
            raise
        logger.warning("Qlib代%s的current指针写入调用报错，但持久化指针已核验为新代", generation_id)
    if _pointer(root) != pointer:
        raise RuntimeError("Qlib当前指针在发布期间被外部修改，保留现场并要求恢复核验")
    intent_path = root / "generations" / f".stage-copy-intent-{generation_id}.json"
    recovery_pending = False
    try:
        intent_path.unlink(missing_ok=True)
    except OSError as error:
        recovery_pending = True
        logger.error("Qlib代%s已提交，但恢复intent清理失败: %s", generation_id, error)
    try:
        _release_stage_lease(root, generation_id)
    except Exception as error:
        recovery_pending = True
        logger.exception("Qlib代%s已提交，但stage lease释放报告异常: %s", generation_id, error)
    return {
        "generation_id": generation_id,
        "data_dir": str(final_dir),
        "pointer": pointer,
        "recovery_pending": recovery_pending,
    }


def rollback_qlib_generation(generation_id, data_root=None, expected_current_generation_id=None):
    root = _root(data_root)
    with _generation_pointer_lock(root):
        return _rollback_qlib_generation(generation_id, root, expected_current_generation_id)


def _rollback_qlib_generation(generation_id, data_root=None, expected_current_generation_id=None):
    root = _root(data_root)
    current = _pointer(root)
    current_id = current.get("generation_id") if current else None
    if current_id != expected_current_generation_id:
        raise RuntimeError("Qlib当前代已变化，拒绝覆盖式回滚")
    if generation_id is None:
        if current is None:
            raise RuntimeError("Qlib当前没有受管代，无法回滚到旧目录")
        pointer_path = root / "current.json"
        history_dir = root / "generations" / "pointer_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        archived_pointer = history_dir / f"current-{current_id}-{uuid.uuid4().hex}.json"
        os.replace(pointer_path, archived_pointer)
        return {"generation_id": None, "legacy_data_dir": str(root), "archived_pointer": str(archived_pointer)}
    if Path(generation_id).name != generation_id:
        raise RuntimeError("Qlib回滚目标代ID无效")
    target_dir = root / "generations" / generation_id
    manifest_path = target_dir / "generation.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Qlib回滚目标代不存在: {target_dir}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("generation_id") != generation_id
        or manifest.get("status") != "ready"
        or not (target_dir / "calendars" / "day.txt").is_file()
        or not (target_dir / "instruments" / "all.txt").is_file()
        or not (target_dir / "features").is_dir()
    ):
        raise RuntimeError(f"Qlib回滚目标代清单无效: {manifest_path}")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer = {
        "generation_id": generation_id,
        "manifest_sha256": manifest_hash,
        "parent_generation_id": current_id,
        "rollback_from_generation_id": current_id,
    }
    save_json_atomic(pointer, root / "current.json")
    return pointer
