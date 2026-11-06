import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from core.adapters.qlib.bridge import FIELDS
from core.adapters.qlib.utils import mysql_to_qlib_code


_TOLERANCES = {
    "open": (0.0, 0.0051), "high": (0.0, 0.0051), "low": (0.0, 0.0051), "close": (0.0, 0.0051),
    "volume": (0.0, 1.0), "amount": (0.0, 16.0), "vwap": (0.0, 0.0051), "factor": (0.0, 0.000001),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_stock_features(data_dir, code):
    data_dir = Path(data_dir)
    qlib_code = mysql_to_qlib_code(str(code).zfill(6)).lower()
    stock_dir = data_dir / "features" / qlib_code
    calendar_path = data_dir / "calendars" / "day.txt"
    if not calendar_path.is_file():
        return {"status": "blocked", "blockers": ["Qlib 日历文件不存在"], "frame": pd.DataFrame()}
    calendar = pd.to_datetime(calendar_path.read_text(encoding="utf-8").splitlines(), errors="coerce")
    if calendar.isna().any() or calendar.duplicated().any():
        return {"status": "blocked", "blockers": ["Qlib 日历无效或重复"], "frame": pd.DataFrame()}
    columns = {"date": pd.Series(calendar)}
    blockers = []
    for field in FIELDS:
        path = stock_dir / f"{field}.day.bin"
        if not path.is_file():
            blockers.append(f"Qlib 缺少整股特征 {field}.day.bin")
            continue
        values = np.fromfile(path, dtype="<f")
        if values.size < 2 or not np.isfinite(values[0]) or not np.isclose(values[0], int(values[0]), rtol=0, atol=0):
            blockers.append(f"Qlib 特征 {field}.day.bin 格式无效")
            continue
        start = int(values[0])
        end = start + values.size - 2
        if start < 0 or end >= len(calendar):
            blockers.append(f"Qlib 特征 {field}.day.bin 超出日历范围")
            continue
        aligned = np.full(len(calendar), np.nan, dtype=np.float64)
        aligned[start:end + 1] = values[1:].astype(np.float64)
        columns[field] = aligned
    if blockers:
        return {"status": "blocked", "blockers": blockers, "frame": pd.DataFrame()}
    return {"status": "ready", "blockers": [], "frame": pd.DataFrame(columns)}


def prepare_stock_patch(data_dir, code, desired, stage_dir, allow_conflicts=False, clear_dates=None, fields=None):
    data_dir = Path(data_dir)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    qlib_code = mysql_to_qlib_code(str(code).zfill(6)).lower()
    stock_dir = data_dir / "features" / qlib_code
    calendar_path = data_dir / "calendars" / "day.txt"
    instruments_path = data_dir / "instruments" / "all.txt"
    if not calendar_path.is_file() or not instruments_path.is_file():
        return {"status": "blocked", "blockers": ["Qlib 缺少行情日历或 instruments/all.txt"], "files": {}}
    calendar = pd.to_datetime(calendar_path.read_text(encoding="utf-8").splitlines(), errors="coerce")
    if calendar.isna().any() or calendar.duplicated().any():
        return {"status": "blocked", "blockers": ["Qlib 日历含无效或重复日期"], "files": {}}
    date_to_index = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(calendar)}
    frame = desired.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        return {"status": "blocked", "blockers": ["Qlib 候选日期无效或重复"], "files": {}}
    blockers = []
    absent_dates = [date.strftime("%Y-%m-%d") for date in frame["date"] if date.strftime("%Y-%m-%d") not in date_to_index]
    if absent_dates:
        blockers.append(f"Qlib 日历缺少候选日期 {len(absent_dates)} 个，需先构造新日历版本")
    if blockers:
        return {"status": "blocked", "blockers": blockers, "files": {}}

    target_indexes = [date_to_index[date.strftime("%Y-%m-%d")] for date in frame["date"]]
    clear_dates = sorted({pd.Timestamp(value).normalize() for value in (clear_dates or [])})
    clear_date_keys = {date.strftime("%Y-%m-%d") for date in clear_dates}
    candidate_date_keys = {date.strftime("%Y-%m-%d") for date in frame["date"]}
    if clear_date_keys & candidate_date_keys:
        return {"status": "blocked", "blockers": ["Qlib 清空日期与行情候选日期重叠"], "files": {}}
    missing_clear_dates = sorted(clear_date_keys - set(date_to_index))
    if missing_clear_dates:
        return {"status": "blocked", "blockers": [f"Qlib 日历缺少已核验停牌日期 {len(missing_clear_dates)} 个"], "files": {}}
    clear_indexes = {date_to_index[date_key] for date_key in clear_date_keys}
    required_start = min([*target_indexes, *clear_indexes])
    required_end = max([*target_indexes, *clear_indexes])
    instrument_lines = instruments_path.read_text(encoding="utf-8").splitlines()
    instrument_matches = []
    for index, line in enumerate(instrument_lines):
        parts = line.split("\t")
        if parts and parts[0].lower() == qlib_code:
            instrument_matches.append((index, parts))
    if len(instrument_matches) > 1:
        return {"status": "blocked", "blockers": [f"Qlib instruments/all.txt 中 {qlib_code} 重复"], "files": {}}
    target_start = frame["date"].min()
    target_end = frame["date"].max()
    instrument_changed = False
    if instrument_matches:
        line_index, parts = instrument_matches[0]
        if len(parts) != 3:
            return {"status": "blocked", "blockers": [f"Qlib instruments/all.txt 中 {qlib_code} 格式无效"], "files": {}}
        try:
            instrument_start = pd.Timestamp(parts[1]).normalize()
            instrument_end = pd.Timestamp(parts[2]).normalize()
        except Exception:
            return {"status": "blocked", "blockers": [f"Qlib instruments/all.txt 中 {qlib_code} 日期无效"], "files": {}}
        if pd.isna(instrument_start) or pd.isna(instrument_end) or instrument_end < instrument_start:
            return {"status": "blocked", "blockers": [f"Qlib instruments/all.txt 中 {qlib_code} 日期区间无效"], "files": {}}
        new_start = min(instrument_start, target_start)
        new_end = max(instrument_end, target_end)
        if new_start != instrument_start or new_end != instrument_end:
            instrument_lines[line_index] = "\t".join((qlib_code, new_start.strftime("%Y-%m-%d"), new_end.strftime("%Y-%m-%d")))
            instrument_changed = True
    else:
        instrument_lines.append("\t".join((qlib_code, target_start.strftime("%Y-%m-%d"), target_end.strftime("%Y-%m-%d"))))
        instrument_lines.sort(key=lambda line: line.split("\t", 1)[0])
        instrument_changed = True

    files = {}
    selected_fields = set(fields or FIELDS)
    if not selected_fields.issubset(set(FIELDS)):
        return {"status": "blocked", "blockers": ["Qlib 修复字段不在允许列表"], "files": {}}
    replaced_conflicts = []
    for field in FIELDS:
        if field not in selected_fields:
            continue
        source = stock_dir / f"{field}.day.bin"
        before_sha256 = _sha256(source) if source.is_file() else None
        if before_sha256 is None:
            start_index = required_start
            end_index = required_end
            values = np.full(end_index - start_index + 1, np.nan, dtype="<f")
        else:
            original = np.fromfile(source, dtype="<f")
            if original.size < 2 or not np.isfinite(original[0]):
                blockers.append(f"Qlib 特征文件 {field}.day.bin 格式无效")
                continue
            old_start = int(original[0])
            if not np.isclose(original[0], old_start, rtol=0, atol=0):
                blockers.append(f"Qlib 特征文件 {field}.day.bin 起始下标无效")
                continue
            old_values = original[1:].copy()
            old_end = old_start + len(old_values) - 1
            start_index = min(old_start, required_start)
            end_index = max(old_end, required_end)
            values = np.full(end_index - start_index + 1, np.nan, dtype="<f")
            values[old_start - start_index:old_end - start_index + 1] = old_values
        if field not in frame:
            blockers.append(f"Qlib 发布候选缺少 {field} 字段")
            continue
        changed = False
        for row, calendar_index in zip(frame.itertuples(index=False), target_indexes):
            date = row.date.strftime("%Y-%m-%d")
            offset = calendar_index - start_index
            desired_value = pd.to_numeric(pd.Series([getattr(row, field)]), errors="coerce").iloc[0]
            if pd.isna(desired_value):
                continue
            current = float(values[offset])
            rtol, atol = _TOLERANCES[field]
            if np.isfinite(current) and current > 0:
                if not np.isclose(current, float(desired_value), rtol=rtol, atol=atol):
                    if allow_conflicts:
                        replaced_conflicts.append({
                            "field": field,
                            "date": date,
                            "previous_value": current,
                            "candidate_value": float(desired_value),
                        })
                        values[offset] = float(desired_value)
                        changed = True
                    else:
                        blockers.append(f"Qlib {field} 在 {date} 与发布候选冲突")
                continue
            values[offset] = float(desired_value)
            changed = True
        if field in ("open", "high", "low", "close", "volume", "amount", "vwap"):
            for calendar_index in clear_indexes:
                offset = calendar_index - start_index
                if 0 <= offset < len(values) and not np.isnan(values[offset]):
                    values[offset] = np.nan
                    changed = True
        if changed:
            staged = stage_dir / f"{field}.day.bin"
            np.concatenate((np.asarray([start_index], dtype="<f"), values.astype("<f"))).tofile(staged)
            files[field] = {
                "source": str(source.resolve()),
                "staged": str(staged.resolve()),
                "before_sha256": before_sha256,
                "after_sha256": _sha256(staged),
                "backup_name": f"{field}.day.bin",
            }
    if instrument_changed and not blockers:
        stage = stage_dir / "instruments_all.txt"
        stage.write_text("\n".join(instrument_lines) + "\n", encoding="utf-8")
        files["instruments"] = {
            "source": str(instruments_path.resolve()),
            "staged": str(stage.resolve()),
            "before_sha256": _sha256(instruments_path),
            "after_sha256": _sha256(stage),
            "backup_name": "instruments_all.txt",
        }
    if blockers:
        files = {}
    return {
        "status": "blocked" if blockers else "ready",
        "blockers": blockers,
        "files": files,
        "qlib_code": qlib_code,
        "replaced_conflicts": replaced_conflicts,
        "cleared_suspension_dates": sorted(clear_date_keys),
    }


def promote_stock_patch(files):
    for field, item in files.items():
        source = Path(item["source"])
        staged = Path(item["staged"])
        before_sha256 = item.get("before_sha256")
        if before_sha256 is None:
            if source.exists():
                raise RuntimeError(f"Qlib {field} 在发布前已出现，拒绝覆盖")
        elif not source.is_file() or _sha256(source) != before_sha256:
            raise RuntimeError(f"Qlib {field} 在发布前已变化，拒绝覆盖")
        if not staged.is_file() or _sha256(staged) != item["after_sha256"]:
            raise RuntimeError(f"Qlib {field} 候选文件哈希不符")
        source.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, source)
        if _sha256(source) != item["after_sha256"]:
            raise RuntimeError(f"Qlib {field} 发布后哈希校验失败")


def restore_stock_patch(files, backup_dir, expected_after=False, allow_partial=False):
    backup_dir = Path(backup_dir)
    for field, item in files.items():
        source = Path(item["source"])
        before_sha256 = item.get("before_sha256")
        if before_sha256 is None:
            current_hash = _sha256(source) if source.is_file() else None
            if current_hash is None:
                continue
            if current_hash != item["after_sha256"]:
                raise RuntimeError(f"Qlib {field} 当前内容变化，拒绝删除恢复")
            source.unlink()
            continue
        expected = item["after_sha256"] if expected_after else item["before_sha256"]
        current_hash = _sha256(source) if source.is_file() else None
        allowed_hashes = {item["before_sha256"], item["after_sha256"]} if allow_partial else {expected}
        if current_hash is not None and current_hash not in allowed_hashes:
            raise RuntimeError(f"Qlib {field} 当前内容变化，拒绝覆盖恢复")
        backup = backup_dir / item.get("backup_name", f"{field}.day.bin")
        if not backup.is_file() or _sha256(backup) != item["before_sha256"]:
            raise RuntimeError(f"Qlib {field} 回滚备份不存在或哈希不符")
        temporary = source.with_suffix(source.suffix + ".restore")
        shutil.copy2(backup, temporary)
        os.replace(temporary, source)
        if _sha256(source) != item["before_sha256"]:
            raise RuntimeError(f"Qlib {field} 回滚后哈希校验失败")

