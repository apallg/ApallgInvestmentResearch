"""
MySQL daily_quotes → qlib 二进制格式 数据桥接
"""
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.adapters.qlib.utils import mysql_to_qlib_code

logger = logging.getLogger(__name__)

FIELDS = ["open", "high", "low", "close", "volume", "amount", "vwap", "factor"]


def _qfq_vwap(amount, volume, adj_factor):
    amount = np.asarray(amount, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    factors = np.asarray(adj_factor, dtype=np.float64)
    valid = np.isfinite(factors) & (factors > 0)
    result = np.full(len(amount), np.nan, dtype=np.float64)
    if not valid.any():
        return result
    raw_vwap = np.full(len(amount), np.nan, dtype=np.float64)
    np.divide(amount, volume, out=raw_vwap, where=volume > 0)
    anchor = factors[np.flatnonzero(valid)[-1]]
    np.multiply(raw_vwap, factors / anchor, out=result, where=valid)
    return result


class QlibDataBridge:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        from core.adapters.qlib.generation import (
            assert_no_active_historical_release,
            has_qlib_generation_pointer,
            qlib_generation_mutation_lock,
        )
        from core.shared import config

        qlib_root = Path(config.QLIB_DATA_ROOT).resolve()
        resolved_dir = self.data_dir.resolve()
        managed_path = resolved_dir == qlib_root or resolved_dir.is_relative_to(qlib_root)
        if managed_path:
            assert_no_active_historical_release(qlib_root)
        self.calendar_dir = self.data_dir / "calendars"
        self.instruments_dir = self.data_dir / "instruments"
        self.features_dir = self.data_dir / "features"
        directories = [self.calendar_dir, self.instruments_dir, self.features_dir]
        if managed_path and has_qlib_generation_pointer(qlib_root):
            if resolved_dir == qlib_root:
                return
            missing = [str(path) for path in directories if not path.is_dir()]
            if missing:
                raise RuntimeError(f"受管Qlib数据代目录不完整，拒绝自动创建: {', '.join(missing)}")
            return
        if managed_path:
            with qlib_generation_mutation_lock(qlib_root, require_legacy=True):
                for directory in directories:
                    directory.mkdir(parents=True, exist_ok=True)
        else:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)

    def _sync_generation(self, operation):
        from core.adapters.qlib.generation import (
            discard_qlib_generation_stage,
            has_qlib_generation_pointer,
            publish_qlib_generation,
            stage_qlib_generation,
        )
        from core.shared import config

        if not has_qlib_generation_pointer(config.QLIB_DATA_ROOT):
            return None
        managed_paths = {
            Path(config.QLIB_DATA_DIR).resolve(),
            Path(config.QLIB_DATA_ROOT).resolve(),
        }
        if self.data_dir.resolve() not in managed_paths:
            return None
        stage = stage_qlib_generation(data_root=config.QLIB_DATA_ROOT, lifecycle_lease=True, owner={"type": "data_sync", "operation": operation})
        try:
            staged_bridge = QlibDataBridge(stage["stage_dir"])
            result = getattr(staged_bridge, operation)()
        except Exception:
            try:
                discard_qlib_generation_stage(stage)
            except Exception as cleanup_error:
                logger.error("Qlib候选代同步失败后清理暂存目录失败: %s", cleanup_error)
            raise
        manifest_result = dict(result) if isinstance(result, dict) else result
        if isinstance(manifest_result, dict):
            manifest_result.pop("data_dir", None)
        published = publish_qlib_generation(
            stage,
            metadata={"operation": operation, "result": manifest_result},
        )
        self.data_dir = Path(published["data_dir"])
        self.calendar_dir = self.data_dir / "calendars"
        self.instruments_dir = self.data_dir / "instruments"
        self.features_dir = self.data_dir / "features"
        if isinstance(result, dict):
            result["generation_id"] = published["generation_id"]
            result["data_dir"] = published["data_dir"]
            result["restart_required"] = True
        return result

    def _sync_legacy(self, operation):
        from core.adapters.qlib.generation import qlib_generation_mutation_lock
        from core.shared import config

        managed_paths = {Path(config.QLIB_DATA_ROOT).resolve(), Path(config.QLIB_DATA_DIR).resolve()}
        mutation_root = config.QLIB_DATA_ROOT if self.data_dir.resolve() in managed_paths else self.data_dir
        with qlib_generation_mutation_lock(mutation_root, require_legacy=True):
            return getattr(self, operation)()

    def sync_all(self):
        published = self._sync_generation("_sync_all_in_place")
        if published is not None:
            return published
        return self._sync_legacy("_sync_all_in_place")

    def _sync_all_in_place(self):
        from core.data import Database
        from core.domain.screening.exclusion import is_excluded

        import shutil
        for d in self.features_dir.iterdir():
            if not d.is_dir():
                continue
            code = d.name[2:] if len(d.name) >= 3 and d.name[:2] in ("sh", "sz") else ""
            if is_excluded(code)[0]:
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass

        # 1. 生成日历
        dates_rows = Database.fetchall("SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date")
        dates = [row["trade_date"] for row in dates_rows]
        if not dates:
            raise RuntimeError("MySQL daily_quotes 表为空，请先采集数据")
        calendar_path = self.calendar_dir / "day.txt"
        with open(calendar_path, "w", encoding="utf-8") as f:
            for d in dates:
                d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                f.write(d_str + "\n")
        date_to_idx = {str(d): i for i, d in enumerate(dates)}

        # 2. 生成 instruments
        instruments_rows = [row for row in Database.fetchall(
            "SELECT code, MIN(trade_date) as start, MAX(trade_date) as end, COUNT(*) as cnt "
            "FROM daily_quotes GROUP BY code ORDER BY code"
        ) if not is_excluded(row["code"])[0]]
        from core.data.repos.instruments import InstrumentRepo
        a500_ranges = InstrumentRepo.get_stable_history_500d_ranges()
        inst_path = self.instruments_dir / "all.txt"
        inst_a500_path = self.instruments_dir / "a500.txt"
        all_stocks = []
        a500_stocks = []
        for row in instruments_rows:
            qlib_code = mysql_to_qlib_code(row["code"])
            start_str = row["start"].strftime("%Y-%m-%d") if hasattr(row["start"], "strftime") else str(row["start"])
            end_str = row["end"].strftime("%Y-%m-%d") if hasattr(row["end"], "strftime") else str(row["end"])
            line = f"{qlib_code}\t{start_str}\t{end_str}\n"
            all_stocks.append(line)
            if row["code"] in a500_ranges and not is_excluded(row["code"])[0]:
                eligible_start, eligible_end = a500_ranges[row["code"]]
                eligible_start = eligible_start.strftime("%Y-%m-%d") if hasattr(eligible_start, "strftime") else str(eligible_start)[:10]
                eligible_end = eligible_end.strftime("%Y-%m-%d") if hasattr(eligible_end, "strftime") else str(eligible_end)[:10]
                a500_stocks.append(f"{qlib_code}\t{eligible_start}\t{eligible_end}\n")
        with open(inst_path, "w", encoding="utf-8") as f:
            f.writelines(all_stocks)
        with open(inst_a500_path, "w", encoding="utf-8") as f:
            f.writelines(a500_stocks)

        # 3. 生成 features 目录（分批查询，每批用独立短连接）
        all_codes = [row["code"] for row in instruments_rows]
        n_total = len(all_codes)
        batch_size = 500
        n_done = 0
        anchor_state = {}

        for batch_start in range(0, n_total, batch_size):
            batch_codes = all_codes[batch_start:batch_start + batch_size]
            placeholders = ",".join(["%s"] * len(batch_codes))
            all_rows = Database.fetchall(
                f"SELECT code, trade_date, open, high, low, close, volume, amount, adj_factor "
                f"FROM daily_quotes WHERE code IN ({placeholders}) ORDER BY code, trade_date",
                tuple(batch_codes),
            )

            # 按 code 分组
            code_data = {}
            for row in all_rows:
                code = row["code"]
                if code not in code_data:
                    code_data[code] = {
                        "dates": [],
                        "values": {**{f: [] for f in FIELDS}, "adj_factor": []},
                    }
                d = row["trade_date"]
                d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                idx = date_to_idx.get(d_str)
                if idx is None:
                    continue
                code_data[code]["dates"].append(idx)
                code_data[code]["values"]["open"].append(float(row["open"]) if row["open"] is not None else np.nan)
                code_data[code]["values"]["high"].append(float(row["high"]) if row["high"] is not None else np.nan)
                code_data[code]["values"]["low"].append(float(row["low"]) if row["low"] is not None else np.nan)
                code_data[code]["values"]["close"].append(float(row["close"]) if row["close"] is not None else np.nan)
                code_data[code]["values"]["volume"].append(float(row["volume"]) if row["volume"] is not None else np.nan)
                code_data[code]["values"]["amount"].append(float(row["amount"]) if row["amount"] is not None else np.nan)
                factor = row.get("adj_factor")
                code_data[code]["values"]["adj_factor"].append(
                    float(factor) if factor is not None else np.nan
                )
                code_data[code]["values"]["factor"].append(1.0)

            for mysql_code, data in code_data.items():
                qlib_code = mysql_to_qlib_code(mysql_code)
                dates_arr = np.array(data["dates"])
                stock_dir = self.features_dir / qlib_code.lower()
                stock_dir.mkdir(parents=True, exist_ok=True)
                valid_factors = [
                    value for value in data["values"]["adj_factor"]
                    if np.isfinite(value) and value > 0
                ]
                if valid_factors:
                    anchor_state[mysql_code] = valid_factors[-1]

                for field in FIELDS:
                    if field == "vwap":
                        amount_arr = np.array(data["values"]["amount"], dtype=np.float64)
                        volume_arr = np.array(data["values"]["volume"], dtype=np.float64)
                        factor_arr = np.array(data["values"]["adj_factor"], dtype=np.float64)
                        values_arr = _qfq_vwap(amount_arr, volume_arr, factor_arr)
                    else:
                        values_arr = np.array(data["values"][field], dtype=np.float64)

                    start_idx = int(dates_arr[0])
                    end_idx = int(dates_arr[-1])
                    full = np.full(end_idx - start_idx + 1, np.nan, dtype=np.float64)
                    for i, idx in enumerate(dates_arr):
                        pos = int(idx) - start_idx
                        full[pos] = values_arr[i]

                    bin_path = stock_dir / f"{field}.day.bin"
                    np.hstack([np.float32(start_idx), full.astype(np.float32)]).astype("<f").tofile(str(bin_path))

                n_done += 1
            if (batch_start + batch_size) % 2000 == 0 or batch_start + batch_size >= n_total:
                logger.info(f"  数据桥接进度: {n_done}/{n_total}")

        from core.shared.persistence import save_json
        save_json(anchor_state, self.data_dir / "incremental_anchor_state.json")

        logger.info(f"数据桥接完成: {len(dates)} 个交易日, {n_total} 只股票")
        try:
            verify_price_adjustment()
        except Exception as e:
            logger.warning(f"[Bridge] 价格调整校验异常 (非致命): {e}")
        return {"dates": len(dates), "stocks": n_total, "data_dir": str(self.data_dir)}

    def _refresh_index_instruments(self, existing_stocks):
        """刷新 csi300.txt / csi500.txt / small_cap.txt 成分股文件。
        从 Tushare/akshare 拉取最新成分股，仅保留存在于 existing_stocks 中的代码。
        返回 True 表示刷新成功，False 表示跳过（获取失败）。
        """
        try:
            from core.data.sources.tushare import get_index_components
            from core.domain.screening.exclusion import is_excluded

            indexes = {
                "csi300": ("000300",),
                "csi500": ("000905",),
                "small_cap": ("000852", "399303"),
            }
            refreshed = 0
            for name, ts_codes in indexes.items():
                constituents = set()
                for ts_code in ts_codes:
                    constituents |= set(get_index_components(ts_code))
                if not constituents:
                    logger.warning(f"[Bridge] 获取 {name} 成分股失败，跳过")
                    continue
                target = self.instruments_dir / f"{name}.txt"
                written = 0
                with open(target, "w", encoding="utf-8") as f:
                    for code in constituents:
                        c = code.strip()
                        if not c.isdigit() or len(c) != 6:
                            continue
                        if is_excluded(c)[0]:
                            continue
                        qcode = f"sh{c}" if c.startswith(("6", "9")) else f"sz{c}"
                        date_range = existing_stocks.get(qcode)
                        if not date_range:
                            continue
                        f.write(f"{qcode}\t{date_range[0]}\t{date_range[1]}\n")
                        written += 1
                logger.info(f"[Bridge] 刷新 {name}.txt: {written} 只成分股")
                refreshed += 1
            return refreshed > 0
        except Exception as e:
            logger.warning(f"[Bridge] 指数成分股刷新失败: {e}")
            return False

    def sync_incremental(self):
        published = self._sync_generation("_sync_incremental_in_place")
        if published is not None:
            return published
        return self._sync_legacy("_sync_incremental_in_place")

    def _sync_incremental_in_place(self):
        """增量同步 — 追加新交易日 + 重写最近 10 天（覆盖 MySQL 价格变更）"""
        from core.data import Database
        from core.domain.screening.exclusion import is_excluded

        cal_path = self.calendar_dir / "day.txt"
        inst_path = self.instruments_dir / "all.txt"

        if not cal_path.exists():
            return self._sync_all_in_place()

        existing_dates = cal_path.read_text().strip().splitlines()
        if not existing_dates:
            return self._sync_all_in_place()

        last_date = existing_dates[-1]
        date_to_idx = {d: i for i, d in enumerate(existing_dates)}

        new_dates_rows = Database.fetchall(
            "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date > %s ORDER BY trade_date",
            (last_date,),
        )
        new_dates = [row["trade_date"] for row in new_dates_rows]

        recent_overlap = existing_dates[-10:] if len(existing_dates) >= 10 else existing_dates
        refresh_dates = [d for d in recent_overlap if d not in (new_dates or [])]
        all_sync_dates = list(refresh_dates) + [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                                                  for d in (new_dates or [])]
        all_sync_dates = sorted(set(all_sync_dates))

        if not new_dates and not refresh_dates:
            n_stocks = len(inst_path.read_text().strip().splitlines()) if inst_path.exists() else 0
            return {"dates": len(existing_dates), "stocks": n_stocks, "new_dates": 0,
                    "message": "数据已是最新"}

        new_date_strs = []
        for d in new_dates:
            d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            new_date_strs.append(d_str)

        _fallback_dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                           for d in (refresh_dates or [])]
        base_date = (new_date_strs[0] if new_date_strs
                     else _fallback_dates[0] if _fallback_dates
                     else existing_dates[-1])

        if new_date_strs:
            with open(cal_path, "a", encoding="utf-8") as f:
                for d in new_date_strs:
                    f.write(d + "\n")

        # 更新 date_to_idx (新日期从 base_idx 开始编号)
        base_idx = len(existing_dates)
        for i, d in enumerate(new_date_strs):
            date_to_idx[d] = base_idx + i

        # 3. 查询新日期 + 最近 N 天 (覆盖 MySQL 价格变更)
        placeholders = ",".join(["%s"] * len(all_sync_dates))
        new_rows = Database.fetchall(
            f"SELECT code, trade_date, open, high, low, close, volume, amount, adj_factor "
            f"FROM daily_quotes WHERE trade_date IN ({placeholders}) ORDER BY trade_date",
            tuple(all_sync_dates),
        )

        factors_by_code = {}
        latest_factors = {}
        for row in new_rows:
            factor = row.get("adj_factor")
            if factor is not None and float(factor) > 0:
                factors_by_code.setdefault(row["code"], set()).add(float(factor))
                latest_factors[row["code"]] = float(factor)
        from core.shared.persistence import load_json, save_json
        anchor_state_path = self.data_dir / "incremental_anchor_state.json"
        anchor_state = load_json(anchor_state_path) or {}
        rebased_codes = set()
        for code, factors in factors_by_code.items():
            latest_factor = latest_factors.get(code)
            previous_factor = anchor_state.get(code)
            if previous_factor is None:
                factor_changed = len(factors) > 1
            else:
                factor_changed = not np.isclose(
                    float(previous_factor), latest_factor, rtol=1e-12, atol=0,
                )
            if factor_changed:
                rebased_codes.add(code)
        if rebased_codes:
            code_placeholders = ",".join(["%s"] * len(rebased_codes))
            calendar_end = (new_date_strs or existing_dates)[-1]
            full_rows = Database.fetchall(
                f"SELECT code, trade_date, open, high, low, close, volume, amount, adj_factor "
                f"FROM daily_quotes WHERE code IN ({code_placeholders}) AND trade_date <= %s "
                f"ORDER BY code, trade_date",
                tuple(sorted(rebased_codes)) + (calendar_end,),
            )
            new_rows = [row for row in new_rows if row["code"] not in rebased_codes] + list(full_rows)

        # 按 code 分组
        code_new_data = {}
        for row in new_rows:
            code = row["code"]
            if code not in code_new_data:
                code_new_data[code] = {
                    "dates": [],
                    "values": {**{f: [] for f in FIELDS}, "adj_factor": []},
                }
            d = row["trade_date"]
            d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            idx = date_to_idx.get(d_str)
            if idx is None:
                continue
            code_new_data[code]["dates"].append(idx)
            code_new_data[code]["values"]["open"].append(float(row["open"]) if row["open"] is not None else np.nan)
            code_new_data[code]["values"]["high"].append(float(row["high"]) if row["high"] is not None else np.nan)
            code_new_data[code]["values"]["low"].append(float(row["low"]) if row["low"] is not None else np.nan)
            code_new_data[code]["values"]["close"].append(float(row["close"]) if row["close"] is not None else np.nan)
            code_new_data[code]["values"]["volume"].append(float(row["volume"]) if row["volume"] is not None else np.nan)
            code_new_data[code]["values"]["amount"].append(float(row["amount"]) if row["amount"] is not None else np.nan)
            factor = row.get("adj_factor")
            code_new_data[code]["values"]["adj_factor"].append(
                float(factor) if factor is not None else np.nan
            )
            code_new_data[code]["values"]["factor"].append(1.0)

        # 4. 更新 instruments/all.txt（可能有新上市的股票）
        inst_path = self.instruments_dir / "all.txt"
        existing_stocks = {}
        for line in inst_path.read_text().strip().splitlines():
            parts = line.split("\t")
            if parts:
                existing_stocks[parts[0]] = parts[1:] if len(parts) > 1 else []

        n_updated = 0
        n_new_stocks = 0

        for mysql_code, data in code_new_data.items():
            qlib_code = mysql_to_qlib_code(mysql_code).lower()
            stock_dir = self.features_dir / qlib_code
            stock_dir.mkdir(parents=True, exist_ok=True)

            dates_arr = np.array(data["dates"])
            if len(dates_arr) == 0:
                continue
            order = np.argsort(dates_arr)
            dates_arr = dates_arr[order]
            start_idx = int(dates_arr.min())
            end_idx = int(dates_arr.max())
            stock_is_new = not any((stock_dir / f"{field}.day.bin").exists() for field in FIELDS)

            for field in FIELDS:
                if field == "vwap":
                    amount_arr = np.array(data["values"]["amount"], dtype=np.float64)[order]
                    volume_arr = np.array(data["values"]["volume"], dtype=np.float64)[order]
                    factor_arr = np.array(data["values"]["adj_factor"], dtype=np.float64)[order]
                    row_vals = _qfq_vwap(amount_arr, volume_arr, factor_arr).astype(np.float32)
                else:
                    row_vals = np.array(data["values"][field], dtype=np.float32)[order]

                new_vals = np.full(end_idx - start_idx + 1, np.nan, dtype=np.float32)
                new_vals[dates_arr - start_idx] = row_vals

                bin_path = stock_dir / f"{field}.day.bin"

                if bin_path.exists() and mysql_code in rebased_codes:
                    merged = np.hstack([np.float32(start_idx), new_vals])
                    merged.astype("<f").tofile(str(bin_path))
                elif bin_path.exists():
                    existing = np.fromfile(str(bin_path), dtype="<f")
                    existing_start = int(existing[0])
                    existing_data = existing[1:]

                    existing_end = existing_start + len(existing_data) - 1
                    merged_start = min(existing_start, start_idx)
                    merged_end = max(existing_end, end_idx)
                    merged_data = np.full(merged_end - merged_start + 1, np.nan, dtype=np.float32)
                    existing_offset = existing_start - merged_start
                    merged_data[existing_offset:existing_offset + len(existing_data)] = existing_data
                    new_offset = start_idx - merged_start
                    merged_data[new_offset:new_offset + len(new_vals)] = new_vals
                    merged = np.hstack([np.float32(merged_start), merged_data])

                    merged.astype("<f").tofile(str(bin_path))
                else:
                    full = np.hstack([np.float32(start_idx), new_vals]).astype("<f")
                    full.tofile(str(bin_path))

            n_updated += 1
            if stock_is_new:
                n_new_stocks += 1

            existing_range = existing_stocks.get(qlib_code) or []
            calendar_dates = existing_dates + new_date_strs
            code_start = calendar_dates[start_idx]
            code_end_idx = int(dates_arr.max())
            code_end = calendar_dates[code_end_idx]
            existing_start = existing_range[0] if existing_range and existing_range[0] else code_start
            existing_end = existing_range[1] if len(existing_range) > 1 and existing_range[1] else code_end
            existing_stocks[qlib_code] = [existing_start, max(existing_end, code_end)]

        # 重写 instruments/all.txt + a500.txt
        from core.data.repos.instruments import InstrumentRepo
        a500_ranges = InstrumentRepo.get_stable_history_500d_ranges()
        with open(inst_path, "w", encoding="utf-8") as f:
            with open(self.instruments_dir / "a500.txt", "w", encoding="utf-8") as fa500:
                for code, date_range in sorted(existing_stocks.items()):
                    start_s = date_range[0] if date_range else base_date
                    end_s = date_range[1] if len(date_range) > 1 and date_range[1] else start_s
                    line = f"{code}\t{start_s}\t{end_s}\n"
                    f.write(line)
                    mysql_code = code[2:]  # sh600519 → 600519
                    if mysql_code in a500_ranges and not is_excluded(mysql_code)[0]:
                        eligible_start, eligible_end = a500_ranges[mysql_code]
                        eligible_start = eligible_start.strftime("%Y-%m-%d") if hasattr(eligible_start, "strftime") else str(eligible_start)[:10]
                        eligible_end = eligible_end.strftime("%Y-%m-%d") if hasattr(eligible_end, "strftime") else str(eligible_end)[:10]
                        fa500.write(f"{code}\t{eligible_start}\t{eligible_end}\n")

        # 刷新指数成分股文件 (csi300.txt / csi500.txt)
        index_refreshed = self._refresh_index_instruments(existing_stocks)
        anchor_state.update(latest_factors)
        save_json(anchor_state, anchor_state_path)

        logger.info(f"增量同步完成: +{len(new_date_strs)} 交易日, 更新 {n_updated} 只股票, 新增 {n_new_stocks} 只"
                    f"{', 成分股已刷新' if index_refreshed else ''}")
        return {
            "dates": len(existing_dates) + len(new_date_strs),
            "stocks": len(existing_stocks),
            "new_dates": len(new_date_strs),
            "updated_stocks": n_updated,
            "new_stocks": n_new_stocks,
            "rebased_stocks": len(rebased_codes),
            "index_refreshed": index_refreshed,
        }

    def get_date_coverage(self, decision_date, expected_codes):
        from core.adapters.qlib.generation import qlib_generation_read_lock
        from core.shared import config

        with qlib_generation_read_lock(config.QLIB_DATA_ROOT):
            decision_date = str(decision_date)[:10]
            cal_path = self.calendar_dir / "day.txt"
            dates = cal_path.read_text(encoding="utf-8").strip().splitlines() if cal_path.exists() else []
            if decision_date not in dates:
                return {
                    "expected": len(expected_codes), "available": 0, "coverage_rate": 0.0,
                    "missing_codes": list(expected_codes)[:100],
                }
            date_idx = dates.index(decision_date)
            missing = []
            available = 0
            for code in expected_codes:
                qlib_code = mysql_to_qlib_code(code).lower()
                bin_path = self.features_dir / qlib_code / "close.day.bin"
                if not bin_path.exists():
                    missing.append(code)
                    continue
                values = np.fromfile(str(bin_path), dtype="<f")
                if len(values) <= 1:
                    missing.append(code)
                    continue
                start_idx = int(values[0])
                offset = date_idx - start_idx + 1
                if offset <= 0 or offset >= len(values) or not np.isfinite(values[offset]):
                    missing.append(code)
                    continue
                available += 1
            expected = len(expected_codes)
            return {
                "expected": expected,
                "available": available,
                "coverage_rate": round(available / expected, 6) if expected else 0.0,
                "missing_codes": missing[:100],
            }

    def get_status(self):
        """返回当前 qlib 数据状态"""
        from core.shared import config
        from core.adapters.qlib.generation import qlib_generation_read_lock

        qlib_root = Path(config.QLIB_DATA_ROOT).resolve()
        with qlib_generation_read_lock(qlib_root):
            cal_path = self.calendar_dir / "day.txt"
            inst_path = self.instruments_dir / "all.txt"
            pointer_path = qlib_root / "current.json"
            current_generation_id = None
            if pointer_path.is_file():
                with pointer_path.open("r", encoding="utf-8") as stream:
                    current_generation_id = json.load(stream).get("generation_id")
            restart_required = current_generation_id is not None and current_generation_id != config.QLIB_DATA_GENERATION_ID
            if not cal_path.exists() or not inst_path.exists():
                return {
                    "synced": False,
                    "dates": 0,
                    "stocks": 0,
                    "generation_id": config.QLIB_DATA_GENERATION_ID,
                    "current_generation_id": current_generation_id,
                    "restart_required": restart_required,
                }

            dates = cal_path.read_text().strip().splitlines()
            stocks = inst_path.read_text().strip().splitlines()
            first_date = dates[0] if dates else None
            last_date = dates[-1] if dates else None
            return {
                "synced": True,
                "dates": len(dates),
                "stocks": len(stocks),
                "first_date": first_date,
                "last_date": last_date,
                "generation_id": config.QLIB_DATA_GENERATION_ID,
                "current_generation_id": current_generation_id,
                "restart_required": restart_required,
            }


def verify_price_adjustment(sample_size=5):
    """抽样校验 MySQL K线是否为前复权价格 (factor 列硬编码 1.0 的前提)。

    对比最近 250 条 MySQL 收盘价与 Tushare 原始价按复权因子计算的前复权价:
    - 偏差 <0.5%: 通过
    - 偏差 <5%: 警告 (可能后复权或少量除权差异)
    - 偏差 >=5%: 错误 (可能未复权, Qlib 收益率计算会出错)
    """
    from core.data import Database
    if not Database.is_available():
        logger.warning("[Bridge] MySQL 不可用, 跳过价格调整校验")
        return

    codes = [r["code"] for r in
             Database.fetchall("SELECT DISTINCT code FROM daily_quotes ORDER BY RAND() LIMIT %s",
                               (sample_size,))]
    if not codes:
        logger.warning("[Bridge] daily_quotes 为空, 跳过校验")
        return

    try:
        from core.data.sources.tushare import get_pro
        pro = get_pro()
        if not pro:
            raise Exception("Tushare 不可用")
    except Exception:
        logger.warning("[Bridge] Tushare 不可用, 跳过价格调整校验")
        return

    ok, warn, fail = 0, 0, 0
    for code in codes:
        try:
            rows = Database.fetchall(
                "SELECT trade_date, close FROM daily_quotes WHERE code=%s "
                "ORDER BY trade_date DESC LIMIT 250", (code,))
            if not rows:
                continue

            start_date = min(row["trade_date"] for row in rows).strftime("%Y%m%d")
            end_date = max(row["trade_date"] for row in rows).strftime("%Y%m%d")

            ts_code = f"{code}.{'SH' if code.startswith(('6','9','5')) else 'SZ'}"
            daily = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            factors = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if daily is None or daily.empty or factors is None or factors.empty:
                continue

            raw_by_date = {
                str(row["trade_date"]): float(row["close"])
                for _, row in daily.iterrows()
                if row.get("close") is not None
            }
            factor_by_date = {
                str(row["trade_date"]): float(row["adj_factor"])
                for _, row in factors.iterrows()
                if row.get("adj_factor") is not None
            }
            if not factor_by_date:
                continue
            anchor_factor = factor_by_date[max(factor_by_date)]
            if anchor_factor <= 0:
                continue

            differences = []
            for row in rows:
                trade_date = row["trade_date"].strftime("%Y%m%d")
                raw_close = raw_by_date.get(trade_date)
                adj_factor = factor_by_date.get(trade_date)
                if raw_close is None or adj_factor is None:
                    continue
                qfq_close = raw_close * adj_factor / anchor_factor
                if qfq_close > 0:
                    differences.append(abs(float(row["close"]) - qfq_close) / qfq_close)
            if not differences:
                continue

            pct_diff = max(differences)
            if pct_diff < 0.005:
                ok += 1
            elif pct_diff < 0.05:
                warn += 1
                logger.warning(f"[Bridge] {code} 最近250条最大前复权价格偏差={pct_diff:.2%}")
            else:
                fail += 1
                logger.error(f"[Bridge] {code} 最近250条最大前复权价格严重偏差={pct_diff:.2%}")
        except Exception as e:
            logger.debug(f"[Bridge] {code} 校验异常: {e}")

    if fail > 0:
        logger.error(f"[Bridge] 价格调整校验 FAIL: {fail}/{sample_size} 只股票偏差>=5%, "
                     f"Qlib factor=1.0 假设可能不成立! 请确认 MySQL daily_quotes 为前复权价格")
    elif warn > 0:
        logger.warning(f"[Bridge] 价格调整校验: {ok} 通过, {warn} 偏差 <5% (可接受)")
    else:
        logger.info(f"[Bridge] 价格调整校验通过 ({ok}/{sample_size}) — factor=1.0 安全")
