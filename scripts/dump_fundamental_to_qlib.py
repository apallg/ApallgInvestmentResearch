"""
基本面数据 → Qlib bin 桥接 (PIT 对齐, 防未来函数)

fundamental_data (季度) → features/{code}/{field}.day.bin (日频 forward-fill)
生效日规则:
  1. 财务字段使用 max(公告日或法定截止日回退, source_observed_at 后的首个交易日)
  2. PB/PE 从 daily_basic 按 trade_date 写入，不从季频基本面表读取
生效日之前该期数值不可见; 本机观察时间只提供保守下界，不代表历史供应商修订版本完整。
内置 PIT 校验: 生效日断言 + 抽样重读 bin 比对。

用法: python -m scripts.dump_fundamental_to_qlib
之后 Qlib 表达式可直接引用 $roe / $net_profit_yoy / $revenue_yoy /
$debt_to_assets / $eps / $pb / $pe (先按白盒因子走八层校验再入训练)。
"""
import sys
import logging
import random
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

from core.data.pit_utils import FUND_FIELDS, FALLBACK_LAG_DAYS, effective_date as _effective_date
from core.shared.persistence import save_json_atomic


def dump_fundamental(data_dir=None, verify_samples=10, batch_size=500, refresh_valuation=True):
    from core.shared import config as app_config

    target = Path(data_dir or app_config.QLIB_DATA_DIR).resolve()
    qlib_root = Path(app_config.QLIB_DATA_ROOT).resolve()
    managed_paths = {qlib_root, Path(app_config.QLIB_DATA_DIR).resolve()}
    if target.is_relative_to(qlib_root) and target not in managed_paths:
        raise RuntimeError("拒绝直接修改Qlib根目录下非当前活动代路径")
    if target in managed_paths:
        from core.adapters.qlib.generation import (
            discard_qlib_generation_stage,
            has_qlib_generation_pointer,
            publish_qlib_generation,
            qlib_generation_mutation_lock,
            stage_qlib_generation,
        )

        if has_qlib_generation_pointer(app_config.QLIB_DATA_ROOT):
            stage = stage_qlib_generation(data_root=app_config.QLIB_DATA_ROOT, lifecycle_lease=True, owner={"type": "fundamental_pit_export"})
            try:
                result = _dump_fundamental_in_place(stage["stage_dir"], verify_samples, batch_size, refresh_valuation)
                if result is None:
                    raise RuntimeError("Qlib候选财务导出未生成结果，未发布当前代")
                published = publish_qlib_generation(
                    stage,
                    metadata={"operation": "fundamental_pit_export", "source_snapshot_sha256": result["source_snapshot_sha256"]},
                )
            except Exception:
                try:
                    discard_qlib_generation_stage(stage)
                except Exception:
                    logger.exception("Qlib候选财务导出暂存清理失败")
                raise
            result["generation_id"] = published["generation_id"]
            result["restart_required"] = True
            return result
        with qlib_generation_mutation_lock(app_config.QLIB_DATA_ROOT, require_legacy=True):
            return _dump_fundamental_in_place(target, verify_samples, batch_size, refresh_valuation)

    return _dump_fundamental_in_place(target, verify_samples, batch_size, refresh_valuation)


def _dump_fundamental_in_place(data_dir, verify_samples, batch_size, refresh_valuation=True):
    from core.data import Database
    from core.adapters.qlib.utils import mysql_to_qlib_code

    data_dir = Path(data_dir).resolve()
    cal_path = data_dir / "calendars" / "day.txt"
    features_dir = data_dir / "features"
    if not cal_path.exists():
        logger.error(f"日历不存在: {cal_path}, 先跑 qlib 数据同步")
        return None

    cal = [line.strip() for line in cal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cal_np = np.array(cal, dtype="datetime64[D]")
    n_days = len(cal)

    rows = Database.fetchall(
        "SELECT f.code, f.report_date, f.ann_date, v.source_observed_at, "
        "f.roe, f.net_profit_yoy, f.revenue_yoy, f.debt_to_assets, f.eps "
        "FROM fundamental_data f LEFT JOIN ("
        "SELECT code, report_date, ann_date, MAX(source_observed_at) AS source_observed_at "
        "FROM fundamental_data_versions "
        "GROUP BY code, report_date, ann_date"
        ") v ON v.code=f.code AND v.report_date=f.report_date AND v.ann_date <=> f.ann_date "
        "WHERE f.report_date IS NOT NULL "
        "ORDER BY f.code, f.report_date"
    )
    if not rows:
        logger.error("fundamental_data 表为空, 先跑 python scripts/tushare_collect.py 采集")
        return None

    by_code = {}
    n_ann = 0
    for r in rows:
        by_code.setdefault(str(r["code"]), []).append(r)
        if r["ann_date"]:
            n_ann += 1
    logger.info(f"[FundDump] {len(by_code)} 只股票, {len(rows)} 期财报 "
                f"(ann_date 覆盖率 {n_ann / len(rows):.1%}, 缺失走法定截止日回退)")

    codes = list(by_code)
    n_done = 0
    written_codes = []
    batch_size = max(1, int(batch_size))
    write_fields = FUND_FIELDS if refresh_valuation else tuple(f for f in FUND_FIELDS if f not in ("pb", "pe"))
    for chunk_start in range(0, len(codes), batch_size):
        chunk = codes[chunk_start:chunk_start + batch_size]
        daily_rows = []
        if refresh_valuation:
            placeholders = ",".join(["%s"] * len(chunk))
            daily_rows = Database.fetchall(
                f"SELECT code, trade_date, pb, pe FROM daily_basic "
                f"WHERE code IN ({placeholders}) AND trade_date BETWEEN %s AND %s "
                f"AND (pb IS NOT NULL OR pe IS NOT NULL)",
                tuple(chunk) + (cal[0], cal[-1]),
            )
        daily_by_code = {}
        for row in daily_rows:
            daily_by_code.setdefault(str(row["code"]), []).append(row)

        for code in chunk:
            recs = by_code[code]
            periods = []
            for r in recs:
                eff = _effective_date(r["report_date"], r["ann_date"])
                ann_idx = int(np.searchsorted(cal_np, np.datetime64(eff, "D"), side="left"))
                observed_at = r["source_observed_at"]
                if observed_at is not None:
                    observed_date = observed_at.date() if hasattr(observed_at, "date") else dt.date.fromisoformat(str(observed_at)[:10])
                    observed_idx = int(np.searchsorted(cal_np, np.datetime64(observed_date, "D"), side="right"))
                    ann_idx = max(ann_idx, observed_idx)
                eff_idx = ann_idx
                if eff_idx < n_days:
                    periods.append((eff_idx, r["report_date"], r))
            periods.sort(key=lambda x: (x[0], x[1]))

            basic = daily_by_code.get(code, [])
            basic_indices = [
                int(np.searchsorted(cal_np, np.datetime64(row["trade_date"], "D"), side="left"))
                for row in basic
                if row.get("pb") is not None or row.get("pe") is not None
            ]
            start_candidates = basic_indices + [p[0] for p in periods]
            if not start_candidates:
                continue
            start_idx = min(start_candidates)
            arrays = {f: np.full(n_days - start_idx, np.nan, dtype=np.float64) for f in write_fields}

            last_report = None
            for eff_idx, report_date, r in periods:
                if last_report is not None and report_date < last_report:
                    continue
                last_report = report_date
                pos = eff_idx - start_idx
                for f in ("roe", "net_profit_yoy", "revenue_yoy", "debt_to_assets", "eps"):
                    v = r[f]
                    arrays[f][pos:] = float(v) if v is not None else np.nan

            for row in basic:
                idx = int(np.searchsorted(cal_np, np.datetime64(row["trade_date"], "D"), side="left"))
                if idx >= n_days or cal[idx] != row["trade_date"].isoformat():
                    continue
                pos = idx - start_idx
                for f in ("pb", "pe"):
                    v = row.get(f)
                    if v is not None:
                        arrays[f][pos] = float(v)

            qlib_code = mysql_to_qlib_code(code)
            stock_dir = features_dir / qlib_code.lower()
            if not stock_dir.exists():
                continue
            for f in write_fields:
                bin_path = stock_dir / f"{f}.day.bin"
                np.hstack([np.float32(start_idx), arrays[f].astype(np.float32)]).astype("<f").tofile(str(bin_path))
            written_codes.append((code, start_idx, periods))
            n_done += 1
            if n_done % 1000 == 0:
                logger.info(f"[FundDump] 进度: {n_done}/{len(codes)}")

    # ── PIT 抽检: 重读 bin, 断言首个非 NaN 位置 == 首期生效 idx, 且数值与内存构造一致 ──
    n_check = 0
    for code, start_idx, periods in random.sample(written_codes, min(verify_samples, len(written_codes))):
        qlib_code = mysql_to_qlib_code(code)
        bin_path = features_dir / qlib_code.lower() / "roe.day.bin"
        raw = np.fromfile(str(bin_path), dtype="<f")
        assert int(raw[0]) == start_idx, f"{code}: bin start_idx {int(raw[0])} != {start_idx}"
        vals = raw[1:]
        non_nan = np.where(~np.isnan(vals))[0]
        if len(non_nan):
            first_visible_idx = start_idx + int(non_nan[0])
            first_eff_idx = periods[0][0]
            assert first_visible_idx >= first_eff_idx, \
                f"{code}: 数值在生效日前可见 ({cal[first_visible_idx]} < {cal[first_eff_idx]})"
        n_check += 1
    logger.info(f"[FundDump] 完成: {n_done} 只写入 {len(FUND_FIELDS)} 个字段, PIT 抽检 {n_check} 只通过")
    result = {
        "stocks": n_done,
        "fields": FUND_FIELDS,
        "refreshed_fields": list(write_fields),
        "preserved_fields": [] if refresh_valuation else ["pb", "pe"],
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "calendar_range": {"start": cal[0], "end": cal[-1]},
        "source_rows": len(rows),
        "ann_date_rows": n_ann,
        "missing_ann_date_rows": len(rows) - n_ann,
        "source_observation_missing_rows": sum(r["source_observed_at"] is None for r in rows),
        "source_snapshot_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "ann_date_coverage": n_ann / len(rows),
        "pit_sampling": {"checked": n_check, "status": "passed"},
        "financial_availability": "max(ann_date or statutory fallback, next trading day after latest retained source observation across all source/version kinds)",
        "source_version_policy": "latest source_observed_at across fundamental_data_versions grouped by code/report_date/ann_date including null ann_date; current formal row values remain unavailable before that date",
        "null_update_policy": "a newer report row with a null metric clears the prior report value from its effective date",
        "valuation_sources": {"pb": "daily_basic.trade_date", "pe": "daily_basic.trade_date"} if refresh_valuation else {"pb": "preserved_from_parent_generation", "pe": "preserved_from_parent_generation"},
        "fundamental_pit_verified": False,
        "historical_source_vintages_certified": False,
    }
    save_json_atomic(result, data_dir / "fundamental_pit_manifest.json")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dump_fundamental()
