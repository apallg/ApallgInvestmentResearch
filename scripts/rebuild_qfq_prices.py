import argparse
import hashlib
import logging
import os
import random
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data import Database, DailyQuotesRepo
from core.data.sources.tushare import get_daily_kline
from core.shared import config
from core.shared.persistence import load_json, load_parquet, now_iso, save_json, save_parquet


logger = logging.getLogger("rebuild_qfq_prices")


def build_shadow_dataset(codes, start, end, output_dir, min_coverage=1.0, resume=False):
    if min_coverage != 1.0:
        raise ValueError("历史修复须逐日核对，min_coverage 固定为 1.0")
    output_dir = Path(output_dir).resolve()
    previous_manifest = None
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        if not resume or not (output_dir / "manifest.json").is_file():
            raise RuntimeError("影子目录非空；续跑必须指定原批次目录和 --resume-shadow")
        previous_manifest = load_json(output_dir / "manifest.json") or {}
        if previous_manifest.get("resume_invalidated"):
            raise RuntimeError("原批次输入曾发生变化且已失效；请使用新目录新建批次")
        contract = {"start": str(start), "end": str(end), "requested_codes": list(codes),
                    "min_coverage": min_coverage, "price_basis": "qfq"}
        if any(previous_manifest.get(key) != value for key, value in contract.items()):
            raise RuntimeError("续跑的日期范围、股票列表或价格契约已变化；请新建影子批次")
    quotes_dir = output_dir / "quotes"
    quotes_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "requested": len(codes), "rebuilt": 0, "reused": 0, "skipped": 0, "failed": 0,
        "output_dir": str(output_dir),
    }
    code_entries = []
    skipped_codes = []
    failed_codes = []
    resume_invalidated = False
    previous_entries = {entry.get("code"): entry for entry in (previous_manifest or {}).get("codes", [])}

    def save_progress():
        ready = result["requested"] > 0 and result["rebuilt"] == result["requested"] and not result["skipped"] and not result["failed"]
        save_json({
            "status": "ready_for_validation" if ready else "incomplete",
            "created_at": (previous_manifest or {}).get("created_at") or now_iso(),
            "updated_at": now_iso(), "start": str(start), "end": str(end),
            "requested_codes": list(codes), "min_coverage": min_coverage,
            "price_basis": "qfq", "resume_contract": "database_rows_and_latest_anchor_sha256",
            "resume_invalidated": resume_invalidated,
            **result, "codes": code_entries, "skipped_codes": skipped_codes,
            "failed_codes": failed_codes,
        }, output_dir / "manifest.json")

    for index, code in enumerate(codes, 1):
        try:
            stored = pd.DataFrame(Database.fetchall(
                "SELECT trade_date,open,high,low,close FROM daily_quotes "
                "WHERE code=%s AND trade_date BETWEEN %s AND %s ORDER BY trade_date",
                (code, pd.Timestamp(start).date(), pd.Timestamp(end).date()),
            ))
            anchor_row = Database.fetchone(
                "SELECT trade_date,adj_factor FROM daily_quotes WHERE code=%s AND close IS NOT NULL "
                "ORDER BY trade_date DESC LIMIT 1", (code,),
            ) or {}
            input_sha256 = hashlib.sha256(
                (stored.to_json(orient="records", date_format="iso") + "|" +
                 str(anchor_row.get("trade_date")) + "|" + str(anchor_row.get("adj_factor"))).encode("utf-8")
            ).hexdigest()
            old_entry = previous_entries.get(code)
            if old_entry:
                path = quotes_dir / f"{code}.parquet"
                file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                if old_entry.get("input_sha256") != input_sha256 or old_entry.get("sha256") != file_sha256:
                    resume_invalidated = True
                    raise RuntimeError("续跑输入或影子文件哈希已变化；请新建批次，不得复用旧分片")
                code_entries.append(old_entry)
                result["rebuilt"] += 1
                result["reused"] += 1
                save_progress()
                continue
            source_end = max(pd.Timestamp(end), pd.Timestamp(anchor_row["trade_date"])) if anchor_row.get("trade_date") else pd.Timestamp(end)
            full_frame = get_daily_kline(
                code, start, source_end.strftime("%Y%m%d"), adj="qfq", include_adj_factor=True,
            )
            anchor_date = str(full_frame["date"].max())[:10] if full_frame is not None and not full_frame.empty else None
            anchor_factor = float(full_frame["adj_factor"].iloc[-1]) if full_frame is not None and not full_frame.empty else None
            frame = full_frame.loc[pd.to_datetime(full_frame["date"]) <= pd.Timestamp(end)].copy() if full_frame is not None else None
            actual = 0 if frame is None else len(frame)
            if stored.empty or frame is None or frame.empty:
                result["skipped"] += 1
                skipped_codes.append({"code": code, "actual": actual, "reason": "source_or_database_empty", "input_sha256": input_sha256})
                continue
            stored_anchor = float(anchor_row.get("adj_factor") or 0)
            if stored_anchor > 0 and abs(anchor_factor - stored_anchor) > 1e-5:
                result["skipped"] += 1
                skipped_codes.append({"code": code, "reason": "anchor_factor_mismatch", "source_anchor": anchor_factor, "stored_anchor": stored_anchor, "input_sha256": input_sha256})
                continue
            frame = frame.copy()
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            stored["date"] = pd.to_datetime(stored["trade_date"]).dt.normalize()
            if frame["date"].duplicated().any() or stored["date"].duplicated().any():
                raise ValueError("日期键重复")
            compared = frame.set_index("date")[["open", "high", "low", "close"]].join(
                stored.set_index("date")[["open", "high", "low", "close"]],
                how="inner", lsuffix="_source", rsuffix="_stored",
            )
            conflict_dates = set()
            for field in ("open", "high", "low", "close"):
                source = pd.to_numeric(compared[f"{field}_source"], errors="coerce")
                current = pd.to_numeric(compared[f"{field}_stored"], errors="coerce")
                difference = (source - current).abs()
                conflict_dates.update(compared.index[(current.notna()) & (difference > (0.011 + source.abs() * 0.001))].strftime("%Y-%m-%d"))
            source_dates = set(frame["date"])
            valid_stored = set(stored.loc[stored["close"].notna(), "date"])
            source_missing_valid = sorted(valid_stored - source_dates)
            unresolved_null = sorted(set(stored.loc[stored["close"].isna(), "date"]) - source_dates)
            suspended_dates = set()
            independent_raw_close = {}
            if unresolved_null or conflict_dates:
                import baostock as bs
                identity = Database.fetchone("SELECT ts_code FROM stock_identity WHERE symbol=%s", (code,)) or {}
                exchange = str(identity.get("ts_code") or "").split(".")[-1].lower()
                if exchange in ("sh", "sz"):
                    session = bs.login()
                    try:
                        if session.error_code == "0":
                            history = bs.query_history_k_data_plus(
                                f"{exchange}.{code}", "date,close,tradestatus",
                                start_date=str(pd.Timestamp(start).date()),
                                end_date=str(pd.Timestamp(end).date()), frequency="d", adjustflag="3",
                            )
                            if history.error_code == "0":
                                while history.next():
                                    day, raw_close, status = history.get_row_data()
                                    if status == "0":
                                        suspended_dates.add(pd.Timestamp(day))
                                    elif raw_close:
                                        independent_raw_close[day] = float(raw_close)
                    finally:
                        bs.logout()
            unresolved_unknown = sorted(set(unresolved_null) - suspended_dates)
            source_by_date = frame.set_index("date")
            confirmed_conflicts = []
            for day in sorted(conflict_dates):
                source_row = source_by_date.loc[pd.Timestamp(day)]
                implied_raw = float(source_row["close"]) * anchor_factor / float(source_row["adj_factor"])
                independent = independent_raw_close.get(day)
                if independent is not None and abs(implied_raw - independent) <= 0.011 + abs(independent) * 0.0005:
                    confirmed_conflicts.append(day)
            if conflict_dates or unresolved_unknown or len(compared.loc[compared["close_stored"].notna()]) < 20:
                quarantine_checksum = None
                if conflict_dates:
                    quarantine_dir = output_dir / "quarantine"
                    quarantine_dir.mkdir(exist_ok=True)
                    quarantine_path = quarantine_dir / f"{code}.parquet"
                    save_parquet(frame, quarantine_path)
                    quarantine_checksum = hashlib.sha256(quarantine_path.read_bytes()).hexdigest()
                result["skipped"] += 1
                skipped_codes.append({
                    "code": code, "actual": actual, "stored_rows": len(stored), "input_sha256": input_sha256,
                    "price_conflict_dates": sorted(conflict_dates)[:20],
                    "price_conflict_count": len(conflict_dates),
                    "independently_confirmed_price_conflict_dates": confirmed_conflicts[:20],
                    "independently_confirmed_price_conflict_count": len(confirmed_conflicts),
                    "quarantine_sha256": quarantine_checksum,
                    "source_missing_valid_dates": [str(day.date()) for day in source_missing_valid[:20]],
                    "unresolved_null_dates": [str(day.date()) for day in unresolved_unknown[:20]],
                    "confirmed_suspension_count": len(set(unresolved_null) & suspended_dates),
                    "overlap_valid": int(compared["close_stored"].notna().sum()),
                })
                continue
            if "amplitude" not in frame.columns:
                frame["amplitude"] = None
            path = quotes_dir / f"{code}.parquet"
            save_parquet(frame, path)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            code_entries.append({
                "code": code,
                "rows": actual,
                "stored_rows": len(stored),
                "source_dates": actual,
                "confirmed_suspension_count": len(set(unresolved_null) & suspended_dates),
                "preserved_database_only_dates": len(source_missing_valid),
                "price_basis": "qfq",
                "price_source": "Tushare pro.daily + pro.adj_factor",
                "anchor_date": anchor_date,
                "anchor_factor": anchor_factor,
                "first_date": str(frame["date"].min())[:10],
                "last_date": str(frame["date"].max())[:10],
                "sha256": checksum,
                "input_sha256": input_sha256,
            })
            result["rebuilt"] += 1
            logger.info(f"[{index}/{len(codes)}] {code} 已写入影子数据 {actual} 条")
        except Exception as error:
            result["failed"] += 1
            failed_codes.append({"code": code, "error": str(error)})
            logger.exception(f"[{index}/{len(codes)}] {code} 影子重建失败: {error}")
        finally:
            save_progress()
    return result


def validate_shadow_dataset(output_dir):
    output_dir = Path(output_dir).resolve()
    manifest = load_json(output_dir / "manifest.json")
    if not manifest:
        return {"valid": False, "checked": 0, "errors": ["manifest_missing"]}
    required = {
        "date", "open", "high", "low", "close", "volume", "amount",
        "amplitude", "pct_chg", "turnover", "adj_factor",
    }
    errors = []
    checked = 0
    for entry in manifest.get("codes") or []:
        code = entry["code"]
        if entry.get("price_basis") != "qfq" or not entry.get("anchor_date") or not entry.get("anchor_factor"):
            errors.append(f"{code}: price_basis_metadata_missing")
            continue
        path = output_dir / "quotes" / f"{code}.parquet"
        if not path.exists():
            errors.append(f"{code}: file_missing")
            continue
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != entry.get("sha256"):
            errors.append(f"{code}: checksum_mismatch")
            continue
        frame = load_parquet(path)
        if not required.issubset(frame.columns):
            errors.append(f"{code}: required_columns_missing")
            continue
        if len(frame) != int(entry.get("rows") or 0):
            errors.append(f"{code}: row_count_mismatch")
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
            errors.append(f"{code}: date_order_invalid")
            continue
        if (frame["date"].isna().any()
                or str(frame["date"].min())[:10] != entry.get("first_date")
                or str(frame["date"].max())[:10] != entry.get("last_date")
                or pd.Timestamp(entry["anchor_date"]) < frame["date"].max()):
            errors.append(f"{code}: date_range_mismatch")
            continue
        numeric = frame[list(required - {"date"})].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([float("inf"), float("-inf")], float("nan"))
        required_numeric = ["open", "high", "low", "close", "volume", "amount", "pct_chg", "adj_factor"]
        optional_numeric = ["amplitude", "turnover"]
        if (numeric[required_numeric].isna().any().any()
                or (frame[optional_numeric].notna() & numeric[optional_numeric].isna()).any().any()):
            errors.append(f"{code}: numeric_value_missing")
            continue
        if ((numeric[["open", "high", "low", "close", "adj_factor"]] <= 0).any().any()
                or (numeric[["volume", "amount"]] < 0).any().any()):
            errors.append(f"{code}: numeric_value_invalid")
            continue
        if ((numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any()
                or (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any()):
            errors.append(f"{code}: ohlc_invariant_invalid")
            continue
        checked += 1
    if manifest.get("status") != "ready_for_validation":
        errors.append("manifest_incomplete")
    return {"valid": not errors, "checked": checked, "errors": errors}


def apply_shadow_dataset(output_dir):
    validation = validate_shadow_dataset(output_dir)
    if not validation["valid"]:
        raise RuntimeError(f"影子数据校验失败: {validation['errors']}")
    output_dir = Path(output_dir).resolve()
    manifest = load_json(output_dir / "manifest.json")
    applied = 0
    backup_dir = output_dir / "backups"
    backup_dir.mkdir(exist_ok=True)
    for entry in manifest["codes"]:
        code = entry["code"]
        frame = load_parquet(output_dir / "quotes" / f"{code}.parquet")
        dates = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        backup_path = backup_dir / f"{code}.parquet"
        if backup_path.exists():
            raise RuntimeError(f"{code} 已存在写库前备份，拒绝重复写入")
        current = pd.DataFrame(Database.fetchall(
            "SELECT trade_date,open,high,low,close FROM daily_quotes WHERE code=%s "
            "AND trade_date BETWEEN %s AND %s",
            (code, dates.iloc[0], dates.iloc[-1]),
        ))
        if not current.empty:
            current["date"] = pd.to_datetime(current["trade_date"])
            compared = frame.assign(date=pd.to_datetime(frame["date"])).set_index("date")[
                ["open", "high", "low", "close"]
            ].join(current.set_index("date")[["open", "high", "low", "close"]], how="inner", lsuffix="_source", rsuffix="_stored")
            for field in ("open", "high", "low", "close"):
                source = pd.to_numeric(compared[f"{field}_source"], errors="coerce")
                stored = pd.to_numeric(compared[f"{field}_stored"], errors="coerce")
                if ((stored.notna()) & ((source - stored).abs() > 0.011 + source.abs() * 0.001)).any():
                    raise RuntimeError(f"{code} 正式库价格已变化，停止回填")
        backup = Database.fetchall(
            "SELECT * FROM daily_quotes WHERE code=%s AND trade_date BETWEEN %s AND %s ORDER BY trade_date",
            (code, dates.iloc[0], dates.iloc[-1]),
        )
        save_parquet(pd.DataFrame(backup), backup_path)
        frame.attrs["provenance"] = {
            "batch_id": str(output_dir.name)[:64],
            "source": "tushare",
            "provider_version": version("tushare"),
            "adjustment_mode": "qfq",
            "anchor_date": str(entry["anchor_date"]),
            "anchor_factor": entry["anchor_factor"],
            "input_payload_sha256": entry["sha256"],
        }
        DailyQuotesRepo.save_batch(code, frame)
        applied += 1
    return {"applied": applied, "output_dir": str(output_dir)}


def rebuild_codes(codes, start, end, min_coverage=1.0):
    raise RuntimeError("直接覆盖正式行情已关闭；须先使用 --shadow-dir 逐日核验")


def audit_codes(codes, start, end):
    rows = []
    for code in codes:
        live = get_daily_kline(code, start, end, adj="qfq")
        if live is None or live.empty:
            rows.append({"code": code, "status": "fetch_failed"})
            continue
        stored_rows = Database.fetchall(
            "SELECT trade_date,close FROM daily_quotes "
            "WHERE code=%s AND trade_date>=%s AND trade_date<=%s ORDER BY trade_date",
            (code, start, end),
        )
        stored = pd.DataFrame(stored_rows)
        if stored.empty:
            rows.append({"code": code, "status": "db_empty"})
            continue
        stored["date"] = pd.to_datetime(stored["trade_date"])
        merged = live[["date", "close"]].merge(
            stored[["date", "close"]], on="date", suffixes=("_live", "_stored"),
        )
        if merged.empty:
            rows.append({"code": code, "status": "no_overlap"})
            continue
        relative = (
            (merged["close_stored"].astype(float) - merged["close_live"].astype(float)).abs()
            / merged["close_live"].astype(float).abs().replace(0, pd.NA)
        ).dropna()
        rows.append({
            "code": code,
            "status": "pass" if not relative.empty and float(relative.max()) <= 0.001 else "mismatch",
            "overlap": len(merged),
            "max_relative_error": round(float(relative.max()), 8) if not relative.empty else None,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", default="")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--shadow-dir")
    parser.add_argument("--resume-shadow", action="store_true")
    parser.add_argument("--validate-shadow")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--sync-qlib", action="store_true")
    args = parser.parse_args()

    if args.validate_shadow:
        print(validate_shadow_dataset(args.validate_shadow))
        return

    bounds = Database.fetchone(
        "SELECT MIN(trade_date) AS first_date,MAX(trade_date) AS last_date FROM daily_quotes"
    ) or {}
    start = (args.start or str(bounds.get("first_date") or "")).replace("-", "")
    end = (args.end or str(bounds.get("last_date") or "")).replace("-", "")
    if not start or not end:
        raise RuntimeError("daily_quotes 没有可重建的数据范围")

    if args.codes:
        codes = [value.strip() for value in args.codes.split(",") if value.strip()]
    else:
        codes = [row["code"] for row in Database.fetchall(
            "SELECT DISTINCT code FROM daily_quotes ORDER BY code"
        )]
        prefixes = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688")
        codes = [code for code in codes if str(code).startswith(prefixes)]
    if args.limit:
        codes = codes[:max(0, args.limit)]

    if args.shadow_dir and not args.apply:
        result = build_shadow_dataset(
            codes, start, end, args.shadow_dir, min_coverage=args.min_coverage,
            resume=args.resume_shadow,
        )
        print(result)
        return

    if not args.apply:
        sample = random.Random(0).sample(codes, min(args.sample_size, len(codes)))
        report = audit_codes(sample, start, end)
        print(pd.DataFrame(report).to_string(index=False))
        return

    if args.confirm != "PRICE_BASIS_REBUILD":
        raise RuntimeError("写库必须同时传 --confirm PRICE_BASIS_REBUILD")
    if not args.shadow_dir:
        raise RuntimeError("写库必须通过 --shadow-dir 指定已校验的影子数据")
    if not args.sync_qlib:
        raise RuntimeError("写库必须同时传 --sync-qlib，确保正式库与Qlib口径一起更新")
    result = apply_shadow_dataset(args.shadow_dir)
    print(result)
    from core.adapters.qlib.bridge import QlibDataBridge
    print(QlibDataBridge(config.QLIB_DATA_DIR).sync_all())


if __name__ == "__main__":
    main()
