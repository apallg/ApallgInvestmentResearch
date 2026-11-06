import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from core.adapters.qlib.horizon_candidate_runtime import CandidateLSTM, _sequence_scores
from core.domain.factors.neutralize import cross_sectional_neutralize
from core.domain.screening.score_pipeline import build_score_result
from scripts.h20_confirmation import ARMS, COMPARISONS, OUTPUT, SEEDS, digest, save_json, sorted_stocks
from scripts.horizon_architecture_research import fit_lgb
from scripts.horizon_multiscale_research import make_calendar_labels, rank_labels
from scripts.model_complementarity_research import paired_block_intervals
from scripts.train_horizon_sequence import fit_sequence


def inputs(args, require_labels):
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    required = ("pit_features_verified", "history_denominator_verified", "price_basis_verified",
                "market_cap_industry_pit_verified")
    if any(provenance.get(key) is not True for key in required):
        raise RuntimeError("输入 PIT、历史股票池或复权口径未通过资格审计")
    panel = pd.read_parquet(args.panel).sort_index()
    denominator = pd.read_parquet(args.denominator).sort_index()
    calendar = pd.DatetimeIndex(pd.to_datetime(Path(args.calendar).read_text(encoding="utf-8").splitlines()))
    if panel.index.names != ["datetime", "instrument"] or denominator.index.names != panel.index.names:
        raise RuntimeError("面板或股票池索引契约不符")
    if panel.index.has_duplicates or denominator.index.has_duplicates or not denominator.index.isin(panel.index).all():
        raise RuntimeError("输入股票池重复或面板缺失")
    if list(panel.columns[:40]) != protocol["factor_keys"] or "open" not in panel:
        raise RuntimeError("原40因子顺序或开盘价缺失")
    if pd.DatetimeIndex(panel.index.get_level_values("datetime").unique()).difference(calendar).size:
        raise RuntimeError("行情日期超出交易日历")
    if require_labels and not panel.index.get_level_values("datetime").max() <= pd.Timestamp(args.as_of):
        raise RuntimeError("训练面板包含截止日之后行情")
    if provenance.get("as_of") != args.as_of:
        raise RuntimeError("来源资格日期与命令日期不符")
    for name, path in (("panel", args.panel), ("denominator", args.denominator), ("calendar", args.calendar)):
        if provenance.get(f"{name}_sha256") != digest(path):
            raise RuntimeError(f"来源资格签名不符: {name}")
    if not require_labels and provenance.get("context_sha256") != digest(args.context):
        raise RuntimeError("PIT中性化输入签名不符")
    return protocol, provenance, panel, denominator, calendar


def retrain(args):
    protocol, provenance, panel, denominator, calendar = inputs(args, True)
    as_of = pd.Timestamp(args.as_of)
    if as_of not in calendar or panel.index.get_level_values("datetime").max() != as_of:
        raise RuntimeError("训练截止日未取得完整行情")
    if not (OUTPUT / "historical.json").exists():
        raise RuntimeError("历史复核未完成")
    version_dir = OUTPUT / "models" / args.as_of
    audit_path = version_dir / "audit.json"
    hashes = {name: digest(path) for name, path in (("panel", args.panel), ("denominator", args.denominator),
              ("calendar", args.calendar), ("provenance", args.provenance), ("protocol", OUTPUT / "protocol.json"))}
    hashes.update({"forward_code": digest(Path(__file__)),
                   "sequence_code": digest(ROOT / "scripts/train_horizon_sequence.py"),
                   "lgb_code": digest(ROOT / "core/adapters/qlib/rank_models.py")})
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("source_sha256") != hashes:
            raise RuntimeError("重训版本签名冲突")
        print(json.dumps({"status": "existing_complete", "version": args.as_of}, ensure_ascii=False))
        return
    version_dir.mkdir(parents=True, exist_ok=True)
    dates = panel.index.get_level_values("datetime")
    universe = panel.index.isin(denominator.index)
    opens = panel[["open"]].copy()
    raw, label_end = make_calendar_labels(opens, calendar, 20)
    raw = raw.where(universe)
    label_rank = rank_labels(raw, calendar)
    labels = pd.DataFrame({"label": raw, "label_rank": label_rank, "label_end": label_end})
    windows = {"train": (as_of - pd.DateOffset(months=43), as_of - pd.DateOffset(months=7)),
               "valid": (as_of - pd.DateOffset(months=7), as_of - pd.DateOffset(months=4)),
               "calibrate": (as_of - pd.DateOffset(months=4), as_of)}
    masks = {}
    for name, (low, high) in windows.items():
        cutoff = windows[{"train": "valid", "valid": "calibrate", "calibrate": "calibrate"}[name]][0] if name != "calibrate" else as_of
        masks[name] = np.asarray((dates >= low) & (dates < high) & raw.notna() & (label_end < cutoff), dtype=bool)
    masks["test"] = masks["calibrate"].copy()
    if min(int(masks[name].sum()) for name in ("train", "valid", "calibrate")) < 1000:
        raise RuntimeError("净化后训练、早停或校准样本不足")
    frame = panel[protocol["factor_keys"]].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    params = {**protocol["lightgbm_parameters"], "num_threads": 4}
    params.pop("seed_policy", None)
    params.pop("verbose_eval", None)
    members = []
    for seed in SEEDS:
        seed_params = {**params, "seed": seed, "bagging_seed": seed,
                       "feature_fraction_seed": seed, "data_random_seed": seed}
        g_path = version_dir / f"{seed}_G.txt"
        n_path = version_dir / f"{seed}_N.pt"
        _, g_meta = fit_lgb(frame, labels, masks, protocol["factor_keys"], seed_params, g_path, False, calendar)
        _, _, n_meta = fit_sequence(frame.to_numpy(dtype=np.float32), frame.index,
            labels["label_rank"].fillna(0).to_numpy(dtype=np.float32), masks, calendar,
            None, "N0", seed, n_path, True, hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            len(protocol["factor_keys"]), None)
        final = n_path.with_suffix(".final.pt")
        if g_meta["restore_max_abs_error"] > 1e-6 or n_meta["restore_max_abs_error"] > 1e-6:
            raise RuntimeError("组件保存恢复误差超限")
        members.append({"seed": seed, "G": {"path": str(g_path), "sha256": digest(g_path)},
                        "N": {"path": str(final), "sha256": digest(final)},
                        "G_audit": g_meta, "N_audit": n_meta})
    result = {"status": "paired_models_validated", "version": args.as_of,
              "validated_at_utc": datetime.now(timezone.utc).isoformat(),
              "validated_local_date": str(datetime.now(ZoneInfo("Asia/Shanghai")).date()),
              "effective_after": args.as_of, "source_sha256": hashes,
              "windows": {key: [str(a.date()), str(b.date())] for key, (a, b) in windows.items()},
              "mature_rows": {name: int(mask.sum()) for name, mask in masks.items() if name != "test"},
              "members": members, "research_only": True, "production_qualified": False}
    save_json(audit_path, result)
    print(json.dumps({"status": result["status"], "version": args.as_of}, ensure_ascii=False))


def record(args):
    protocol, provenance, panel, denominator, calendar = inputs(args, False)
    date = pd.Timestamp(args.as_of)
    if date not in calendar or panel.index.get_level_values("datetime").max() != date:
        raise RuntimeError("决策日行情或日历不完整")
    local_now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if local_now.date() != date.date() or local_now.time() < time(15, 0):
        raise RuntimeError("预测必须在当日收盘后记录，禁止追认历史日期")
    models = sorted((OUTPUT / "models").glob("*/audit.json"))
    eligible = [p for p in models if pd.Timestamp(p.parent.name) < date
                and pd.Timestamp(json.loads(p.read_text(encoding="utf-8"))["validated_local_date"]) < date]
    if not eligible:
        raise RuntimeError("没有决策日前完成验收的配对模型")
    version_path = eligible[-1]
    version = json.loads(version_path.read_text(encoding="utf-8"))
    if version.get("status") != "paired_models_validated":
        raise RuntimeError("模型版本未完成验收")
    sessions_since = int(((calendar > pd.Timestamp(version["validated_local_date"])) & (calendar <= date)).sum())
    retrain_overdue = sessions_since > protocol["retrain_trading_days"]
    day = denominator.index.get_level_values("datetime") == date
    expected = denominator.index[day]
    if len(expected) < 50:
        raise RuntimeError("决策日共同股票池不足")
    frame = panel[protocol["factor_keys"]].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    codes = expected.get_level_values("instrument").str.lower().tolist()
    if len(set(codes)) != len(codes):
        raise RuntimeError("股票代码规范化后重复")
    context = pd.read_parquet(args.context).reindex(expected)
    if list(context.columns) != ["market_cap", "industry", "known_at"] or context.isna().any().any():
        raise RuntimeError("PIT中性化字段或覆盖不足")
    if (pd.to_datetime(context["known_at"]) > date + pd.Timedelta(hours=15)).any() or not (context["market_cap"] > 0).all():
        raise RuntimeError("中性化输入含未来信息或市值无效")
    by_arm = {arm: [] for arm in ARMS}
    for member in version["members"]:
        for name in ("G", "N"):
            artifact = member[name]
            if digest(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError("配对组件签名冲突")
        g = lgb.Booster(model_str=Path(member["G"]["path"]).read_text(encoding="utf-8"))
        state = torch.load(member["N"]["path"], map_location="cpu", weights_only=False)
        n = CandidateLSTM(len(protocol["factor_keys"]))
        n.load_state_dict(state["model"])
        n.eval()
        gr = pd.Series(g.predict(frame.loc[expected].to_numpy(dtype=np.float32), num_threads=4), index=expected).rank(pct=True)
        nr = _sequence_scores(n, state["mean"], state["scale"], frame, expected, calendar).rank(pct=True)
        for arm, weight in protocol["arms"].items():
            by_arm[arm].append(((1 - weight) * gr + weight * nr).to_numpy(dtype=float))
    output = pd.DataFrame(index=expected)
    audits = {}
    mktcap = dict(zip(codes, context["market_cap"].to_numpy(dtype=float)))
    industry = dict(zip(codes, context["industry"].astype(str)))
    for arm in ARMS:
        raw = np.mean(by_arm[arm], axis=0)
        if not np.isfinite(raw).all():
            raise RuntimeError("模型预测非有限")
        raw_map = dict(zip(codes, raw))
        neutral, audit = cross_sectional_neutralize(codes, raw, mktcap, industry, return_details=True)
        if audit.get("status") != "applied" or len(neutral) != len(expected):
            raise RuntimeError("实际评分中性化未完整应用")
        result = build_score_result(raw_map, neutral, neutralize_requested=True,
                                    neutralize_status="applied", model_input_semantics="rank_blend")
        if len(result["scores"]) != len(expected):
            raise RuntimeError("评分后共同股票池变化")
        output[f"{arm}_raw"] = raw
        output[f"{arm}_score"] = [result["scores"][code] for code in codes]
        for seed, values in zip(SEEDS, by_arm[arm]):
            output[f"{arm}_{seed}_raw"] = values
        audits[arm] = audit
    path = OUTPUT / "records" / args.as_of / "predictions.parquet"
    meta = path.with_suffix(".json")
    signatures = {name: digest(source) for name, source in (("panel", args.panel), ("denominator", args.denominator),
        ("calendar", args.calendar), ("context", args.context), ("provenance", args.provenance),
        ("model", version_path), ("protocol", OUTPUT / "protocol.json"))}
    if path.exists() or meta.exists():
        if not path.exists() or not meta.exists() or json.loads(meta.read_text(encoding="utf-8")).get("sha256") != signatures:
            raise RuntimeError("前瞻记录签名冲突，拒绝覆盖")
        print(json.dumps({"status": "already_recorded", "date": args.as_of}, ensure_ascii=False))
        return
    temp_dir = path.parent.with_name(path.parent.name + ".tmp")
    if temp_dir.exists():
        raise RuntimeError("上次记录留下临时目录，须先审计后恢复")
    temp_dir.mkdir(parents=True)
    output.to_parquet(temp_dir / path.name, compression="zstd")
    save_json(temp_dir / meta.name, {"status": "prediction_recorded", "decision_date": args.as_of,
                     "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                     "model_version": version["version"], "rows": len(output),
                     "trading_days_since_validation": sessions_since,
                     "quarterly_retrain_overdue": retrain_overdue,
                     "sha256": signatures, "neutralization": audits,
                     "label_matured": False, "execution_verified": False, "production_qualified": False})
    temp_dir.replace(path.parent)
    observation_path = OUTPUT / "observation.json"
    if not observation_path.exists():
        save_json(observation_path, {"status": "observing", "start_date": args.as_of,
            "quality_check_after": str((date + timedelta(weeks=26)).date()),
            "end_after": str((date + timedelta(weeks=52)).date()),
            "initial_model_version": version["version"], "protocol_sha256": digest(OUTPUT / "protocol.json"),
            "production_qualified": False})
    print(json.dumps({"status": "prediction_recorded", "date": args.as_of, "rows": len(output)}, ensure_ascii=False))


def evaluate(args):
    observation_path = OUTPUT / "observation.json"
    if not observation_path.exists():
        print(json.dumps({"status": "pending", "reason": "observation_not_started"}))
        return
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    calendar = pd.DatetimeIndex(pd.to_datetime(Path(args.calendar).read_text(encoding="utf-8").splitlines()))
    start = pd.Timestamp(observation["start_date"])
    end = pd.Timestamp(observation["end_after"])
    sessions = calendar[(calendar >= start) & (calendar < end)]
    if len(sessions) < 200 or pd.Timestamp(datetime.now(ZoneInfo("Asia/Shanghai")).date()) < end:
        print(json.dumps({"status": "pending", "reason": "52_weeks_not_finished", "observed_sessions": len(sessions)}))
        return
    paths = [OUTPUT / "records" / str(day.date()) / "predictions.parquet" for day in sessions]
    missing = [str(day.date()) for day, path in zip(sessions, paths) if not path.exists() or not path.with_suffix(".json").exists()]
    if missing:
        print(json.dumps({"status": "pending", "reason": "daily_records_missing", "first_missing": missing[:10]}))
        return
    if not args.quotes or not args.eval_provenance:
        raise RuntimeError("评价需要完整后续复权开盘价及来源资格")
    qualification = json.loads(Path(args.eval_provenance).read_text(encoding="utf-8"))
    if (qualification.get("price_basis_verified") is not True or qualification.get("as_of") != args.as_of
            or qualification.get("quotes_sha256") != digest(args.quotes)
            or qualification.get("calendar_sha256") != digest(args.calendar)):
        raise RuntimeError("评价价格来源资格不符")
    quotes = pd.read_parquet(args.quotes).sort_index()
    if quotes.index.names != ["datetime", "instrument"] or "open" not in quotes or quotes.index.has_duplicates:
        raise RuntimeError("评价价格输入无效")
    formed = sessions[::20]
    last_position = calendar.get_loc(sessions[-1]) + 21
    if last_position >= len(calendar) or quotes.index.get_level_values("datetime").max() < calendar[last_position]:
        print(json.dumps({"status": "pending", "reason": "last_label_not_matured"}))
        return
    rows = []
    daily_ic = []
    formed_set = set(formed)
    for day in sessions:
        frame = pd.read_parquet(OUTPUT / "records" / str(day.date()) / "predictions.parquet")
        if frame.index.has_duplicates or len(frame) < 50:
            raise RuntimeError(f"形成日记录不完整: {day.date()}")
        pos = calendar.get_loc(day)
        entry, exit_ = calendar[pos + 1], calendar[pos + 21]
        stocks = frame.index.get_level_values("instrument")
        entry_idx = pd.MultiIndex.from_arrays([[entry] * len(stocks), stocks], names=frame.index.names)
        exit_idx = pd.MultiIndex.from_arrays([[exit_] * len(stocks), stocks], names=frame.index.names)
        first = quotes["open"].reindex(entry_idx).to_numpy(dtype=float)
        last = quotes["open"].reindex(exit_idx).to_numpy(dtype=float)
        label = last / first - 1
        if not np.isfinite(label).all() or (first <= 0).any():
            raise RuntimeError(f"形成日股票池缺少准确开盘标签: {day.date()}")
        ic_row = {"date": str(day.date())}
        for arm in ARMS:
            ic_row[arm] = float(frame[f"{arm}_score"].corr(pd.Series(label, index=frame.index), method="spearman"))
        if not all(np.isfinite(ic_row[arm]) for arm in ARMS):
            raise RuntimeError(f"每日RankIC无效: {day.date()}")
        daily_ic.append(ic_row)
        if day not in formed_set:
            continue
        record = {"date": str(day.date()), "entry": str(entry.date()), "exit": str(exit_.date()),
                  "pool": float(np.mean(label))}
        for arm in ARMS:
            sorted_frame = sorted_stocks(pd.DataFrame({"score": frame[f"{arm}_score"], "label": label}, index=frame.index))
            record[arm] = float(sorted_frame["label"].iloc[:5].mean())
            for seed in SEEDS:
                seed_frame = pd.DataFrame({"score": frame[f"{arm}_{seed}_raw"], "label": label}, index=frame.index)
                record[f"{arm}_{seed}"] = float(sorted_stocks(seed_frame)["label"].iloc[:5].mean())
        rows.append(record)
    table = pd.DataFrame(rows)
    daily_table = pd.DataFrame(daily_ic)
    comparisons = []
    for candidate, baseline in COMPARISONS:
        delta = table[candidate].to_numpy() - table[baseline].to_numpy()
        intervals = paired_block_intervals([delta], (2, 3, 4), family_size=3)
        semivar_c = np.mean(np.minimum(table[candidate], 0) ** 2)
        semivar_b = np.mean(np.minimum(table[baseline], 0) ** 2)
        ratio = float(semivar_c / semivar_b) if semivar_b else (0.0 if semivar_c == 0 else None)
        seed_pos = sum((table[f"{candidate}_{seed}"] - table[f"{baseline}_{seed}"]).mean() > 0 for seed in SEEDS)
        ic_diff = float((daily_table[candidate] - daily_table[baseline]).mean())
        lower = intervals["2"]["bonferroni_95_family26"][0]
        screen = delta.mean() >= 0.002 and seed_pos >= 2 and ic_diff >= -0.005 and ratio is not None and ratio <= 1.1
        stable = bool(screen and lower > 0 and all(intervals[str(size)]["ordinary_95"][1] > 0 for size in (3, 4)))
        comparisons.append({"comparison": f"{candidate}-{baseline}", "mean_bp": float(delta.mean() * 10000),
            "seed_positive": int(seed_pos), "rank_ic_diff": ic_diff, "downside_semivariance_ratio": ratio,
            "intervals": intervals, "screen_pass": bool(screen), "stable_improvement": stable})
    passed = {row["comparison"] for row in comparisons if row["stable_improvement"]}
    choice = "F75" if {"F75-N", "F75-G"} <= passed else "N" if "N-G" in passed else "G"
    result = {"status": "forward_prediction_evaluated", "start_date": str(start.date()),
              "end_date": str(end.date()), "formed_baskets": rows, "daily_rank_ic": daily_ic,
              "comparisons": comparisons,
              "research_structure_decision": choice, "trade_execution_verified": False,
              "production_qualified": False, "quotes_sha256": digest(args.quotes),
              "calendar_sha256": digest(args.calendar)}
    save_json(OUTPUT / "forward_analysis.json", result)
    print(json.dumps({"status": result["status"], "choice": choice, "baskets": len(rows)}, ensure_ascii=False))


def quality_status():
    path = OUTPUT / "observation.json"
    if not path.exists():
        return {"status": "not_started"}
    observation = json.loads(path.read_text(encoding="utf-8"))
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Shanghai")).date())
    start = pd.Timestamp(observation["start_date"])
    checkpoint = pd.Timestamp(observation["quality_check_after"])
    calendar_path = ROOT / "data/qlib_data/calendars/day.txt"
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text(encoding="utf-8").splitlines()))
    settled = today if datetime.now(ZoneInfo("Asia/Shanghai")).time() >= time(15, 0) else today - pd.Timedelta(days=1)
    expected = calendar[(calendar >= start) & (calendar <= min(settled, checkpoint))]
    missing = [str(day.date()) for day in expected if not (OUTPUT / "records" / str(day.date()) / "predictions.json").exists()]
    records = list((OUTPUT / "records").glob("*/predictions.json"))
    overdue = sum(json.loads(p.read_text(encoding="utf-8")).get("quarterly_retrain_overdue", False) for p in records)
    return {"status": "quality_check_due" if today >= checkpoint else "observing",
            "calendar_through": str(calendar.max().date()), "expected_days_known": len(expected),
            "missing_first10": missing[:10], "quarterly_retrain_overdue_days": overdue,
            "performance_selection_allowed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("retrain", "record", "evaluate"))
    parser.add_argument("--panel")
    parser.add_argument("--denominator")
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--provenance")
    parser.add_argument("--context")
    parser.add_argument("--quotes")
    parser.add_argument("--eval-provenance")
    parser.add_argument("--as-of")
    args = parser.parse_args()
    if args.stage == "evaluate":
        evaluate(args)
    elif not all((args.panel, args.denominator, args.provenance, args.as_of)):
        raise RuntimeError("重训和记录需要面板、股票池、来源审计及截止日")
    elif args.stage == "retrain":
        retrain(args)
    else:
        if not args.context:
            raise RuntimeError("影子预测必须提供PIT中性化上下文")
        record(args)


if __name__ == "__main__":
    main()
