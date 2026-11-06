import argparse
import hashlib
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.adapters.qlib.model_contract import get_model_label_mode
from core.data import Database
from core.domain.screening._screener import BuySideScreener
from core.shared import config
from core.shared.trading_time import get_latest_trading_day, get_trading_days_between, is_trading_day
from backend.service.qlib_access import qlib_read_locked


MODEL = ("656749665990832865", "7e5e67ba9b714c76b0808a2626d0f7d1")
POOL = "stable_history_500d"
PORTFOLIO_CODE_PREFIXES = ["600", "601", "603", "605", "000", "001", "002", "003"]
TOP_DEPTH = 20
CONFIRM_SESSIONS = 3
REQUIRED_APPEARANCES = 2
OUTPUT = ROOT / "data" / "experiments" / "rank_persistence_shadow_20260925"
CONTRACT_FILES = [
    "scripts/analyze_rank_persistence_ab.py",
    "scripts/capture_rank_persistence_shadow.py",
    "scripts/run_rank_persistence_forward.py",
    "scripts/run_rank_persistence_ab.py",
    "core/adapters/qlib/config.py",
    "core/adapters/qlib/custom_handler.py",
    "core/adapters/qlib/factor_expression.py",
    "core/adapters/qlib/factor_resolver.py",
    "core/adapters/qlib/lstm_fusion.py",
    "core/adapters/qlib/model_contract.py",
    "core/adapters/qlib/models.py",
    "core/data/fundamental.py",
    "core/domain/backtest/common.py",
    "core/domain/backtest/engine.py",
    "core/domain/backtest/metrics.py",
    "core/domain/factors/expression.py",
    "core/domain/factors/factor_pool_state.json",
    "core/domain/factors/registry.py",
    "core/domain/screening/_screener.py",
    "core/domain/screening/exclusion.py",
    "core/domain/screening/scoring.py",
    "core/shared/trading_time.py",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_fingerprint(ranked):
    return hashlib.sha256(json.dumps(
        ranked, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def contract_fingerprint_for(files_sha256):
    return hashlib.sha256(json.dumps(
        files_sha256, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def security_reference_fingerprint(references):
    return hashlib.sha256(json.dumps(
        references, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def snapshot_metadata_fingerprint(snapshot):
    fields = (
        "schema_version", "captured_at_utc", "captured_after_close_or_holiday",
        "earliest_execution_date", "prediction_date", "market_data_date",
        "expected_latest_trading_date", "market_quote_rows", "source", "model",
        "model_fingerprint", "contract_fingerprint", "prediction_artifact_sha256",
        "pool", "pool_fingerprint", "rank_scope", "portfolio_code_prefixes",
        "neutralize_qlib", "score_semantics", "score_count", "scores_sha256",
        "security_reference_source", "security_reference_observed_at_utc",
        "security_references_sha256", "confirmation_rule", "confirmation_ready",
        "previous_prediction_dates", "missing_previous_prediction_dates",
        "confirmed_top20", "confirmed_top20_in_portfolio_prefixes",
    )
    payload = {key: snapshot.get(key) for key in fields}
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def validate_prior_snapshot(prior, previous_date, model_fingerprint,
                            pool_fingerprint, contract_fingerprint):
    if (prior.get("prediction_date") != previous_date
            or not prior.get("scores_sha256")
            or prior.get("model_fingerprint") != model_fingerprint
            or prior.get("pool_fingerprint") != pool_fingerprint
            or prior.get("pool") != POOL):
        raise RuntimeError(f"历史快照模型或股票池合同不一致: {previous_date}")
    rows = prior.get("ranked_scores")
    if not isinstance(rows, list) or len(rows) != prior.get("score_count"):
        raise RuntimeError(f"历史快照排名行数异常: {previous_date}")
    try:
        ranked = [(str(row["code"]), float(row["score"])) for row in rows]
        ranks_valid = all(row.get("rank") == index for index, row in enumerate(rows, 1))
        sorted_valid = all(
            ranked[index - 1][1] > ranked[index][1]
            or (ranked[index - 1][1] == ranked[index][1]
                and ranked[index - 1][0] < ranked[index][0])
            for index in range(1, len(ranked))
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"历史快照排名数据损坏: {previous_date}") from error
    if (not ranks_valid or not sorted_valid
            or len({code for code, _ in ranked}) != len(ranked)
            or score_fingerprint(ranked) != prior.get("scores_sha256")):
        raise RuntimeError(f"历史快照排名哈希或顺序校验失败: {previous_date}")
    if int(prior.get("schema_version", 1)) < 3:
        return False
    if (not prior.get("snapshot_metadata_sha256")
            or snapshot_metadata_fingerprint(prior) != prior.get("snapshot_metadata_sha256")):
        raise RuntimeError(f"历史快照元数据哈希校验失败: {previous_date}")
    model = prior.get("model") or {}
    try:
        captured_at = datetime.fromisoformat(
            str(prior.get("captured_at_utc") or "").replace("Z", "+00:00")
        )
        earliest_execution_date = date.fromisoformat(
            str(prior.get("earliest_execution_date") or "")[:10]
        )
    except ValueError as error:
        raise RuntimeError(f"历史快照时间戳或执行日无效: {previous_date}") from error
    if (
        prior.get("market_data_date") != previous_date
        or prior.get("expected_latest_trading_date") != previous_date
        or prior.get("captured_after_close_or_holiday") is not True
        or captured_at.tzinfo is None
        or earliest_execution_date <= date.fromisoformat(previous_date)
        or prior.get("source") != "live_model"
        or model.get("experiment_id") != MODEL[0]
        or model.get("run_id") != MODEL[1]
        or model.get("label_mode") != "tradable_open_to_open"
        or prior.get("pool") != POOL
        or prior.get("neutralize_qlib") is not False
        or int(prior.get("market_quote_rows") or 0) < 1000
        or not prior.get("prediction_artifact_sha256")
    ):
        raise RuntimeError(f"历史快照决策时点或数据来源合同不匹配: {previous_date}")
    references = prior.get("security_references")
    if not isinstance(references, dict):
        raise RuntimeError(f"历史快照缺少证券参考信息: {previous_date}")
    reference_rows = [
        {"code": code, **value}
        for code, value in sorted(references.items())
        if isinstance(value, dict)
    ]
    if (len(reference_rows) != len(references)
            or set(references) != {code for code, _ in ranked}
            or any(
                not value.get("name") or not value.get("industry")
                or not isinstance(value.get("is_st"), bool)
                or not value.get("update_date")
                for value in references.values() if isinstance(value, dict)
            )
            or security_reference_fingerprint(reference_rows)
            != prior.get("security_references_sha256")):
        raise RuntimeError(f"历史快照证券参考信息覆盖或哈希校验失败: {previous_date}")
    if (prior.get("security_reference_source") != "BaoStock.query_stock_industry"
            or not prior.get("security_reference_observed_at_utc")):
        raise RuntimeError(f"历史快照证券参考信息来源缺失: {previous_date}")
    prior_contract = prior.get("contract_fingerprint")
    if not prior_contract:
        if int(prior.get("schema_version", 1)) >= 2:
            raise RuntimeError(f"新版本历史快照缺少代码合同指纹: {previous_date}")
        return False
    files_sha256 = prior.get("contract_files_sha256")
    if (not isinstance(files_sha256, dict)
            or contract_fingerprint_for(files_sha256) != prior_contract):
        raise RuntimeError(f"历史快照代码合同清单哈希校验失败: {previous_date}")
    if prior_contract != contract_fingerprint:
        raise RuntimeError(f"历史快照代码合同指纹不一致: {previous_date}")
    return True


def capture_security_references(codes):
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock登录失败: {login.error_msg}")
    try:
        result = bs.query_stock_industry()
        if result is None or result.error_code != "0":
            raise RuntimeError("BaoStock query_stock_industry 查询失败")
        requested = set(codes)
        references = {}
        while result.next():
            row = result.get_row_data()
            raw_code = row[1]
            code = raw_code.split(".")[-1]
            if code not in requested:
                continue
            name = str(row[2] or "")
            industry = str(row[3] or row[4] or "")
            references[code] = {
                "name": name,
                "industry": industry,
                "is_st": "ST" in name.upper(),
                "update_date": str(row[0] or ""),
            }
    finally:
        bs.logout()

    missing = sorted(requested - set(references))
    incomplete = sorted(
        code for code, value in references.items()
        if not value["name"] or not value["industry"]
    )
    if missing or incomplete:
        raise RuntimeError(
            f"BaoStock证券参考信息覆盖不足: missing={len(missing)}, incomplete={len(incomplete)}"
        )
    return references


@qlib_read_locked
def _load_qlib_inputs(target):
    qlib_dir = Path(config.QLIB_DATA_DIR)
    calendar_path = qlib_dir / "calendars" / "day.txt"
    calendar = calendar_path.read_text(encoding="utf-8").splitlines() if calendar_path.exists() else []
    if not calendar:
        raise RuntimeError("Qlib交易日历为空或不存在，停止采集")
    target_key = target.isoformat()
    if target_key not in calendar:
        raise RuntimeError(f"Qlib日历未覆盖{target_key}，停止采集")
    if calendar[-1] != target_key:
        raise RuntimeError(f"Qlib最新日期{calendar[-1]}与行情最新日期{target_key}不一致，停止采集")

    pool_path = qlib_dir / "instruments" / "a500.txt"
    if not pool_path.exists():
        raise RuntimeError(f"Qlib股票池文件不存在: {pool_path}")
    contract_files_sha256 = {}
    for relative_path in CONTRACT_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            raise RuntimeError(f"实验合同代码文件缺失: {relative_path}")
        contract_files_sha256[relative_path] = file_sha256(path)
    contract_fingerprint = contract_fingerprint_for(contract_files_sha256)
    run_dir = Path(str(config.QLIB_MLFLOW_URI).replace("file:", "")) / MODEL[0] / MODEL[1]
    label_mode = get_model_label_mode(run_dir)
    if label_mode != "tradable_open_to_open":
        raise RuntimeError(f"模型标签合同不匹配: {label_mode}")

    artifact_paths = [
        run_dir / "artifacts" / "params.pkl",
        run_dir / "artifacts" / "params_lstm.pkl",
        run_dir / "artifacts" / "lstm_bundle.pkl",
        run_dir / "artifacts" / "factor_keys.json",
        run_dir / "artifacts" / "pred.pkl",
    ]
    missing_artifacts = [str(path) for path in artifact_paths if not path.exists()]
    if missing_artifacts:
        raise RuntimeError(f"固定模型文件不完整: {missing_artifacts}")
    model_fingerprint = hashlib.sha256("".join(
        f"{path.name}:{file_sha256(path)}\n" for path in artifact_paths
    ).encode("utf-8")).hexdigest()
    pool_fingerprint = file_sha256(pool_path)
    details = BuySideScreener()._load_qlib_predictions(
        MODEL,
        as_of=target,
        pool=POOL,
        neutralize_qlib=False,
        return_details=True,
    )
    scores = details.get("scores", {}) if isinstance(details, dict) else {}
    context = details.get("audit", {}).get("context", {}) if isinstance(details, dict) else {}
    if context.get("source") != "live_model":
        raise RuntimeError(f"实验只接受现场模型预测，拒绝来源={context.get('source')!r}")
    prediction_date = str(context.get("prediction_date") or "")[:10]
    if not scores or prediction_date != target_key:
        raise RuntimeError(
            f"预测为空或日期不精确匹配: rows={len(scores)}, prediction_date={prediction_date}, requested={target_key}"
        )
    return {
        "calendar": calendar,
        "target_key": target_key,
        "pool_path": pool_path,
        "contract_fingerprint": contract_fingerprint,
        "model_fingerprint": model_fingerprint,
        "pool_fingerprint": pool_fingerprint,
        "scores": scores,
        "context": context,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="必须是daily_quotes和Qlib日历的同一个最新交易日")
    args = parser.parse_args()

    latest = Database.fetchone("SELECT MAX(trade_date) AS latest FROM daily_quotes")
    if not latest or not latest.get("latest"):
        raise RuntimeError("daily_quotes无最新交易日，停止采集")
    market_date = latest["latest"]
    if hasattr(market_date, "date"):
        market_date = market_date.date()
    target = date.fromisoformat(args.date) if args.date else market_date
    if target != market_date:
        raise RuntimeError(f"请求日期{target}不是daily_quotes最新交易日{market_date}，停止采集")
    expected_latest = get_latest_trading_day(date.today())
    if target != expected_latest:
        raise RuntimeError(
            f"daily_quotes最新日期{target}与交易日历最新日期{expected_latest}不一致，停止采集"
        )
    captured_at = datetime.now()
    if is_trading_day(captured_at) and captured_at.time() < time(15, 30):
        raise RuntimeError("当前交易日尚未收盘，拒绝生成日度影子信号")
    next_sessions = get_trading_days_between(
        date.today(), date.today() + timedelta(days=14)
    )
    if not next_sessions:
        raise RuntimeError("交易日历未能确认下一可执行交易日，停止采集")

    qlib_inputs = _load_qlib_inputs(target)
    calendar = qlib_inputs["calendar"]
    target_key = qlib_inputs["target_key"]
    pool_path = qlib_inputs["pool_path"]
    contract_fingerprint = qlib_inputs["contract_fingerprint"]
    model_fingerprint = qlib_inputs["model_fingerprint"]
    pool_fingerprint = qlib_inputs["pool_fingerprint"]
    scores = qlib_inputs["scores"]
    context = qlib_inputs["context"]

    ranked = sorted(
        ((str(code), float(score)) for code, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    scores_fingerprint = score_fingerprint(ranked)
    output_path = OUTPUT / f"snapshot_{target.strftime('%Y%m%d')}.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        is_contract_compatible = validate_prior_snapshot(
            existing, target_key, model_fingerprint, pool_fingerprint,
            contract_fingerprint,
        )
        if existing.get("scores_sha256") != scores_fingerprint:
            raise RuntimeError(f"日期快照已存在且评分不同，拒绝覆盖: {output_path}")
        print(json.dumps({
            "status": "already_captured",
            "path": str(output_path),
            "legacy_without_code_fingerprint": not is_contract_compatible,
        }, ensure_ascii=False))
        return

    security_references = capture_security_references(code for code, _ in ranked)
    security_reference_rows = [
        {"code": code, **value}
        for code, value in sorted(security_references.items())
    ]
    date_index = calendar.index(target_key)
    previous_dates = calendar[max(0, date_index - CONFIRM_SESSIONS + 1):date_index]
    required_previous = CONFIRM_SESSIONS - 1
    previous_snapshots = []
    missing_previous = []
    for previous_date in previous_dates[-required_previous:]:
        previous_path = OUTPUT / f"snapshot_{previous_date.replace('-', '')}.json"
        if not previous_path.exists():
            missing_previous.append(previous_date)
            continue
        prior = json.loads(previous_path.read_text(encoding="utf-8"))
        if not validate_prior_snapshot(
                prior, previous_date, model_fingerprint, pool_fingerprint,
                contract_fingerprint):
            missing_previous.append(previous_date)
            continue
        previous_snapshots.append(prior)

    confirmation_ready = not missing_previous and len(previous_snapshots) == required_previous
    top_depth_codes = {code for code, _ in ranked[:TOP_DEPTH]}
    confirmed = []
    if confirmation_ready:
        appearance_counts = {code: 1 for code in top_depth_codes}
        for prior in previous_snapshots:
            prior_top = {row["code"] for row in prior["ranked_scores"][:TOP_DEPTH]}
            for code in top_depth_codes & prior_top:
                appearance_counts[code] += 1
        confirmed = sorted(
            (code for code, count in appearance_counts.items()
             if count >= REQUIRED_APPEARANCES),
            key=lambda code: next(i for i, row in enumerate(ranked) if row[0] == code),
        )
    confirmed_portfolio_codes = [
        code for code in confirmed
        if any(code.startswith(prefix) for prefix in PORTFOLIO_CODE_PREFIXES)
    ]

    quote_count = Database.fetchone(
        "SELECT COUNT(*) AS n FROM daily_quotes WHERE trade_date=%s", (target,)
    )
    if not quote_count or int(quote_count.get("n") or 0) < 1000:
        raise RuntimeError("最新交易日daily_quotes覆盖不足，拒绝采集")
    snapshot = {
        "schema_version": 3,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "captured_after_close_or_holiday": True,
        "earliest_execution_date": next_sessions[0].isoformat(),
        "prediction_date": target_key,
        "market_data_date": target_key,
        "expected_latest_trading_date": expected_latest.isoformat(),
        "market_quote_rows": int(quote_count["n"] if quote_count else 0),
        "source": context.get("source"),
        "model": {"experiment_id": MODEL[0], "run_id": MODEL[1], "label_mode": label_mode},
        "model_fingerprint": model_fingerprint,
        "contract_fingerprint": contract_fingerprint,
        "contract_files_sha256": contract_files_sha256,
        "prediction_artifact_sha256": file_sha256(artifact_paths[-1]),
        "pool": POOL,
        "pool_fingerprint": pool_fingerprint,
        "rank_scope": "固定模型a500全截面排名；组合回放再应用代码前缀和当日可交易过滤",
        "portfolio_code_prefixes": PORTFOLIO_CODE_PREFIXES,
        "neutralize_qlib": False,
        "score_semantics": "daily_cross_sectional_rank_fusion",
        "score_count": len(ranked),
        "scores_sha256": scores_fingerprint,
        "security_reference_source": "BaoStock.query_stock_industry",
        "security_reference_observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "security_references_sha256": security_reference_fingerprint(security_reference_rows),
        "security_references": security_references,
        "confirmation_rule": {
            "depth": TOP_DEPTH,
            "sessions": CONFIRM_SESSIONS,
            "required_appearances": REQUIRED_APPEARANCES,
        },
        "confirmation_ready": confirmation_ready,
        "previous_prediction_dates": [row["prediction_date"] for row in previous_snapshots],
        "missing_previous_prediction_dates": missing_previous,
        "confirmed_top20": confirmed,
        "confirmed_top20_in_portfolio_prefixes": confirmed_portfolio_codes,
        "audit_context": context,
        "ranked_scores": [
            {"rank": rank, "code": code, "score": score}
            for rank, (code, score) in enumerate(ranked, 1)
        ],
    }
    snapshot["snapshot_metadata_sha256"] = snapshot_metadata_fingerprint(snapshot)

    if not validate_prior_snapshot(
            snapshot, target_key, model_fingerprint, pool_fingerprint,
            contract_fingerprint):
        raise RuntimeError("新生成快照未通过自身数据与代码合同校验")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False)
    temp_path = output_path.with_suffix(".tmp")
    temp_path.write_text(serialized, encoding="utf-8")
    temp_path.replace(output_path)
    print(json.dumps({
        "status": "captured",
        "path": str(output_path),
        "prediction_date": target_key,
        "score_count": len(ranked),
        "source": context.get("source"),
        "confirmation_ready": confirmation_ready,
        "confirmed_top20_count": len(confirmed),
        "missing_previous_prediction_dates": missing_previous,
        "scores_sha256": scores_fingerprint,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
