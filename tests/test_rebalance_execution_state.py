from unittest.mock import patch
from types import SimpleNamespace
import datetime

import pytest

from backend.service.rebalance import (
    RebalanceService,
    _execution_advice,
    _intraday_execution_blockers,
    _load_half_sold,
)
from core.domain.portfolio.execution import ACTION_LIMIT, ACTION_MARKET, PHASE_CLOSED, ExecutionDecision


@pytest.mark.p0
@pytest.mark.parametrize("price,commission,expected_error", [
    ("10.0000000000000000000000000000001", "0.30", "成交价格最多三位小数"),
    ("10.000", "0.30000000000000000000000000001", "实际费用必须为非负且最多两位小数"),
])
def test_manual_fill_rejects_decimal_precision_without_float_rounding(price, commission, expected_error):
    signal = {"id": 7, "account_id": "LOCAL-LIVE", "planned_shares": 100,
              "execution_status": "pending", "created_at": datetime.datetime.now(),
              "code": "000001", "side": "sell", "account_version": 1}
    with patch("core.data.RebalanceSignalRepo.get_by_id", return_value=signal), \
         patch("core.data.RebalanceSignalRepo.record_execution") as record:
        result = RebalanceService.mark_executed(7, {
            "shares": 100, "price": price, "request_id": "00000000-0000-4000-8000-000000000001",
            "expected_account_version": 1, "commission": commission,
            "stamp_duty": "0", "actor": "admin",
        })
    assert expected_error in result["error"]
    record.assert_not_called()


@pytest.mark.p0
def test_batch_fill_forwards_broker_trade_id():
    with patch.object(RebalanceService, "mark_executed", return_value={"execution_status": "executed"}) as record:
        result = RebalanceService.batch_execute([{
            "signal_id": 7, "shares": 100, "price": "10.000", "request_id": "req-1",
            "expected_account_version": 2, "commission": "0.30", "stamp_duty": "0",
            "broker_trade_id": "TRADE-1", "actor": "admin",
        }])
    assert result["executed"] == 1
    assert record.call_args.args[1]["broker_trade_id"] == "TRADE-1"


@pytest.mark.p0
def test_load_half_sold_accepts_repo_deserialized_list():
    with patch("core.data.repos.settings.SettingsRepo.get", return_value=["600001"]) as get:
        assert _load_half_sold() == {"600001"}
    get.assert_called_once_with("rebalance_half_sold:LOCAL-LIVE", None)


@pytest.mark.unit
def test_load_half_sold_is_scoped_to_requested_account():
    with patch("core.data.repos.settings.SettingsRepo.get", return_value=["600002"]) as get:
        assert _load_half_sold("LOCAL-OTHER") == {"600002"}
    get.assert_called_once_with("rebalance_half_sold:LOCAL-OTHER", None)


@pytest.mark.p0
def test_buy_limit_in_range_is_marked_ready():
    result = SimpleNamespace(
        current_price=9.9, vwap=10.0, open_price=9.8,
        entry_quality_score=70, risk_warning=None, nearest_support=None,
        code="600001",
    )
    with patch("core.domain.portfolio.execution.decide_execution",
               return_value=ExecutionDecision(ACTION_LIMIT, 10.0, "test")):
        advice = _execution_advice(result, {"close": 9.8}, side="buy", blockers=[],
                                   quote={"prev_close": 10.0, "name": "测试股份"})
    assert advice["status"] == "ready"
    assert advice["can_execute"] is True


@pytest.mark.p0
def test_sell_limit_below_target_is_marked_waiting():
    result = SimpleNamespace(
        current_price=9.8, vwap=9.9, open_price=9.7,
        entry_quality_score=None, risk_warning=None, nearest_support=None,
        code="600001",
    )
    with patch("core.domain.portfolio.execution.decide_execution",
               return_value=ExecutionDecision(ACTION_LIMIT, 10.0, "test")):
        advice = _execution_advice(result, {"close": 9.7}, side="sell", blockers=[],
                                   quote={"prev_close": 10.0, "name": "测试股份"})
    assert advice["status"] == "waiting"
    assert advice["can_execute"] is False
    assert "10.00" in advice["trigger_text"]


@pytest.mark.p0
def test_execution_advice_fails_closed_without_readiness_evidence():
    result = SimpleNamespace(
        current_price=10.0, vwap=10.0, open_price=10.0,
        entry_quality_score=70, risk_warning=None, nearest_support=None,
    )
    advice = _execution_advice(result, {"close": 10.0}, side="sell", urgency="stop_loss")
    assert advice["status"] == "blocked"
    assert advice["can_execute"] is False
    assert "执行条件未核验" in advice["blockers"]


@pytest.mark.p0
def test_closed_session_blocks_execution_but_preserves_stop_loss_intent():
    result = SimpleNamespace(
        current_price=10.0, vwap=10.0, open_price=10.0,
        entry_quality_score=None, risk_warning=None, nearest_support=None,
    )
    advice = _execution_advice(
        result, {"close": 10.0}, side="sell", urgency="stop_loss",
        phase=PHASE_CLOSED, blockers=["当前不是有效交易时段"],
    )
    assert advice["action"] == "wait"
    assert advice["intent_action"] == ACTION_MARKET
    assert advice["can_execute"] is False
    assert "当前不是有效交易时段" in advice["trigger_text"]


@pytest.mark.p0
def test_intraday_readiness_requires_current_verified_bars_and_fresh_quote():
    import pandas as pd
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 10, 30, 10, 0, tzinfo=timezone(timedelta(hours=8)))
    bars = pd.DataFrame({"close": [9.9]}, index=pd.DatetimeIndex(["2026-10-30 09:59:00"]))
    quote = {
        "price": 9.9,
        "prev_close": 10.0,
        "name": "测试股份",
        "session_date": "2026-10-30",
        "observed_at": "2026-10-30T10:00:00",
    }
    assert _intraday_execution_blockers(
        "600001", bars, quote, "2026-10-30", now, True, "buy", bars_verified=True,
        semantics_verified=True,
    ) == []

    blockers = _intraday_execution_blockers(
        "600001", None, None, "2026-10-30", now, False, "buy",
    )
    assert "当前不是有效交易时段" in blockers
    assert "缺少新鲜实时执行报价" in blockers
    assert "缺少当日已验证分钟线" in blockers
    unverified_bars = _intraday_execution_blockers(
        "600001", bars, quote, "2026-10-30", now, True, "buy",
    )
    assert "分钟线未通过verified状态校验" in unverified_bars
    assert "分钟数据单位、累计口径或价格基准未完成语义核验" in unverified_bars

    for stale_quote in (
        {**quote, "observed_at": "2026-10-30T09:59:54"},
        {**quote, "observed_at": "2026-10-30T10:00:06"},
        {key: value for key, value in quote.items() if key != "observed_at"},
    ):
        stale_blockers = _intraday_execution_blockers(
            "600001", bars, stale_quote, "2026-10-30", now, True, "buy", bars_verified=True,
            semantics_verified=True,
        )
        assert "实时报价时间缺失、过期或不属于当前交易日" in stale_blockers


@pytest.mark.p0
def test_intraday_preview_with_only_previous_day_data_is_not_executable():
    result_obj = SimpleNamespace(
        current_price=10.0, vwap=None, open_price=None, entry_quality_score=35,
        risk_warning=None, nearest_support=None, code=None,
    )
    with patch("backend.service.rebalance._fetch_intraday_data",
               return_value={"600001": {"df": None, "prev_day": {"close": 10.0},
                                         "quantity_status": "unknown", "semantic_status": "unverified",
                                         "data_envelope": None, "source": "test"}}), \
         patch("backend.service.market_data.MarketDataService.get_live_quotes",
               return_value={"data": {}, "meta": {"freshness": "unavailable"}}), \
         patch("core.shared.trading_time.is_trading_time", return_value=False), \
         patch("core.domain.portfolio.intraday.IntradayPullbackDetector.analyze",
               return_value=result_obj), \
         patch("backend.service.rebalance._intraday_result_to_dict", return_value={
             "code": "600001", "entry_quality_score": 35, "entry_quality_label": "一般",
             "signals": [{"type": "volume_shrink"}], "supports": [{"name": "VWAP"}],
             "nearest_support": {"name": "VWAP"}, "recent_bars": [{"close": 10.0}],
             "vwap": 10.0, "volume_ma": 100,
         }):
        result = RebalanceService.analyze_intraday_entry(["600001"])

    item = result["results"][0]
    execution = item["execution"]
    assert item["analysis_status"] == "unverified"
    assert item["entry_quality_score"] is None
    assert item["entry_quality_label"] == "数据未认证"
    assert item["signals"] == []
    assert item["supports"] == []
    assert item["nearest_support"] is None
    assert item["recent_bars"] == []
    assert item["vwap"] is None
    assert item["volume_ma"] is None
    assert "未核验" in item["risk_warning"]
    assert execution["can_execute"] is False
    assert execution["status"] == "blocked"
    assert "缺少新鲜实时执行报价" in execution["blockers"]
    assert "缺少当日已验证分钟线" in execution["blockers"]


def test_rebalance_api_exposes_definitive_noncommit_outcome_header():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api.rebalance import router

    app = FastAPI()
    app.include_router(router)
    payload = {
        "screening_save_id": 1,
        "decision_id": "decision-test",
        "preview_only": False,
        "commit_request_id": "00000000-0000-4000-8000-000000000123",
        "expected_plan_fingerprint": "a" * 64,
    }
    with patch("backend.service.rebalance.RebalanceService.run_rebalance", return_value={
        "error": "账户版本已变化",
        "commit_outcome": "not_committed",
    }):
        response = TestClient(app).post("/run", json=payload)

    assert response.status_code == 400
    assert response.headers["x-commit-outcome"] == "not_committed"


@pytest.mark.p0
def test_unverified_minute_analysis_is_removed_before_buy_and_sell_execution_advice():
    import pandas as pd
    from datetime import datetime, timedelta

    execution_inputs = []

    def analyze(*args, **kwargs):
        result = SimpleNamespace(
            current_price=8.0, vwap=80.0, open_price=90.0,
            entry_quality_score=99, risk_warning=None,
            nearest_support=SimpleNamespace(price=70.0), code="600001",
        )
        return result

    def capture_execution_advice(result, *args, **kwargs):
        execution_inputs.append({
            "current_price": result.current_price,
            "vwap": result.vwap,
            "open_price": result.open_price,
            "entry_quality_score": result.entry_quality_score,
            "nearest_support": result.nearest_support,
            "risk_warning": result.risk_warning,
        })
        return {"status": "blocked", "can_execute": False, "blockers": ["测试阻断"]}

    bars = pd.DataFrame(
        {"open": [90.0], "high": [91.0], "low": [89.0], "close": [8.0], "volume": [1000]},
        index=pd.DatetimeIndex([datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=1)]),
    )
    fetch_result = {
        "df": bars, "prev_day": {"close": 10.0}, "source": "test",
        "quantity_status": "unknown", "semantic_status": "unverified",
        "data_envelope": None,
    }

    with patch("backend.service.rebalance._fetch_intraday_data", return_value={"600001": fetch_result}), \
         patch("backend.service.market_data.MarketDataService.get_live_quotes",
               return_value={"data": {}, "meta": {"freshness": "unavailable"}}), \
         patch("core.shared.trading_time.is_trading_time", return_value=False), \
         patch("core.domain.portfolio.intraday.IntradayPullbackDetector.analyze", side_effect=analyze), \
         patch("backend.service.rebalance._intraday_result_to_dict", return_value={"code": "600001"}), \
         patch("backend.service.rebalance._execution_advice", side_effect=capture_execution_advice), \
         patch("backend.service.local_ledger.LocalLedgerService.get_t0_account_snapshot",
               return_value={"holdings": [{"code": "600001", "sellable_shares": 100}]}):
        result = RebalanceService.analyze_intraday_entry(
            ["600001"],
            sell_signals=[{"code": "600001", "side": "sell", "suggested_shares": 100}],
        )

    assert len(execution_inputs) == 2
    for item in execution_inputs:
        assert item["current_price"] is None
        assert item["vwap"] == 0.0
        assert item["open_price"] == 0.0
        assert item["entry_quality_score"] is None
        assert item["nearest_support"] is None
        assert "语义未核验" in item["risk_warning"]
    assert result["sell_results"][0]["analysis_status"] == "unverified"


@pytest.mark.p0
def test_intraday_preview_can_be_ready_when_all_readiness_gates_pass():
    import pandas as pd
    from datetime import datetime, timedelta, timezone
    from core.domain.portfolio.execution import PHASE_MID

    now = datetime.now(timezone(timedelta(hours=8)))
    reference_date = now.date().isoformat()
    result_obj = SimpleNamespace(
        current_price=9.9, vwap=10.0, open_price=9.8, entry_quality_score=70,
        risk_warning=None, nearest_support=None, code=None,
    )
    bars = pd.DataFrame({"close": [9.9]}, index=pd.DatetimeIndex([
        (now.replace(tzinfo=None) - timedelta(minutes=1)).isoformat(),
    ]))
    with patch("backend.service.rebalance._fetch_intraday_data", return_value={
            "600001": {"df": bars, "prev_day": {"close": 10.0},
                       "source": "ths", "quantity_status": "verified", "semantic_status": "verified"}}), \
         patch("backend.service.market_data.MarketDataService.get_live_quotes", return_value={
             "data": {"600001": {"price": 9.9, "prev_close": 10.0, "name": "测试股份",
                                  "session_date": reference_date,
                                  "observed_at": now.replace(tzinfo=None).isoformat()}},
             "meta": {"freshness": "fresh"}}), \
         patch("core.shared.trading_time.is_trading_time", return_value=True), \
         patch("core.shared.trading_time.market_date", return_value=now.date()), \
         patch("core.domain.portfolio.execution.resolve_phase", return_value=PHASE_MID), \
         patch("core.domain.portfolio.intraday.IntradayPullbackDetector.analyze",
               return_value=result_obj), \
         patch("backend.service.rebalance._intraday_result_to_dict", return_value={"code": "600001"}):
        result = RebalanceService.analyze_intraday_entry(["600001"], reference_date=reference_date)

    execution = result["results"][0]["execution"]
    assert execution["status"] == "ready"
    assert execution["can_execute"] is True
    assert execution["blockers"] == []


@pytest.mark.p0
def test_intraday_readiness_blocks_limit_down_and_excess_sell_quantity():
    import pandas as pd
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 10, 30, 10, 0, tzinfo=timezone(timedelta(hours=8)))
    bars = pd.DataFrame({"close": [9.0]}, index=pd.DatetimeIndex(["2026-10-30 09:59:00"]))
    quote = {
        "price": 9.0,
        "prev_close": 10.0,
        "name": "测试股份",
        "session_date": "2026-10-30",
        "observed_at": "2026-10-30T10:00:00",
    }
    blockers = _intraday_execution_blockers(
        "600001", bars, quote, "2026-10-30", now, True, "sell",
        requested_shares=200, sellable_shares=100,
        bars_verified=True,
    )
    assert "现价触及或低于跌停价，卖出不可执行" in blockers
    assert "卖出计划超过本地核验的可卖股数" in blockers
    missing_sellable = _intraday_execution_blockers(
        "600001", bars, {**quote, "price": 9.9}, "2026-10-30", now, True, "sell",
        requested_shares=100, sellable_shares=None,
        bars_verified=True,
    )
    assert "可卖股数缺失或为零" in missing_sellable


@pytest.mark.p0
def test_get_cooldown_returns_names_and_remaining_periods():
    payload = {"as_of": "2026-08-18", "periods": {"600001": 2}}
    with patch("core.data.repos.settings.SettingsRepo.get", return_value=payload), \
         patch("core.data.connection.Database.fetchall",
               return_value=[{"code": "600001", "name": "测试股"}]):
        result = RebalanceService.get_cooldown()
    assert result == {
        "as_of": "2026-08-18",
        "items": [{"code": "600001", "name": "测试股", "remaining_periods": 2}],
    }


@pytest.mark.p0
def test_unmanaged_live_signal_skips_champion_gate_when_recording_buy():
    import datetime

    signal = {
        "id": 7,
        "batch_id": "batch",
        "code": "000001",
        "name": "测试股",
        "side": "buy",
        "urgency": "normal",
        "execution_status": "pending",
        "source_mode": "live_unmanaged",
        "created_at": datetime.datetime.now(),
        "planned_price": 10.0,
        "amount": 1000.0,
        "input_fingerprint": None,
        "account_id": "LOCAL-LIVE",
        "account_version": 1,
        "planned_shares": 100,
    }
    with patch("core.data.RebalanceSignalRepo.get_by_id", return_value=signal), \
         patch("core.data.RebalanceSignalRepo.record_execution",
               return_value={"execution_status": "executed", "already_recorded": False, "remaining_shares": 100}) as record, \
         patch("core.data.HoldingsRepo.get_by_code", return_value=None), \
         patch("core.data.HoldingsRepo.get_all", return_value=[]), \
          patch("backend.service.strategy_profile.StrategyProfileService.ensure_execution_allowed") as gate:
        result = RebalanceService.mark_executed(7, {"shares": 100, "price": 10.0,
            "request_id": "00000000-0000-4000-8000-000000000001", "expected_account_version": 1,
            "commission": "0.30", "stamp_duty": "0", "actor": "admin"})

    assert result["signal_id"] == 7
    assert result["execution_status"] == "executed"
    gate.assert_called_once_with(None, None, None, enforce=False)
    record.assert_called_once()


@pytest.mark.p0
@pytest.mark.parametrize("status", ["cancelled", "skipped"])
def test_non_pending_signal_cannot_change_holdings(status):
    signal = {"id": 7, "code": "000001", "side": "sell", "execution_status": status}
    with patch("core.data.RebalanceSignalRepo.get_by_id", return_value=signal), \
         patch("core.data.RebalanceSignalRepo.record_execution") as record:
        result = RebalanceService.mark_executed(7, {"shares": 100, "price": 10})
    assert "error" in result
    record.assert_not_called()


@pytest.mark.p0
def test_partial_sell_records_remaining_holding_in_one_transaction():
    import datetime
    from unittest.mock import Mock
    from core.data.repos.rebalance import RebalanceSignalRepo

    signal = {"id": 7, "batch_id": "batch", "code": "000001", "side": "sell",
              "execution_status": "pending", "created_at": datetime.datetime.now(),
              "input_fingerprint": "fingerprint", "account_id": "LOCAL-LIVE",
              "account_version": 1, "planned_shares": 500, "filled_shares": 0}
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1, "cash": 1000, "frozen": 0}, None, None, signal,
        {"shares": 0},
    ]
    cursor.fetchall.side_effect = [
        [{"code": "000001", "shares": 1000, "cost_price": 9, "buy_date": "2020-01-01"}],
        [{"code": "000001", "shares": 700, "cost_price": 9, "buy_date": "2020-01-01"}],
        [],
    ]
    cursor.rowcount = 1
    cursor.execute.side_effect = lambda sql, params=None: setattr(
        cursor, "rowcount", 0 if "execution_status='cancelled'" in sql else 1,
    )
    with patch("core.data.repos.local_ledger.LocalLedgerRepo.audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda operation: operation(cursor)) as transaction:
        result = RebalanceSignalRepo.record_execution(7, {"shares": 300, "price": 10,
            "request_id": "00000000-0000-4000-8000-000000000001", "expected_account_version": 1,
            "commission": "1.00", "stamp_duty": "1.50", "actor": "admin"}, lambda rows: "fingerprint")

    assert result["execution_status"] == "partial"
    assert result["remaining_plan_shares"] == 200
    assert result["cash_after"] == 3997.50
    statements = [call.args[0] for call in cursor.execute.call_args_list]
    assert any(
        "UPDATE holdings SET shares=%s,version=version+1 WHERE code=%s AND account_id=%s" in sql
        for sql in statements
    )
    assert not any("DELETE FROM holdings" in sql for sql in statements)
    assert any("UPDATE rebalance_signals SET filled_shares" in sql for sql in statements)
    assert any("INSERT INTO local_ledger_events" in sql for sql in statements)
    transaction.assert_called_once()


@pytest.mark.p0
def test_execution_state_write_failure_stops_transaction():
    import datetime
    from unittest.mock import Mock
    from core.data.repos.rebalance import RebalanceSignalRepo

    signal = {"id": 7, "batch_id": "batch", "code": "000001", "side": "sell",
              "execution_status": "pending", "created_at": datetime.datetime.now(),
              "input_fingerprint": "fingerprint", "account_id": "LOCAL-LIVE",
              "account_version": 1, "planned_shares": 300, "filled_shares": 0}
    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 1, "cash": 1000, "frozen": 0}, None, None, signal,
        {"shares": 0},
    ]
    cursor.fetchall.return_value = [{"code": "000001", "shares": 1000, "cost_price": 9, "buy_date": "2020-01-01"}]

    def execute(sql, params=None):
        cursor.rowcount = 0 if "UPDATE rebalance_signals SET filled_shares" in sql or \
            "execution_status='cancelled'" in sql else 1

    cursor.execute.side_effect = execute
    with patch("core.data.repos.local_ledger.LocalLedgerRepo.audit_integrity", return_value={"ok": True}), \
         patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda operation: operation(cursor)):
        with pytest.raises(RuntimeError, match="信号状态写入失败"):
            RebalanceSignalRepo.record_execution(7, {"shares": 300, "price": 10,
                "request_id": "00000000-0000-4000-8000-000000000001", "expected_account_version": 1,
                "commission": "1.00", "stamp_duty": "1.50", "actor": "admin"}, lambda rows: "fingerprint")
    assert not any("INSERT INTO local_ledger_events" in call.args[0] for call in cursor.execute.call_args_list)


@pytest.mark.p0
def test_repeated_execution_does_not_touch_holdings():
    from unittest.mock import Mock
    from core.data.repos.rebalance import RebalanceSignalRepo

    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "active", "version": 2, "cash": 3997.50, "frozen": 0},
        {"signal_id": 7, "payload_json": {"shares": 300, "price": 10,
            "request_id": "00000000-0000-4000-8000-000000000001", "expected_account_version": 1,
            "commission": "1.00", "stamp_duty": "1.50", "actor": "admin",
            "execution_status": "partial", "remaining_plan_shares": 200}, "account_version": 2},
    ]
    with patch("core.data.repos.rebalance.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = RebalanceSignalRepo.record_execution(7, {"shares": 300, "price": 10,
            "request_id": "00000000-0000-4000-8000-000000000001", "expected_account_version": 1,
            "commission": "1.00", "stamp_duty": "1.50", "actor": "admin"}, lambda rows: "fingerprint")
    assert result["already_recorded"] is True
    assert cursor.fetchall.call_count == 0
    assert cursor.execute.call_count == 2


@pytest.mark.p0
def test_skip_cannot_override_execution_after_concurrent_state_change():
    signal = {"id": 7, "account_id": "LOCAL-LIVE", "execution_status": "pending"}
    with patch("core.data.RebalanceSignalRepo.get_by_id", return_value=signal), \
         patch("core.data.RebalanceSignalRepo.mark_skipped", return_value=0):
        result = RebalanceService.mark_skipped(7)
    assert "error" in result
