from unittest.mock import Mock, patch

import pytest

from core.data.repos.holdings import HoldingsRepo
from core.data.repos.local_ledger import ACCOUNT_ID


def test_holdings_default_reads_use_local_account_scope():
    with patch("core.data.repos.holdings.Database.fetchall", return_value=[]) as fetchall, \
         patch("core.data.repos.holdings.Database.fetchone", return_value=None) as fetchone:
        HoldingsRepo.get_all()
        HoldingsRepo.get_by_code("000001")

    assert "WHERE account_id=%s" in fetchall.call_args.args[0]
    assert fetchall.call_args.args[1] == (ACCOUNT_ID,)
    assert "account_id=%s" in fetchone.call_args.args[0]
    assert fetchone.call_args.args[1] == ("000001", ACCOUNT_ID)


def test_empty_account_id_is_rejected_before_holding_query():
    with patch("core.data.repos.holdings.Database.fetchall", return_value=[]) as fetchall:
        with pytest.raises(ValueError, match="必须指定账户"):
            HoldingsRepo.get_all("")

    fetchall.assert_not_called()


def test_holding_count_defaults_to_local_account_scope():
    with patch("core.data.repos.holdings.Database.fetchone", return_value={"cnt": 2}) as fetchone:
        assert HoldingsRepo.count() == 2

    assert "WHERE account_id=%s" in fetchone.call_args.args[0]
    assert fetchone.call_args.args[1] == (ACCOUNT_ID,)


def test_holding_adjustment_reads_use_local_account_scope():
    with patch("core.data.repos.holdings.Database.fetchall", return_value=[]) as fetchall:
        HoldingsRepo.get_adjustments("000001", 20)

    assert "WHERE account_id=%s AND code=%s" in fetchall.call_args.args[0]
    assert fetchall.call_args.args[1] == (ACCOUNT_ID, "000001", 20)


def test_holding_creation_binds_position_and_adjustment_to_local_account():
    cursor = Mock()
    cursor.fetchone.return_value = {"status": "uninitialized"}
    cursor.lastrowid = 4
    with patch("core.data.repos.holdings.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = HoldingsRepo.add("000001", "测试", 100, "10.00", "2026-01-01")

    assert result == 4
    inserts = [call for call in cursor.execute.call_args_list if call.args[0].startswith("INSERT INTO")]
    assert len(inserts) == 2
    assert "account_id,code" in inserts[0].args[0]
    assert inserts[0].args[1][0] == ACCOUNT_ID
    assert "account_id,code" in inserts[1].args[0]
    assert inserts[1].args[1][0] == ACCOUNT_ID


def test_holding_update_locks_and_updates_only_local_account_row():
    cursor = Mock()
    before = {"id": 4, "code": "000001", "account_id": ACCOUNT_ID, "name": "旧名", "shares": 100,
              "cost_price": "10.00", "buy_date": "2026-01-01", "version": 1}
    after = {**before, "name": "新名", "version": 2}
    cursor.rowcount = 1
    cursor.fetchone.side_effect = [before, after]
    with patch("core.data.repos.holdings.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = HoldingsRepo.update("000001", name="新名", expected_id=4, expected_version=1)

    assert result == 1
    reads = [call for call in cursor.execute.call_args_list if call.args[0].startswith("SELECT * FROM holdings")]
    updates = [call for call in cursor.execute.call_args_list if call.args[0].startswith("UPDATE holdings")]
    assert len(reads) == 2
    assert all("account_id=%s" in call.args[0] for call in reads)
    assert all(call.args[1] == ("000001", ACCOUNT_ID) for call in reads)
    assert len(updates) == 1
    assert "version=version+1" in updates[0].args[0]
    assert "WHERE code=%s AND account_id=%s AND id=%s AND version=%s" in updates[0].args[0]
    assert updates[0].args[1][-4:] == ("000001", ACCOUNT_ID, 4, 1)


def test_holding_delete_locks_and_deletes_only_local_account_row():
    cursor = Mock()
    cursor.rowcount = 1
    cursor.fetchone.side_effect = [
        {"status": "uninitialized"},
        {"id": 4, "code": "000001", "account_id": ACCOUNT_ID, "shares": 100,
         "cost_price": "10.00", "buy_date": "2026-01-01", "version": 1},
    ]
    with patch("core.data.repos.holdings.Database.transaction", side_effect=lambda operation: operation(cursor)):
        result = HoldingsRepo.delete(
            "000001", "核对期初前清理旧快照", 100, "10.00", "2026-01-01", 4, 1,
        )

    assert result == 1
    delete = next(call for call in cursor.execute.call_args_list if call.args[0].startswith("DELETE FROM holdings"))
    assert "WHERE code=%s AND account_id=%s AND id=%s AND version=%s" in delete.args[0]
    assert delete.args[1] == ("000001", ACCOUNT_ID, 4, 1)


def test_holding_update_rejects_stale_row_version_for_non_position_fields():
    from core.data.repos.holdings import StaleHoldingError

    cursor = Mock()
    cursor.fetchone.return_value = {
        "id": 4, "code": "000001", "account_id": ACCOUNT_ID, "name": "新名", "shares": 100,
        "cost_price": "10.00", "buy_date": "2026-01-01", "version": 2,
    }
    with patch("core.data.repos.holdings.Database.transaction", side_effect=lambda operation: operation(cursor)):
        try:
            HoldingsRepo.update("000001", name="旧请求名", expected_id=4, expected_version=1)
        except StaleHoldingError:
            pass
        else:
            raise AssertionError("stale row version must be rejected")

    assert not any(call.args[0].startswith("UPDATE holdings") for call in cursor.execute.call_args_list)


def test_holding_delete_rejects_stale_row_version_before_delete():
    from core.data.repos.holdings import StaleHoldingError

    cursor = Mock()
    cursor.fetchone.side_effect = [
        {"status": "uninitialized"},
        {"id": 4, "code": "000001", "account_id": ACCOUNT_ID, "shares": 100,
         "cost_price": "10.00", "buy_date": "2026-01-01", "version": 2},
    ]
    with patch("core.data.repos.holdings.Database.transaction", side_effect=lambda operation: operation(cursor)):
        try:
            HoldingsRepo.delete(
                "000001", "并发更新后的过期删除请求", 100, "10.00", "2026-01-01", 4, 1,
            )
        except StaleHoldingError:
            pass
        else:
            raise AssertionError("stale row version must be rejected")

    assert not any(call.args[0].startswith("DELETE FROM holdings") for call in cursor.execute.call_args_list)


def test_holding_adjustment_route_passes_local_account_scope():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.portfolio import router

    app = FastAPI()
    app.include_router(router)
    with patch("core.data.HoldingsRepo.get_adjustments", return_value=[]) as get_adjustments:
        response = TestClient(app).get("/holdings/adjustments?code=000001&limit=20")

    assert response.status_code == 200
    get_adjustments.assert_called_once_with("000001", 20, ACCOUNT_ID)
