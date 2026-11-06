import asyncio


def test_stalled_backtest_is_reported_as_stalled(monkeypatch):
    from backend.api import backtest

    monkeypatch.setattr(
        backtest.task_manager,
        "get_dict",
        lambda job_id: {
            "status": "stalled",
            "progress": {"message": "超过 10 分钟未收到心跳"},
            "description": "策略回测",
            "error": "",
        },
    )

    result = asyncio.run(backtest.get_direction_backtest_result("job-1"))

    assert result["status"] == "stalled"
    assert result["error"] == ""
    assert "心跳" in result["message"]


def test_failed_backtest_keeps_real_error(monkeypatch):
    from backend.api import backtest

    monkeypatch.setattr(
        backtest.task_manager,
        "get_dict",
        lambda job_id: {"status": "failed", "error": "模型加载失败"},
    )

    result = asyncio.run(backtest.get_direction_backtest_result("job-2"))

    assert result == {"status": "error", "error": "模型加载失败"}
