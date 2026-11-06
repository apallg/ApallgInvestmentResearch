import pandas as pd

from scripts.analyze_top5_top50_hold import simulate, weekly_decisions


def test_weekly_decisions_snap_holiday_thursday_backward():
    dates = [
        "2026-09-21",
        "2026-09-22",
        "2026-09-23",
        "2026-09-28",
        "2026-09-29",
        "2026-09-30",
    ]
    assert weekly_decisions(dates) == ["2026-09-23"]


def test_top50_buffer_retains_name_that_daily_top5_replaces():
    codes = [f"60000{i}" for i in range(1, 7)]
    dates = pd.to_datetime(["2026-09-21", "2026-09-22", "2026-09-23"])
    frames = {}
    for index, code in enumerate(codes, 1):
        frames[code] = pd.DataFrame({
            "trade_date": dates,
            "open": [10.0 + index] * 3,
            "close": [10.0 + index] * 3,
            "adj_factor": [1.0] * 3,
        })
    eligible = {
        "2026-09-21": codes,
        "2026-09-22": [codes[5], codes[1], codes[2], codes[3], codes[4], codes[0]],
    }
    calendar = ["2026-09-21", "2026-09-22", "2026-09-23"]
    top5 = simulate(eligible, {}, frames, {}, calendar, "day", 5)
    top50 = simulate(eligible, {}, frames, {}, calendar, "day", 50)
    top5_sells = [row["code"] for row in top5["trades"] if row["action"] == "sell"]
    top50_sells = [row["code"] for row in top50["trades"] if row["action"] == "sell"]
    assert top5_sells == [codes[0]]
    assert top50_sells == []
