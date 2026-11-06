import pandas as pd

from scripts.analyze_top5_top50_hold import simulate
from scripts.analyze_top50_exit_grace import simulate_grace


def build_case(days=7):
    dates = pd.bdate_range("2026-09-21", periods=days)
    calendar = [value.strftime("%Y-%m-%d") for value in dates]
    codes = [f"600{i:03d}" for i in range(1, 52)]
    frames = {}
    for index, code in enumerate(codes[:6], 1):
        frames[code] = pd.DataFrame({
            "trade_date": dates,
            "open": [10.0 + index] * days,
            "close": [10.0 + index] * days,
            "adj_factor": [1.0] * days,
        })
    initial = codes[:]
    outside = [codes[5], codes[1], codes[2], codes[3], codes[4], *codes[6:], codes[0]]
    eligible = {calendar[0]: initial}
    for date in calendar[1:-1]:
        eligible[date] = outside
    return codes, calendar, frames, eligible


def test_one_day_grace_matches_immediate_top50_exit():
    _, calendar, frames, eligible = build_case()
    immediate = simulate(eligible, {}, frames, {}, calendar, "day", 50)
    grace = simulate_grace(eligible, frames, {}, calendar, 1)
    assert [row["equity"] for row in immediate["equity"]] == [row["equity"] for row in grace["equity"]]
    assert [row["code"] for row in immediate["trades"]] == [row["code"] for row in grace["trades"]]


def test_five_day_grace_sells_after_five_consecutive_days_outside_top50():
    codes, calendar, frames, eligible = build_case()
    result = simulate_grace(eligible, frames, {}, calendar, 5)
    sells = [row for row in result["trades"] if row["action"] == "sell"]
    assert len(sells) == 1
    assert sells[0]["code"] == codes[0]
    assert sells[0]["decision_date"] == calendar[5]
    assert sells[0]["outside_top50_days"] == 5
