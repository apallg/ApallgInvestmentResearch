import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_router_four_groups_20260917 import _latest_value, _load_trade_data

RESULT = ROOT / "data/experiments/router_four_groups_20260917.json"
OUT = ROOT / "data/experiments/router_account_ledger_verification_20260917.json"

payload = json.loads(RESULT.read_text(encoding="utf-8"))
all_daily = {name: rows["daily"] for name, rows in payload["accounts"].items()}
codes = set()
dates = set()
for rows in all_daily.values():
    for row in rows:
        dates.add(pd.Timestamp(row["following_execution"]))
        codes.update(row.get("holdings", {}).keys())
        codes.update(order["code"] for order in row.get("orders", []))
trade_data = _load_trade_data(codes, sorted(dates))
open_df, close_df = trade_data["$open"], trade_data["$close"]

checks = {}
for name, rows in all_daily.items():
    cash = 1_000_000.0
    holdings = {}
    max_cash_diff = 0.0
    max_nav_diff = 0.0
    holding_mismatches = 0
    cash_mismatches = 0
    nav_mismatches = 0
    for row in rows:
        for order in row.get("orders", []):
            code = order["code"]
            shares = int(order["shares"])
            gross = float(order["gross"])
            cost = float(order["cost"])
            if order["side"] == "sell":
                holdings[code] = holdings.get(code, 0) - shares
                cash += gross - cost
            else:
                holdings[code] = holdings.get(code, 0) + shares
                cash -= gross + cost
            if holdings.get(code) == 0:
                holdings.pop(code, None)
        expected_holdings = {str(code): int(qty) for code, qty in holdings.items()}
        observed_holdings = {str(code): int(qty) for code, qty in row.get("holdings", {}).items() if int(qty) != 0}
        if expected_holdings != observed_holdings:
            holding_mismatches += 1
        cash_diff = abs(cash - float(row["cash"]))
        max_cash_diff = max(max_cash_diff, cash_diff)
        if cash_diff > 1e-8:
            cash_mismatches += 1
        valuation = cash
        following = pd.Timestamp(row["following_execution"])
        for code, qty in holdings.items():
            px = _latest_value(open_df, following, code)
            if not np.isfinite(px) or px <= 0:
                px = _latest_value(close_df, following, code)
            if np.isfinite(px) and px > 0:
                valuation += qty * px
        nav_diff = abs(valuation / 1_000_000.0 - float(row["nav"]))
        max_nav_diff = max(max_nav_diff, nav_diff)
        if nav_diff > 1e-8:
            nav_mismatches += 1
    checks[name] = {
        "rows": len(rows),
        "holding_mismatches": holding_mismatches,
        "cash_mismatches": cash_mismatches,
        "nav_mismatches": nav_mismatches,
        "max_cash_abs_diff": max_cash_diff,
        "max_nav_abs_diff": max_nav_diff,
        "passed": holding_mismatches == 0 and cash_mismatches == 0 and nav_mismatches == 0,
    }

result = {
    "status": "passed" if all(item["passed"] for item in checks.values()) else "failed",
    "source": str(RESULT.relative_to(ROOT)),
    "contract": payload["contract"],
    "checks": checks,
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
