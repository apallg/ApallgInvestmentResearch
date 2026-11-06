def assess_legacy_alias_row_cleanup(alias, old_row, canonical_row):
    blockers = []
    if not alias or not old_row or not canonical_row:
        blockers.append("旧代码别名、待清理行或规范代码行情缺失")
        return {"status": "blocked", "blockers": blockers}

    old_code = str(alias.get("old_symbol", "")).zfill(6)
    canonical_code = str(alias.get("canonical_symbol", "")).zfill(6)
    effective_date = str(alias.get("effective_date", ""))[:10]
    trade_date = str(old_row.get("trade_date", ""))[:10]
    if str(old_row.get("code", "")).zfill(6) != old_code:
        blockers.append("待清理行情代码与别名清单不一致")
    if str(canonical_row.get("code", "")).zfill(6) != canonical_code:
        blockers.append("规范行情代码与别名清单不一致")
    if not effective_date or not trade_date or trade_date < effective_date:
        blockers.append("待清理日期早于代码切换生效日或日期无效")

    old_prices = [old_row.get(field) for field in ("open", "high", "low", "close")]
    canonical_prices = [canonical_row.get(field) for field in ("open", "high", "low", "close")]
    try:
        old_values = [float(value) for value in old_prices]
        canonical_values = [float(value) for value in canonical_prices]
    except (TypeError, ValueError):
        old_values = []
        canonical_values = []
    if len(old_values) != 4 or any(value != 0 for value in old_values):
        blockers.append("待清理旧代码行不是已核验的全零 OHLC 异常")
    if len(canonical_values) != 4 or any(value <= 0 for value in canonical_values):
        blockers.append("规范代码行情 OHLC 不完整或非正")
    elif (canonical_values[1] < canonical_values[2]
          or not canonical_values[2] <= canonical_values[0] <= canonical_values[1]
          or not canonical_values[2] <= canonical_values[3] <= canonical_values[1]):
        blockers.append("规范代码行情 OHLC 关系非法")

    return {"status": "ready" if not blockers else "blocked", "blockers": blockers}
