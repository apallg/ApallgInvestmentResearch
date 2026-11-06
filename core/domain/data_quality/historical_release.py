import math

import pandas as pd

HISTORICAL_VOLUME_TOLERANCE_SHARES = 10
HISTORICAL_QFQ_PRICE_TOLERANCE = 0.0051


def invalid_or_duplicate_date_axes(date_axes):
    invalid = []
    for name, values in date_axes.items():
        if values is None:
            invalid.append(name)
            continue
        dates = pd.to_datetime(pd.Series(values), errors="coerce", utc=True).dt.normalize()
        if dates.isna().any() or dates.duplicated().any():
            invalid.append(name)
    return invalid


def classify_database_qfq_price(database_row_present, database_qfq_close, candidate_qfq_close):
    if pd.isna(database_row_present) or not bool(database_row_present):
        return "no_database_row"
    try:
        database_close = float(database_qfq_close)
        candidate_close = float(candidate_qfq_close)
    except (TypeError, ValueError):
        return "unverifiable"
    if not math.isfinite(database_close) or not math.isfinite(candidate_close):
        return "unverifiable"
    return "matched" if abs(database_close - candidate_close) <= HISTORICAL_QFQ_PRICE_TOLERANCE else "database_price_conflict"


def assess_quote_shard_release(expected, candidate, comparison, stored_rows, replace_price_conflicts=False,
                               verified_delist_dates=()):
    blockers = []
    if expected.empty:
        blockers.append("G0 未提供该股票年度的预期日期")
    if candidate.empty:
        blockers.append("影子候选为空")
    if comparison.empty:
        blockers.append("缺少双源逐日比较证据")
    if blockers:
        return {"status": "blocked", "blockers": blockers, "expected_rows": 0, "candidate_rows": len(candidate)}

    expected = expected.copy()
    candidate = candidate.copy()
    comparison = comparison.copy()
    stored_rows = stored_rows.copy()
    expected["trade_date"] = pd.to_datetime(expected["trade_date"], errors="coerce").dt.normalize()
    candidate["trade_date"] = pd.to_datetime(candidate["date"], errors="coerce").dt.normalize()
    comparison_date = "date_key" if "date_key" in comparison else "trade_date"
    comparison["trade_date"] = pd.to_datetime(comparison[comparison_date], errors="coerce").dt.normalize()
    if not stored_rows.empty:
        stored_rows["trade_date"] = pd.to_datetime(stored_rows["trade_date"], errors="coerce").dt.normalize()

    if expected["trade_date"].isna().any() or expected["trade_date"].duplicated().any():
        blockers.append("G0 预期日期含无效值或重复键")
    if candidate["trade_date"].isna().any() or candidate["trade_date"].duplicated().any():
        blockers.append("影子候选日期含无效值或重复键")
    if comparison["trade_date"].isna().any() or comparison["trade_date"].duplicated().any():
        blockers.append("双源比较日期含无效值或重复键")

    if "coverage_status" not in expected or "full_day_suspended" not in expected:
        blockers.append("G0 分母缺少停牌或覆盖分类")
        tradable = expected.iloc[0:0]
        suspended = expected.iloc[0:0]
    else:
        unresolved = expected["coverage_status"].astype(str).eq("missing_row_unresolved")
        full_day_suspended = expected["full_day_suspended"].fillna(False).astype(bool)
        tradable = expected.loc[~full_day_suspended]
        suspended = expected.loc[full_day_suspended]
        allowed = {"observed_price_complete", "observed_row_missing_price", "missing_row_unresolved"}
        if not set(tradable["coverage_status"].astype(str)).issubset(allowed):
            blockers.append("G0 存在不支持发布的覆盖状态")

    all_expected_dates = set(expected["trade_date"].dt.strftime("%Y-%m-%d"))
    expected_dates = set(tradable["trade_date"].dt.strftime("%Y-%m-%d"))
    suspended_dates = set(suspended["trade_date"].dt.strftime("%Y-%m-%d"))
    candidate_dates = set(candidate["trade_date"].dt.strftime("%Y-%m-%d"))

    comparison_dates = comparison["trade_date"].dt.strftime("%Y-%m-%d")
    comparison_status = comparison.get("raw_close_status", pd.Series(dtype="object")).astype(str)
    status_by_date = pd.Series(comparison_status.to_numpy(), index=comparison_dates)
    verified_delist_dates = {
        pd.Timestamp(value).strftime("%Y-%m-%d")
        for value in verified_delist_dates
        if pd.notna(pd.to_datetime(value, errors="coerce"))
    }
    comparison_date_set = set(comparison_dates)
    extra_comparison_dates = comparison_date_set - all_expected_dates
    incomplete_dates = set(expected.loc[
        expected.get("coverage_status", pd.Series(index=expected.index, dtype="object"))
        .astype(str).isin({"observed_row_missing_price", "missing_row_unresolved"}),
        "trade_date",
    ].dt.strftime("%Y-%m-%d"))
    targeted_partial = candidate_dates < expected_dates
    if targeted_partial:
        unverified_comparison_dates = comparison_date_set - candidate_dates - suspended_dates
        if candidate_dates - comparison_date_set or unverified_comparison_dates:
            blockers.append("部分修复的双源比较日期必须覆盖候选日期，额外日期只能是 G0 全天停牌日")
        if candidate_dates != incomplete_dates:
            blockers.append("部分修复必须覆盖该股票该年度全部 G0 缺失或部分行情日期")
    elif all_expected_dates - comparison_date_set or extra_comparison_dates - verified_delist_dates:
        blockers.append("双源比较日期与 G0 预期交易日不一致")
    verified_extra_dates = extra_comparison_dates & verified_delist_dates
    if verified_extra_dates and not status_by_date.reindex(sorted(verified_extra_dates)).eq("baostock_suspended").all():
        blockers.append("已核验摘牌日的额外双源记录缺少 BaoStock 全天停牌状态")
    source_matched_dates = set(status_by_date.index[status_by_date.eq("matched")])
    publishable_dates = expected_dates & source_matched_dates
    if candidate_dates != publishable_dates:
        blockers.append(
            f"候选日期与双源匹配的 G0 预期日期不一致：候选 {len(candidate_dates)} 日，双源匹配 {len(publishable_dates)} 日"
        )
    if candidate_dates - expected_dates:
        blockers.append(f"候选包含 {len(candidate_dates - expected_dates)} 个 G0 分母外日期")
    unresolved_dates = set(expected.loc[
        expected.get("coverage_status", pd.Series(index=expected.index, dtype="object")).astype(str).eq("missing_row_unresolved"),
        "trade_date",
    ].dt.strftime("%Y-%m-%d"))
    source_resolved_unresolved = len(unresolved_dates & source_matched_dates)
    unresolved_after_sources = len(unresolved_dates - source_matched_dates)
    unresolved_unconfirmed = unresolved_dates - source_matched_dates
    unresolved_without_nontrading_evidence = {
        date for date in unresolved_unconfirmed
        if status_by_date.get(date) != "baostock_suspended"
    }
    unpublished_dates = expected_dates - candidate_dates
    if unresolved_without_nontrading_evidence:
        blockers.append(
            f"G0 有 {len(unresolved_without_nontrading_evidence)} 个缺行交易日既未双源匹配，也无 BaoStock 非交易证据"
        )
    suspended_dates_to_verify = suspended_dates & comparison_date_set if targeted_partial else suspended_dates
    if not status_by_date.reindex(sorted(suspended_dates_to_verify)).eq("baostock_suspended").all():
        blockers.append("G0 全天停牌日缺少 BaoStock 停牌状态证据")

    prices = candidate[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    valid_prices = prices.notna().all(axis=1) & prices.gt(0).all(axis=1)
    valid_prices &= prices["high"].ge(prices["low"])
    valid_prices &= prices["open"].between(prices["low"], prices["high"])
    valid_prices &= prices["close"].between(prices["low"], prices["high"])
    if not valid_prices.all():
        blockers.append(f"候选存在 {int((~valid_prices).sum())} 行非法 OHLC")
    factors = pd.to_numeric(candidate.get("adj_factor"), errors="coerce")
    if factors.isna().any() or factors.le(0).any():
        blockers.append("候选存在缺失或非正复权因子")
    volumes = pd.to_numeric(candidate.get("volume"), errors="coerce")
    if volumes.isna().any() or volumes.le(0).any():
        blockers.append("候选存在缺失或非正成交量")

    price_conflict_rows = 0
    if not stored_rows.empty:
        current = stored_rows.set_index("trade_date")
        proposed = candidate.set_index("trade_date")
        overlap = current.index.intersection(proposed.index)
        if len(overlap):
            current_prices = current.loc[overlap, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
            proposed_prices = proposed.loc[overlap, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
            present = current_prices.notna()
            conflicts = present & ((current_prices - proposed_prices).abs() > HISTORICAL_QFQ_PRICE_TOLERANCE)
            if conflicts.any(axis=None):
                price_conflict_rows = int(conflicts.any(axis=1).sum())
                if not replace_price_conflicts:
                    blockers.append(f"现库已有价格与双源候选冲突 {price_conflict_rows} 日")
                elif not expected_dates.issubset(source_matched_dates):
                    blockers.append("双源未逐日匹配全部 G0 可交易日期，不能覆盖现库价格冲突")

    return {
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "price_conflict_rows": price_conflict_rows,
        "price_conflicts_replacement_eligible": price_conflict_rows if replace_price_conflicts and not blockers else 0,
        "expected_rows": len(expected_dates),
        "candidate_rows": len(candidate),
        "matched_source_rows": int(status_by_date.reindex(sorted(expected_dates)).eq("matched").sum()),
        "g0_unresolved_expected_rows": len(unresolved_dates),
        "source_resolved_expected_rows": source_resolved_unresolved,
        "unresolved_expected_rows": unresolved_after_sources,
        "unpublished_expected_rows": len(unpublished_dates),
        "unpublished_source_status_counts": {
            str(key): int(value)
            for key, value in status_by_date.reindex(sorted(unpublished_dates)).value_counts(dropna=False).items()
        },
        "partial_release": bool(unpublished_dates or unresolved_after_sources),
    }
